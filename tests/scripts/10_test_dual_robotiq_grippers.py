#!/usr/bin/env python3
"""Run small standalone diagnostics on two Robotiq grippers concurrently.

This script orchestrates two instances of
``9_test_single_robotiq_gripper.py``. It does not duplicate the Modbus
implementation. Each gripper must use a different USB-RS485 adapter and an
explicit, stable serial path.

The processes run concurrently, but their individual Modbus transactions and
motion commands are not synchronized by a real-time barrier. Reported
frequency measurements therefore describe each gripper independently while
both devices are active on the host.


Find both adapters under ``/dev/serial/by-id`` and pass their stable paths
explicitly. For example::

    ls -l /dev/serial/by-id/
    export P1_ROBOTIQ_PORT="/dev/serial/by-id/<p1-adapter-id>"
    export P2_ROBOTIQ_PORT="/dev/serial/by-id/<p2-adapter-id>"

Current defaults: 
    export P1_ROBOTIQ_PORT="/dev/serial/by-id/usb-FTDI_USB_TO_RS-485_DAAL8XY5-if00-port0"
    export P1_ROBOTIQ_SERVER_PORT=1235

    export P2_ROBOTIQ_PORT="/dev/serial/by-id/usb-FTDI_USB_TO_RS-485_DAANTFDG-if00-port0"
    export P2_ROBOTIQ_SERVER_PORT=4322

Examples::

    # Read-only FC04 frequency test on both grippers.
    .venv-standalone/bin/python 10_test_dual_robotiq_grippers.py \
        --p1-port "${P1_ROBOTIQ_PORT}" \
        --p2-port "${P2_ROBOTIQ_PORT}" \
        --diagnostic frequency \
        --frequency-hz 100 \
        --frequency-samples 500

    # Read-only preflight, then concurrent open/half/close/open motion.
    .venv-standalone/bin/python 10_test_dual_robotiq_grippers.py \
        --p1-port "${P1_ROBOTIQ_PORT}" \
        --p2-port "${P2_ROBOTIQ_PORT}" \
        --diagnostic movement

Stop Polymetis and every other serial client before running this script. Dual
motion resets and activates both grippers, so calibration can move both sets
of fingers. The movement workspace must be clear around both grippers.
"""

import argparse
import math
import os
import shlex
import signal
import stat
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path


CHILD_SCRIPT = Path(__file__).with_name(
    "9_test_single_robotiq_gripper.py"
)
DIAGNOSTIC_CHOICES = ("read-only", "frequency", "movement", "all")
SUPPORTED_BAUDRATES = frozenset(
    {1200, 2400, 4800, 9600, 19200, 38400, 57600, 115200}
)
PRINT_LOCK = threading.Lock()
STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)


class TerminationRequested(Exception):
    def __init__(self, signum: int):
        super().__init__(signum)
        self.signum = signum


@contextmanager
def forwardable_termination_signals():
    """Convert the first parent stop signal into a cleanup-aware exception."""
    stop_requested = False
    previous_handlers = {
        handled_signal: signal.getsignal(handled_signal)
        for handled_signal in STOP_SIGNALS
    }

    def request_termination(signum: int, _frame: object) -> None:
        nonlocal stop_requested
        if stop_requested:
            return
        stop_requested = True
        raise TerminationRequested(signum)

    try:
        for handled_signal in STOP_SIGNALS:
            signal.signal(handled_signal, request_termination)
        yield
    finally:
        for handled_signal, previous_handler in previous_handlers.items():
            signal.signal(handled_signal, previous_handler)


def ignore_stop_signals() -> None:
    """Prevent repeated stop signals from interrupting child cleanup."""
    for handled_signal in STOP_SIGNALS:
        signal.signal(handled_signal, signal.SIG_IGN)


def parse_integer(value: str) -> int:
    """Parse a decimal or prefixed integer such as ``9`` or ``0x09``."""
    return int(value, 0)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the existing standalone Robotiq diagnostic concurrently "
            "on P1 and P2."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Safety:\n"
            "  frequency is read-only. movement first runs a read-only\n"
            "  preflight, then requires the token MOVE BOTH. Activation and\n"
            "  calibration can move both grippers before the toy sequence.\n"
            "  rFR=0 is minimum physical force, not zero physical force."
        ),
    )
    parser.add_argument(
        "--p1-port",
        required=True,
        help="P1 USB-RS485 path; prefer /dev/serial/by-id/...",
    )
    parser.add_argument(
        "--p2-port",
        required=True,
        help="P2 USB-RS485 path; prefer /dev/serial/by-id/...",
    )
    parser.add_argument(
        "--diagnostic",
        choices=DIAGNOSTIC_CHOICES,
        default="read-only",
        help=(
            "read-only status, concurrent FC04 frequency, concurrent toy "
            "movement, or frequency followed by movement (default: read-only)"
        ),
    )
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument(
        "--p1-slave-id",
        type=parse_integer,
        default=0x09,
        help="P1 Modbus slave ID (default: 0x09)",
    )
    parser.add_argument(
        "--p2-slave-id",
        type=parse_integer,
        default=0x09,
        help="P2 Modbus slave ID (default: 0x09)",
    )
    parser.add_argument("--io-timeout", type=float, default=1.0)
    parser.add_argument(
        "--baseline-samples",
        type=int,
        default=3,
        help="FC04 samples in each baseline health check (default: 3)",
    )
    parser.add_argument("--sample-interval", type=float, default=0.2)
    parser.add_argument("--activation-timeout", type=float, default=5.0)
    parser.add_argument("--motion-timeout", type=float, default=10.0)
    parser.add_argument("--poll-interval", type=float, default=0.02)
    parser.add_argument("--motion-dwell", type=float, default=0.2)
    parser.add_argument(
        "--speed-request",
        type=int,
        default=127,
        help="raw rSP for the movement test (default: 127, near midpoint)",
    )
    parser.add_argument(
        "--mid-position",
        type=int,
        default=128,
        help="raw rPR midpoint in the movement test (default: 128)",
    )
    parser.add_argument(
        "--frequency-samples",
        type=int,
        default=500,
        help="FC04 transactions per gripper (default: 500)",
    )
    parser.add_argument(
        "--frequency-hz",
        type=float,
        default=100.0,
        help="requested FC04 rate per gripper, at most 200 Hz (default: 100)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print child commands without checking or opening serial devices",
    )
    return parser.parse_args(argv)


def validate_arguments(args: argparse.Namespace) -> None:
    if args.p1_port == args.p2_port:
        raise ValueError("--p1-port and --p2-port must be different")
    if args.baudrate not in SUPPORTED_BAUDRATES:
        supported = ", ".join(str(value) for value in sorted(SUPPORTED_BAUDRATES))
        raise ValueError(
            f"--baudrate must be one of {supported}; received {args.baudrate}"
        )
    if not 1 <= args.p1_slave_id <= 247:
        raise ValueError("--p1-slave-id must be between 1 and 247")
    if not 1 <= args.p2_slave_id <= 247:
        raise ValueError("--p2-slave-id must be between 1 and 247")
    if args.baseline_samples < 1:
        raise ValueError("--baseline-samples must be at least 1")
    if args.frequency_samples < 2:
        raise ValueError("--frequency-samples must be at least 2")
    if not 1.0 <= args.frequency_hz <= 200.0:
        raise ValueError("--frequency-hz must be between 1 and 200")
    if not 0 <= args.speed_request <= 255:
        raise ValueError("--speed-request must be between 0 and 255")
    if not 1 <= args.mid_position <= 254:
        raise ValueError("--mid-position must be between 1 and 254")

    time_values = {
        "--io-timeout": args.io_timeout,
        "--sample-interval": args.sample_interval,
        "--activation-timeout": args.activation_timeout,
        "--motion-timeout": args.motion_timeout,
        "--poll-interval": args.poll_interval,
        "--motion-dwell": args.motion_dwell,
    }
    for option, value in time_values.items():
        if not math.isfinite(value):
            raise ValueError(f"{option} must be finite")
    if args.io_timeout <= 0.0:
        raise ValueError("--io-timeout must be positive")
    if args.sample_interval < 0.005:
        raise ValueError("--sample-interval must be at least 0.005 seconds")
    if args.activation_timeout <= 0.0 or args.motion_timeout <= 0.0:
        raise ValueError("activation and motion timeouts must be positive")
    if args.poll_interval < 0.005:
        raise ValueError("--poll-interval must be at least 0.005 seconds")
    if args.poll_interval > args.activation_timeout:
        raise ValueError(
            "--poll-interval must not exceed --activation-timeout; "
            f"received {args.poll_interval} and {args.activation_timeout} seconds"
        )
    if args.motion_dwell < 0.0:
        raise ValueError("--motion-dwell cannot be negative")


def validate_serial_devices(args: argparse.Namespace) -> dict[str, str]:
    """Validate two distinct character devices without opening either one."""
    ports = {
        "P1": str(Path(args.p1_port).expanduser()),
        "P2": str(Path(args.p2_port).expanduser()),
    }
    resolved_paths: dict[str, Path] = {}

    for label, port in ports.items():
        path = Path(port)
        try:
            resolved = path.resolve(strict=True)
        except FileNotFoundError as error:
            raise ValueError(f"{label} serial path does not exist: {port}") from error
        if not stat.S_ISCHR(resolved.stat().st_mode):
            raise ValueError(
                f"{label} serial path is not a character device: {port}"
            )
        if not os.access(path, os.R_OK | os.W_OK):
            raise ValueError(
                f"{label} serial path is not readable and writable: {port}"
            )
        resolved_paths[label] = resolved

        if not str(path).startswith("/dev/serial/by-id/"):
            print_locked(
                f"[DUAL | WARNING] {label} uses {port}; prefer a stable "
                "/dev/serial/by-id/... path.",
                file=sys.stderr,
            )

    if resolved_paths["P1"] == resolved_paths["P2"]:
        raise ValueError(
            "P1 and P2 resolve to the same serial device: "
            f"{resolved_paths['P1']}"
        )
    return ports


def phases_for_diagnostic(diagnostic: str) -> tuple[str, ...]:
    if diagnostic == "read-only":
        return ("read-only",)
    if diagnostic == "frequency":
        return ("frequency",)
    if diagnostic == "movement":
        return ("read-only", "movement")
    if diagnostic == "all":
        return ("frequency", "movement")
    raise ValueError(f"unsupported diagnostic: {diagnostic}")


def build_child_command(
    *,
    port: str,
    slave_id: int,
    phase: str,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(CHILD_SCRIPT),
        "--port",
        port,
        "--baudrate",
        str(args.baudrate),
        "--slave-id",
        str(slave_id),
        "--io-timeout",
        str(args.io_timeout),
        "--num_consecutive_read_only_samples",
        str(args.baseline_samples),
        "--sample-interval",
        str(args.sample_interval),
    ]

    if phase == "frequency":
        command.extend(
            [
                "--diagnostic",
                "fc04-frequency",
                "--frequency-samples",
                str(args.frequency_samples),
                "--frequency-hz",
                str(args.frequency_hz),
            ]
        )
    elif phase == "movement":
        command.extend(
            [
                "--diagnostic",
                "basic-motion",
                "--activation-timeout",
                str(args.activation_timeout),
                "--motion-timeout",
                str(args.motion_timeout),
                "--poll-interval",
                str(args.poll_interval),
                "--motion-dwell",
                str(args.motion_dwell),
                "--speed-request",
                str(args.speed_request),
                "--mid-position",
                str(args.mid_position),
                "--yes",
            ]
        )
    elif phase != "read-only":
        raise ValueError(f"unsupported child phase: {phase}")

    return command


def print_locked(*values: object, **kwargs: object) -> None:
    with PRINT_LOCK:
        try:
            print(*values, **kwargs, flush=True)
        except BrokenPipeError:
            # Output consumers may exit while children still require cleanup.
            pass


def stream_output(label: str, pipe: object) -> None:
    try:
        for line in pipe:
            # Keep draining even after the wrapper's output consumer exits, so
            # a child cannot block on a full stdout pipe.
            print_locked(f"[{label}] {line.rstrip()}")
    finally:
        pipe.close()


def signal_process_group(process: subprocess.Popen, sig: signal.Signals) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, sig)
    except (OSError, ProcessLookupError):
        if process.poll() is None:
            process.send_signal(sig)


def stop_and_reap(
    processes: dict[str, subprocess.Popen],
    *,
    grace_seconds: float,
) -> bool:
    """Interrupt children, escalating only if cleanup does not finish."""
    forced_stop = False
    for process in processes.values():
        signal_process_group(process, signal.SIGINT)

    deadline = time.monotonic() + grace_seconds
    for label, process in processes.items():
        remaining = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
            continue
        except subprocess.TimeoutExpired:
            forced_stop = True
            print_locked(
                f"[DUAL | WARNING] {label} did not finish cleanup after "
                f"{grace_seconds:.1f} s; sending SIGTERM.",
                file=sys.stderr,
            )

        signal_process_group(process, signal.SIGTERM)
        try:
            process.wait(timeout=2.0)
            continue
        except subprocess.TimeoutExpired:
            print_locked(
                f"[DUAL | WARNING] {label} ignored SIGTERM; sending SIGKILL. "
                "Gripper stopping and cleanup were not verified.",
                file=sys.stderr,
            )
        signal_process_group(process, signal.SIGKILL)
        process.wait()
    return forced_stop


def aggregate_exit_codes(
    exit_codes: dict[str, int],
    *,
    parent_interrupted: bool = False,
    forced_stop: bool = False,
) -> int:
    if parent_interrupted:
        return 130
    if forced_stop:
        return 1
    if any(code not in (0, 2) for code in exit_codes.values()):
        return 1
    if any(code == 2 for code in exit_codes.values()):
        return 2
    return 0


def run_dual_phase(
    phase: str,
    commands: dict[str, list[str]],
    *,
    io_timeout: float,
) -> int:
    print_locked(
        f"\n[DUAL | {phase.upper()} | START] Launching P1 and P2 concurrently."
    )
    processes: dict[str, subprocess.Popen] = {}
    output_threads: list[threading.Thread] = []
    parent_interrupted = False
    termination_exit_code = None
    forced_stop = False

    with forwardable_termination_signals():
        try:
            for label in ("P1", "P2"):
                process = subprocess.Popen(
                    commands[label],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    start_new_session=True,
                )
                processes[label] = process
                output_thread = threading.Thread(
                    target=stream_output,
                    args=(label, process.stdout),
                    daemon=True,
                )
                output_thread.start()
                output_threads.append(output_thread)

            while True:
                exit_codes = {
                    label: process.poll()
                    for label, process in processes.items()
                }
                if all(code is not None for code in exit_codes.values()):
                    break

                hard_failure = any(
                    code is not None and code not in (0, 2)
                    for code in exit_codes.values()
                )
                if hard_failure:
                    ignore_stop_signals()
                    print_locked(
                        "[DUAL | WARNING] One child failed; interrupting its "
                        "live peer so the child cleanup path can run.",
                        file=sys.stderr,
                    )
                    forced_stop = stop_and_reap(
                        processes,
                        grace_seconds=max(5.0, 2.0 * io_timeout + 2.0),
                    )
                    break
                time.sleep(0.05)
        except KeyboardInterrupt:
            parent_interrupted = True
            ignore_stop_signals()
            print_locked(
                "\n[DUAL | ABORTED] Interrupting both child diagnostics.",
                file=sys.stderr,
            )
            forced_stop = stop_and_reap(
                processes,
                grace_seconds=max(5.0, 2.0 * io_timeout + 2.0),
            )
        except TerminationRequested as error:
            termination_exit_code = 128 + error.signum
            ignore_stop_signals()
            print_locked(
                f"\n[DUAL | ABORTED] Received signal {error.signum}; "
                "interrupting both child diagnostics.",
                file=sys.stderr,
            )
            forced_stop = stop_and_reap(
                processes,
                grace_seconds=max(5.0, 2.0 * io_timeout + 2.0),
            )
        except Exception as error:
            ignore_stop_signals()
            print_locked(
                f"[DUAL | FAIL] Could not run both diagnostics: {error}",
                file=sys.stderr,
            )
            forced_stop = stop_and_reap(
                processes,
                grace_seconds=max(5.0, 2.0 * io_timeout + 2.0),
            )
            for output_thread in output_threads:
                output_thread.join()
            return 1

    if not parent_interrupted:
        for process in processes.values():
            process.wait()
    for output_thread in output_threads:
        output_thread.join()

    exit_codes = {
        label: process.returncode
        for label, process in processes.items()
    }
    if termination_exit_code is not None:
        result = termination_exit_code
    else:
        result = aggregate_exit_codes(
            exit_codes,
            parent_interrupted=parent_interrupted,
            forced_stop=forced_stop,
        )
    print_locked(
        f"[DUAL | {phase.upper()} | RESULT] "
        f"P1 exit={exit_codes.get('P1')}, P2 exit={exit_codes.get('P2')}."
    )
    return result


def confirm_dual_motion(args: argparse.Namespace, ports: dict[str, str]) -> bool:
    print_locked("\n[DUAL | MOTION SAFETY | START]")
    print_locked(
        "Both grippers will reset and activate concurrently; calibration can "
        "move both sets of fingers."
    )
    print_locked(
        "Toy sequence per gripper: raw rPR "
        f"0 -> {args.mid_position} -> 255 -> 0, "
        f"rSP={args.speed_request}, rFR=0 (minimum-force mode)."
    )
    print_locked(f"P1: {ports['P1']}")
    print_locked(f"P2: {ports['P2']}")
    print_locked(
        "Clear both complete finger sweeps of hands, objects, cables, and tools."
    )
    try:
        answer = input("Type MOVE BOTH to continue: ").strip()
    except EOFError:
        return False
    return answer == "MOVE BOTH"


def build_phase_commands(
    phase: str,
    ports: dict[str, str],
    args: argparse.Namespace,
) -> dict[str, list[str]]:
    return {
        "P1": build_child_command(
            port=ports["P1"],
            slave_id=args.p1_slave_id,
            phase=phase,
            args=args,
        ),
        "P2": build_child_command(
            port=ports["P2"],
            slave_id=args.p2_slave_id,
            phase=phase,
            args=args,
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        validate_arguments(args)
        if not CHILD_SCRIPT.is_file():
            raise ValueError(f"child diagnostic is missing: {CHILD_SCRIPT}")

        if args.dry_run:
            ports = {
                "P1": str(Path(args.p1_port).expanduser()),
                "P2": str(Path(args.p2_port).expanduser()),
            }
            if Path(ports["P1"]).resolve(strict=False) == Path(
                ports["P2"]
            ).resolve(strict=False):
                raise ValueError("P1 and P2 resolve to the same serial path")
        else:
            ports = validate_serial_devices(args)
    except (OSError, ValueError) as error:
        print(f"[DUAL | VALIDATION FAIL] {error}", file=sys.stderr)
        return 1

    phases = phases_for_diagnostic(args.diagnostic)
    if args.dry_run:
        print("[DUAL | DRY RUN] No serial device will be opened.")
        for phase in phases:
            for label, command in build_phase_commands(
                phase, ports, args
            ).items():
                print(f"[{label} | {phase}] {shlex.join(command)}")
        return 0

    for phase in phases:
        if phase == "movement":
            try:
                motion_confirmed = confirm_dual_motion(args, ports)
            except KeyboardInterrupt:
                print(
                    "\n[DUAL | MOTION SAFETY | ABORTED] Interrupted before "
                    "the motion diagnostic started.",
                    file=sys.stderr,
                )
                return 130
            if not motion_confirmed:
                print(
                    "[DUAL | MOTION SAFETY | ABORTED] Confirmation did not "
                    "match; no motion diagnostic was started.",
                    file=sys.stderr,
                )
                return 1

        result = run_dual_phase(
            phase,
            build_phase_commands(phase, ports, args),
            io_timeout=args.io_timeout,
        )
        if result != 0:
            return result

    print("\n[DUAL | PASS] Every requested dual-gripper phase passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

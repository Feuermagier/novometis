"""Standalone Robotiq 2F Modbus RTU hardware diagnostics.

This script talks directly to a Robotiq 2F-85/2F-140 through PyModbus so its results can later be compared with
or incorporated into another real-world codebase.

Register layout and examples: Robotiq 2F manual, pages 31-39 and 54-57:
https://blog.robotiq.com/hubfs/support-files/2F-85_2F-140_General_PDF_20210623.pdf

With no diagnostic selector, the script only opens the serial connection,
collects FC04 status samples, decodes them, checks baseline health, and closes
the connection. Exactly one optional diagnostic can be selected per run.

Find the adapter under ``/dev/serial/by-id`` and pass that stable path
explicitly. For example::

    ls -l /dev/serial/by-id/
    export ROBOTIQ_PORT="/dev/serial/by-id/<adapter-id>"

Current defaults: 
    export P1_ROBOTIQ_PORT="/dev/serial/by-id/usb-FTDI_USB_TO_RS-485_DAAL8XY5-if00-port0"
    export P1_ROBOTIQ_SERVER_PORT=1235

    export P2_ROBOTIQ_PORT="/dev/serial/by-id/usb-FTDI_USB_TO_RS-485_DAANTFDG-if00-port0"
    export P2_ROBOTIQ_SERVER_PORT=4322

Examples (rPR, rSP, and rFR are raw values from 0 to 255):

    # 1. Read-only: no writes or motion, only read three FC04 samples 0.2 s apart.
    # Decode every field, the final state must be fault-free and reset or ready.
    python 9_test_single_robotiq_gripper.py \
        --port "$ROBOTIQ_PORT" \
        --slave-id 0x09 \
        --num_consecutive_read_only_samples 3 \
        --sample-interval 0.2

    # 2. Activation: reset and prepare the gripper for use.
    # The fingers move during calibration; the gripper finishes ready.
    python 9_test_single_robotiq_gripper.py \
        --port "$ROBOTIQ_PORT" \
        --diagnostic activation \
        --activation-timeout 5 \
        --poll-interval 0.1
    # Short alias: --activate

    # 3. Basic motion: after calibration, slowly open fully,
    # move halfway closed, close fully, then reopen without touching an object.
    python 9_test_single_robotiq_gripper.py \
        --port "$ROBOTIQ_PORT" \
        --diagnostic basic-motion \
        --mid-position 128 \
        --speed-request 0 \
        --motion-timeout 10

    # 4. Position tracking: slowly move through several opening widths
    # in both directions and check that reported positions follow commands.
    python 9_test_single_robotiq_gripper.py \
        --port "$ROBOTIQ_PORT" \
        --diagnostic position \
        --position-requests 0,64,128,192,128,64,0 \
        --speed-request 0 \
        --position-tolerance 10 \
        --endpoint-tolerance 30

    # 5. Speed comparison: repeat the same closing movement at three
    # speed settings, returning to the start between trials and opening at end.
    # Compare measured durations; this is not calibrated physical speed.
    python 9_test_single_robotiq_gripper.py \
        --port "$ROBOTIQ_PORT" \
        --diagnostic speed \
        --speed-requests 0,64,128 \
        --speed-repetitions 3 \
        --poll-interval 0.02 \
        --motion-dwell 0.2

    # 6. Force request (object required): first the gripper is activated, then the script pauses
    # to put an object between the grippers. Never use a hand-held object or enter the pinch zone.
    # Finally, it tests three force settings and reopens, physical force is not measured.
    python 9_test_single_robotiq_gripper.py \
        --port "$ROBOTIQ_PORT" \
        --diagnostic force-request \
        --force-requests 0,32,64 \
        --speed-request 0 \
        --contact-test-fixture-ready

    # 7. Object detection (object required): first the gripper is activated, then the script pauses
    # to put an object between the grippers.
    # Close slowly, verify contact stays detected for three readings, then reopen.
    # Automatic re-grasp is disabled so the observed contact state stays stable.
    python 9_test_single_robotiq_gripper.py \
        --port "$ROBOTIQ_PORT" \
        --diagnostic object-detection \
        --force-request 0 \
        --speed-request 0 \
        --num_consecutive_read_only_samples 3 \
        --sample-interval 0.2 \
        --contact-test-fixture-ready

    # 8. Motion-state trace: perform one slow closing movement while
    # recording position, object detection, current, and faults; then reopen.
    python 9_test_single_robotiq_gripper.py \
        --port "$ROBOTIQ_PORT" \
        --diagnostic motion-state \
        --motion-trace-interval 0.02 \
        --motion-timeout 10

    # 9. Read-frequency test: keep the gripper still and repeatedly read status.
    # Report communication speed and reliability, not controller frequency.
    python 9_test_single_robotiq_gripper.py \
        --port "$ROBOTIQ_PORT" \
        --diagnostic fc04-frequency \
        --frequency-samples 100 \
        --frequency-hz 50

    # 10. Write-frequency test: activate/open, then repeatedly resend open.
    # No repeated movement is intended; report command acknowledgement rate.
    python 9_test_single_robotiq_gripper.py \
        --port "$ROBOTIQ_PORT" \
        --diagnostic fc16-frequency \
        --frequency-samples 100 \
        --frequency-hz 50

    # Optional motion trace: print every state sample from another motion test.
    # This changes only the output, not the commanded movement.
    python 9_test_single_robotiq_gripper.py \
        --port "$ROBOTIQ_PORT" \
        --diagnostic position \
        --trace-motion

Position results use the raw encoder gPO, not a measured aperture. Speed
results use host-observed completion timing, not calibrated physical velocity.
gCU is sampled motor current, not gripping force; physical force validation
requires an external calibrated load cell.

Run motion only with the gripper firmly mounted and visible. Keep the workspace
clear unless running a separately confirmed contact diagnostic, and be ready
to remove power if motion is unexpected.

Frequency diagnostics require the selected FTDI adapter's Linux
``latency_timer`` to be configured to 1. The script verifies this before
opening the serial port; ask the system administrator if it is not configured.

Runtime dependency: ``python -m pip install "pymodbus>=3,<4" pyserial``.
"""


import argparse
import inspect
import math
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any


OUTPUT_REGISTER_ADDRESS = 0x03E8
INPUT_REGISTER_ADDRESS = 0x07D0
REGISTER_COUNT = 3
SUPPORTED_BAUDRATES = frozenset(
    {1200, 2400, 4800, 9600, 19200, 38400, 57600, 115200}
)
MIN_REQUEST_INTERVAL_SECONDS = 0.005

_last_request_started_at: float | None = None

DIAGNOSTIC_CHOICES = (
    "activation",
    "basic-motion",
    "position",
    "speed",
    "force-request",
    "object-detection",
    "motion-state",
    "fc04-frequency",
    "fc16-frequency",
)
DEFAULT_POSITION_REQUESTS = (0, 64, 128, 192, 128, 64, 0)
DEFAULT_SPEED_REQUESTS = (0, 64, 128)
DEFAULT_FORCE_REQUESTS = (0, 32, 64)
SPEED_TEST_START_POSITION = 32
SPEED_TEST_TARGET_POSITION = 224

FAULT_DESCRIPTIONS = {
    0x00: "no fault",
    0x05: "action delayed; activation must complete first",
    0x07: "activation bit is not set",
    0x08: "maximum operating temperature exceeded",
    0x09: "no communication for at least one second",
    0x0A: "operating voltage below minimum",
    0x0B: "automatic release in progress",
    0x0C: "internal fault",
    0x0D: "activation fault",
    0x0E: "overcurrent",
    0x0F: "automatic release completed",
}

GRIPPER_STATE_DESCRIPTIONS = {
    0: "reset",
    1: "activation in progress",
    2: "reserved",
    3: "activation complete",
}

OBJECT_STATE_DESCRIPTIONS = {
    0: "fingers moving",
    1: "contact while opening",
    2: "contact while closing",
    3: "requested position reached",
}

def read_usb_serial_latency_timer(port: str) -> int | None:
    """Return the Linux USB-serial latency timer when it is available."""
    device_name = Path(port).resolve().name
    timer_path = (
        Path("/sys/bus/usb-serial/devices")
        / device_name
        / "latency_timer"
    )

    if not timer_path.exists():
        return None

    try:
        return int(timer_path.read_text().strip())
    except (OSError, ValueError):
        return None


def parse_integer(value: str) -> int:
    """Parse a decimal or prefixed integer such as ``9`` or ``0x09``."""
    return int(value, 0)


def parse_raw_request_values(value: str) -> tuple[int, ...]:
    """Parse comma-separated raw requests in decimal or hexadecimal."""
    try:
        values = tuple(
            int(item.strip(), 0)
            for item in value.split(",")
            if item.strip()
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"invalid comma-separated raw request list: {value}"
        ) from error

    if not values:
        raise argparse.ArgumentTypeError(
            "at least one raw request value is required"
        )
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone Robotiq 2F Modbus hardware diagnostic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Safety:\n"
            "  With no diagnostic selector, the script is read-only. Every\n"
            "  motion diagnostic performs activation calibration first.\n"
            "  Force and object tests require a fixture that can be safely\n"
            "  positioned after activation and cannot be run with --yes."
        ),
    )
    parser.add_argument(
        "--port",
        required=True,
        help=(
            "USB-RS485 serial port; use a stable /dev/serial/by-id/... path"
        ),
    )
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument(
        "--slave-id",
        type=parse_integer,
        default=0x09,
        help="Modbus slave ID in decimal or hexadecimal (default: 0x09)",
    )
    parser.add_argument(
        "--io-timeout",
        type=float,
        default=1.0,
        help="Timeout for one Modbus transaction in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--num_consecutive_read_only_samples",
        type=int,
        default=3,
        help="Number of consecutive baseline status samples (default: 3)",
    )
    parser.add_argument("--sample-interval", type=float, default=0.2)
    parser.add_argument("--activation-timeout", type=float, default=5.0)
    parser.add_argument("--motion-timeout", type=float, default=10.0)
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.1,
        help="Motion status-polling interval; minimum 0.005 s",
    )
    parser.add_argument(
        "--motion-dwell",
        type=float,
        default=0.2,
        help="Pause between diagnostic motions in seconds (default: 0.2)",
    )

    diagnostic_group = parser.add_mutually_exclusive_group()
    diagnostic_group.add_argument(
        "--diagnostic",
        choices=DIAGNOSTIC_CHOICES,
        help=(
            "Run exactly one independent diagnostic after baseline checks; "
            "omit for read-only operation"
        ),
    )
    diagnostic_group.add_argument(
        "--activate",
        dest="diagnostic",
        action="store_const",
        const="activation",
        help="Alias for --diagnostic activation",
    )
    diagnostic_group.add_argument(
        "--control",
        dest="diagnostic",
        action="store_const",
        const="basic-motion",
        help="Alias for --diagnostic basic-motion",
    )
    parser.set_defaults(diagnostic="read-only")

    parser.add_argument(
        "--speed-request",
        type=int,
        default=0,
        help="Fixed raw rSP for non-speed diagnostics (default: 0)",
    )
    parser.add_argument(
        "--force-request",
        type=int,
        default=0,
        help=(
            "Fixed raw rFR for object detection; stable-state mode requires "
            "0 to disable automatic re-grasp (default: 0)"
        ),
    )
    parser.add_argument(
        "--mid-position",
        type=int,
        default=128,
        help="Raw midpoint for the basic-motion diagnostic (default: 128)",
    )
    parser.add_argument(
        "--position-requests",
        type=parse_raw_request_values,
        default=DEFAULT_POSITION_REQUESTS,
        help=(
            "Comma-separated rPR values for the position diagnostic "
            "(default: 0,64,128,192,128,64,0)"
        ),
    )
    parser.add_argument(
        "--speed-requests",
        type=parse_raw_request_values,
        default=DEFAULT_SPEED_REQUESTS,
        help=(
            "Comma-separated rSP values for identical closing strokes "
            "(default: 0,64,128)"
        ),
    )
    parser.add_argument(
        "--force-requests",
        type=parse_raw_request_values,
        default=DEFAULT_FORCE_REQUESTS,
        help=(
            "Comma-separated rFR values for secured-object contact trials "
            "(default: 0,32,64)"
        ),
    )
    parser.add_argument(
        "--speed-repetitions",
        type=int,
        default=1,
        help="Measured closing strokes per rSP value (default: 1)",
    )
    parser.add_argument(
        "--position-tolerance",
        type=int,
        default=10,
        help="Allowed gPO error for an interior target (default: 10)",
    )
    parser.add_argument(
        "--endpoint-tolerance",
        type=int,
        default=30,
        help="Allowed gPO error for target 0 or 255 (default: 30)",
    )
    parser.add_argument(
        "--trace-motion",
        action="store_true",
        help="Print all decoded FC04 samples collected during each motion",
    )
    parser.add_argument(
        "--motion-trace-interval",
        type=float,
        default=0.02,
        help="Polling interval for the motion-state diagnostic (default: 0.02)",
    )
    parser.add_argument(
        "--contact-test-fixture-ready",
        action="store_true",
        help=(
            "Declare that a rigid, secured fixture can be positioned after "
            "activation for a force-request or object-detection diagnostic"
        ),
    )
    parser.add_argument(
        "--frequency-samples",
        type=int,
        default=100,
        help="Transactions for an FC04/FC16 frequency diagnostic (default: 100)",
    )
    parser.add_argument(
        "--frequency-hz",
        type=float,
        default=50.0,
        help="Requested FC04/FC16 frequency, at most 200 Hz (default: 50)",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip typed confirmation for non-contact motion diagnostics",
    )
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    if not args.port or args.port != args.port.strip():
        raise ValueError(
            f"--port must be a non-empty path without surrounding whitespace; "
            f"received {args.port!r}"
        )
    if not 1 <= args.slave_id <= 247:
        raise ValueError("--slave-id must be between 1 and 247")
    if args.baudrate not in SUPPORTED_BAUDRATES:
        supported = ", ".join(str(value) for value in sorted(SUPPORTED_BAUDRATES))
        raise ValueError(
            f"--baudrate must be one of {supported}; received {args.baudrate}"
        )
    time_values = {
        "--io-timeout": args.io_timeout,
        "--sample-interval": args.sample_interval,
        "--activation-timeout": args.activation_timeout,
        "--motion-timeout": args.motion_timeout,
        "--poll-interval": args.poll_interval,
        "--motion-dwell": args.motion_dwell,
        "--motion-trace-interval": args.motion_trace_interval,
        "--frequency-hz": args.frequency_hz,
    }
    for option, value in time_values.items():
        if not math.isfinite(value):
            raise ValueError(f"{option} must be finite")

    if args.io_timeout <= 0:
        raise ValueError("--io-timeout must be positive")
    if args.num_consecutive_read_only_samples < 1:
        raise ValueError(
            "--num_consecutive_read_only_samples must be at least 1"
        )
    if args.sample_interval < 0.005:
        raise ValueError(
            "--sample-interval must be at least 0.005 seconds"
        )
    if args.activation_timeout <= 0 or args.motion_timeout <= 0:
        raise ValueError("timeouts must be positive")
    if args.poll_interval < 0.005:
        raise ValueError("--poll-interval must be at least 0.005 seconds")
    if args.poll_interval > args.activation_timeout:
        raise ValueError(
            "--poll-interval must not exceed --activation-timeout; "
            f"received {args.poll_interval} and {args.activation_timeout} seconds"
        )
    if args.motion_trace_interval < 0.005:
        raise ValueError(
            "--motion-trace-interval must be at least 0.005 seconds"
        )
    if args.motion_dwell < 0:
        raise ValueError("--motion-dwell cannot be negative")
    if not 0 <= args.speed_request <= 255:
        raise ValueError("--speed-request must be between 0 and 255")
    if not 0 <= args.force_request <= 255:
        raise ValueError("--force-request must be between 0 and 255")
    if not 1 <= args.mid_position <= 254:
        raise ValueError("--mid-position must be between 1 and 254")
    if not 0 <= args.position_tolerance < 255:
        raise ValueError("--position-tolerance must be between 0 and 254")
    if not 0 <= args.endpoint_tolerance < 255:
        raise ValueError("--endpoint-tolerance must be between 0 and 254")
    if args.speed_repetitions < 1:
        raise ValueError("--speed-repetitions must be at least 1")
    if args.frequency_samples < 2:
        raise ValueError("--frequency-samples must be at least 2")
    if not 1.0 <= args.frequency_hz <= 200.0:
        raise ValueError("--frequency-hz must be between 1 and 200")

    request_lists = {
        "--position-requests": args.position_requests,
        "--speed-requests": args.speed_requests,
        "--force-requests": args.force_requests,
    }
    for option, values in request_lists.items():
        if any(not 0 <= value <= 255 for value in values):
            raise ValueError(f"{option} values must be between 0 and 255")

    if (
        args.diagnostic == "position"
        and len(set(args.position_requests)) < 2
    ):
        raise ValueError(
            "the position diagnostic requires at least two distinct rPR "
            "values"
        )
    if (
        args.diagnostic == "speed"
        and len(set(args.speed_requests)) < 2
    ):
        raise ValueError(
            "the speed diagnostic requires at least two distinct rSP values"
        )
    if (
        args.diagnostic == "force-request"
        and len(set(args.force_requests)) < 2
    ):
        raise ValueError(
            "the force-request diagnostic requires at least two distinct "
            "rFR values"
        )

    contact_diagnostics = {"force-request", "object-detection"}
    is_contact_diagnostic = args.diagnostic in contact_diagnostics
    if is_contact_diagnostic and not args.contact_test_fixture_ready:
        raise ValueError(
            "contact diagnostics require --contact-test-fixture-ready"
        )
    if (
        args.contact_test_fixture_ready
        and not is_contact_diagnostic
    ):
        raise ValueError(
            "--contact-test-fixture-ready is valid only for a "
            "force-request or object-detection diagnostic"
        )
    if is_contact_diagnostic and args.yes:
        raise ValueError(
            "--yes cannot bypass confirmation for a contact diagnostic"
        )
    if args.diagnostic == "object-detection" and args.force_request != 0:
        raise ValueError(
            "the object-detection diagnostic requires --force-request 0 "
            "so automatic re-grasp cannot invalidate the stable-state check"
        )

    motion_diagnostics = {
        "basic-motion",
        "position",
        "speed",
        "force-request",
        "object-detection",
        "motion-state",
    }
    if args.trace_motion and args.diagnostic not in motion_diagnostics:
        raise ValueError(
            "--trace-motion requires a motion diagnostic"
        )


def slave_address_kwargs(
    method: Callable[..., Any],
    slave_id: int,
) -> dict[str, int]:
    """Return the PyModbus keyword argument for the slave address."""
    parameters = inspect.signature(method).parameters
    keyword = "device_id" if "device_id" in parameters else "slave"
    return {keyword: slave_id}


def wait_for_request_interval() -> float:
    """Wait for Robotiq's minimum interval and return the request start time."""
    global _last_request_started_at

    now = time.monotonic()
    if _last_request_started_at is not None:
        delay = MIN_REQUEST_INTERVAL_SECONDS - (
            now - _last_request_started_at
        )
        if delay > 0.0:
            time.sleep(delay)

    _last_request_started_at = time.monotonic()
    return _last_request_started_at


def encode_action_request(*, activate: bool, go_to: bool) -> int:
    """Return the complete 16-bit ACTION REQUEST register value."""
    # Robotiq manual p. 32: rACT is bit 0 and rGTO is bit 3 of the ACTION
    # REQUEST byte. Pages 54-57 show the resulting reset, activation, and
    # go-to Modbus register values.
    if go_to and not activate:
        raise ValueError("a go-to request requires rACT=1")

    if activate and go_to:
        action_request_byte = 0x09  # rGTO bit 3 + rACT bit 0.
    elif activate:
        action_request_byte = 0x01  # rACT bit 0.
    else:
        action_request_byte = 0x00  # Clear all action bits for reset.

    # Register 0x03E8 stores ACTION REQUEST in its high byte. Its low options
    # byte remains clear, giving 0x0000 (reset), 0x0100 (activate), or 0x0900
    # (go-to).
    return action_request_byte << 8


def decode_status_from_registers(
    register_values: list[int],
) -> dict[str, int]:
    """Decode the three FC04 registers beginning at address 0x07D0."""
    # Robotiq manual pp. 36-39 defines the six returned status bytes.
    if len(register_values) != REGISTER_COUNT:
        raise ValueError(
            f"expected {REGISTER_COUNT} status registers, "
            f"received {register_values}"
        )

    for register_value in register_values:
        if not isinstance(register_value, int) or not 0 <= register_value <= 0xFFFF:
            raise ValueError(
                "status registers must be 16-bit integers: "
                f"{register_values}"
            )

    (
        status_register,
        fault_request_register,
        position_current_register,
    ) = register_values

    status_byte = (status_register >> 8) & 0xFF
    fault_byte = (fault_request_register >> 8) & 0xFF

    if status_register & 0x00FF:
        raise ValueError(
            "reserved low byte of status register 0x07D0 is nonzero: "
            f"0x{status_register:04X}"
        )
    if status_byte & 0x06:
        raise ValueError(
            "reserved GRIPPER STATUS bits 1-2 are nonzero: "
            f"0x{status_byte:02X}"
        )

    status = {
        "gACT": status_byte & 0x01,
        "gGTO": (status_byte >> 3) & 0x01,
        "gSTA": (status_byte >> 4) & 0x03,
        "gOBJ": (status_byte >> 6) & 0x03,
        "gFLT": fault_byte & 0x0F,
        "kFLT": (fault_byte >> 4) & 0x0F,
        "gPR": fault_request_register & 0xFF,
        "gPO": (position_current_register >> 8) & 0xFF,
        "gCU": position_current_register & 0xFF,
    }
    if status["gSTA"] == 2:
        raise ValueError("gSTA returned reserved state 2")
    return status


def format_status(status: dict[str, int]) -> str:
    object_description = (
        OBJECT_STATE_DESCRIPTIONS[status["gOBJ"]]
        if status["gGTO"] == 1
        else "ignored while gGTO=0"
    )
    fault = status["gFLT"]
    return (
        f"gACT={status['gACT']} "
        f"gGTO={status['gGTO']} "
        f"gSTA={status['gSTA']}"
        f"({GRIPPER_STATE_DESCRIPTIONS[status['gSTA']]}) "
        f"gOBJ={status['gOBJ']}({object_description}) "
        f"gFLT=0x{fault:02X}({FAULT_DESCRIPTIONS.get(fault, 'unknown')}) "
        f"kFLT=0x{status['kFLT']:X} "
        f"gPR={status['gPR']} "
        f"gPO={status['gPO']} "
        f"gCU={status['gCU']} (~{status['gCU'] * 10} mA)"
    )


def write_command_registers(
    client: Any,
    slave_id: int,
    registers: list[int],
    description: str,
    *,
    pace_request: bool = True,
) -> None:
    if len(registers) != REGISTER_COUNT or any(
        not isinstance(value, int) or not 0 <= value <= 0xFFFF
        for value in registers
    ):
        raise ValueError(
            f"command must contain exactly {REGISTER_COUNT} 16-bit "
            f"registers: {registers}"
        )

    if pace_request:
        wait_for_request_interval()

    response = client.write_registers(
        address=OUTPUT_REGISTER_ADDRESS,
        values=registers,
        **slave_address_kwargs(client.write_registers, slave_id),
    )
    if response is None or response.isError():
        values = ", ".join(f"0x{value:04X}" for value in registers)
        raise RuntimeError(
            f"FC16 {description} write failed for [{values}]: {response}"
        )

    response_address = getattr(response, "address", None)
    response_count = getattr(response, "count", None)
    if (
        response_address != OUTPUT_REGISTER_ADDRESS
        or response_count != REGISTER_COUNT
    ):
        raise RuntimeError(
            f"FC16 {description} returned unexpected address/count: {response}"
        )


def activate_gripper(
    client: Any,
    *,
    timeout: float,
    poll_interval: float,
    slave_id: int,
) -> dict[str, int]:
    """Reset and activate the gripper, waiting for both acknowledgements."""
    print(
        "\n[PHASE 5 | ACTIVATE GRIPPER | START] "
        f"Resetting and activating Modbus slave 0x{slave_id:02X}; "
        f"timeout={timeout:.1f} s per stage."
    )

    # Clear the complete command block: action/options, options/position,
    # and speed/force.
    reset_registers = [
        encode_action_request(activate=False, go_to=False),
        0x0000,
        0x0000,
    ]
    print(
        "[PHASE 5 | ACTIVATE GRIPPER | RESET COMMAND] "
        "Clearing rACT and the remaining command registers."
    )
    write_command_registers(client, slave_id, reset_registers, "reset")

    reset_deadline = time.monotonic() + timeout
    reset_status = None
    last_read_error = None

    while time.monotonic() < reset_deadline:
        remaining_time = reset_deadline - time.monotonic()
        time.sleep(min(poll_interval, max(remaining_time, 0.0)))

        try:
            register_values = read_status_registers(client, slave_id)
            reset_status = decode_status_from_registers(register_values)
        except (RuntimeError, ValueError) as error:
            last_read_error = error
            continue

        last_read_error = None

        if reset_status["gACT"] == 0 and reset_status["gSTA"] == 0:
            break
    else:
        raise RuntimeError(
            "[PHASE 5 | ACTIVATE GRIPPER | FAIL] "
            f"Reset was not acknowledged within {timeout:.1f} s; "
            f"last status={reset_status}, last read error={last_read_error}"
        )

    print(
        "[PHASE 5 | ACTIVATE GRIPPER | RESET ACKNOWLEDGED] "
        f"{format_status(reset_status)}"
    )

    # Set only rACT; keep position, speed, and force cleared.
    activation_registers = [
        encode_action_request(activate=True, go_to=False),
        0x0000,
        0x0000,
    ]
    print(
        "[PHASE 5 | ACTIVATE GRIPPER | ACTIVATION COMMAND] "
        "Setting rACT to begin automatic calibration."
    )
    write_command_registers(client, slave_id, activation_registers, "activation")

    activation_deadline = time.monotonic() + timeout
    activation_status = None
    last_read_error = None

    while time.monotonic() < activation_deadline:
        remaining_time = activation_deadline - time.monotonic()
        time.sleep(min(poll_interval, max(remaining_time, 0.0)))

        try:
            register_values = read_status_registers(client, slave_id)
            activation_status = decode_status_from_registers(register_values)
        except (RuntimeError, ValueError) as error:
            last_read_error = error
            continue

        last_read_error = None

        gripper_fault = activation_status["gFLT"]
        controller_fault = activation_status["kFLT"]

        if controller_fault != 0:
            raise RuntimeError(
                "[PHASE 5 | ACTIVATE GRIPPER | FAIL] "
                f"Controller fault kFLT=0x{controller_fault:X}; "
                f"status={activation_status}"
            )

        if gripper_fault >= 0x08:
            raise RuntimeError(
                "[PHASE 5 | ACTIVATE GRIPPER | FAIL] "
                f"Gripper fault gFLT=0x{gripper_fault:02X}: "
                f"{FAULT_DESCRIPTIONS.get(gripper_fault, 'unknown')}"
            )

        if (
            activation_status["gACT"] == 1
            and activation_status["gSTA"] == 3
            and gripper_fault == 0
        ):
            print(
                "[PHASE 5 | ACTIVATE GRIPPER | FINISHED] "
                f"Activation completed successfully: "
                f"{format_status(activation_status)}"
            )
            return activation_status

    raise RuntimeError(
        "[PHASE 5 | ACTIVATE GRIPPER | FAIL] "
        f"Activation did not complete within {timeout:.1f} s; "
        f"last status={activation_status}, last read error={last_read_error}"
    )


def execute_motion(
    client: Any,
    *,
    diagnostic_name: str,
    label: str,
    command_number: int,
    command_count: int,
    starting_position: int,
    target_position: int,
    speed_request: int,
    force_request: int,
    allow_contact: bool,
    position_tolerance: int,
    timeout: float,
    poll_interval: float,
    slave_id: int,
    trace_motion: bool,
) -> tuple[dict[str, int], list[tuple[float, dict[str, int]]]]:
    """Send one go-to command and return its status and sampled trajectory."""
    if target_position < starting_position:
        movement_direction = "opening"
        expected_contact_state = 1
    elif target_position > starting_position:
        movement_direction = "closing"
        expected_contact_state = 2
    else:
        movement_direction = "no position change"
        expected_contact_state = None

    speed_force_register = (speed_request << 8) | force_request
    registers = [
        encode_action_request(activate=True, go_to=True),
        target_position,
        speed_force_register,
    ]
    message_prefix = f"[PHASE 6 | {diagnostic_name}"

    print(
        f"\n{message_prefix} | COMMAND {command_number}/{command_count}] "
        f"{label}: direction={movement_direction}, "
        f"start gPO={starting_position}, rPR={target_position}, "
        f"rSP={speed_request}, rFR={force_request}."
    )

    write_started_at = time.monotonic()
    write_command_registers(client, slave_id, registers, label)
    command_started_at = time.monotonic()

    register_text = " ".join(f"0x{value:04X}" for value in registers)
    print(
        f"{message_prefix} | WRITE {command_number}/{command_count} "
        f"ACKNOWLEDGED] FC16 acknowledged [{register_text}]."
    )

    write_duration = command_started_at - write_started_at
    deadline = command_started_at + timeout
    last_status = None
    last_read_error = None
    last_recovered_read_error = None
    successful_poll_count = 0
    failed_poll_count = 0
    motion_samples: list[tuple[float, dict[str, int]]] = []

    while time.monotonic() < deadline:
        remaining_time = deadline - time.monotonic()
        time.sleep(min(poll_interval, max(remaining_time, 0.0)))

        try:
            register_values = read_status_registers(client, slave_id)
            last_status = decode_status_from_registers(register_values)
        except RuntimeError as error:
            last_read_error = error
            last_recovered_read_error = error
            failed_poll_count += 1
            continue

        last_read_error = None
        successful_poll_count += 1
        elapsed_seconds = time.monotonic() - command_started_at
        sampled_status = last_status.copy()
        motion_samples.append((elapsed_seconds, sampled_status))

        if last_status["kFLT"] != 0:
            raise RuntimeError(
                f"{message_prefix} | FAIL] {label} reported controller fault "
                f"kFLT=0x{last_status['kFLT']:X}; "
                f"status={format_status(last_status)}."
            )

        if last_status["gFLT"] != 0:
            fault = last_status["gFLT"]
            raise RuntimeError(
                f"{message_prefix} | FAIL] {label} reported gripper fault "
                f"gFLT=0x{fault:02X}: "
                f"{FAULT_DESCRIPTIONS.get(fault, 'unknown')}; "
                f"status={format_status(last_status)}."
            )

        if last_status["gACT"] != 1 or last_status["gSTA"] != 3:
            raise RuntimeError(
                f"{message_prefix} | FAIL] The gripper left its "
                f"activated-ready state during {label}: "
                f"{format_status(last_status)}."
            )

        command_received = last_status["gPR"] == target_position
        motion_finished = (
            last_status["gGTO"] == 1 and last_status["gOBJ"] != 0
        )
        if not (command_received and motion_finished):
            continue

        object_state = last_status["gOBJ"]
        if object_state in (1, 2):
            actual_contact = OBJECT_STATE_DESCRIPTIONS[object_state]

            if not allow_contact:
                raise RuntimeError(
                    f"{message_prefix} | FAIL] {label} ended with unexpected "
                    f"{actual_contact}; clear the workspace."
                )

            if expected_contact_state is None:
                raise RuntimeError(
                    f"{message_prefix} | FAIL] {label} reported "
                    f"{actual_contact}, although rPR already matched the "
                    "starting gPO."
                )

            if object_state != expected_contact_state:
                expected_contact = OBJECT_STATE_DESCRIPTIONS[
                    expected_contact_state
                ]
                raise RuntimeError(
                    f"{message_prefix} | FAIL] {label} reported "
                    f"{actual_contact}, but its direction requires "
                    f"{expected_contact}; status={format_status(last_status)}."
                )
        elif (
            object_state == 3
            and abs(last_status["gPO"] - target_position)
            > position_tolerance
        ):
            raise RuntimeError(
                f"{message_prefix} | FAIL] {label} ended at "
                f"gPO={last_status['gPO']}, which differs from "
                f"rPR={target_position} by more than "
                f"{position_tolerance} counts."
            )

        if failed_poll_count:
            print(
                f"{message_prefix} | WARNING "
                f"{command_number}/{command_count}] Recovered after "
                f"{failed_poll_count} failed FC04 read(s); "
                "most recent recovered read error="
                f"{last_recovered_read_error}."
            )

        highest_sampled_current = max(
            status["gCU"] for _, status in motion_samples
        )
        if trace_motion:
            for sample_index, (sample_time, sample_status) in enumerate(
                motion_samples,
                start=1,
            ):
                print(
                    f"{message_prefix} | TRACE {command_number}/"
                    f"{command_count} | SAMPLE {sample_index}/"
                    f"{len(motion_samples)}] t={sample_time:.4f} s "
                    f"{format_status(sample_status)}"
                )
        print(
            f"{message_prefix} | RESULT {command_number}/{command_count}] "
            f"{format_status(last_status)}"
        )
        print(
            f"{message_prefix} | COMMAND {command_number}/{command_count} "
            f"FINISHED] {label} completed after "
            f"{successful_poll_count} valid and {failed_poll_count} failed "
            f"status read(s); host-observed duration="
            f"{elapsed_seconds:.4f} s after FC16 acknowledgement; "
            f"FC16 round trip={write_duration:.4f} s; "
            "highest sampled gCU="
            f"{highest_sampled_current} "
            f"(~{highest_sampled_current * 10} mA)."
        )
        return last_status, motion_samples

    raise RuntimeError(
        f"{message_prefix} | FAIL] {label} did not complete within "
        f"approximately {timeout:.1f} s; valid reads="
        f"{successful_poll_count}, failed reads={failed_poll_count}, "
        f"last status={last_status}, last read error={last_read_error}."
    )


def run_position_diagnostic(
    client: Any,
    args: argparse.Namespace,
    initial_status: dict[str, int],
    *,
    diagnostic_name: str,
    target_positions: tuple[int, ...],
) -> dict[str, int]:
    """Test raw position requests while speed and force remain fixed."""
    # Motion sequence: visit every rPR target in order. Basic motion supplies
    # open -> midpoint -> closed -> open; the position diagnostic supplies
    # its configurable multi-position sequence.
    command_count = len(target_positions)
    current_status = initial_status
    observations: list[tuple[int, int, float]] = []

    print(
        f"\n[PHASE 6 | {diagnostic_name} | START] "
        f"Testing rPR targets={target_positions} with fixed "
        f"rSP={args.speed_request} and rFR=0."
    )

    for command_number, target_position in enumerate(
        target_positions,
        start=1,
    ):
        position_tolerance = (
            args.endpoint_tolerance
            if target_position in (0, 255)
            else args.position_tolerance
        )
        current_status, samples = execute_motion(
            client,
            diagnostic_name=diagnostic_name,
            label=f"position target {target_position}",
            command_number=command_number,
            command_count=command_count,
            starting_position=current_status["gPO"],
            target_position=target_position,
            speed_request=args.speed_request,
            force_request=0,
            allow_contact=False,
            position_tolerance=position_tolerance,
            timeout=args.motion_timeout,
            poll_interval=args.poll_interval,
            slave_id=args.slave_id,
            trace_motion=args.trace_motion,
        )
        observations.append(
            (target_position, current_status["gPO"], samples[-1][0])
        )

        if command_number < command_count:
            time.sleep(args.motion_dwell)

    print(f"[PHASE 6 | {diagnostic_name} | SUMMARY]")
    for target_position, actual_position, duration in observations:
        print(
            f"  rPR={target_position:3d} -> gPO={actual_position:3d}, "
            f"error={abs(actual_position - target_position):3d}, "
            f"duration={duration:.4f} s"
        )
    print(
        f"[PHASE 6 | {diagnostic_name} | FINISHED] "
        f"All {command_count} position commands completed without contact."
    )
    return current_status


def run_speed_diagnostic(
    client: Any,
    args: argparse.Namespace,
    initial_status: dict[str, int],
) -> tuple[dict[str, int], bool]:
    """Compare identical closing strokes while varying only rSP."""
    # Motion sequence: move to 32, measure 32 -> 224 at each rSP, return to 32
    # between measurements, and finish fully open at rPR=0.
    diagnostic_name = "SPEED DIAGNOSTIC"
    speed_requests = tuple(sorted(set(args.speed_requests)))
    command_count = 2 + (
        len(speed_requests) * args.speed_repetitions * 2
    )
    command_number = 1
    current_status = initial_status
    measured_durations: dict[int, list[float]] = {
        speed_request: [] for speed_request in speed_requests
    }

    print(
        f"\n[PHASE 6 | {diagnostic_name} | START] "
        f"Comparing rSP={speed_requests} over identical closing strokes "
        f"{SPEED_TEST_START_POSITION}->{SPEED_TEST_TARGET_POSITION}; "
        "rFR remains 0."
    )

    current_status, _ = execute_motion(
        client,
        diagnostic_name=diagnostic_name,
        label="setup at common start position",
        command_number=command_number,
        command_count=command_count,
        starting_position=current_status["gPO"],
        target_position=SPEED_TEST_START_POSITION,
        speed_request=args.speed_request,
        force_request=0,
        allow_contact=False,
        position_tolerance=args.position_tolerance,
        timeout=args.motion_timeout,
        poll_interval=args.poll_interval,
        slave_id=args.slave_id,
        trace_motion=False,
    )
    command_number += 1
    time.sleep(args.motion_dwell)

    for speed_request in speed_requests:
        for repetition in range(1, args.speed_repetitions + 1):
            current_status, samples = execute_motion(
                client,
                diagnostic_name=diagnostic_name,
                label=(
                    f"measured closing stroke rSP={speed_request}, "
                    f"repetition {repetition}"
                ),
                command_number=command_number,
                command_count=command_count,
                starting_position=current_status["gPO"],
                target_position=SPEED_TEST_TARGET_POSITION,
                speed_request=speed_request,
                force_request=0,
                allow_contact=False,
                position_tolerance=args.position_tolerance,
                timeout=args.motion_timeout,
                poll_interval=args.poll_interval,
                slave_id=args.slave_id,
                trace_motion=args.trace_motion,
            )
            measured_durations[speed_request].append(samples[-1][0])
            command_number += 1
            time.sleep(args.motion_dwell)

            current_status, _ = execute_motion(
                client,
                diagnostic_name=diagnostic_name,
                label="return to common start position",
                command_number=command_number,
                command_count=command_count,
                starting_position=current_status["gPO"],
                target_position=SPEED_TEST_START_POSITION,
                speed_request=args.speed_request,
                force_request=0,
                allow_contact=False,
                position_tolerance=args.position_tolerance,
                timeout=args.motion_timeout,
                poll_interval=args.poll_interval,
                slave_id=args.slave_id,
                trace_motion=False,
            )
            command_number += 1

            if command_number <= command_count:
                time.sleep(args.motion_dwell)

    current_status, _ = execute_motion(
        client,
        diagnostic_name=diagnostic_name,
        label="fully open after speed trials",
        command_number=command_number,
        command_count=command_count,
        starting_position=current_status["gPO"],
        target_position=0,
        speed_request=0,
        force_request=0,
        allow_contact=False,
        position_tolerance=args.endpoint_tolerance,
        timeout=args.motion_timeout,
        poll_interval=args.poll_interval,
        slave_id=args.slave_id,
        trace_motion=False,
    )

    median_durations = {
        speed_request: statistics.median(durations)
        for speed_request, durations in measured_durations.items()
    }
    print(f"[PHASE 6 | {diagnostic_name} | SUMMARY]")
    for speed_request in speed_requests:
        durations = ", ".join(
            f"{duration:.4f}" for duration in measured_durations[speed_request]
        )
        print(
            f"  rSP={speed_request:3d}: durations=[{durations}] s, "
            f"median={median_durations[speed_request]:.4f} s"
        )

    slowest_request = speed_requests[0]
    fastest_request = speed_requests[-1]
    speed_relationship_observed = (
        median_durations[fastest_request]
        < median_durations[slowest_request]
    )
    if not speed_relationship_observed:
        print(
            f"[PHASE 6 | {diagnostic_name} | WARNING] The highest rSP did "
            "not produce a shorter host-observed median duration. Timing is "
            "quantized by Modbus polling; repeat the diagnostic before "
            "drawing a hardware conclusion."
        )
    else:
        print(
            f"[PHASE 6 | {diagnostic_name} | RESULT] The largest tested "
            "rSP produced a shorter host-observed median duration than the "
            "smallest tested rSP across the fixed stroke."
        )

    print(
        f"[PHASE 6 | {diagnostic_name} | FINISHED] Speed requests were "
        "compared independently and the gripper finished fully open; exact "
        "physical mm/s was not measured."
    )
    return current_status, speed_relationship_observed


def confirm_contact_fixture_positioned(
    *,
    diagnostic_name: str,
    confirmation_token: str,
) -> None:
    """Gate contact motion after activation and a verified open command."""
    print(
        f"\n[PHASE 6 | {diagnostic_name} | FIXTURE SETUP | START] "
        "Activation is complete and the gripper is open."
    )
    print(
        f"[PHASE 6 | {diagnostic_name} | FIXTURE SETUP | WARNING] "
        "Position the pre-mounted fixture using a safe external mechanism. "
        "Do not place a hand in the energized gripper's pinch zone."
    )
    print(
        f"[PHASE 6 | {diagnostic_name} | FIXTURE SETUP | WARNING] "
        "If the fixture cannot be positioned without entering the pinch "
        "zone, stop now; this prompt is not a safety interlock."
    )
    answer = input(
        f"[PHASE 6 | {diagnostic_name} | FIXTURE SETUP | INPUT] "
        f"Type {confirmation_token} after the fixture is secured: "
    ).strip()
    if answer != confirmation_token:
        raise RuntimeError(
            f"[PHASE 6 | {diagnostic_name} | FIXTURE SETUP | ABORTED] "
            "Confirmation did not match; no contact command was sent."
        )
    print(
        f"[PHASE 6 | {diagnostic_name} | FIXTURE SETUP | FINISHED] "
        "The secured-fixture gate passed; contact motion is permitted."
    )


def run_force_request_diagnostic(
    client: Any,
    args: argparse.Namespace,
    initial_status: dict[str, int],
) -> dict[str, int]:
    """Test rFR requests against one rigid, secured contact object."""
    # Motion sequence: open, confirm the secured fixture, close toward rPR=255
    # at each rFR until gOBJ=2 reports contact, and reopen after every trial.
    diagnostic_name = "FORCE-REQUEST DIAGNOSTIC"
    force_requests = tuple(sorted(set(args.force_requests)))
    command_count = 1 + (2 * len(force_requests))
    command_number = 1
    current_status = initial_status
    observations: list[tuple[int, int, int, float]] = []

    print(
        f"\n[PHASE 6 | {diagnostic_name} | START] "
        f"Testing rFR={force_requests} with fixed rSP={args.speed_request}, "
        "rPR=255, and one secured object."
    )

    current_status, _ = execute_motion(
        client,
        diagnostic_name=diagnostic_name,
        label="open before contact trials",
        command_number=command_number,
        command_count=command_count,
        starting_position=current_status["gPO"],
        target_position=0,
        speed_request=args.speed_request,
        force_request=0,
        allow_contact=False,
        position_tolerance=args.endpoint_tolerance,
        timeout=args.motion_timeout,
        poll_interval=args.poll_interval,
        slave_id=args.slave_id,
        trace_motion=False,
    )
    command_number += 1
    confirm_contact_fixture_positioned(
        diagnostic_name=diagnostic_name,
        confirmation_token="FORCE FIXTURE SECURED",
    )

    for force_request in force_requests:
        time.sleep(args.motion_dwell)
        contact_status, samples = execute_motion(
            client,
            diagnostic_name=diagnostic_name,
            label=f"closing contact trial rFR={force_request}",
            command_number=command_number,
            command_count=command_count,
            starting_position=current_status["gPO"],
            target_position=255,
            speed_request=args.speed_request,
            force_request=force_request,
            allow_contact=True,
            position_tolerance=args.endpoint_tolerance,
            timeout=args.motion_timeout,
            poll_interval=args.poll_interval,
            slave_id=args.slave_id,
            trace_motion=args.trace_motion,
        )
        command_number += 1
        highest_sampled_current = max(
            status["gCU"] for _, status in samples
        )
        contact_detected = contact_status["gOBJ"] == 2

        time.sleep(args.motion_dwell)
        current_status, _ = execute_motion(
            client,
            diagnostic_name=diagnostic_name,
            label=f"release after rFR={force_request}",
            command_number=command_number,
            command_count=command_count,
            starting_position=contact_status["gPO"],
            target_position=0,
            speed_request=args.speed_request,
            force_request=0,
            allow_contact=False,
            position_tolerance=args.endpoint_tolerance,
            timeout=args.motion_timeout,
            poll_interval=args.poll_interval,
            slave_id=args.slave_id,
            trace_motion=False,
        )
        command_number += 1

        if not contact_detected:
            raise RuntimeError(
                f"[PHASE 6 | {diagnostic_name} | FAIL] rFR={force_request} "
                "reached the closing target without reporting gOBJ=2. The "
                "object was released before this failure was reported."
            )

        observations.append(
            (
                force_request,
                contact_status["gPO"],
                highest_sampled_current,
                samples[-1][0],
            )
        )

    print(f"[PHASE 6 | {diagnostic_name} | SUMMARY]")
    for (
        force_request,
        contact_position,
        highest_sampled_current,
        duration,
    ) in observations:
        print(
            f"  rFR={force_request:3d}: contact gPO={contact_position:3d}, "
            f"highest sampled gCU={highest_sampled_current:3d} "
            f"(~{highest_sampled_current * 10} mA), "
            f"duration={duration:.4f} s"
        )
    print(
        f"[PHASE 6 | {diagnostic_name} | FINISHED] Every request produced "
        "closing-contact detection and was released. Associated sampled "
        "current was recorded; rFR application and physical force were not "
        "independently measured."
    )
    return current_status


def run_object_detection_diagnostic(
    client: Any,
    args: argparse.Namespace,
    initial_status: dict[str, int],
) -> dict[str, int]:
    """Require stable closing-contact detection with a secured object."""
    # Motion sequence: open, confirm the secured fixture, close toward rPR=255,
    # verify several stable gOBJ=2 hold samples, and reopen to release it.
    diagnostic_name = "OBJECT-DETECTION DIAGNOSTIC"
    command_count = 3
    current_status = initial_status

    print(
        f"\n[PHASE 6 | {diagnostic_name} | START] "
        f"Using fixed rSP={args.speed_request}, rFR={args.force_request}, "
        "and rPR=255 with the secured test object."
    )

    current_status, _ = execute_motion(
        client,
        diagnostic_name=diagnostic_name,
        label="open before object-detection trial",
        command_number=1,
        command_count=command_count,
        starting_position=current_status["gPO"],
        target_position=0,
        speed_request=args.speed_request,
        force_request=0,
        allow_contact=False,
        position_tolerance=args.endpoint_tolerance,
        timeout=args.motion_timeout,
        poll_interval=args.poll_interval,
        slave_id=args.slave_id,
        trace_motion=False,
    )
    time.sleep(args.motion_dwell)
    confirm_contact_fixture_positioned(
        diagnostic_name=diagnostic_name,
        confirmation_token="OBJECT FIXTURE SECURED",
    )

    contact_status, samples = execute_motion(
        client,
        diagnostic_name=diagnostic_name,
        label="close toward secured object",
        command_number=2,
        command_count=command_count,
        starting_position=current_status["gPO"],
        target_position=255,
        speed_request=args.speed_request,
        force_request=args.force_request,
        allow_contact=True,
        position_tolerance=args.endpoint_tolerance,
        timeout=args.motion_timeout,
        poll_interval=args.poll_interval,
        slave_id=args.slave_id,
        trace_motion=args.trace_motion,
    )
    contact_detected = contact_status["gOBJ"] == 2
    hold_failure = None

    if contact_detected:
        for sample_number in range(
            1,
            args.num_consecutive_read_only_samples + 1,
        ):
            time.sleep(args.sample_interval)
            try:
                status = decode_status_from_registers(
                    read_status_registers(client, args.slave_id)
                )
            except (RuntimeError, ValueError) as error:
                hold_failure = f"hold sample {sample_number} failed: {error}"
                break
            print(
                f"[PHASE 6 | {diagnostic_name} | HOLD SAMPLE "
                f"{sample_number}/"
                f"{args.num_consecutive_read_only_samples}] "
                f"{format_status(status)}"
            )
            if (
                status["gACT"] != 1
                or status["gSTA"] != 3
                or status["gFLT"] != 0
                or status["kFLT"] != 0
                or status["gGTO"] != 1
                or status["gOBJ"] != 2
                or status["gPR"] != 255
            ):
                hold_failure = (
                    "contact status was not retained: "
                    f"{format_status(status)}"
                )
                break

    current_status, _ = execute_motion(
        client,
        diagnostic_name=diagnostic_name,
        label="release secured object",
        command_number=3,
        command_count=command_count,
        starting_position=contact_status["gPO"],
        target_position=0,
        speed_request=args.speed_request,
        force_request=0,
        allow_contact=False,
        position_tolerance=args.endpoint_tolerance,
        timeout=args.motion_timeout,
        poll_interval=args.poll_interval,
        slave_id=args.slave_id,
        trace_motion=False,
    )

    if not contact_detected:
        raise RuntimeError(
            f"[PHASE 6 | {diagnostic_name} | FAIL] The close command "
            "reached its requested position without reporting closing "
            "contact. The gripper reopened before reporting this failure."
        )
    if hold_failure is not None:
        raise RuntimeError(
            f"[PHASE 6 | {diagnostic_name} | FAIL] {hold_failure}. "
            "The secured object was released before this failure was "
            "reported."
        )

    highest_sampled_current = max(
        status["gCU"] for _, status in samples
    )
    print(
        f"[PHASE 6 | {diagnostic_name} | RESULT] Closing contact was "
        f"detected at gPO={contact_status['gPO']}; highest sampled "
        f"gCU={highest_sampled_current} "
        f"(~{highest_sampled_current * 10} mA)."
    )
    print(
        f"[PHASE 6 | {diagnostic_name} | FINISHED] Stable gOBJ=2 was "
        "observed and the secured object was released. This does not prove "
        "grasp retention or measured gripping force."
    )
    return current_status


def run_motion_state_diagnostic(
    client: Any,
    args: argparse.Namespace,
    initial_status: dict[str, int],
) -> tuple[dict[str, int], bool]:
    """Trace FC04 status throughout one slow, long closing stroke."""
    # Motion sequence: move to 32, trace the slow 32 -> 224 closing stroke,
    # analyze its sampled state transitions, and finish fully open at rPR=0.
    diagnostic_name = "MOTION-STATE DIAGNOSTIC"
    command_count = 3
    current_status = initial_status

    print(
        f"\n[PHASE 6 | {diagnostic_name} | START] "
        f"Tracing a fixed {SPEED_TEST_START_POSITION}->"
        f"{SPEED_TEST_TARGET_POSITION} stroke at rSP=0 and rFR=0."
    )

    current_status, _ = execute_motion(
        client,
        diagnostic_name=diagnostic_name,
        label="setup at trace start position",
        command_number=1,
        command_count=command_count,
        starting_position=current_status["gPO"],
        target_position=SPEED_TEST_START_POSITION,
        speed_request=0,
        force_request=0,
        allow_contact=False,
        position_tolerance=args.position_tolerance,
        timeout=args.motion_timeout,
        poll_interval=args.poll_interval,
        slave_id=args.slave_id,
        trace_motion=False,
    )
    time.sleep(args.motion_dwell)

    traced_status, samples = execute_motion(
        client,
        diagnostic_name=diagnostic_name,
        label="traced slow closing stroke",
        command_number=2,
        command_count=command_count,
        starting_position=current_status["gPO"],
        target_position=SPEED_TEST_TARGET_POSITION,
        speed_request=0,
        force_request=0,
        allow_contact=False,
        position_tolerance=args.position_tolerance,
        timeout=args.motion_timeout,
        poll_interval=args.motion_trace_interval,
        slave_id=args.slave_id,
        trace_motion=True,
    )

    moving_samples = [
        (elapsed, status)
        for elapsed, status in samples
        if (
            status["gPR"] == SPEED_TEST_TARGET_POSITION
            and status["gGTO"] == 1
            and status["gOBJ"] == 0
        )
    ]
    observed_positions = [
        status["gPO"] for _, status in moving_samples
    ] + [traced_status["gPO"]]
    direction_violations = sum(
        later + 1 < earlier
        for earlier, later in zip(
            observed_positions,
            observed_positions[1:],
        )
    )
    analysis_failure = None
    if direction_violations:
        analysis_failure = (
            f"Observed {direction_violations} backward gPO transition(s) "
            "during the commanded closing stroke"
        )

    current_status, _ = execute_motion(
        client,
        diagnostic_name=diagnostic_name,
        label="reopen after state trace",
        command_number=3,
        command_count=command_count,
        starting_position=traced_status["gPO"],
        target_position=0,
        speed_request=0,
        force_request=0,
        allow_contact=False,
        position_tolerance=args.endpoint_tolerance,
        timeout=args.motion_timeout,
        poll_interval=args.poll_interval,
        slave_id=args.slave_id,
        trace_motion=args.trace_motion,
    )

    if analysis_failure is not None:
        raise RuntimeError(
            f"[PHASE 6 | {diagnostic_name} | FAIL] {analysis_failure}. "
            "The gripper reopened before this failure was reported."
        )

    if moving_samples:
        print(
            f"[PHASE 6 | {diagnostic_name} | RESULT] Captured "
            f"{len(moving_samples)} in-motion gOBJ=0 sample(s) with "
            "directionally consistent gPO."
        )
    else:
        print(
            f"[PHASE 6 | {diagnostic_name} | INCONCLUSIVE] No gOBJ=0 "
            "sample was captured; the motion may have completed between "
            "FC04 polls."
        )
    print(
        f"[PHASE 6 | {diagnostic_name} | FINISHED] The sampled trajectory "
        "was printed and the gripper reopened."
    )
    return current_status, bool(moving_samples)


def request_motion_stop(client: Any, slave_id: int) -> None:
    """Request a non-safety-rated stop by retaining rACT and clearing rGTO."""
    registers = [
        encode_action_request(activate=True, go_to=False),
        0x0000,
        0x0000,
    ]
    write_command_registers(
        client,
        slave_id,
        registers,
        "non-safety-rated motion-stop request",
    )


def request_gripper_reset(client: Any, slave_id: int) -> None:
    """Request a gripper reset after interrupted activation by clearing rACT."""
    registers = [
        encode_action_request(activate=False, go_to=False),
        0x0000,
        0x0000,
    ]
    write_command_registers(
        client,
        slave_id,
        registers,
        "activation-abort reset request",
    )


def check_action_request_encoding() -> None:
    """Check ACTION REQUEST values against the Robotiq manual examples."""
    expected_values = (
        ("reset", False, False, 0x0000),
        ("activation", True, False, 0x0100),
        ("go-to", True, True, 0x0900),
    )

    for name, activate, go_to, expected in expected_values:
        actual = encode_action_request(
            activate=activate,
            go_to=go_to,
        )

        if actual != expected:
            raise RuntimeError(
                f"{name} encoded as 0x{actual:04X}, "
                f"expected 0x{expected:04X}"
            )


def read_status_registers(
    client: Any,
    slave_id: int,
    *,
    pace_request: bool = True,
) -> list[int]:
    """Read one sample containing the three raw FC04 registers."""
    if pace_request:
        wait_for_request_interval()

    response = client.read_input_registers(
        address=INPUT_REGISTER_ADDRESS,
        count=REGISTER_COUNT,
        **slave_address_kwargs(client.read_input_registers, slave_id),
    )

    if response is None or response.isError():
        raise RuntimeError(f"FC04 status read failed: {response}")

    response_registers = getattr(response, "registers", None)
    if response_registers is None:
        raise RuntimeError(f"FC04 response contains no registers: {response}")

    registers = list(response_registers)

    if len(registers) != REGISTER_COUNT:
        raise RuntimeError(
            f"expected {REGISTER_COUNT} registers, received {registers}"
        )

    return registers


def read_gripper_status(
    client: Any,
    args: argparse.Namespace,
) -> list[list[int]]:
    """Read multiple raw FC04 status-register samples."""

    register_samples: list[list[int]] = []

    for sample_index in range(args.num_consecutive_read_only_samples):
        registers = read_status_registers(
            client,
            args.slave_id,
        )
        register_samples.append(registers)

        register_text = " ".join(
            f"0x{value:04X}" for value in registers
        )
        print(
            "[PHASE 2 | READ GRIPPER STATUS | SAMPLE] "
            f"{sample_index + 1}/"
            f"{args.num_consecutive_read_only_samples}: "
            f"raw registers=[{register_text}]"
        )

        if sample_index + 1 < args.num_consecutive_read_only_samples:
            time.sleep(args.sample_interval)

    return register_samples


def analyze_gripper_status(
    register_samples: list[list[int]],
) -> tuple[dict[str, int], bool]:
    """Decode collected samples and evaluate the final gripper state."""
    if not register_samples:
        raise ValueError("at least one status-register sample is required")

    decoded_status_samples: list[dict[str, int]] = []

    for sample_index, register_values in enumerate(register_samples, start=1):
        status = decode_status_from_registers(register_values)
        decoded_status_samples.append(status)

        print(
            "[PHASE 3 | ANALYZE GRIPPER STATUS | SAMPLE] "
            f"{sample_index}/{len(register_samples)}: "
            f"{format_status(status)}"
        )

    last_status = decoded_status_samples[-1]

    state_is_activated_and_ready = (
        last_status["gACT"] == 1
        and last_status["gSTA"] == 3
        and last_status["gFLT"] == 0
        and last_status["kFLT"] == 0
    )
    state_is_reset_and_fault_free = (
        last_status["gACT"] == 0
        and last_status["gSTA"] == 0
        and last_status["gFLT"] == 0
        and last_status["kFLT"] == 0
    )
    state_is_acceptable = (
        state_is_activated_and_ready or state_is_reset_and_fault_free
    )

    return last_status, state_is_acceptable


def run_fc04_frequency_diagnostic(
    client: Any,
    *,
    slave_id: int,
    sample_count: int,
    requested_frequency_hz: float,
) -> bool:
    """Measure sequential host-to-gripper FC04 transaction frequency."""
    # Measures the complete status-read round trip:
    # Python -> PyModbus -> operating system -> USB/RS-485 adapter
    # -> Robotiq gripper -> FC04 response -> Python.
    diagnostic_name = "FC04 FREQUENCY DIAGNOSTIC"
    requested_period = 1.0 / requested_frequency_hz
    request_start_times: list[float] = []
    latencies: list[float] = []
    failures: list[str] = []
    starts_over_one_period_late = 0

    print(
        f"\n[PHASE 6 | {diagnostic_name} | START] Requesting "
        f"{sample_count} sequential reads at {requested_frequency_hz:.1f} Hz "
        f"({requested_period * 1000:.2f} ms period) without catch-up bursts."
    )

    benchmark_started_at = time.monotonic()
    next_request_start = benchmark_started_at

    for sample_index in range(sample_count):
        sleep_duration = next_request_start - time.monotonic()
        if sleep_duration > 0:
            time.sleep(sleep_duration)

        request_started_at = wait_for_request_interval()
        request_start_times.append(request_started_at)
        if request_started_at - next_request_start > requested_period:
            starts_over_one_period_late += 1
        # Schedule from the actual start. If one transaction is slow, the
        # benchmark does not catch up with a burst of faster requests.
        next_request_start = request_started_at + requested_period

        response_received_at = None
        try:
            register_values = read_status_registers(
                client,
                slave_id,
                pace_request=False,
            )
            response_received_at = time.monotonic()
            status = decode_status_from_registers(register_values)
            state_is_healthy = (
                status["gFLT"] == 0
                and status["kFLT"] == 0
                and (
                    (
                        status["gACT"] == 0
                        and status["gSTA"] == 0
                    )
                    or (
                        status["gACT"] == 1
                        and status["gSTA"] == 3
                    )
                )
            )
            if not state_is_healthy:
                failures.append(
                    f"sample {sample_index + 1}: unhealthy "
                    f"{format_status(status)}"
                )
        except (RuntimeError, ValueError) as error:
            failures.append(f"sample {sample_index + 1}: {error}")

        if response_received_at is None:
            response_received_at = time.monotonic()
        latencies.append(response_received_at - request_started_at)

    benchmark_elapsed = time.monotonic() - benchmark_started_at
    start_intervals = [
        later - earlier
        for earlier, later in zip(
            request_start_times,
            request_start_times[1:],
        )
    ]
    achieved_frequency_hz = (
        (len(request_start_times) - 1)
        / (request_start_times[-1] - request_start_times[0])
    )
    ordered_latencies = sorted(latencies)
    p95_index = round(0.95 * (len(ordered_latencies) - 1))
    p99_index = round(0.99 * (len(ordered_latencies) - 1))

    print(
        f"[PHASE 6 | {diagnostic_name} | RESULT] "
        f"healthy valid samples={sample_count - len(failures)}/"
        f"{sample_count}, "
        f"elapsed={benchmark_elapsed:.4f} s, "
        f"achieved start rate={achieved_frequency_hz:.2f} Hz, "
        "starts over one period late="
        f"{starts_over_one_period_late}."
    )
    print(
        f"[PHASE 6 | {diagnostic_name} | LATENCY] "
        f"min={min(latencies) * 1000:.3f} ms, "
        f"median={statistics.median(latencies) * 1000:.3f} ms, "
        f"mean={statistics.fmean(latencies) * 1000:.3f} ms, "
        f"p95={ordered_latencies[p95_index] * 1000:.3f} ms, "
        f"p99={ordered_latencies[p99_index] * 1000:.3f} ms, "
        f"max={max(latencies) * 1000:.3f} ms."
    )
    print(
        f"[PHASE 6 | {diagnostic_name} | START-INTERVAL] "
        f"min={min(start_intervals) * 1000:.3f} ms, "
        f"mean={statistics.fmean(start_intervals) * 1000:.3f} ms, "
        f"max={max(start_intervals) * 1000:.3f} ms."
    )

    if failures:
        failure_summary = "; ".join(failures[:3])
        if len(failures) > 3:
            failure_summary += f"; plus {len(failures) - 3} more"
        raise RuntimeError(
            f"[PHASE 6 | {diagnostic_name} | FAIL] "
            f"{len(failures)} read/status failure(s): {failure_summary}"
        )

    frequency_target_observed = (
        starts_over_one_period_late == 0
        and achieved_frequency_hz >= 0.9 * requested_frequency_hz
    )
    if starts_over_one_period_late:
        print(
            f"[PHASE 6 | {diagnostic_name} | WARNING] "
            f"{starts_over_one_period_late} request(s) started more than one "
            "requested period late."
        )
    if achieved_frequency_hz < 0.9 * requested_frequency_hz:
        print(
            f"[PHASE 6 | {diagnostic_name} | WARNING] Achieved frequency "
            "was below 90% of the requested frequency."
        )

    print(
        f"[PHASE 6 | {diagnostic_name} | FINISHED] All FC04 transactions "
        "were valid. This measures host/OS/USB/Modbus round trips, not the "
        "gripper's internal control-loop frequency."
    )
    return frequency_target_observed


def run_fc16_frequency_diagnostic(
    client: Any,
    args: argparse.Namespace,
    initial_status: dict[str, int],
) -> tuple[dict[str, int], bool]:
    """Measure stationary FC16 write-acknowledgement frequency."""
    # Motion/setup sequence: verify the gripper is fully open, repeatedly send
    # the same open target without deliberate motion, then verify it stayed open.
    # Measures the complete command-write round trip:
    # Python -> PyModbus -> operating system -> USB/RS-485 adapter
    # -> Robotiq gripper -> FC16 acknowledgement -> Python.
    diagnostic_name = "FC16 WRITE-FREQUENCY DIAGNOSTIC"
    print(
        f"\n[PHASE 6 | {diagnostic_name} | START] "
        "Establishing a verified fully open state before stationary writes."
    )
    execute_motion(
        client,
        diagnostic_name=diagnostic_name,
        label="open before stationary write test",
        command_number=1,
        command_count=1,
        starting_position=initial_status["gPO"],
        target_position=0,
        speed_request=0,
        force_request=0,
        allow_contact=False,
        position_tolerance=args.endpoint_tolerance,
        timeout=args.motion_timeout,
        poll_interval=args.poll_interval,
        slave_id=args.slave_id,
        trace_motion=False,
    )

    requested_period = 1.0 / args.frequency_hz
    stationary_registers = [
        encode_action_request(activate=True, go_to=True),
        0x0000,
        0x0000,
    ]
    request_start_times: list[float] = []
    latencies: list[float] = []
    failures: list[str] = []
    starts_over_one_period_late = 0

    print(
        f"\n[PHASE 6 | {diagnostic_name} | BENCHMARK START] Sending "
        f"{args.frequency_samples} identical open-target FC16 writes at "
        f"{args.frequency_hz:.1f} Hz "
        f"({requested_period * 1000:.2f} ms period) without catch-up bursts."
    )

    benchmark_started_at = time.monotonic()
    next_request_start = benchmark_started_at

    for sample_index in range(args.frequency_samples):
        sleep_duration = next_request_start - time.monotonic()
        if sleep_duration > 0:
            time.sleep(sleep_duration)

        request_started_at = wait_for_request_interval()
        request_start_times.append(request_started_at)
        if request_started_at - next_request_start > requested_period:
            starts_over_one_period_late += 1
        next_request_start = request_started_at + requested_period

        try:
            write_command_registers(
                client,
                args.slave_id,
                stationary_registers,
                f"stationary frequency sample {sample_index + 1}",
                pace_request=False,
            )
        except RuntimeError as error:
            failures.append(f"sample {sample_index + 1}: {error}")
        latencies.append(time.monotonic() - request_started_at)

    benchmark_elapsed = time.monotonic() - benchmark_started_at
    start_intervals = [
        later - earlier
        for earlier, later in zip(
            request_start_times,
            request_start_times[1:],
        )
    ]
    achieved_frequency_hz = (
        (len(request_start_times) - 1)
        / (request_start_times[-1] - request_start_times[0])
    )
    ordered_latencies = sorted(latencies)
    p95_index = round(0.95 * (len(ordered_latencies) - 1))
    p99_index = round(0.99 * (len(ordered_latencies) - 1))

    print(
        f"[PHASE 6 | {diagnostic_name} | RESULT] "
        f"acknowledged={args.frequency_samples - len(failures)}/"
        f"{args.frequency_samples}, elapsed={benchmark_elapsed:.4f} s, "
        f"achieved start rate={achieved_frequency_hz:.2f} Hz, "
        "starts over one period late="
        f"{starts_over_one_period_late}."
    )
    print(
        f"[PHASE 6 | {diagnostic_name} | LATENCY] "
        f"min={min(latencies) * 1000:.3f} ms, "
        f"median={statistics.median(latencies) * 1000:.3f} ms, "
        f"mean={statistics.fmean(latencies) * 1000:.3f} ms, "
        f"p95={ordered_latencies[p95_index] * 1000:.3f} ms, "
        f"p99={ordered_latencies[p99_index] * 1000:.3f} ms, "
        f"max={max(latencies) * 1000:.3f} ms."
    )
    print(
        f"[PHASE 6 | {diagnostic_name} | START-INTERVAL] "
        f"min={min(start_intervals) * 1000:.3f} ms, "
        f"mean={statistics.fmean(start_intervals) * 1000:.3f} ms, "
        f"max={max(start_intervals) * 1000:.3f} ms."
    )

    if failures:
        failure_summary = "; ".join(failures[:3])
        if len(failures) > 3:
            failure_summary += f"; plus {len(failures) - 3} more"
        raise RuntimeError(
            f"[PHASE 6 | {diagnostic_name} | FAIL] "
            f"{len(failures)} FC16 write failure(s): {failure_summary}"
        )

    final_deadline = time.monotonic() + args.motion_timeout
    final_status = None
    last_read_error = None

    while time.monotonic() < final_deadline:
        time.sleep(args.poll_interval)
        try:
            final_status = decode_status_from_registers(
                read_status_registers(client, args.slave_id)
            )
        except RuntimeError as error:
            last_read_error = error
            continue

        last_read_error = None
        if (
            final_status["gFLT"] != 0
            or final_status["kFLT"] != 0
            or final_status["gACT"] != 1
            or final_status["gSTA"] != 3
        ):
            raise RuntimeError(
                f"[PHASE 6 | {diagnostic_name} | FAIL] Final state became "
                f"unhealthy: {format_status(final_status)}."
            )
        if (
            final_status["gGTO"] == 1
            and final_status["gPR"] == 0
            and final_status["gOBJ"] != 0
        ):
            break
    else:
        raise RuntimeError(
            f"[PHASE 6 | {diagnostic_name} | FAIL] Final open target did "
            f"not become terminal: {final_status}; "
            f"last read error={last_read_error}."
        )

    if (
        final_status["gOBJ"] != 3
        or final_status["gPO"] > args.endpoint_tolerance
    ):
        raise RuntimeError(
            f"[PHASE 6 | {diagnostic_name} | FAIL] Final open state was "
            f"not reached without contact: {format_status(final_status)}."
        )

    frequency_target_observed = (
        starts_over_one_period_late == 0
        and achieved_frequency_hz >= 0.9 * args.frequency_hz
    )
    if starts_over_one_period_late:
        print(
            f"[PHASE 6 | {diagnostic_name} | WARNING] "
            f"{starts_over_one_period_late} write(s) started more than one "
            "requested period late."
        )
    if achieved_frequency_hz < 0.9 * args.frequency_hz:
        print(
            f"[PHASE 6 | {diagnostic_name} | WARNING] Achieved frequency "
            "was below 90% of the requested frequency."
        )

    print(
        f"[PHASE 6 | {diagnostic_name} | FINISHED] Every stationary FC16 "
        "write was acknowledged and the final open state was healthy. This "
        "measures write-transport throughput; it does not prove mechanical "
        "tracking or execution of every repeated identical target."
    )
    return final_status, frequency_target_observed


def get_diagnostic_prompt(
    args: argparse.Namespace,
) -> tuple[str, str]:
    """Return the confirmation token and prompt text for the diagnostic."""
    if args.diagnostic == "activation":
        return (
            "ACTIVATE",
            "Reset and activate; activation performs calibration motion.",
        )
    if args.diagnostic == "basic-motion":
        return (
            "CONTROL",
            "Activate, then run rPR "
            f"0 -> {args.mid_position} -> 255 -> 0 with "
            f"fixed rSP={args.speed_request} and rFR=0.",
        )
    if args.diagnostic == "position":
        return (
            "POSITION",
            "Activate, then test rPR "
            f"{args.position_requests} with fixed "
            f"rSP={args.speed_request} and rFR=0.",
        )
    if args.diagnostic == "speed":
        return (
            "SPEED",
            "Activate, then compare rSP "
            f"{tuple(sorted(set(args.speed_requests)))} over "
            f"{args.speed_repetitions} identical closing stroke(s) each; "
            "rFR remains 0.",
        )
    if args.diagnostic == "force-request":
        return (
            "CALIBRATE",
            "Activate with a clear sweep, open, safely position the secured "
            "fixture, then close on it at rFR "
            f"{tuple(sorted(set(args.force_requests)))} with fixed "
            f"rSP={args.speed_request}; reopen after every trial.",
        )
    if args.diagnostic == "object-detection":
        return (
            "CALIBRATE",
            "Activate with a clear sweep, open, safely position the secured "
            "fixture, then close on it at "
            f"rSP={args.speed_request}, rFR={args.force_request}, "
            "verify stable gOBJ=2, then reopen.",
        )
    if args.diagnostic == "motion-state":
        return (
            "TRACE",
            "Activate, then trace one slow "
            f"{SPEED_TEST_START_POSITION}->"
            f"{SPEED_TEST_TARGET_POSITION} closing stroke and reopen.",
        )
    if args.diagnostic == "fc16-frequency":
        return (
            "WRITE RATE",
            "Activate, open, then measure stationary identical FC16 write "
            f"acknowledgements at {args.frequency_hz:.1f} Hz without "
            "commanding repeated physical strokes.",
        )
    raise ValueError(
        f"no motion-safety plan exists for {args.diagnostic}"
    )


def main() -> int:
    args = parse_args()
    client = None
    activation_in_progress = False
    motion_in_progress = False
    cleanup_error = None
    diagnostic_conclusive = True

    try:
        print(
            "[PHASE 0 | VALIDATE TEST SETUP | START] "
            f"Selected diagnostic={args.diagnostic}; checking arguments, "
            "ACTION REQUEST encoding, PyModbus, and USB latency configuration."
        )

        validate_arguments(args)
        print(
            "[PHASE 0 | VALIDATE TEST SETUP | CHECK 1/4] "
            "Command-line arguments and diagnostic isolation are valid."
        )

        check_action_request_encoding()
        print(
            "[PHASE 0 | VALIDATE TEST SETUP | CHECK 2/4] "
            "ACTION REQUEST encoding matches the expected register values."
        )

        try:
            import pymodbus
            from pymodbus.client import ModbusSerialClient
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "[PHASE 0 | VALIDATE TEST SETUP | FAIL] "
                "PyModbus is not installed; install pymodbus>=3,<4."
            ) from error

        pymodbus_version = getattr(pymodbus, "__version__", "unknown")
        if not pymodbus_version.startswith("3."):
            raise RuntimeError(
                "[PHASE 0 | VALIDATE TEST SETUP | FAIL] "
                "This diagnostic requires pymodbus>=3,<4; found "
                f"{pymodbus_version}."
            )

        print(
            "[PHASE 0 | VALIDATE TEST SETUP | CHECK 3/4] "
            f"PyModbus {pymodbus_version} is supported."
        )

        frequency_diagnostics = {"fc04-frequency", "fc16-frequency"}
        if args.diagnostic in frequency_diagnostics:
            latency_timer = read_usb_serial_latency_timer(args.port)

            if latency_timer is None:
                raise RuntimeError(
                    "[PHASE 0 | VALIDATE TEST SETUP | CHECK 4/4 | FAIL] "
                    "USB-serial latency_timer is unavailable for the selected "
                    "port. Ask the system administrator to configure the "
                    "selected FTDI adapter with latency_timer=1."
                )
            elif latency_timer != 1:
                raise RuntimeError(
                    "[PHASE 0 | VALIDATE TEST SETUP | CHECK 4/4 | FAIL] "
                    f"FTDI latency_timer={latency_timer}; ask the system "
                    "administrator to configure latency_timer=1 before "
                    "frequency diagnostics."
                )
            else:
                print(
                    "[PHASE 0 | VALIDATE TEST SETUP | CHECK 4/4] "
                    "FTDI latency_timer=1 is configured."
                )
        else:
            print(
                "[PHASE 0 | VALIDATE TEST SETUP | CHECK 4/4] "
                "The FTDI latency check is not required for this diagnostic."
            )

        print(
            "[PHASE 0 | VALIDATE TEST SETUP | FINISHED] "
            "Test setup validation completed successfully."
        )

        print(
            "\n[PHASE 1 | OPEN SERIAL TRANSPORT | START] "
            f"Opening {args.port} at {args.baudrate} baud, 8-N-1."
        )
        client = ModbusSerialClient(
            port=args.port,
            baudrate=args.baudrate,
            bytesize=8,
            parity="N",
            stopbits=1,
            timeout=args.io_timeout,
        )
        if not client.connect():
            raise ConnectionError(
                "[PHASE 1 | OPEN SERIAL TRANSPORT | FAIL] "
                f"Could not open serial port {args.port}."
            )

        print(
            "[PHASE 1 | OPEN SERIAL TRANSPORT | FINISHED] "
            f"Serial transport {args.port} opened successfully."
        )

        print(
            "\n[PHASE 2 | READ GRIPPER STATUS | START] "
            f"Requesting {args.num_consecutive_read_only_samples} FC04 "
            f"baseline sample(s) from slave 0x{args.slave_id:02X}."
        )
        register_samples = read_gripper_status(client, args)
        print(
            "[PHASE 2 | READ GRIPPER STATUS | FINISHED] "
            f"Received {len(register_samples)} valid FC04 status sample(s)."
        )

        print(
            "\n[PHASE 3 | ANALYZE GRIPPER STATUS | START] "
            f"Decoding and evaluating {len(register_samples)} collected "
            "sample(s)."
        )
        last_baseline_status, state_is_acceptable = analyze_gripper_status(
            register_samples
        )

        if not state_is_acceptable:
            baseline_is_fault_free = (
                last_baseline_status["gFLT"] == 0
                and last_baseline_status["kFLT"] == 0
            )
            diagnostic_requires_activation = args.diagnostic not in {
                "read-only",
                "fc04-frequency",
            }
            if baseline_is_fault_free and diagnostic_requires_activation:
                print(
                    "[PHASE 3 | ANALYZE GRIPPER STATUS | WARNING] "
                    "The final baseline is fault-free but not ready/reset. "
                    "The selected write diagnostic may continue because "
                    "Phase 5 begins with an explicit reset."
                )
            else:
                raise RuntimeError(
                    "[PHASE 3 | ANALYZE GRIPPER STATUS | FAIL] "
                    "The final baseline state is neither activated-ready nor "
                    "reset-fault-free, or it contains a fault. No diagnostic "
                    "operation will continue."
                )
        else:
            print(
                "[PHASE 3 | ANALYZE GRIPPER STATUS | RESULT] "
                "The final baseline state is fault-free and acceptable."
            )
        print(
            "[PHASE 3 | ANALYZE GRIPPER STATUS | FINISHED] "
            "Status decoding and baseline health gating completed."
        )

        if args.diagnostic == "read-only":
            print(
                "\n[READ-ONLY DIAGNOSTIC | OPERATIONS FINISHED] "
                "Phases 0-3 completed successfully."
            )
        elif args.diagnostic == "fc04-frequency":
            diagnostic_conclusive = run_fc04_frequency_diagnostic(
                client,
                slave_id=args.slave_id,
                sample_count=args.frequency_samples,
                requested_frequency_hz=args.frequency_hz,
            )
            print(
                "\n[DIAGNOSTIC | OPERATIONS FINISHED] "
                "The FC04 frequency diagnostic completed successfully."
            )
        else:
            confirmation_token, prompt_text = (
                get_diagnostic_prompt(args)
            )
            contact_diagnostic = args.diagnostic in {
                "force-request",
                "object-detection",
            }

            print(
                "\n[PHASE 4 | CONFIRM MOTION SAFETY | START] "
                f"Preparing the {args.diagnostic} diagnostic."
            )
            print(
                "[PHASE 4 | CONFIRM MOTION SAFETY | PLAN] "
                f"{prompt_text}"
            )
            print(
                "[PHASE 4 | CONFIRM MOTION SAFETY | WARNING] "
                "Activation calibration moves the fingers before the selected "
                "diagnostic begins."
            )

            if contact_diagnostic:
                print(
                    "[PHASE 4 | CONFIRM MOTION SAFETY | WARNING] "
                    "Keep the entire calibration sweep clear. The fixture is "
                    "positioned only after activation and a verified open "
                    "command. Never use a hand as the test object."
                )
                print(
                    "[PHASE 4 | CONFIRM MOTION SAFETY | WARNING] "
                    "The fixture must be rigid, centrally aligned, secured "
                    "against ejection or falling, and rated for the gripper "
                    "model's maximum possible force."
                )
                print(
                    "[PHASE 4 | CONFIRM MOTION SAFETY | LIMITATION] "
                    "gCU is sampled motor current, not measured gripping "
                    "force. Newtons require an external calibrated load cell."
                )
                print(
                    "[PHASE 4 | CONFIRM MOTION SAFETY | LIMITATION] "
                    "rFR=0 requests the gripper's minimum force; it does not "
                    "mean zero physical force."
                )
                print(
                    "[PHASE 4 | CONFIRM MOTION SAFETY | RECOVERY LIMIT] "
                    "On failure or Ctrl+C, cleanup requests only a non-safety-"
                    "rated rGTO=0 stop. The fixture may remain held, and "
                    "removing power may release it."
                )
            else:
                print(
                    "[PHASE 4 | CONFIRM MOTION SAFETY | WARNING] "
                    "Keep hands, cables, tools, and objects clear of the "
                    "fingers."
                )

            print(
                "[PHASE 4 | CONFIRM MOTION SAFETY | WARNING] "
                "Keep the gripper visible and be ready to remove power."
            )

            if args.yes:
                print(
                    "[PHASE 4 | CONFIRM MOTION SAFETY | SKIPPED] "
                    "Typed confirmation was bypassed by --yes."
                )
            else:
                answer = input(
                    "[PHASE 4 | CONFIRM MOTION SAFETY | INPUT] "
                    f"Type {confirmation_token} to continue: "
                ).strip()
                if answer != confirmation_token:
                    raise RuntimeError(
                        "[PHASE 4 | CONFIRM MOTION SAFETY | ABORTED] "
                        "Confirmation did not match; no Modbus write was sent."
                    )

            print(
                "[PHASE 4 | CONFIRM MOTION SAFETY | FINISHED] "
                "The safety gate passed; activation writes are permitted."
            )

            activation_in_progress = True
            activation_status = activate_gripper(
                client,
                timeout=args.activation_timeout,
                poll_interval=args.poll_interval,
                slave_id=args.slave_id,
            )
            activation_in_progress = False

            if args.diagnostic != "activation":
                motion_in_progress = True

                if args.diagnostic == "basic-motion":
                    run_position_diagnostic(
                        client,
                        args,
                        activation_status,
                        diagnostic_name="BASIC MOTION DIAGNOSTIC",
                        target_positions=(0, args.mid_position, 255, 0),
                    )
                elif args.diagnostic == "position":
                    run_position_diagnostic(
                        client,
                        args,
                        activation_status,
                        diagnostic_name="POSITION DIAGNOSTIC",
                        target_positions=args.position_requests,
                    )
                elif args.diagnostic == "speed":
                    _, diagnostic_conclusive = run_speed_diagnostic(
                        client,
                        args,
                        activation_status,
                    )
                elif args.diagnostic == "force-request":
                    run_force_request_diagnostic(
                        client,
                        args,
                        activation_status,
                    )
                elif args.diagnostic == "object-detection":
                    run_object_detection_diagnostic(
                        client,
                        args,
                        activation_status,
                    )
                elif args.diagnostic == "motion-state":
                    _, diagnostic_conclusive = run_motion_state_diagnostic(
                        client,
                        args,
                        activation_status,
                    )
                elif args.diagnostic == "fc16-frequency":
                    _, diagnostic_conclusive = run_fc16_frequency_diagnostic(
                        client,
                        args,
                        activation_status,
                    )
                else:
                    raise RuntimeError(
                        f"unsupported diagnostic: {args.diagnostic}"
                    )

                motion_in_progress = False

            print(
                "\n[DIAGNOSTIC | OPERATIONS FINISHED] "
                f"The {args.diagnostic} diagnostic procedure completed."
            )
    except KeyboardInterrupt:
        print(
            "\n[DIAGNOSTIC | ABORTED] Interrupted by user.",
            file=sys.stderr,
        )
        return 130
    except Exception as error:
        print(f"\n[DIAGNOSTIC | FAIL] {error}", file=sys.stderr)
        return 1
    finally:
        if client is not None:
            print(
                "\n[PHASE 7 | CLEAN UP | START] Performing recovery if "
                "needed and releasing serial-client resources."
            )

            if activation_in_progress:
                try:
                    print(
                        "[PHASE 7 | CLEAN UP | RECOVERY] "
                        "Requesting rACT=0 after interrupted activation.",
                        file=sys.stderr,
                    )
                    request_gripper_reset(client, args.slave_id)
                    print(
                        "[PHASE 7 | CLEAN UP | RECOVERY ACKNOWLEDGED] "
                        "The FC16 reset request was accepted; the resulting "
                        "gripper state was not verified.",
                        file=sys.stderr,
                    )
                except Exception as reset_error:
                    print(
                        "[PHASE 7 | CLEAN UP | WARNING] "
                        f"Activation-abort reset request failed: {reset_error}",
                        file=sys.stderr,
                    )
            elif motion_in_progress:
                try:
                    print(
                        "[PHASE 7 | CLEAN UP | RECOVERY] "
                        "Requesting rGTO=0 after interrupted motion. This is "
                        "not a safety-rated stop.",
                        file=sys.stderr,
                    )
                    request_motion_stop(client, args.slave_id)
                    print(
                        "[PHASE 7 | CLEAN UP | RECOVERY ACKNOWLEDGED] "
                        "The FC16 motion-stop request was accepted; physical "
                        "stopping and any held-object state were not verified.",
                        file=sys.stderr,
                    )
                except Exception as stop_error:
                    print(
                        "[PHASE 7 | CLEAN UP | WARNING] "
                        f"Motion-stop request failed: {stop_error}",
                        file=sys.stderr,
                    )

            try:
                client.close()
                print(
                    "[PHASE 7 | CLEAN UP | FINISHED] "
                    "Serial-client resources were released. Normal cleanup "
                    "does not reset or deactivate the gripper."
                )
            except Exception as close_error:
                cleanup_error = close_error
                print(
                    "[PHASE 7 | CLEAN UP | WARNING] "
                    f"Serial cleanup failed: {close_error}",
                    file=sys.stderr,
                )

    if cleanup_error is not None:
        print(
            "\n[DIAGNOSTIC | FAIL] Requested diagnostic operations "
            "completed, but serial-client cleanup failed.",
            file=sys.stderr,
        )
        return 1

    if not diagnostic_conclusive:
        print(
            "\n[DIAGNOSTIC | INCONCLUSIVE] The requested procedure and "
            "cleanup completed, but the expected comparative or in-motion "
            "evidence was not observed."
        )
        return 2

    print(
        "\n[DIAGNOSTIC | PASS] All requested standalone diagnostics and "
        "cleanup completed successfully."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

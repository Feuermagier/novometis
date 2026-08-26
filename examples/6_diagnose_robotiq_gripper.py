#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Run focused Robotiq diagnostics through the public Polymetis API.

Start ``launch_gripper.py`` in another terminal and leave it running. Set
``ROBOTIQ_PORT`` to this installation's stable ``/dev/serial/by-id/...`` path.
For example, from the repository root::

    mamba activate polymetis
    python polymetis/python/scripts/launch_gripper.py \
        gripper=robotiq_2f \
        gripper.port="${ROBOTIQ_PORT}" \
        gripper.hz=100

The five diagnostics intentionally have different meanings:

* ``position`` verifies several commanded widths through hardware feedback.
* ``speed`` compares durations over identical unloaded closing strokes.
* ``force-request`` verifies several force requests during unloaded motion;
  it does not measure physical force.
* ``state-frequency`` measures cached GetState RPCs and distinct hardware
  snapshots. The latter approximates the idle hardware-client FC04 loop rate.
* ``control-frequency`` repeatedly sends a stationary open target while also
  observing distinct hardware snapshots. This loads the FC04 + gRPC + FC16
  path, but the last-value command cache prevents proving that every submitted
  command became a physical FC16 write.

Do not run the standalone Modbus diagnostic at the same time: only one process
may own the serial port. All motion tests require a clear finger workspace and
finish with the gripper open.

An exit status of 2 means a frequency diagnostic completed safely but did not
provide conclusive evidence that its target rate was achieved.
"""

import argparse
import math
import statistics
import sys
import threading
import time

from polymetis import GripperInterface


ERROR_DESCRIPTIONS = {
    -3: "latest command failed or was rejected",
    -2: "gripper is not activated and ready",
    -1: "hardware-state communication failed",
    0: "healthy",
}
EXPECTED_GRIPPER_TYPE = "robotiq_2f"


def parse_float_list(text, option_name):
    """Parse a nonempty comma-separated list of finite floats."""
    try:
        values = tuple(float(item.strip()) for item in text.split(","))
    except ValueError as error:
        raise ValueError(
            f"{option_name} must contain comma-separated numbers"
        ) from error

    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError(f"{option_name} must contain finite numbers")
    return values


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diagnostic",
        choices=(
            "position",
            "speed",
            "force-request",
            "state-frequency",
            "control-frequency",
        ),
        default="state-frequency",
    )
    parser.add_argument("--server-ip", default="localhost")
    parser.add_argument("--server-port", type=int, default=50052)
    parser.add_argument(
        "--positions",
        default="1.0,0.75,0.5,0.25,0.5,0.75,1.0",
        help="opening fractions for the position diagnostic",
    )
    parser.add_argument(
        "--speeds",
        default="0.020,0.085,0.150",
        help="m/s values for the speed diagnostic",
    )
    parser.add_argument(
        "--forces",
        default="0,50,100",
        help="nominal N requests for the unloaded force-request diagnostic",
    )
    parser.add_argument(
        "--motion-speed",
        type=float,
        default=0.020,
        help="m/s for position, force-request, setup, and return motions",
    )
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--stroke-open-fraction", type=float, default=0.75)
    parser.add_argument("--stroke-closed-fraction", type=float, default=0.25)
    parser.add_argument("--motion-timeout", type=float, default=15.0)
    parser.add_argument("--startup-timeout", type=float, default=15.0)
    parser.add_argument("--poll-interval", type=float, default=0.02)
    parser.add_argument("--width-tolerance", type=float, default=0.003)
    parser.add_argument("--dwell", type=float, default=0.2)
    parser.add_argument(
        "--frequency-hz",
        type=float,
        default=100.0,
        help="stationary Goto submission rate for control-frequency",
    )
    parser.add_argument("--frequency-duration", type=float, default=3.0)
    parser.add_argument(
        "--monitor-hz",
        type=float,
        default=500.0,
        help="GetState polling rate used to observe hardware snapshots",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="skip the typed confirmation for clear-workspace tests",
    )
    return parser.parse_args()


def validate_args(args):
    if not 1 <= args.server_port <= 65535:
        raise ValueError("--server-port must be between 1 and 65535")
    if not 0.020 <= args.motion_speed <= 0.150:
        raise ValueError("--motion-speed must be between 0.020 and 0.150 m/s")
    if args.repetitions < 1:
        raise ValueError("--repetitions must be at least 1")
    if not (
        0.0 <= args.stroke_closed_fraction
        < args.stroke_open_fraction
        <= 1.0
    ):
        raise ValueError(
            "stroke fractions must satisfy 0 <= closed < open <= 1"
        )
    positive_values = {
        "--motion-timeout": args.motion_timeout,
        "--startup-timeout": args.startup_timeout,
        "--poll-interval": args.poll_interval,
        "--width-tolerance": args.width_tolerance,
        "--frequency-hz": args.frequency_hz,
        "--frequency-duration": args.frequency_duration,
        "--monitor-hz": args.monitor_hz,
    }
    for name, value in positive_values.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if args.frequency_hz > 500.0 or args.monitor_hz > 2000.0:
        raise ValueError("frequency-hz must be <=500 and monitor-hz <=2000")
    if args.dwell < 0.0:
        raise ValueError("--dwell cannot be negative")

    positions = parse_float_list(args.positions, "--positions")
    if len(positions) < 2 or len(set(positions)) < 2:
        raise ValueError("--positions requires at least two distinct values")
    if any(not 0.0 <= value <= 1.0 for value in positions):
        raise ValueError("--positions values must be between 0 and 1")

    speeds = parse_float_list(args.speeds, "--speeds")
    if len(set(speeds)) < 2:
        raise ValueError("--speeds requires at least two distinct values")
    if any(not 0.020 <= value <= 0.150 for value in speeds):
        raise ValueError("--speeds values must be between 0.020 and 0.150 m/s")

    forces = parse_float_list(args.forces, "--forces")
    if len(set(forces)) < 2:
        raise ValueError("--forces requires at least two distinct values")
    for force in forces:
        if force < 0.0 or force > 235.0 or 0.0 < force < 20.0:
            raise ValueError(
                "--forces values must be 0 or between 20 and 235 nominal N"
            )

    return positions, speeds, forces


def validate_gripper_type(gripper, args):
    reported_type = getattr(gripper.metadata, "gripper_type", "").strip()
    if reported_type != EXPECTED_GRIPPER_TYPE:
        raise RuntimeError(
            f"Expected a {EXPECTED_GRIPPER_TYPE} service at "
            f"{args.server_ip}:{args.server_port}, received "
            f"{reported_type or '<missing>'}."
        )


def timestamp_key(state):
    return state.timestamp.seconds, state.timestamp.nanos


def timestamp_seconds(timestamp):
    seconds, nanos = timestamp
    return seconds + nanos / 1_000_000_000.0


def describe_error(error_code):
    if error_code in ERROR_DESCRIPTIONS:
        return ERROR_DESCRIPTIONS[error_code]
    if error_code > 0:
        return f"Robotiq vendor fault byte 0x{error_code:02X}"
    return "unknown adapter error"


def format_state(state):
    return (
        f"width={state.width * 1000.0:6.2f} mm, "
        f"moving={state.is_moving}, grasped={state.is_grasped}, "
        f"write_accepted={state.prev_command_successful}, "
        f"error={state.error_code} ({describe_error(state.error_code)})"
    )


def connect_to_server(args):
    try:
        return GripperInterface(
            ip_address=args.server_ip,
            port=args.server_port,
        )
    except Exception as error:
        raise RuntimeError(
            "Could not initialize GripperInterface. Start launch_gripper.py "
            "and wait until its hardware client has registered."
        ) from error


def wait_for_initial_state(gripper, timeout, poll_interval):
    """Require two observations so stale cached state cannot look healthy."""
    deadline = time.monotonic() + timeout
    first_timestamp = None
    last_state = None

    while time.monotonic() < deadline:
        state = gripper.get_state()
        last_state = state
        current_timestamp = timestamp_key(state)
        if current_timestamp == (0, 0):
            time.sleep(poll_interval)
            continue
        if first_timestamp is None:
            first_timestamp = current_timestamp
        elif current_timestamp != first_timestamp:
            if state.error_code != 0:
                raise RuntimeError("Unhealthy initial state: " + format_state(state))
            return state
        time.sleep(poll_interval)

    detail = format_state(last_state) if last_state is not None else "no state"
    raise TimeoutError(
        "No advancing hardware timestamp arrived; the server cache may be "
        f"stale. Last state: {detail}"
    )


def wait_for_target(
    gripper,
    target_width,
    previous_timestamp,
    timeout,
    poll_interval,
    width_tolerance,
):
    """Wait for post-command hardware feedback and a terminal target width."""
    deadline = time.monotonic() + timeout
    last_timestamp = previous_timestamp
    new_snapshot_count = 0
    last_state = None

    while time.monotonic() < deadline:
        state = gripper.get_state()
        last_state = state
        current_timestamp = timestamp_key(state)
        if current_timestamp != last_timestamp:
            new_snapshot_count += 1
            last_timestamp = current_timestamp

        if state.error_code != 0:
            raise RuntimeError("Gripper reported an error: " + format_state(state))
        if state.is_grasped:
            raise RuntimeError(
                "Unexpected closing contact in a clear-workspace test: "
                + format_state(state)
            )

        target_reached = abs(state.width - target_width) <= width_tolerance
        if (
            new_snapshot_count >= 2
            and state.prev_command_successful
            and not state.is_moving
            and target_reached
        ):
            return state
        time.sleep(poll_interval)

    detail = format_state(last_state) if last_state is not None else "no state"
    raise TimeoutError(
        f"Target {target_width * 1000.0:.2f} mm was not verified within "
        f"{timeout:.1f} s; last state: {detail}"
    )


def execute_motion(gripper, label, width, speed, force, args):
    """Send one command and verify it through subsequent hardware feedback."""
    before = gripper.get_state()
    if before.error_code != 0:
        raise RuntimeError("Refusing motion: " + format_state(before))

    started_at = time.monotonic()
    print(
        f"[COMMAND] {label}: width={width * 1000.0:.2f} mm, "
        f"speed={speed:.3f} m/s, nominal force={force:.1f} N"
    )
    # One command at a time keeps the original interface's one-slot queue
    # empty. Mechanical completion is established from state, not queue timing.
    gripper.goto(width=width, speed=speed, force=force, blocking=False)
    state = wait_for_target(
        gripper=gripper,
        target_width=width,
        previous_timestamp=timestamp_key(before),
        timeout=args.motion_timeout,
        poll_interval=args.poll_interval,
        width_tolerance=args.width_tolerance,
    )
    duration = time.monotonic() - started_at
    print(f"[REACHED] {label}: duration={duration:.4f} s, {format_state(state)}")
    return duration, state


def confirm_motion(args, detail):
    print("\nThis diagnostic sends motion commands with no object expected.")
    print(detail)
    print("Clear the complete finger sweep of hands, objects, cables, and tools.")
    print("A force request is not a measured or guaranteed physical force.")
    if args.yes:
        return
    if input("Type RUN to continue: ").strip() != "RUN":
        raise RuntimeError("Diagnostic cancelled; no motion command was sent")


def run_position_diagnostic(gripper, max_width, positions, args):
    confirm_motion(
        args,
        "Plan: test opening fractions " + ", ".join(f"{p:.2f}" for p in positions),
    )
    observations = []
    for index, fraction in enumerate(positions, start=1):
        target = fraction * max_width
        duration, state = execute_motion(
            gripper,
            f"position {index}/{len(positions)} (fraction={fraction:.2f})",
            target,
            args.motion_speed,
            0.0,
            args,
        )
        observations.append((fraction, target, state.width, duration))
        time.sleep(args.dwell)

    if positions[-1] != 1.0:
        execute_motion(
            gripper, "final reopen", max_width, args.motion_speed, 0.0, args
        )

    print("\n[POSITION | SUMMARY]")
    for fraction, target, observed, duration in observations:
        print(
            f"  fraction={fraction:.2f}: target={target * 1000.0:6.2f} mm, "
            f"observed={observed * 1000.0:6.2f} mm, "
            f"error={abs(observed - target) * 1000.0:5.2f} mm, "
            f"duration={duration:.4f} s"
        )


def run_speed_diagnostic(gripper, max_width, speeds, args):
    open_width = args.stroke_open_fraction * max_width
    closed_width = args.stroke_closed_fraction * max_width
    confirm_motion(
        args,
        f"Plan: compare speeds {speeds} m/s over "
        f"{open_width * 1000.0:.1f}->{closed_width * 1000.0:.1f} mm strokes.",
    )
    measured = {speed: [] for speed in speeds}

    execute_motion(
        gripper, "speed-test setup", open_width, args.motion_speed, 0.0, args
    )
    for speed in speeds:
        for repetition in range(1, args.repetitions + 1):
            duration, _ = execute_motion(
                gripper,
                f"speed={speed:.3f}, repetition={repetition}",
                closed_width,
                speed,
                0.0,
                args,
            )
            measured[speed].append(duration)
            execute_motion(
                gripper,
                "return to common start",
                open_width,
                args.motion_speed,
                0.0,
                args,
            )
            time.sleep(args.dwell)

    execute_motion(gripper, "final reopen", max_width, args.motion_speed, 0.0, args)
    print("\n[SPEED | SUMMARY]")
    medians = {}
    for speed in speeds:
        medians[speed] = statistics.median(measured[speed])
        durations = ", ".join(f"{value:.4f}" for value in measured[speed])
        print(
            f"  {speed:.3f} m/s: durations=[{durations}] s, "
            f"median={medians[speed]:.4f} s"
        )
    if medians[max(speeds)] >= medians[min(speeds)]:
        print("[SPEED | WARNING] Highest speed was not faster than lowest speed.")
    else:
        print("[SPEED | RESULT] Higher requested speed reduced stroke duration.")


def run_force_request_diagnostic(gripper, max_width, forces, args):
    open_width = args.stroke_open_fraction * max_width
    closed_width = args.stroke_closed_fraction * max_width
    confirm_motion(
        args,
        "Plan: send unloaded nominal force requests "
        + ", ".join(f"{force:g} N" for force in forces)
        + ". This validates request handling, not force output.",
    )
    observations = []
    execute_motion(
        gripper, "force-test setup", open_width, args.motion_speed, 0.0, args
    )
    for force in forces:
        duration, state = execute_motion(
            gripper,
            f"nominal force request={force:g} N",
            closed_width,
            args.motion_speed,
            force,
            args,
        )
        observations.append((force, duration, state.width))
        execute_motion(
            gripper,
            "return to common start",
            open_width,
            args.motion_speed,
            0.0,
            args,
        )
        time.sleep(args.dwell)

    execute_motion(gripper, "final reopen", max_width, args.motion_speed, 0.0, args)
    print("\n[FORCE REQUEST | SUMMARY]")
    for force, duration, width in observations:
        print(
            f"  request={force:6.1f} nominal N: accepted, "
            f"terminal width={width * 1000.0:6.2f} mm, "
            f"unloaded duration={duration:.4f} s"
        )
    print(
        "[FORCE REQUEST | LIMIT] No object or load cell was used, so this does "
        "not validate contact force, force accuracy, or force limits."
    )


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[round(fraction * (len(ordered) - 1))]


def rate_from_times(times):
    if len(times) < 2 or times[-1] <= times[0]:
        return 0.0
    return (len(times) - 1) / (times[-1] - times[0])


def collect_state_samples(gripper, duration, requested_hz, start_at=None):
    """Poll cached state at a bounded rate without catch-up bursts."""
    if start_at is None:
        start_at = time.monotonic()
    deadline = start_at + duration
    period = 1.0 / requested_hz
    next_start = start_at
    samples = []
    late_starts = 0

    while True:
        sleep_duration = next_start - time.monotonic()
        if sleep_duration > 0.0:
            time.sleep(sleep_duration)
        started_at = time.monotonic()
        if started_at >= deadline:
            break
        if started_at - next_start > period:
            late_starts += 1
        next_start = started_at + period
        state = gripper.get_state()
        finished_at = time.monotonic()
        samples.append((started_at, finished_at, state))

    return samples, late_starts


def summarize_state_frequency(label, samples, late_starts, requested_hz):
    if len(samples) < 2:
        raise RuntimeError(f"{label}: fewer than two state samples")
    errors = [sample[2] for sample in samples if sample[2].error_code != 0]
    if errors:
        raise RuntimeError(f"{label}: unhealthy sample: {format_state(errors[0])}")

    starts = [sample[0] for sample in samples]
    latencies = [sample[1] - sample[0] for sample in samples]
    unique_snapshots = []
    previous_key = None
    for started_at, finished_at, state in samples:
        key = timestamp_key(state)
        if key != (0, 0) and key != previous_key:
            unique_snapshots.append((key, finished_at))
            previous_key = key

    host_unique_times = [item[1] for item in unique_snapshots]
    hardware_times = [timestamp_seconds(item[0]) for item in unique_snapshots]
    query_rate = rate_from_times(starts)
    host_update_rate = rate_from_times(host_unique_times)
    hardware_update_rate = rate_from_times(hardware_times)

    print(f"\n[{label} | RESULT]")
    print(
        f"  GetState samples={len(samples)}, requested query rate="
        f"{requested_hz:.2f} Hz, achieved query start rate={query_rate:.2f} Hz, "
        f"starts over one period late={late_starts}"
    )
    print(
        f"  distinct hardware snapshots={len(unique_snapshots)}, "
        f"host-observed update rate={host_update_rate:.2f} Hz, "
        f"hardware-timestamp rate={hardware_update_rate:.2f} Hz"
    )
    print(
        "  GetState latency ms: "
        f"min={min(latencies) * 1000.0:.3f}, "
        f"median={statistics.median(latencies) * 1000.0:.3f}, "
        f"mean={statistics.fmean(latencies) * 1000.0:.3f}, "
        f"p95={percentile(latencies, 0.95) * 1000.0:.3f}, "
        f"max={max(latencies) * 1000.0:.3f}"
    )
    return {
        "query_rate": query_rate,
        "host_update_rate": host_update_rate,
        "hardware_update_rate": hardware_update_rate,
        "unique_count": len(unique_snapshots),
    }


def run_state_frequency_diagnostic(gripper, args):
    print(
        "[STATE FREQUENCY | START] Read-only sampling of the server's cached "
        "GripperState. Distinct timestamps correspond to successful FC04 "
        "snapshots published by the hardware client."
    )
    samples, late = collect_state_samples(
        gripper,
        duration=args.frequency_duration,
        requested_hz=args.monitor_hz,
    )
    result = summarize_state_frequency(
        "STATE FREQUENCY", samples, late, args.monitor_hz
    )
    declared_hz = float(gripper.metadata.hz)
    print(f"  hardware client declared rate={declared_hz:.2f} Hz")
    if args.monitor_hz <= declared_hz:
        print(
            "[STATE FREQUENCY | INCONCLUSIVE] Monitor rate is not above the "
            "declared hardware rate, so updates may be missed."
        )
        return False
    elif result["host_update_rate"] >= 0.9 * declared_hz:
        print("[STATE FREQUENCY | PASS] At least 90% of declared rate observed.")
        return True
    else:
        print(
            "[STATE FREQUENCY | INCONCLUSIVE] Distinct update rate was below "
            "90% of the declared best-effort rate."
        )
        return False


def run_stationary_command_phase(gripper, max_width, args, start_at):
    period = 1.0 / args.frequency_hz
    deadline = start_at + args.frequency_duration
    next_start = start_at
    starts = []
    latencies = []
    late_starts = 0

    while True:
        sleep_duration = next_start - time.monotonic()
        if sleep_duration > 0.0:
            time.sleep(sleep_duration)
        started_at = time.monotonic()
        if started_at >= deadline:
            break
        starts.append(started_at)
        if started_at - next_start > period:
            late_starts += 1
        next_start = started_at + period
        # The target is already open, so this loads command transport without
        # deliberately cycling the mechanics.
        gripper.goto(
            width=max_width,
            speed=args.motion_speed,
            force=0.0,
            blocking=True,
        )
        latencies.append(time.monotonic() - started_at)

    return starts, latencies, late_starts


def run_control_frequency_diagnostic(gripper, max_width, args):
    confirm_motion(
        args,
        f"Plan: establish full open, then submit stationary open commands at "
        f"{args.frequency_hz:.1f} Hz for {args.frequency_duration:.1f} s.",
    )
    execute_motion(
        gripper, "stationary-frequency setup", max_width, args.motion_speed, 0.0, args
    )

    print("\n[CONTROL FREQUENCY | IDLE BASELINE]")
    idle_samples, idle_late = collect_state_samples(
        gripper,
        duration=args.frequency_duration,
        requested_hz=args.monitor_hz,
    )
    idle = summarize_state_frequency(
        "CONTROL FREQUENCY IDLE", idle_samples, idle_late, args.monitor_hz
    )

    print("\n[CONTROL FREQUENCY | STATIONARY COMMAND LOAD]")
    phase_start = time.monotonic() + 0.1
    monitor_result = {}

    def monitor_state():
        samples, late = collect_state_samples(
            gripper,
            duration=args.frequency_duration,
            requested_hz=args.monitor_hz,
            start_at=phase_start,
        )
        monitor_result["samples"] = samples
        monitor_result["late"] = late

    monitor_thread = threading.Thread(target=monitor_state, daemon=True)
    monitor_thread.start()
    starts, latencies, late_starts = run_stationary_command_phase(
        gripper, max_width, args, phase_start
    )
    monitor_thread.join(timeout=args.frequency_duration + 2.0)
    if monitor_thread.is_alive():
        raise RuntimeError("state monitor did not finish")
    if len(starts) < 2:
        raise RuntimeError("fewer than two stationary commands completed")

    loaded = summarize_state_frequency(
        "CONTROL FREQUENCY LOADED",
        monitor_result["samples"],
        monitor_result["late"],
        args.monitor_hz,
    )
    command_rate = rate_from_times(starts)
    print("\n[CONTROL FREQUENCY | COMMAND RESULT]")
    print(
        f"  Goto RPCs completed={len(starts)}, requested start rate="
        f"{args.frequency_hz:.2f} Hz, achieved start rate={command_rate:.2f} Hz, "
        f"starts over one period late={late_starts}"
    )
    print(
        "  Goto latency ms: "
        f"min={min(latencies) * 1000.0:.3f}, "
        f"median={statistics.median(latencies) * 1000.0:.3f}, "
        f"mean={statistics.fmean(latencies) * 1000.0:.3f}, "
        f"p95={percentile(latencies, 0.95) * 1000.0:.3f}, "
        f"max={max(latencies) * 1000.0:.3f}"
    )
    print(
        f"  distinct state rate: idle={idle['host_update_rate']:.2f} Hz, "
        f"under stationary command load={loaded['host_update_rate']:.2f} Hz"
    )

    final_before = gripper.get_state()
    wait_for_target(
        gripper,
        target_width=max_width,
        previous_timestamp=timestamp_key(final_before),
        timeout=args.motion_timeout,
        poll_interval=args.poll_interval,
        width_tolerance=args.width_tolerance,
    )

    observed = (
        command_rate >= 0.9 * args.frequency_hz
        and loaded["host_update_rate"] >= 0.9 * args.frequency_hz
    )
    if observed:
        print(
            "[CONTROL FREQUENCY | OBSERVED] Command ingress and distinct "
            "loaded hardware snapshots both reached at least 90% of the target."
        )
    else:
        print(
            "[CONTROL FREQUENCY | NOT OBSERVED] The complete loaded architecture "
            "did not show 90% of the requested rate."
        )
    print(
        "[CONTROL FREQUENCY | LIMIT] Goto only updates a last-value cache. "
        "Completed Goto RPCs are not a count of FC16 writes, and this does not "
        "measure the gripper's internal control-loop frequency."
    )
    return observed


def main():
    args = parse_args()
    gripper = None
    success_criterion_observed = True
    motion_diagnostics = {
        "position",
        "speed",
        "force-request",
        "control-frequency",
    }

    try:
        positions, speeds, forces = validate_args(args)
        gripper = connect_to_server(args)
        validate_gripper_type(gripper, args)
        state = wait_for_initial_state(
            gripper, args.startup_timeout, args.poll_interval
        )
        max_width = float(gripper.metadata.max_width)
        if max_width <= 0.0 or args.width_tolerance >= max_width:
            raise RuntimeError("Invalid metadata max_width or width tolerance")

        print(
            f"Connected: max_width={max_width * 1000.0:.2f} mm, "
            f"declared hardware-client rate={gripper.metadata.hz} Hz"
        )
        print("Initial state: " + format_state(state))

        if args.diagnostic == "position":
            run_position_diagnostic(gripper, max_width, positions, args)
        elif args.diagnostic == "speed":
            run_speed_diagnostic(gripper, max_width, speeds, args)
        elif args.diagnostic == "force-request":
            run_force_request_diagnostic(gripper, max_width, forces, args)
        elif args.diagnostic == "state-frequency":
            success_criterion_observed = run_state_frequency_diagnostic(
                gripper, args
            )
        elif args.diagnostic == "control-frequency":
            success_criterion_observed = run_control_frequency_diagnostic(
                gripper, max_width, args
            )
        else:
            raise AssertionError(f"unsupported diagnostic {args.diagnostic}")

        if not success_criterion_observed:
            print(
                f"\n[DIAGNOSTIC INCONCLUSIVE] {args.diagnostic} completed, but "
                "its success criterion was not observed."
            )
            return 2

        print(f"\n[DIAGNOSTIC PASS] {args.diagnostic} completed.")
        return 0
    except KeyboardInterrupt:
        print(
            "\nInterrupted. No automatic recovery command was sent; the gripper "
            "may continue its last requested motion.",
            file=sys.stderr,
        )
        return 130
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        if args.diagnostic in motion_diagnostics:
            print(
                "No automatic recovery command was sent; inspect the gripper "
                "before issuing another command.",
                file=sys.stderr,
            )
        return 1
    finally:
        if gripper is not None:
            gripper.channel.close()


if __name__ == "__main__":
    raise SystemExit(main())

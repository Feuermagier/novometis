#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Read and optionally exercise a Robotiq gripper through Polymetis.

From the Novometis repository root, start the server and hardware client in
another terminal first. Set ``ROBOTIQ_PORT`` to this installation's stable
``/dev/serial/by-id/...`` path:

    python polymetis/python/scripts/launch_gripper.py \
        gripper=robotiq_2f \
        gripper.port="${ROBOTIQ_PORT}" \
        gripper.hz=100

Launching the hardware client resets and activates the gripper, which performs
calibration motion. Keep the complete finger sweep clear before launching it.

This example is read-only unless ``--move`` is supplied. Its motion sequence is
contact-free: full open -> half open -> full open. It deliberately uses only
the public ``GripperInterface`` rather than bypassing Polymetis with Modbus.
"""

import argparse
import sys
import time

from polymetis import GripperInterface


ERROR_DESCRIPTIONS = {
    -3: "latest command failed or was rejected",
    -2: "gripper is not activated and ready",
    -1: "hardware-state communication failed",
    0: "healthy",
}
EXPECTED_GRIPPER_TYPE = "robotiq_2f"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-ip", default="localhost")
    parser.add_argument("--server-port", type=int, default=50052)
    parser.add_argument(
        "--move",
        action="store_true",
        help="run the clear-workspace open/half-open/open motion test",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="skip the typed motion confirmation (for deliberate automation)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=0.020,
        help="motion speed in m/s (2F-85 range: 0.020 to 0.150)",
    )
    parser.add_argument("--motion-timeout", type=float, default=15.0)
    parser.add_argument("--startup-timeout", type=float, default=15.0)
    parser.add_argument("--poll-interval", type=float, default=0.05)
    parser.add_argument("--width-tolerance", type=float, default=0.003)
    parser.add_argument("--dwell", type=float, default=0.5)
    return parser.parse_args()


def validate_args(args):
    if not 1 <= args.server_port <= 65535:
        raise ValueError("--server-port must be between 1 and 65535")
    if not 0.020 <= args.speed <= 0.150:
        raise ValueError("--speed must be between 0.020 and 0.150 m/s")
    if args.motion_timeout <= 0.0 or args.startup_timeout <= 0.0:
        raise ValueError("timeouts must be positive")
    if args.poll_interval <= 0.0:
        raise ValueError("--poll-interval must be positive")
    if args.width_tolerance <= 0.0:
        raise ValueError("--width-tolerance must be positive")
    if args.dwell < 0.0:
        raise ValueError("--dwell cannot be negative")


def validate_gripper_type(gripper, args):
    reported_type = getattr(gripper.metadata, "gripper_type", "").strip()
    if reported_type != EXPECTED_GRIPPER_TYPE:
        raise RuntimeError(
            f"Expected a {EXPECTED_GRIPPER_TYPE} service at "
            f"{args.server_ip}:{args.server_port}, received "
            f"{reported_type or '<missing>'}."
        )


def timestamp_key(state):
    """Return a comparable key for one hardware-observation timestamp."""
    return state.timestamp.seconds, state.timestamp.nanos


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
    """Connect through the unchanged public Polymetis interface."""
    try:
        return GripperInterface(
            ip_address=args.server_ip,
            port=args.server_port,
        )
    except Exception as error:
        raise RuntimeError(
            "Could not initialize GripperInterface. Start launch_gripper.py "
            "and wait until the Robotiq hardware client has registered."
        ) from error


def wait_for_initial_state(gripper, timeout, poll_interval):
    """Require two observations so a stale server cache cannot look healthy."""
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
                raise RuntimeError(
                    "Initial gripper state is unhealthy: " + format_state(state)
                )
            return state
        time.sleep(poll_interval)

    detail = format_state(last_state) if last_state is not None else "no state"
    raise TimeoutError(
        "No advancing hardware-state timestamp arrived before timeout; the "
        f"server cache may be stale. Last state: {detail}"
    )


def wait_for_target(
    gripper,
    target_width,
    previous_timestamp,
    timeout,
    poll_interval,
    width_tolerance,
):
    """Verify a new command through cached hardware feedback, not RPC timing."""
    deadline = time.monotonic() + timeout
    last_state = None
    saw_new_state = False

    while time.monotonic() < deadline:
        state = gripper.get_state()
        last_state = state
        saw_new_state = saw_new_state or timestamp_key(state) > previous_timestamp

        # Communication failures deliberately preserve the last good timestamp,
        # so error handling must not depend on seeing a newer observation.
        if state.error_code != 0:
            raise RuntimeError("Gripper reported an error: " + format_state(state))
        if saw_new_state and state.is_grasped:
            raise RuntimeError(
                "Unexpected closing contact in a clear-workspace test: "
                + format_state(state)
            )

        target_reached = abs(state.width - target_width) <= width_tolerance
        if (
            saw_new_state
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


def move_and_verify(gripper, label, target_width, args):
    before = gripper.get_state()
    if before.error_code != 0:
        raise RuntimeError(
            "Refusing motion from unhealthy state: " + format_state(before)
        )

    print(
        f"[COMMAND] {label}: target={target_width * 1000.0:.2f} mm, "
        f"speed={args.speed:.3f} m/s"
    )

    # The unchanged interface sends through a worker thread. We avoid its
    # blocking queue wait here and establish completion from newer hardware
    # state samples instead.
    gripper.goto(
        width=target_width,
        speed=args.speed,
        force=0.0,
        blocking=False,
    )
    state = wait_for_target(
        gripper=gripper,
        target_width=target_width,
        previous_timestamp=timestamp_key(before),
        timeout=args.motion_timeout,
        poll_interval=args.poll_interval,
        width_tolerance=args.width_tolerance,
    )
    print(f"[REACHED] {label}: {format_state(state)}")


def confirm_motion(args, max_width):
    print("\nMotion plan (no object is required):")
    print(
        f"  full open ({max_width * 1000.0:.1f} mm) -> "
        f"half open ({max_width * 500.0:.1f} mm) -> full open"
    )
    print(f"  speed={args.speed:.3f} m/s; force request=0 (minimum-force mode)")
    print("Clear the complete finger sweep of hands, objects, cables, and tools.")
    print("The Robotiq rFR=0 request is minimum force, not zero physical force.")

    if args.yes:
        return True
    return input("Type MOVE to continue: ").strip() == "MOVE"


def main():
    args = parse_args()
    gripper = None

    try:
        validate_args(args)
        gripper = connect_to_server(args)
        validate_gripper_type(gripper, args)
        state = wait_for_initial_state(
            gripper,
            timeout=args.startup_timeout,
            poll_interval=args.poll_interval,
        )

        max_width = float(gripper.metadata.max_width)
        if max_width <= 0.0 or args.width_tolerance >= max_width:
            raise RuntimeError(
                "Server reported an invalid max width or the configured width "
                "tolerance is too large"
            )

        print(f"Connected: max_width={max_width * 1000.0:.2f} mm")
        print("State:     " + format_state(state))

        if not args.move:
            print("Read-only check complete. Add --move to run the motion test.")
            return 0

        if not confirm_motion(args, max_width):
            print("Motion test cancelled; no command was sent.")
            return 2

        targets = (
            ("full open", max_width),
            ("half open", max_width / 2.0),
            ("reopen", max_width),
        )
        for index, (label, target_width) in enumerate(targets):
            move_and_verify(gripper, label, target_width, args)
            if index + 1 < len(targets):
                time.sleep(args.dwell)

        print("Robotiq Polymetis smoke test passed; the gripper finished open.")
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
        if args.move:
            print(
                "No automatic recovery command was sent; inspect the gripper "
                "before issuing another command.",
                file=sys.stderr,
            )
        return 1
    finally:
        if gripper is not None:
            # The original GripperInterface has no close() method. Closing its
            # public gRPC channel releases this example's transport resources.
            gripper.channel.close()


if __name__ == "__main__":
    raise SystemExit(main())

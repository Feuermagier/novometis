# Copyright (c) Facebook, Inc. and its affiliates.
# In case of problems, please contact Emiliyan Gospodinov.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import math
import struct
import time
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import grpc
from google.protobuf import timestamp_pb2

import polymetis
import polymetis_pb2
import polymetis_pb2_grpc

from polymetis.utils import Spinner
from .third_party.robotiq_2finger_grippers.robotiq_2f_gripper_refactored import (
    Robotiq2FingerGripper,
    RobotiqError,
)

log = logging.getLogger(__name__)


SERVER_PORT_MIN = 1
SERVER_PORT_MAX = 65535
# Robotiq control-rate guidance:
# https://assets.robotiq.com/website-assets/support_documents/document/2F-85_2F-140_Instruction_Manual_CB-Series_PDF_20190206.pdf
CONTROL_RATE_HZ_MIN = 1
CONTROL_RATE_HZ_MAX = 200
ROBOTIQ_RAW_POSITION_MIN = 0
ROBOTIQ_RAW_POSITION_MAX = 255

# Reserve negative values for client errors, Robotiq hardware faults occupy the unsigned fault byte.
COMMUNICATION_ERROR_CODE = -1
NOT_READY_ERROR_CODE = -2
COMMAND_ERROR_CODE = -3


@dataclass(frozen=True)
class RobotiqClientConfig:
    """Validated settings for the Robotiq Polymetis hardware client.

    The low-level driver validates serial and activation settings. This
    configuration validates the gRPC endpoint, control rate, SI-unit limits,
    and their protobuf float32 encodings.
    """

    server_ip: str
    server_port: int
    max_width: float
    hz: int
    model_min_speed_m_s: float
    model_max_speed_m_s: float
    model_min_force_n: float
    model_max_force_n: float
    max_command_speed_m_s: float
    max_command_force_n: float
    grpc_timeout: float
    max_command_state_age: float
    _protobuf_limits: Mapping[str, float] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self):
        """Validate types, ranges, relationships, and float32 encodings."""
        if type(self.server_ip) is not str:
            raise TypeError(
                "server_ip must be str, received "
                f"{type(self.server_ip).__name__}: {self.server_ip!r}"
            )
        if not self.server_ip:
            raise ValueError("server_ip must not be empty")
        if self.server_ip != self.server_ip.strip():
            raise ValueError(
                "server_ip must not contain leading or trailing whitespace, "
                f"received {self.server_ip!r}"
            )

        for name, minimum, maximum in (
            ("server_port", SERVER_PORT_MIN, SERVER_PORT_MAX),
            ("hz", CONTROL_RATE_HZ_MIN, CONTROL_RATE_HZ_MAX),
        ):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(
                    f"{name} must be int, received "
                    f"{type(value).__name__}: {value!r}"
                )
            if not minimum <= value <= maximum:
                raise ValueError(
                    f"{name} must be between {minimum} and {maximum}, "
                    f"received {value}"
                )

        for name in ("grpc_timeout", "max_command_state_age"):
            self._validate_finite_float(name, getattr(self, name))

        semantic_limits = {
            "max_width": self.max_width,
            "model_min_speed_m_s": self.model_min_speed_m_s,
            "model_max_speed_m_s": self.model_max_speed_m_s,
            "model_min_force_n": self.model_min_force_n,
            "model_max_force_n": self.model_max_force_n,
            "max_command_speed_m_s": self.max_command_speed_m_s,
            "max_command_force_n": self.max_command_force_n,
        }
        protobuf_limits = {
            name: self._to_protobuf_float32(name, value)
            for name, value in semantic_limits.items()
        }
        self._validate_limit_relationships(semantic_limits)
        if self.grpc_timeout <= 0.0:
            raise ValueError(
                f"grpc_timeout must be positive, received {self.grpc_timeout}"
            )
        if self.max_command_state_age <= 0.0:
            raise ValueError(
                "max_command_state_age must be positive, received "
                f"{self.max_command_state_age}"
            )
        self._validate_limit_relationships(
            protobuf_limits,
            context="after protobuf float32 encoding",
        )
        object.__setattr__(
            self,
            "_protobuf_limits",
            MappingProxyType(protobuf_limits),
        )

    @staticmethod
    def _validate_finite_float(name, value):
        """Require a finite built-in float without coercion."""
        if type(value) is not float:
            raise TypeError(
                f"{name} must be float, received " f"{type(value).__name__}: {value!r}"
            )
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, received {value!r}")

    @classmethod
    def _to_protobuf_float32(cls, name, value):
        """Return the finite float32 representation used by protobuf."""
        cls._validate_finite_float(name, value)
        try:
            encoded = struct.unpack("!f", struct.pack("!f", value))[0]
        except (OverflowError, struct.error) as error:
            raise ValueError(
                f"{name} cannot be represented as protobuf float32: " f"{value!r}"
            ) from error
        if not math.isfinite(encoded):
            raise ValueError(
                f"{name} cannot be represented as a finite protobuf "
                f"float32: {value!r}"
            )
        if value != 0.0 and encoded == 0.0:
            raise ValueError(
                f"{name} underflows to zero as protobuf float32: {value!r}"
            )
        return encoded

    @staticmethod
    def _validate_limit_relationships(limits, context=""):
        """Validate relationships among semantic or float32-encoded limits."""
        suffix = f" {context}" if context else ""
        if limits["max_width"] <= 0.0:
            raise ValueError(
                f"max_width must be positive{suffix}, received "
                f"{limits['max_width']}"
            )
        if not (0.0 < limits["model_min_speed_m_s"] < limits["model_max_speed_m_s"]):
            raise ValueError(
                "model speed limits must satisfy "
                "0 < model_min_speed_m_s < model_max_speed_m_s"
                f"{suffix}, received {limits['model_min_speed_m_s']} and "
                f"{limits['model_max_speed_m_s']}"
            )
        if not (
            limits["model_min_speed_m_s"]
            <= limits["max_command_speed_m_s"]
            <= limits["model_max_speed_m_s"]
        ):
            raise ValueError(
                "max_command_speed_m_s must be between the model speed "
                f"limits{suffix} [{limits['model_min_speed_m_s']}, "
                f"{limits['model_max_speed_m_s']}], received "
                f"{limits['max_command_speed_m_s']}"
            )
        if not (0.0 <= limits["model_min_force_n"] < limits["model_max_force_n"]):
            raise ValueError(
                "model force limits must satisfy "
                "0 <= model_min_force_n < model_max_force_n"
                f"{suffix}, received {limits['model_min_force_n']} and "
                f"{limits['model_max_force_n']}"
            )
        if not (
            limits["model_min_force_n"]
            <= limits["max_command_force_n"]
            <= limits["model_max_force_n"]
        ):
            raise ValueError(
                "max_command_force_n must be between the model force limits"
                f"{suffix} [{limits['model_min_force_n']}, "
                f"{limits['model_max_force_n']}], received "
                f"{limits['max_command_force_n']}"
            )

    @property
    def protobuf_limits(self) -> Mapping[str, float]:
        """Return the immutable float32 view used at the protobuf boundary."""
        return self._protobuf_limits


class RobotiqGripperClient:
    """Hardware-side Client between Polymetis and the low-level Robotiq2FingerGripper.

    Polymetis supplies width, speed, and nominal force in SI units. This
    adapter validates that API contract, converts it to raw Robotiq requests,
    and publishes one coherent protobufF state from each FC04 snapshot.
    This way, the low-level implementation remains independent of protobuf and gRPC.

    A successful FC16 response means that the write was accepted, it does not
    mean that the requested mechanical motion has completed.
    """

    # Public client API

    def __init__(
        self,
        server_ip,
        server_port,
        port,
        hz=60,
        baudrate=115200,
        slave_id=0x09,
        max_width=0.085,
        model_min_speed_m_s=0.020,
        model_max_speed_m_s=0.150,
        model_min_force_n=20.0,
        model_max_force_n=235.0,
        max_command_speed_m_s=0.150,
        max_command_force_n=235.0,
        response_timeout=1.0,
        activation_timeout=5.0,
        poll_interval=0.1,
        grpc_timeout=1.0,
        max_command_state_age=0.25,
    ):
        """Connect to and activate the gripper, then register with the server.

        Default width and model limits are documented Robotiq 2F-85 values.
        Position request and feedback conversion use the documented raw range:
        0 is fully open and 255 is fully closed.

        Command ceilings are independent application limits. Speed and force
        conversion always use the model ranges, so lowering a ceiling does not
        remap it to the raw maximum. Mapping nominal newtons to rFR is an
        approximate request conversion, not a calibrated force measurement.
        """
        # Validate client settings before activation; the low-level driver
        # validates serial settings.
        self._config = RobotiqClientConfig(
            server_ip=server_ip,
            server_port=server_port,
            max_width=max_width,
            hz=hz,
            model_min_speed_m_s=model_min_speed_m_s,
            model_max_speed_m_s=model_max_speed_m_s,
            model_min_force_n=model_min_force_n,
            model_max_force_n=model_max_force_n,
            max_command_speed_m_s=max_command_speed_m_s,
            max_command_force_n=max_command_force_n,
            grpc_timeout=grpc_timeout,
            max_command_state_age=max_command_state_age,
        )
        self.hz = self._config.hz

        # FC04 reads return independent state readings rather than updating driver
        # fields. Cache the latest valid state, readiness, and command result.
        self._last_valid_state = None
        self._previous_command_successful = False
        self._last_command_failed = False
        self._gripper_ready = False
        self._last_status_monotonic = None
        self._status_failure_logged = False

        self.gripper = None
        self.channel = None
        self.connection = None
        self._closed = False

        try:
            # The driver owns the serial lifecycle: construction connects,
            # resets, activates, and raises if any stage fails.
            self.gripper = Robotiq2FingerGripper(
                port=port,
                baudrate=baudrate,
                slave_id=slave_id,
                response_timeout=response_timeout,
                activation_timeout=activation_timeout,
                poll_interval=poll_interval,
            )

            # The gRPC server exchanges commands and state; this client owns
            # the serial device and executes motion commands.
            self.channel = grpc.insecure_channel(
                f"{self._config.server_ip}:{self._config.server_port}"
            )
            self.connection = polymetis_pb2_grpc.GripperServerStub(self.channel)

            metadata = polymetis_pb2.GripperMetadata(
                polymetis_version=polymetis.__version__,
                hz=self.hz,
                max_width=self._config.protobuf_limits["max_width"],
                gripper_type="robotiq_2f",
            )
            self.connection.InitRobotClient(
                metadata,
                timeout=self._config.grpc_timeout,
            )
        except BaseException:
            # Release resources acquired so far before propagating the
            # initialization error.
            try:
                self.cleanup()
            except Exception:
                log.exception(
                    "Robotiq cleanup also failed during client initialization"
                )
            raise

    def get_gripper_state(self):
        """Return a fresh FC04 state or cached state with a read error."""
        read_error = None
        try:
            status = self.gripper.read_status()
        except RobotiqError as error:
            read_error = error
            status = None

        if status is None:
            # A failed FC04 read has no current physical values, so reuse the
            # last valid snapshot instead of reporting protobuf defaults.
            state = polymetis_pb2.GripperState()
            self._gripper_ready = False

            if self._last_valid_state is not None:
                state.CopyFrom(self._last_valid_state)

            # Preserve the cached observation's timestamp so consumers can
            # detect stale data; it remains zero if no valid snapshot exists.
            state.error_code = COMMUNICATION_ERROR_CODE
            state.prev_command_successful = self._previous_command_successful

            if not self._status_failure_logged:
                log.warning(
                    "Failed to read Robotiq status (%s); returning the last "
                    "valid physical state with a communication error.",
                    read_error or "no status returned",
                )
                self._status_failure_logged = True
            return state

        if self._status_failure_logged:
            log.info("Robotiq status communication recovered")
            self._status_failure_logged = False

        self._last_status_monotonic = time.monotonic()
        state = polymetis_pb2.GripperState()

        # Timestamp successful reads so users can distinguish fresh
        # hardware data from cached fallback data.
        state.timestamp.GetCurrentTime()
        state.width = self._status_to_polymetis_width(status)

        # gOBJ is meaningful only while a go-to operation is active on an
        # activated, ready, fault-free gripper.
        ready = status.is_ready
        self._gripper_ready = ready

        # gOBJ=2 means the fingers stopped after contact while closing.
        state.is_grasped = ready and status.gGTO == 1 and status.gOBJ == 2

        # gOBJ=0 while gGTO=1 means the requested movement is still running.
        state.is_moving = ready and status.gGTO == 1 and status.gOBJ == 0

        fault_byte = status.fault_code
        if fault_byte != 0:
            # Report the full fault byte: kFLT is the high nibble and gFLT is
            # the low nibble.
            state.error_code = fault_byte
        elif not ready:
            state.error_code = NOT_READY_ERROR_CODE
        elif self._last_command_failed:
            state.error_code = COMMAND_ERROR_CODE
        else:
            state.error_code = 0

        # This field reports FC16 acknowledgement, not finger-motion completion.
        state.prev_command_successful = self._previous_command_successful

        # Cache only valid FC04 snapshots so failed reads cannot replace
        # physical values with protobuf defaults.
        self._last_valid_state = polymetis_pb2.GripperState()
        self._last_valid_state.CopyFrom(state)

        return state

    def apply_gripper_command(self, command):
        """Validate, convert, and send one Polymetis command via Modbus FC16."""
        self._previous_command_successful = False

        try:
            if not self._gripper_ready:
                raise RuntimeError(
                    "Refusing a Robotiq command because the latest state is "
                    "not activated, ready, and fault-free."
                )

            status_age = time.monotonic() - self._last_status_monotonic
            if status_age > self._config.max_command_state_age:
                raise RuntimeError(
                    "Refusing a Robotiq command because the latest valid status "
                    f"sample is {status_age:.3f} s old; maximum allowed age is "
                    f"{self._config.max_command_state_age:.3f} s."
                )
            # GripperInterface stores goto and grasp targets in command.width.
            # Robotiq treats grasp like goto and ignores the epsilon fields.
            raw_command = self._command_to_robotiq_arguments(command)

            log.debug(
                "Sending Robotiq command: width=%s m, speed=%s m/s, force=%s N "
                "-> rPR=%d, rSP=%d, rFR=%d.",
                command.width,
                command.speed,
                command.force,
                raw_command["position_request"],
                raw_command["speed_request"],
                raw_command["force_request"],
            )

            self.gripper.move(**raw_command)
        except Exception:
            self._last_command_failed = True
            raise
        else:
            # move() returning means FC16 was acknowledged; physical motion may
            # still be in progress.
            self._previous_command_successful = True
            self._last_command_failed = False

    def run(self):
        """Publish state and execute each newly timestamped command once.

        The loop runs at a best-effort rate. Command timestamps identify new
        commands; they are not deadlines or motion acknowledgements.
        """
        spinner = Spinner(self.hz)

        try:
            last_seen_command_timestamp = self._prime_command_cache()

            while True:
                command = self._exchange_state_for_command()
                self._process_server_command(
                    command,
                    last_seen_command_timestamp,
                )
                spinner.spin()
        finally:
            self.cleanup()

    def cleanup(self):
        """Close resources without issuing a motion-stop command."""
        if self._closed:
            return
        self._closed = True

        cleanup_errors = []
        try:
            if self.channel is not None:
                self.channel.close()
        except Exception as error:
            cleanup_errors.append(error)
        try:
            if self.gripper is not None:
                self.gripper.cleanup()
        except Exception as error:
            cleanup_errors.append(error)

        if cleanup_errors:
            raise RuntimeError(
                "Failed to close one or more Robotiq client resources"
            ) from cleanup_errors[0]

    # Methods for Polymetis <-> Robotiq API conversion as they use different units
    # TODO: later remove when APIs are matched
    @staticmethod
    def _validate_api_command_value(name, value, minimum, maximum):
        """Require a finite float within the inclusive command range."""
        if type(value) is not float:
            raise TypeError(
                f"{name} must be float, received " f"{type(value).__name__}: {value!r}"
            )
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, received {value!r}")
        if not minimum <= value <= maximum:
            raise ValueError(
                f"{name} must be between {minimum} and {maximum}, " f"received {value}"
            )
        return value

    def _status_to_polymetis_width(self, status):
        """Convert FC04 gPO from 0 (open) to 255 (closed) into metres."""
        position = status.gPO
        if type(position) is not int:
            raise TypeError(
                "status position must be int, received "
                f"{type(position).__name__}: {position!r}"
            )
        if not ROBOTIQ_RAW_POSITION_MIN <= position <= ROBOTIQ_RAW_POSITION_MAX:
            raise ValueError(
                "status position must be between "
                f"{ROBOTIQ_RAW_POSITION_MIN} and "
                f"{ROBOTIQ_RAW_POSITION_MAX}, received {position}"
            )
        position_span = ROBOTIQ_RAW_POSITION_MAX - ROBOTIQ_RAW_POSITION_MIN
        opening_fraction = (ROBOTIQ_RAW_POSITION_MAX - position) / position_span
        return self._config.max_width * opening_fraction

    def _command_to_robotiq_arguments(self, command):
        """Convert one Polymetis SI command to low-level Robotiq arguments."""
        limits = self._config.protobuf_limits

        # Width: metres -> raw rPR.
        width = self._validate_api_command_value(
            "width", command.width, 0.0, limits["max_width"]
        )
        opening_fraction = width / limits["max_width"]
        position_request = (
            ROBOTIQ_RAW_POSITION_MAX
            + (ROBOTIQ_RAW_POSITION_MIN - ROBOTIQ_RAW_POSITION_MAX) * opening_fraction
        )

        # Speed: metres/second -> raw rSP.
        speed = self._validate_api_command_value(
            "speed",
            command.speed,
            limits["model_min_speed_m_s"],
            limits["max_command_speed_m_s"],
        )
        normalized_speed = (speed - limits["model_min_speed_m_s"]) / (
            limits["model_max_speed_m_s"] - limits["model_min_speed_m_s"]
        )

        # Nominal force: newtons -> raw rFR. This is not a force calibration.
        force = self._validate_api_command_value(
            "force",
            command.force,
            0.0,
            limits["max_command_force_n"],
        )
        if 0.0 < force < limits["model_min_force_n"]:
            raise ValueError(
                f"force must be 0 (minimum-force mode) or at least "
                f"{limits['model_min_force_n']} N, received "
                f"{force} N"
            )

        # rFR=0 requests minimum hardware force; a zero-newton API request does
        # not mean zero physical force.
        if force <= limits["model_min_force_n"]:
            force_request = 0
        else:
            normalized_force = (force - limits["model_min_force_n"]) / (
                limits["model_max_force_n"] - limits["model_min_force_n"]
            )
            force_request = int(round(255.0 * normalized_force))

        return {
            "position_request": int(round(position_request)),
            "speed_request": int(255.0 * normalized_speed),
            "force_request": force_request,
        }

    # Control-loop helpers handle server I/O, startup replay protection, and
    # per-command de-duplication.
    def _exchange_state_for_command(self):
        """Publish the latest state and fetch the server's cached command."""
        return self.connection.ControlUpdate(
            self.get_gripper_state(),
            timeout=self._config.grpc_timeout,
        )

    def _prime_command_cache(self):
        """Record a retained command without executing it at startup."""
        cached_command = self._exchange_state_for_command()
        last_seen_command_timestamp = timestamp_pb2.Timestamp()
        last_seen_command_timestamp.CopyFrom(cached_command.timestamp)

        if cached_command.timestamp.seconds or cached_command.timestamp.nanos:
            log.warning(
                "Ignoring a Robotiq command cached before this hardware-client "
                "session, send a fresh command before moving the gripper."
            )

        return last_seen_command_timestamp

    def _process_server_command(self, command, last_seen_command_timestamp):
        """Apply a command when its nonzero timestamp differs from the last.

        Record the timestamp before applying the command so a rejected command
        is not retried on every control-loop iteration.
        """
        command_has_timestamp = bool(
            command.timestamp.seconds or command.timestamp.nanos
        )
        if not command_has_timestamp:
            # A zero timestamp marks an empty server cache, as after a restart.
            # Record the reset without treating the default width as a close.
            last_seen_command_timestamp.CopyFrom(command.timestamp)
            return

        if command.timestamp == last_seen_command_timestamp:
            return

        last_seen_command_timestamp.CopyFrom(command.timestamp)
        try:
            self.apply_gripper_command(command)
        except Exception:
            log.exception("Failed to apply the latest Robotiq command")

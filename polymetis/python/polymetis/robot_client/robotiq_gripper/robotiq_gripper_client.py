# Copyright (c) Facebook, Inc. and its affiliates.
# In case of problems please contact Emiliyan Gospodinov.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import math
import struct
import time
from dataclasses import dataclass

import grpc
from google.protobuf import timestamp_pb2

import polymetis
import polymetis_pb2
import polymetis_pb2_grpc

from polymetis.utils import Spinner
from .third_party.robotiq_2finger_grippers.robotiq_2f_gripper_refactored import (
    Robotiq2FingerGripper,
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

# Negative adapter errors remain distinct from Robotiq's vendor fault byte.
COMMUNICATION_ERROR_CODE = -1
NOT_READY_ERROR_CODE = -2
COMMAND_ERROR_CODE = -3


@dataclass
class RobotiqAdapterConfig:
    """Validated settings for the Robotiq Polymetis hardware adapter.

    The low-level driver owns serial transport, Modbus communication, and
    activation. This configuration defines the gRPC endpoint, control rate,
    calibration, and SI-unit command limits used by the hardware client.
    """

    server_ip: str
    server_port: int
    max_width: float
    hz: int
    open_position_request: int
    closed_position_request: int
    open_position_feedback: int
    closed_position_feedback: int
    model_min_speed_m_s: float
    model_max_speed_m_s: float
    model_min_force_n: float
    model_max_force_n: float
    max_command_speed_m_s: float
    max_command_force_n: float
    grpc_timeout: float
    max_command_state_age: float

    @staticmethod
    def normalize_finite_number(name, value):
        """Return a finite float while preserving a useful field-name error."""
        if isinstance(value, bool):
            raise ValueError(f"{name} must be numeric, received {value!r}")
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                f"{name} must be numeric, received {value!r}"
            ) from error
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, received {value}")
        return value

    @classmethod
    def normalize_integer_in_range(cls, name, value, minimum, maximum):
        """Return an integer constrained to an inclusive range."""
        value = cls.normalize_finite_number(name, value)
        if not value.is_integer():
            raise ValueError(f"{name} must be an integer, received {value}")
        value = int(value)
        if not minimum <= value <= maximum:
            raise ValueError(
                f"{name} must be between {minimum} and {maximum}, "
                f"received {value}"
            )
        return value

    @classmethod
    def normalize_protobuf_float(cls, name, value):
        """Normalize a policy limit to protobuf's IEEE-754 float32 value."""
        value = cls.normalize_finite_number(name, value)
        try:
            return struct.unpack("!f", struct.pack("!f", value))[0]
        except (OverflowError, struct.error) as error:
            raise ValueError(
                f"{name} must be representable by a protobuf float field"
            ) from error

    def __post_init__(self):
        """Normalize fields and validate relationships before hardware starts."""
        if not isinstance(self.server_ip, str) or not self.server_ip.strip():
            raise ValueError(
                "server_ip must be a non-empty string, "
                f"received {self.server_ip!r}"
            )
        self.server_ip = self.server_ip.strip()

        self.server_port = self.normalize_integer_in_range(
            "server_port",
            self.server_port,
            SERVER_PORT_MIN,
            SERVER_PORT_MAX,
        )
        self.hz = self.normalize_integer_in_range(
            "hz",
            self.hz,
            CONTROL_RATE_HZ_MIN,
            CONTROL_RATE_HZ_MAX,
        )
        self.open_position_request = self.normalize_integer_in_range(
            "open_position_request",
            self.open_position_request,
            ROBOTIQ_RAW_POSITION_MIN,
            ROBOTIQ_RAW_POSITION_MAX,
        )
        self.closed_position_request = self.normalize_integer_in_range(
            "closed_position_request",
            self.closed_position_request,
            ROBOTIQ_RAW_POSITION_MIN,
            ROBOTIQ_RAW_POSITION_MAX,
        )
        self.open_position_feedback = self.normalize_integer_in_range(
            "open_position_feedback",
            self.open_position_feedback,
            ROBOTIQ_RAW_POSITION_MIN,
            ROBOTIQ_RAW_POSITION_MAX,
        )
        self.closed_position_feedback = self.normalize_integer_in_range(
            "closed_position_feedback",
            self.closed_position_feedback,
            ROBOTIQ_RAW_POSITION_MIN,
            ROBOTIQ_RAW_POSITION_MAX,
        )

        # These limits are compared directly with protobuf float fields.
        self.max_width = self.normalize_protobuf_float(
            "max_width", self.max_width
        )
        self.model_min_speed_m_s = self.normalize_protobuf_float(
            "model_min_speed_m_s", self.model_min_speed_m_s
        )
        self.model_max_speed_m_s = self.normalize_protobuf_float(
            "model_max_speed_m_s", self.model_max_speed_m_s
        )
        self.model_min_force_n = self.normalize_protobuf_float(
            "model_min_force_n", self.model_min_force_n
        )
        self.model_max_force_n = self.normalize_protobuf_float(
            "model_max_force_n", self.model_max_force_n
        )
        self.max_command_speed_m_s = self.normalize_protobuf_float(
            "max_command_speed_m_s", self.max_command_speed_m_s
        )
        self.max_command_force_n = self.normalize_protobuf_float(
            "max_command_force_n", self.max_command_force_n
        )

        self.grpc_timeout = self.normalize_finite_number(
            "grpc_timeout", self.grpc_timeout
        )
        self.max_command_state_age = self.normalize_finite_number(
            "max_command_state_age", self.max_command_state_age
        )

        self.validate_relationships()

    def validate_relationships(self):
        """Validate relationships that cannot be expressed field by field."""
        if self.max_width <= 0.0:
            raise ValueError(
                "max_width must be positive and representable by a protobuf "
                "float field"
            )
        if not 0.0 < self.model_min_speed_m_s < self.model_max_speed_m_s:
            raise ValueError(
                "model speed limits must satisfy "
                "0 < model_min_speed_m_s < model_max_speed_m_s"
            )
        if not (
            self.model_min_speed_m_s
            <= self.max_command_speed_m_s
            <= self.model_max_speed_m_s
        ):
            raise ValueError(
                "max_command_speed_m_s must be between the model speed limits"
            )
        if not 0.0 <= self.model_min_force_n < self.model_max_force_n:
            raise ValueError(
                "model force limits must satisfy "
                "0 <= model_min_force_n < model_max_force_n"
            )
        if not (
            self.model_min_force_n
            <= self.max_command_force_n
            <= self.model_max_force_n
        ):
            raise ValueError(
                "max_command_force_n must be between the model force limits"
            )
        if self.grpc_timeout <= 0.0:
            raise ValueError("grpc_timeout must be positive")
        if self.max_command_state_age <= 0.0:
            raise ValueError("max_command_state_age must be positive")
        if self.open_position_request >= self.closed_position_request:
            raise ValueError(
                "open_position_request must be less than "
                "closed_position_request"
            )
        if self.open_position_feedback >= self.closed_position_feedback:
            raise ValueError(
                "open_position_feedback must be less than "
                "closed_position_feedback"
            )


class RobotiqGripperClient:
    """Hardware-side Client between Polymetis and the low-level Robotiq2FingerGripper.

    Polymetis supplies width, speed, and nominal force in SI units. This
    adapter validates that API contract, converts it to raw Robotiq requests,
    and publishes one coherent protobufF state from each FC04 snapshot.
    This way, the low-level implementation remains independent of protobuf and gRPC.

    A successful FC16 response means that the write was accepted, it does not
    mean that the requested mechanical motion has completed.
    """

    # Public API

    def __init__(
        self,
        server_ip,
        server_port,
        port,
        hz=60,
        baudrate=115200,
        slave_id=0x09,
        max_width=0.085,
        open_position_request=3,
        closed_position_request=230,
        open_position_feedback=3,
        closed_position_feedback=230,
        model_min_speed_m_s=0.020,
        model_max_speed_m_s=0.150,
        model_min_force_n=20.0,
        model_max_force_n=235.0,
        max_command_speed_m_s=0.150,
        max_command_force_n=235.0,
        activation_timeout=5.0,
        poll_interval=0.1,
        grpc_timeout=1.0,
        max_command_state_age=0.25,
    ):
        """Connect to and activate the gripper, then register with the server.

        Width and model limits are nominal Robotiq 2F-85 values.
        Request and feedback endpoints 3/230 are calibrated raw values from
        this installation and must be recalibrated for a different gripper or
        finger setup. The independent low-level driver still exposes the full
        protocol request range 0/255.

        Command ceilings are independent safety-policy limits. Raw conversion
        always uses the model limits, so lowering a command ceiling cannot map
        that ceiling to raw maximum. Mapping rFR to newtons remains an
        approximate request conversion, not a calibrated force measurement.
        """
        # Validate adapter configuration before activation. The driver owns
        # validation of serial-specific values.
        self._config = RobotiqAdapterConfig(
            server_ip=server_ip,
            server_port=server_port,
            max_width=max_width,
            hz=hz,
            open_position_request=open_position_request,
            closed_position_request=closed_position_request,
            open_position_feedback=open_position_feedback,
            closed_position_feedback=closed_position_feedback,
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

        # FC04 returns independent snapshots rather than updating mutable
        # fields on the driver. The adapter therefore owns its protobuf cache
        # and remembers whether the most recent FC16 call succeeded.
        self._last_valid_state = None
        self._previous_command_successful = False
        self._last_command_failed = False
        self._gripper_ready = False
        self._last_status_monotonic = None

        self.gripper = None
        self.channel = None
        self.connection = None
        self._closed = False

        try:
            # The driver owns the complete serial lifecycle. Construction
            # connects, waits for reset acknowledgement, activates, and raises
            # if any stage fails.
            self.gripper = Robotiq2FingerGripper(
                port=port,
                baudrate=baudrate,
                slave_id=slave_id,
                max_width=self._config.max_width,
                activation_timeout=activation_timeout,
                poll_interval=poll_interval,
            )
            self._max_width = self.gripper.max_width

            # The gRPC server is only a command/state broker. It does not open
            # the serial device or execute motion itself.
            self.channel = grpc.insecure_channel(
                f"{self._config.server_ip}:{self._config.server_port}"
            )
            self.connection = polymetis_pb2_grpc.GripperServerStub(self.channel)

            metadata = polymetis_pb2.GripperMetadata(
                polymetis_version=polymetis.__version__,
                hz=self.hz,
                max_width=self._max_width,
                gripper_type="robotiq_2f",
            )
            self.connection.InitRobotClient(
                metadata,
                timeout=self._config.grpc_timeout,
            )
        except Exception:
            # If gRPC setup fails after activation, release serial ownership as
            # well as the partially-created channel before propagating the error.
            try:
                self.cleanup()
            except Exception:
                log.exception(
                    "Robotiq cleanup also failed during client initialization"
                )
            raise

    def get_gripper_state(self):
        # One FC04 snapshot keeps width, motion, grasp, and fault information
        # synchronized to the same physical observation.
        try:
            status = self.gripper.read_status()
        except Exception as error:
            log.warning("Robotiq status read raised an exception: %s", error)
            status = None

        if status is None:
            # The refactored driver returns snapshots instead of retaining mutable
            # gPO/gOBJ fields, so the hardware client owns the last-valid-state cache.
            state = polymetis_pb2.GripperState()
            self._gripper_ready = False

            if self._last_valid_state is not None:
                state.CopyFrom(self._last_valid_state)

            # Keep the original timestamp because the copied physical values are
            # stale, so only the communication error is new.
            state.error_code = COMMUNICATION_ERROR_CODE
            state.prev_command_successful = self._previous_command_successful

            log.warning(
                "Failed to read Robotiq status; returning the last valid "
                "physical state with a communication error."
            )
            return state

        self._last_status_monotonic = time.monotonic()
        state = polymetis_pb2.GripperState()

        # Timestamp only successful hardware observations.
        state.timestamp.GetCurrentTime()
        state.width = self._status_to_polymetis_width(status)

        # gOBJ is meaningful only while a go-to operation is active on an
        # activated, ready, fault-free gripper.
        ready = (
            status["gACT"] == 1
            and status["gSTA"] == 3
            and status["fault"] == 0
            and status["kFLT"] == 0
        )
        self._gripper_ready = ready

        # gOBJ=2 means the fingers stopped after contact while closing.
        state.is_grasped = (
            ready
            and status["gGTO"] == 1
            and status["gOBJ"] == 2
        )

        # gOBJ=0 while gGTO=1 means the requested movement is still running.
        state.is_moving = (
            ready
            and status["gGTO"] == 1
            and status["gOBJ"] == 0
        )

        fault_byte = (status["kFLT"] << 4) | status["fault"]
        if fault_byte != 0:
            # Preserve the complete Robotiq vendor fault byte. Human-readable
            # details remain in the hardware-client log.
            state.error_code = fault_byte
        elif not ready:
            state.error_code = NOT_READY_ERROR_CODE
        elif self._last_command_failed:
            state.error_code = COMMAND_ERROR_CODE
        else:
            state.error_code = 0

        # For now this means the latest FC16 write was acknowledged; it does not
        # prove that the mechanical movement completed.
        state.prev_command_successful = self._previous_command_successful

        # Failed reads must never overwrite the last known valid physical state.
        self._last_valid_state = polymetis_pb2.GripperState()
        self._last_valid_state.CopyFrom(state)

        return state

    def apply_gripper_command(self, command):
        """Validate, convert, and send one Polymetis command via FC16."""
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

            # GripperInterface already stores grasp_width in command.width.
            # Robotiq does not implement the epsilon fields, but the requested
            # width itself must not be overwritten.
            raw_command = self._command_to_robotiq_arguments(command)

            log.debug(
                "Sending Robotiq command: width=%s m, speed=%s m/s, force=%s N "
                "-> rPR=%d, normalized speed=%.4f, rFR=%d.",
                command.width,
                command.speed,
                command.force,
                raw_command["position"],
                raw_command["speed"],
                raw_command["force"],
            )

            self.gripper.move(**raw_command)
        except Exception:
            self._last_command_failed = True
            raise
        else:
            # This currently means the driver's FC16 call returned without an
            # error response; it does not mean mechanical motion completed.
            self._previous_command_successful = True
            self._last_command_failed = False

    def run(self):
        """Exchange cached state/commands with the server at best-effort rate.

        The timestamp makes each cached command execute at most once during
        this client session. It is not a real-time deadline or a mechanical
        completion acknowledgement.
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
        """Idempotently close gRPC and serial resources without commanding motion."""
        if self._closed:
            return

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

        self._closed = True

    # Methods for Polymetis <-> Robotiq API conversion
    # TODO: later remove when APIs are matched
    @staticmethod
    def _validate_api_command_value(name, value, minimum, maximum):
        """Validate one runtime value from a Polymetis GripperCommand."""
        value = RobotiqAdapterConfig.normalize_finite_number(name, value)
        if not minimum <= value <= maximum:
            raise ValueError(
                f"{name} must be between {minimum} and {maximum}, "
                f"received {value}"
            )
        return value

    def _status_to_polymetis_width(self, status):
        """Convert FC04 gPO to the metre-based width required by Polymetis.

        Request and feedback calibration remain separate configuration values
        because a future installation may measure a repeatable offset between
        them. Both default to the observed physical endpoints 3/230 here.
        """
        position = float(status["position"])
        position = max(
            self._config.open_position_feedback,
            min(position, self._config.closed_position_feedback),
        )
        position_span = (
            self._config.closed_position_feedback
            - self._config.open_position_feedback
        )
        opening_fraction = (
            self._config.closed_position_feedback - position
        ) / position_span
        return self._max_width * opening_fraction

    def _command_to_robotiq_arguments(self, command):
        """Convert one Polymetis SI command to Robotiq driver arguments."""
        # Width: metres -> raw rPR.
        width = self._validate_api_command_value(
            "width", command.width, 0.0, self._max_width
        )
        if width == 0.0:
            # A binary close must request the Robotiq protocol endpoint rather
            # than the measured/calibrated position reached by the fingers.
            position_request = ROBOTIQ_RAW_POSITION_MAX
        else:
            # Preserve calibrated interpolation for arbitrary SI widths.
            opening_fraction = width / self._max_width
            position_request = self._config.closed_position_request + (
                self._config.open_position_request
                - self._config.closed_position_request
            ) * opening_fraction

        # Speed: metres/second -> normalized input encoded by the driver as rSP.
        speed = self._validate_api_command_value(
            "speed",
            command.speed,
            self._config.model_min_speed_m_s,
            self._config.max_command_speed_m_s,
        )
        normalized_speed = (speed - self._config.model_min_speed_m_s) / (
            self._config.model_max_speed_m_s
            - self._config.model_min_speed_m_s
        )

        # Nominal force: newtons -> raw rFR. This is not a force calibration.
        force = self._validate_api_command_value(
            "force", command.force, 0.0, self._config.max_command_force_n
        )
        if 0.0 < force < self._config.model_min_force_n:
            raise ValueError(
                f"force must be 0 (minimum-force mode) or at least "
                f"{self._config.model_min_force_n} N, received {force} N"
            )

        # rFR=0 selects minimum hardware force and disables automatic re-grasp;
        # a zero-newton API request therefore does not mean zero physical force.
        if force <= self._config.model_min_force_n:
            force_request = 0
        else:
            normalized_force = (force - self._config.model_min_force_n) / (
                self._config.model_max_force_n
                - self._config.model_min_force_n
            )
            force_request = int(round(255.0 * normalized_force))

        return {
            "position": int(round(position_request)),
            "speed": normalized_speed,
            "force": force_request,
        }

    # Control-loop internals

    def _exchange_state_for_command(self):
        """Publish one FC04 snapshot and fetch the broker's cached command."""
        return self.connection.ControlUpdate(
            self.get_gripper_state(),
            timeout=self._config.grpc_timeout,
        )

    def _prime_command_cache(self):
        """Discard retained commands so a client restart cannot replay motion."""
        cached_command = self._exchange_state_for_command()
        last_seen_command_timestamp = timestamp_pb2.Timestamp()
        last_seen_command_timestamp.CopyFrom(cached_command.timestamp)

        if cached_command.timestamp.seconds or cached_command.timestamp.nanos:
            log.warning(
                "Ignoring a Robotiq command cached before this hardware-client "
                "session; send a fresh command before moving the gripper."
            )

        return last_seen_command_timestamp

    def _process_server_command(self, command, last_seen_command_timestamp):
        """Execute each nonempty command timestamp at most once per session.

        Timestamp zero represents the server's empty command. A new nonzero
        timestamp is recorded before execution so a rejected command cannot be
        retried on every hardware-client iteration.
        """
        command_has_timestamp = bool(
            command.timestamp.seconds or command.timestamp.nanos
        )
        if not command_has_timestamp:
            # A server restart resets its cache to an empty protobuf. Remember
            # that reset, but never interpret it as a close command.
            last_seen_command_timestamp.CopyFrom(command.timestamp)
            return

        if command.timestamp == last_seen_command_timestamp:
            return

        last_seen_command_timestamp.CopyFrom(command.timestamp)
        try:
            self.apply_gripper_command(command)
        except Exception:
            log.exception("Failed to apply the latest Robotiq command")

# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Derived from the RLinf Robotiq driver and substantially modified for
# Novometis. See the adjacent NOTICE and LICENSE files.

"""Robotiq 2F-85 / 2F-140 gripper via direct Modbus RTU over USB-RS485.

No ROS dependency — communicates with the gripper through ``pymodbus``
and a USB-RS485 adapter, preferably addressed through a stable
``/dev/serial/by-id/...`` path.

Robotiq input/output register mapping: pages 32-39 of the 2F instruction
manual available from ``https://robotiq.com/support``.

Modbus register map (Robotiq 2F series)
---------------------------------------
**Output (Control) registers** (Function Code (FC16), base address 0x03E8):

====== ========= ===========================================
Byte   Register  Description
====== ========= ===========================================
0      reg0 hi   Action request:
                    Bit 0 rACT (Activate/Deactivate gripper)
                    Bit 3 rGTO (Go to position)
                    Bit 4 rATR (Automatic release)
                    Bit 5 rARD (Automatic release direction)
1      reg0 lo   Reserved
2      reg1 hi   Reserved
3      reg1 lo   rPR — position request  (0=open, 255=closed)
4      reg2 hi   rSP — speed             (0=min,  255=max)
5      reg2 lo   rFR — force             (0=min,  255=max)
====== ========= ===========================================

**Input (Status) registers** (FC04, base address 0x07D0 / 2000):

====== ========= ===========================================
Byte   Register  Description
====== ========= ===========================================
0      reg0 hi   Status:
                    Bit 0 gACT (Activation status)
                    Bits 1-2 Reserved
                    Bit 3 gGTO (Action status)
                    Bits 4-5 gSTA (Gripper status)
                    Bits 6-7 gOBJ (Object detection status)
1      reg0 lo   Reserved
2      reg1 hi   gFLT — fault status
3      reg1 lo   gPR  — position request echo
4      reg2 hi   gPO  — actual position  (0=open, 255=closed)
5      reg2 lo   gCU  — motor current    (×10 mA)
====== ========= ===========================================
"""

import inspect
import logging
import math
import numbers
import time
from dataclasses import dataclass
from typing import Optional, TypedDict


# Polymetis pins PyModbus 2.5 while the standalone code uses 3.x;
# their slave-address keywords differ. Resolve the keyword once.
# This branch can disappear after both environments use one API.
try:
    from pymodbus.client import ModbusSerialClient
except ImportError:  # PyModbus 2.x
    from pymodbus.client.sync import ModbusSerialClient
    PYMODBUS_V2 = True
else:
    PYMODBUS_V2 = False


SERIAL_BAUDRATE_MIN = 1
SERIAL_BAUDRATE_MAX = 2**31 - 1
MODBUS_SLAVE_ID_MIN = 1
MODBUS_SLAVE_ID_MAX = 247
MODBUS_POLL_INTERVAL_MIN_SECONDS = 0.005
# Robotiq model limits:
# https://assets.robotiq.com/website-assets/support_documents/document/2F-85_2F-140_Instruction_Manual_CB-Series_PDF_20190206.pdf
ROBOTIQ_REGISTER_COUNT = 3
ROBOTIQ_OUTPUT_REGISTER_ADDRESS = 0x03E8
ROBOTIQ_INPUT_REGISTER_ADDRESS = 0x07D0
ROBOTIQ_VALID_ACTION_REQUESTS = frozenset({0x0000, 0x0100, 0x0900})

log = logging.getLogger(__name__)


class RobotiqStatus(TypedDict):
    """Decoded values returned by one Robotiq FC04 status read."""

    gACT: int
    gGTO: int
    gSTA: int
    gOBJ: int
    fault: int
    gFLT: int
    kFLT: int
    position_echo: int
    position: int
    current: int


@dataclass
class RobotiqDriverConfig:
    """Validated serial and activation settings for the Robotiq driver."""

    port: str
    baudrate: int
    slave_id: int
    max_width: float
    activation_timeout: float
    poll_interval: float

    @staticmethod
    def normalize_finite_number(name, value):
        """Return a finite float with a field-specific validation error."""
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

    def __post_init__(self):
        """Normalize individual fields, then validate their relationships."""
        if not isinstance(self.port, str) or not self.port.strip():
            raise ValueError(
                f"port must be a non-empty string, received {self.port!r}"
            )
        self.port = self.port.strip()

        self.baudrate = self.normalize_integer_in_range(
            "baudrate",
            self.baudrate,
            SERIAL_BAUDRATE_MIN,
            SERIAL_BAUDRATE_MAX,
        )
        self.slave_id = self.normalize_integer_in_range(
            "slave_id",
            self.slave_id,
            MODBUS_SLAVE_ID_MIN,
            MODBUS_SLAVE_ID_MAX,
        )
        self.max_width = self.normalize_finite_number(
            "max_width", self.max_width
        )
        self.activation_timeout = self.normalize_finite_number(
            "activation_timeout", self.activation_timeout
        )
        self.poll_interval = self.normalize_finite_number(
            "poll_interval", self.poll_interval
        )

        self.validate_relationships()

    def validate_relationships(self):
        """Validate constraints involving normalized configuration fields."""
        if self.max_width <= 0.0:
            raise ValueError("max_width must be positive")
        if self.activation_timeout <= 0.0:
            raise ValueError("activation_timeout must be positive")
        if self.poll_interval < MODBUS_POLL_INTERVAL_MIN_SECONDS:
            raise ValueError(
                "poll_interval must be at least "
                f"{MODBUS_POLL_INTERVAL_MIN_SECONDS} seconds"
            )
        if self.poll_interval > self.activation_timeout:
            raise ValueError(
                "poll_interval must not exceed activation_timeout"
            )


class Robotiq2FingerGripper:
    """Synchronous low-level control for a Robotiq 2F gripper over Modbus RTU.

    Creating an instance opens the serial connection, resets the gripper, and
    waits for activation to complete. Activation performs an automatic
    calibration and may move the fingers.

    ``move`` sends one validated FC16 command using a raw position request, a
    normalized speed, and a raw force request. ``read_status`` reads and
    decodes one validated FC04 status snapshot.

    The Polymetis hardware client owns protobuf/gRPC communication, command
    caching, and conversion between SI units and Robotiq request values.

    Args:
        port: Stable serial device path, preferably under
            ``/dev/serial/by-id/``.
        baudrate: Modbus baud rate assuming ACC-ADT-USB-RS485 (default 115200).
        slave_id: Modbus slave address (default 0x09).
        max_width: Physical opening of the fully-open gripper in metres.
            0.085 for the 2F-85, 0.140 for the 2F-140.
        activation_timeout: Maximum duration of each activation stage in
            seconds.
        poll_interval: Delay between activation-status reads in seconds.
    """

    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        slave_id: int = 0x09,
        max_width: float = 0.085,
        activation_timeout: float = 5.0,
        poll_interval: float = 0.1,
    ):
        # This device-level validation runs once, before serial construction.
        configuration = RobotiqDriverConfig(
            port=port,
            baudrate=baudrate,
            slave_id=slave_id,
            max_width=max_width,
            activation_timeout=activation_timeout,
            poll_interval=poll_interval,
        )

        self._port = configuration.port
        self._baudrate = configuration.baudrate
        self._slave_id = configuration.slave_id
        self._max_width = configuration.max_width
        self._activation_timeout = configuration.activation_timeout
        self._poll_interval = configuration.poll_interval

        # PyModbus 2.x defaults serial clients to ASCII, whereas 3.x uses the
        # modern RTU API. Select RTU explicitly for the pinned 2.5.x runtime.
        client_arguments = {
            "port": self._port,
            "baudrate": self._baudrate,
            "bytesize": 8,
            "parity": "N",
            "stopbits": 1,
            "timeout": 1,
        }
        if PYMODBUS_V2:
            client_arguments["method"] = "rtu"
        self._client = ModbusSerialClient(**client_arguments)

        try:
            # Polymetis pins PyModbus 2.5 while the standalone code uses 3.x;
            # their slave-address keywords differ. Resolve the keyword once.
            # This branch can disappear after both environments use one API.
            if PYMODBUS_V2:
                self._slave_kwarg = "unit"
            else:
                parameters = inspect.signature(
                    self._client.write_registers
                ).parameters
                if "device_id" in parameters:
                    self._slave_kwarg = "device_id"
                elif "slave" in parameters:
                    self._slave_kwarg = "slave"
                else:
                    raise RuntimeError(
                        "Unsupported PyModbus write_registers signature"
                    )

            if not self._client.connect():
                raise ConnectionError(
                    f"Could not connect to Robotiq gripper on {self._port}"
                )

            self._activate()

        except Exception:
            # __init__ failed, so the caller never receives an object on which
            # cleanup() could be called.
            self._client.close()
            raise

    # Public gripper API
    @property
    def port(self) -> str:
        """Configured serial-device path."""
        return self._port

    @property
    def baudrate(self) -> int:
        """Configured serial baud rate."""
        return self._baudrate

    @property
    def slave_id(self) -> int:
        """Configured Modbus slave address."""
        return self._slave_id

    @property
    def max_width(self) -> float:
        """Configured maximum physical opening in metres."""
        return self._max_width

    @property
    def activation_timeout(self) -> float:
        """Configured timeout for each activation stage in seconds."""
        return self._activation_timeout

    @property
    def poll_interval(self) -> float:
        """Configured delay between status polls in seconds."""
        return self._poll_interval

    def open(self, speed: float = 0.3) -> None:
        """Request full opening at minimum force."""
        self.move(position=0, speed=speed, force=0)

    def close(self, speed: float = 0.3, force: int = 130) -> None:
        """Request full closing with a raw force request from 0 to 255."""
        self.move(position=255, speed=speed, force=force)

    def move(self, position: int, speed: float = 0.3, force: int = 130) -> None:
        """Send one raw go-to-position request.

        Args:
            position: Raw position request from 0 (open) to 255 (closed).
            speed: Normalized speed from 0.0 (minimum) to 1.0 (maximum).
            force: Raw force request from 0 (minimum) to 255 (maximum).
        """
        (
            position_request,
            speed,
            force_request,
        ) = self._validate_raw_motion_request(position, speed, force)

        # rSP occupies the high byte and rFR occupies the low byte.
        speed_request = int(speed * 255)
        speed_force_request = (speed_request << 8) | force_request
        action_request = self._encode_action_request(
            activate=True,
            go_to=True,
        )

        self._write_command_registers(
            action_request=action_request,
            position_request=position_request,
            speed_force_request=speed_force_request,
        )

    def read_status(self) -> Optional[RobotiqStatus]:
        """Read and validate one three-register FC04 status snapshot.

        All checks operate on the response already received. Invalid or
        malformed responses return ``None`` rather than being decoded as a
        healthy state.
        """
        # FC04 reads the verified input-register status block at 0x07D0.
        response = self._client.read_input_registers(
            address=ROBOTIQ_INPUT_REGISTER_ADDRESS,
            count=ROBOTIQ_REGISTER_COUNT,
            **{self._slave_kwarg: self._slave_id},
        )
        if response is None or response.isError():
            log.warning("Robotiq Modbus read error: %s", response)
            return None

        return self._decode_status_registers(
            getattr(response, "registers", None)
        )

    @property
    def position(self) -> float:
        """Estimate width using ideal raw endpoints 0 (open) and 255 (closed).

        A higher-level adapter should use measured gPO endpoints when calibrated
        physical width is required.
        """
        status = self.read_status()

        if status is None:
            raise RuntimeError(
                f"Could not read Robotiq position on {self._port}"
            )

        return self._max_width * (1.0 - status["position"] / 255.0)

    @property
    def is_open(self) -> bool:
        """Return whether the gripper has completed an opening request."""
        status = self.read_status()

        return (
            status is not None
            and status["gACT"] == 1
            and status["gSTA"] == 3
            and status["fault"] == 0
            and status["kFLT"] == 0
            and status["position_echo"] == 0
            and status["gGTO"] == 1
            and status["gOBJ"] == 3
        )

    def is_ready(self) -> bool:
        """Return whether the gripper is activated and fault-free."""
        status = self.read_status()

        return (
            status is not None
            and status["gACT"] == 1
            and status["gSTA"] == 3
            and status["fault"] == 0
            and status["kFLT"] == 0
        )

    def cleanup(self) -> None:
        """Release the serial transport; this does not command a motion stop."""
        self._client.close()

    # Private Modbus implementation

    def _activate(self) -> None:
        """Reset and activate the gripper, waiting for both acknowledgements."""
        # Clear the complete six-byte command block so no stale position,
        # speed, or force request survives the reset.
        action_request = self._encode_action_request(activate=False, go_to=False)
        self._write_command_registers(
            action_request=action_request,
            position_request=0x0000,
            speed_force_request=0x0000,
        )

        # Wait until the gripper acknowledges the reset command.
        reset_deadline = time.monotonic() + self._activation_timeout
        reset_status = None
        while time.monotonic() < reset_deadline:
            remaining = reset_deadline - time.monotonic()
            time.sleep(min(self._poll_interval, max(0.0, remaining)))
            reset_status = self.read_status()

            if (
                reset_status is not None
                and reset_status["gACT"] == 0
                and reset_status["gSTA"] == 0
            ):
                break
        else:
            raise RuntimeError(
                f"Robotiq gripper on {self._port} did not acknowledge reset "
                f"within {self._activation_timeout:.1f} s; "
                f"last status: {reset_status}"
            )

        # Set only rACT and keep position, speed, and force cleared.
        action_request = self._encode_action_request(activate=True, go_to=False)
        self._write_command_registers(
            action_request=action_request,
            position_request=0x0000,
            speed_force_request=0x0000,
        )

        # Wait until automatic calibration completes and the gripper is ready.
        activation_deadline = time.monotonic() + self._activation_timeout
        last_status = None
        while time.monotonic() < activation_deadline:
            remaining = activation_deadline - time.monotonic()
            time.sleep(min(self._poll_interval, max(0.0, remaining)))
            last_status = self.read_status()

            if last_status is not None:
                fault = last_status["fault"]
                controller_fault = last_status["kFLT"]

                if controller_fault != 0:
                    raise RuntimeError(
                        f"Robotiq controller fault 0x{controller_fault:X} "
                        f"during activation on {self._port}: {last_status}"
                    )

                if fault >= 0x08:
                    raise RuntimeError(
                        f"Robotiq activation fault 0x{fault:02X} "
                        f"on {self._port}: {last_status}"
                    )

                if (
                    last_status["gACT"] == 1
                    and last_status["gSTA"] == 3
                    and fault == 0
                ):
                    return

        raise RuntimeError(
            f"Robotiq gripper on {self._port} did not activate within "
            f"{self._activation_timeout:.1f} s; "
            f"last status: {last_status}"
        )

    @staticmethod
    def _validate_raw_motion_request(position, speed, force):
        """Validate one raw motion request before it can reach Modbus.

        Values are rejected instead of clipped so an invalid input cannot turn
        silently into an endpoint, maximum-speed, or maximum-force command.
        """
        if isinstance(position, bool) or not isinstance(
            position, numbers.Integral
        ):
            raise ValueError(
                "position must be an integer from 0 to 255, "
                f"received {position!r}"
            )
        if not 0 <= position <= 255:
            raise ValueError(
                f"position must be between 0 and 255, received {position}"
            )

        if isinstance(speed, bool) or not isinstance(speed, numbers.Real):
            raise ValueError(
                f"speed must be numeric from 0.0 to 1.0, received {speed!r}"
            )
        speed = float(speed)
        if not math.isfinite(speed) or not 0.0 <= speed <= 1.0:
            raise ValueError(
                "speed must be finite and between 0.0 and 1.0, "
                f"received {speed}"
            )

        if isinstance(force, bool) or not isinstance(
            force, numbers.Integral
        ):
            raise ValueError(
                f"force must be an integer from 0 to 255, received {force!r}"
            )
        if not 0 <= force <= 255:
            raise ValueError(
                f"force must be between 0 and 255, received {force}"
            )

        return int(position), speed, int(force)

    @staticmethod
    def _validate_modbus_command_registers(
        action_request,
        position_request,
        speed_force_request,
    ):
        """Validate the final packed command block before FC16.

        This second boundary also protects reset/activation writes that do not
        pass through ``move``. It performs no I/O and adds no serial delay.
        """
        register_values = {
            "action_request": action_request,
            "position_request": position_request,
            "speed_force_request": speed_force_request,
        }
        for name, value in register_values.items():
            if isinstance(value, bool) or not isinstance(
                value, numbers.Integral
            ):
                raise ValueError(
                    f"{name} must be a 16-bit integer, received {value!r}"
                )

        action_request = int(action_request)
        position_request = int(position_request)
        speed_force_request = int(speed_force_request)

        if action_request not in ROBOTIQ_VALID_ACTION_REQUESTS:
            valid_values = ", ".join(
                f"0x{value:04X}"
                for value in sorted(ROBOTIQ_VALID_ACTION_REQUESTS)
            )
            raise ValueError(
                f"action_request must be one of {valid_values}, received "
                f"0x{action_request:04X}"
            )
        if not 0 <= position_request <= 0x00FF:
            raise ValueError(
                "position_request must keep its reserved high byte clear and "
                f"contain rPR from 0 to 255, received 0x{position_request:04X}"
            )
        if not 0 <= speed_force_request <= 0xFFFF:
            raise ValueError(
                "speed_force_request must be a 16-bit rSP/rFR value, "
                f"received {speed_force_request}"
            )

        return action_request, position_request, speed_force_request

    def _write_command_registers(
        self,
        action_request: int,
        position_request: int,
        speed_force_request: int,
    ) -> None:
        """Write one command block and verify its existing FC16 response.

        Checking address/count does not add another Modbus transaction. It
        confirms only protocol acknowledgement, not physical completion.
        """
        (
            action_request,
            position_request,
            speed_force_request,
        ) = self._validate_modbus_command_registers(
            action_request,
            position_request,
            speed_force_request,
        )
        values = [
            action_request,
            position_request,
            speed_force_request,
        ]
        response = self._client.write_registers(
            address=ROBOTIQ_OUTPUT_REGISTER_ADDRESS,
            values=values,
            **{self._slave_kwarg: self._slave_id},
        )

        if response is None or response.isError():
            raise RuntimeError(
                f"Robotiq Modbus write failed on {self._port}: "
                f"address=0x{ROBOTIQ_OUTPUT_REGISTER_ADDRESS:04X}, "
                f"values={[f'0x{value:04X}' for value in values]}, "
                f"response={response}"
            )

        response_address = getattr(response, "address", None)
        response_count = getattr(response, "count", None)
        if (
            response_address != ROBOTIQ_OUTPUT_REGISTER_ADDRESS
            or response_count != ROBOTIQ_REGISTER_COUNT
        ):
            raise RuntimeError(
                "Robotiq Modbus write returned an unexpected acknowledgement "
                f"on {self._port}: address={response_address}, "
                f"count={response_count}, response={response}"
            )

    def _decode_status_registers(
        self,
        registers,
    ) -> Optional[RobotiqStatus]:
        """Validate and decode one raw three-register status block."""
        if registers is None or len(registers) != ROBOTIQ_REGISTER_COUNT:
            log.warning(
                "Expected %d Robotiq status registers, received: %s",
                ROBOTIQ_REGISTER_COUNT,
                registers,
            )
            return None

        if any(
            isinstance(value, bool)
            or not isinstance(value, numbers.Integral)
            or not 0 <= value <= 0xFFFF
            for value in registers
        ):
            log.warning(
                "Robotiq status registers must be 16-bit integers: %s",
                registers,
            )
            return None

        (
            gripper_status_register,
            fault_and_request_register,
            position_and_current_register,
        ) = registers

        # Register 0x07D0: GRIPPER STATUS in the high byte.
        gripper_status_byte = (gripper_status_register >> 8) & 0xFF
        fault_status_byte = (fault_and_request_register >> 8) & 0xFF

        if gripper_status_register & 0x00FF:
            log.warning(
                "Reserved low byte of Robotiq status register is nonzero: "
                "0x%04X",
                gripper_status_register,
            )
            return None
        if gripper_status_byte & 0x06:
            log.warning(
                "Reserved Robotiq status bits 1-2 are nonzero: 0x%02X",
                gripper_status_byte,
            )
            return None

        gripper_state = (gripper_status_byte >> 4) & 0x03
        if gripper_state == 2:
            log.warning("Robotiq gSTA returned reserved state 2")
            return None

        gripper_fault = fault_status_byte & 0x0F
        controller_fault = (fault_status_byte >> 4) & 0x0F

        return {
            # Register 0x07D0: gripper status byte
            "gACT": gripper_status_byte & 0x01,
            "gGTO": (gripper_status_byte >> 3) & 0x01,
            "gSTA": gripper_state,
            "gOBJ": (gripper_status_byte >> 6) & 0x03,
            # Register 0x07D1: fault status and position-request echo
            "fault": gripper_fault,
            "gFLT": gripper_fault,
            "kFLT": controller_fault,
            "position_echo": fault_and_request_register & 0xFF,
            # Register 0x07D2: actual position and motor current
            "position": (position_and_current_register >> 8) & 0xFF,
            "current": position_and_current_register & 0xFF,
        }

    def _encode_action_request(
        self,
        activate: bool = False,
        go_to: bool = False,
    ) -> int:
        """Return the ACTION REQUEST value for Modbus address 0x03E8."""
        # See the module-linked manual: ACTION REQUEST bits on p. 32 and the
        # corresponding Modbus command examples on pp. 54-57.

        if go_to and not activate:
            raise ValueError(
                "A go-to request requires the gripper to remain activated."
            )

        if activate and go_to:
            # 0x0900: rGTO bit 3 + rACT bit 0.
            action_request_byte = 0x09
        elif activate:
            # 0x0100: rACT bit 0.
            action_request_byte = 0x01
        else:
            # 0x0000: clear all action bits for reset.
            action_request_byte = 0x00

        # Pack ACTION REQUEST in the high byte (bits 15-8); keep the options clear.
        return action_request_byte << 8

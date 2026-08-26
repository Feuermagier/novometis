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
# Derived from the RLinf Robotiq class:
# https://github.com/RLinf/RLinf/blob/main/rlinf/envs/realworld/common/gripper/robotiq_gripper.py
# and adapted by fixing all bugs, validation with useful error messages, diagnostics, and polymetis integration.
# See the adjacent NOTICE and LICENSE files.

"""Robotiq 2F-85 / 2F-140 gripper via direct Modbus RTU over USB-RS485.

No ROS dependency — communicates with the gripper through ``pymodbus``
and a USB-RS485 adapter, preferably addressed through a stable
``/dev/serial/by-id/...`` path.

Robotiq input/output register mapping: https://assets.robotiq.com/website-assets/support_documents/document/2F-85_2F-140_Instruction_Manual_CB-Series_PDF_20190206.pdf, Section 42, Page 48.

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
2      reg1 hi   Fault status: kFLT high nibble, gFLT low nibble
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
from typing import Optional


# Polymetis pins PyModbus 2.5 while the standalone code uses 3.x,
# their slave-address keywords differ. Resolve the keyword once.
# This branch can disappear after both environments use one API.
try:
    from pymodbus.client import ModbusSerialClient
except ImportError:  # PyModbus 2.x
    from pymodbus.client.sync import ModbusSerialClient
    PYMODBUS_V2 = True
else:
    PYMODBUS_V2 = False


ROBOTIQ_SUPPORTED_BAUDRATES = frozenset(
    {1200, 2400, 4800, 9600, 19200, 38400, 57600, 115200}
)
MODBUS_SLAVE_ID_MIN = 1
MODBUS_SLAVE_ID_MAX = 247
MODBUS_MIN_REQUEST_INTERVAL_SECONDS = 0.005
# Robotiq register protocol:
# https://assets.robotiq.com/website-assets/support_documents/document/2F-85_2F-140_Instruction_Manual_CB-Series_PDF_20190206.pdf
ROBOTIQ_REGISTER_COUNT = 3
ROBOTIQ_OUTPUT_REGISTER_ADDRESS = 0x03E8
ROBOTIQ_INPUT_REGISTER_ADDRESS = 0x07D0
ROBOTIQ_VALID_ACTION_REQUESTS = frozenset({0x0000, 0x0100, 0x0900})

log = logging.getLogger(__name__)


class RobotiqError(RuntimeError):
    """A low-level Robotiq communication or device operation failed."""


@dataclass
class RobotiqStatus:
    """Decoded values returned by one Robotiq FC04 state reading."""

    gACT: int
    gGTO: int
    gSTA: int
    gOBJ: int
    gFLT: int
    kFLT: int
    gPR: int
    gPO: int
    gCU: int

    @property
    def fault_code(self) -> int:
        """Return the complete Robotiq fault byte."""
        return (self.kFLT << 4) | self.gFLT

    @property
    def is_ready(self) -> bool:
        """Return whether the gripper is activated and fault-free."""
        return self.gACT == 1 and self.gSTA == 3 and self.fault_code == 0


@dataclass
class RobotiqDriverConfig:
    """Validated serial and activation settings for the Robotiq low-level control class."""

    port: str
    baudrate: int
    slave_id: int
    response_timeout: float
    activation_timeout: float
    poll_interval: float

    def __post_init__(self):
        """Reject invalid settings without coercing caller-provided values."""
        if type(self.port) is not str:
            raise TypeError(
                "port must be str, received "
                f"{type(self.port).__name__}: {self.port!r}"
            )
        if not self.port or self.port != self.port.strip():
            raise ValueError(
                "port must be non-empty and contain no surrounding "
                f"whitespace, received {self.port!r}"
            )

        if type(self.baudrate) is not int:
            raise TypeError(
                "baudrate must be int, received "
                f"{type(self.baudrate).__name__}: {self.baudrate!r}"
            )
        if self.baudrate not in ROBOTIQ_SUPPORTED_BAUDRATES:
            supported = ", ".join(
                str(value) for value in sorted(ROBOTIQ_SUPPORTED_BAUDRATES)
            )
            raise ValueError(
                f"baudrate must be one of {supported}, received {self.baudrate}"
            )

        if type(self.slave_id) is not int:
            raise TypeError(
                "slave_id must be int, received "
                f"{type(self.slave_id).__name__}: {self.slave_id!r}"
            )
        if not MODBUS_SLAVE_ID_MIN <= self.slave_id <= MODBUS_SLAVE_ID_MAX:
            raise ValueError(
                f"slave_id must be between {MODBUS_SLAVE_ID_MIN} and "
                f"{MODBUS_SLAVE_ID_MAX}, received {self.slave_id}"
            )

        for name in (
            "response_timeout",
            "activation_timeout",
            "poll_interval",
        ):
            value = getattr(self, name)
            if type(value) is not float:
                raise TypeError(
                    f"{name} must be float, received "
                    f"{type(value).__name__}: {value!r}"
                )
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, received {value!r}")

        if self.response_timeout <= 0.0:
            raise ValueError(
                "response_timeout must be positive, received "
                f"{self.response_timeout}"
            )
        if self.activation_timeout <= 0.0:
            raise ValueError(
                "activation_timeout must be positive, received "
                f"{self.activation_timeout}"
            )
        if self.poll_interval < MODBUS_MIN_REQUEST_INTERVAL_SECONDS:
            raise ValueError(
                "poll_interval must be at least "
                f"{MODBUS_MIN_REQUEST_INTERVAL_SECONDS} seconds, received "
                f"{self.poll_interval}"
            )
        if self.poll_interval > self.activation_timeout:
            raise ValueError(
                "poll_interval must not exceed activation_timeout, received "
                f"poll_interval={self.poll_interval} and "
                f"activation_timeout={self.activation_timeout}"
            )


class Robotiq2FingerGripper:
    """Synchronous low-level control for a Robotiq 2F gripper over Modbus RTU.

    Creating an instance opens the serial connection, resets the gripper, and
    waits for activation to complete. Activation performs an automatic
    calibration and may move the fingers.

    ``move`` sends one validated FC16 command using raw rPR, rSP, and rFR
    values. ``read_status`` reads and decodes one validated FC04 snapshot.

    The Polymetis hardware client owns protobuf/gRPC communication, command
    caching, and conversion between SI units and Robotiq request values.

    Args:
        port: Stable serial device path, preferably under
            ``/dev/serial/by-id/``.
        baudrate: Standard Modbus baud rate (default 115200).
        slave_id: Modbus slave address (default 0x09).
        response_timeout: Serial response timeout in seconds.
        activation_timeout: Maximum duration of each activation stage in
            seconds.
        poll_interval: Delay between activation-status reads in seconds.
    """

    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        slave_id: int = 0x09,
        response_timeout: float = 1.0,
        activation_timeout: float = 5.0,
        poll_interval: float = 0.1,
    ):
        # Validate all parameters once before constructing the serial client.
        configuration = RobotiqDriverConfig(
            port=port,
            baudrate=baudrate,
            slave_id=slave_id,
            response_timeout=response_timeout,
            activation_timeout=activation_timeout,
            poll_interval=poll_interval,
        )
        self._port = configuration.port
        self._baudrate = configuration.baudrate
        self._slave_id = configuration.slave_id
        self._response_timeout = configuration.response_timeout
        self._activation_timeout = configuration.activation_timeout
        self._poll_interval = configuration.poll_interval
        self._closed = False
        self._last_request_started_at = None

        # PyModbus 2.5 defaults serial clients to ASCII, while 3.x uses the
        # modern RTU API. Select RTU explicitly for the pinned 2.5.x runtime.
        client_arguments = {
            "port": self._port,
            "baudrate": self._baudrate,
            "bytesize": 8,
            "parity": "N",
            "stopbits": 1,
            "timeout": self._response_timeout,
        }
        if PYMODBUS_V2:
            client_arguments["method"] = "rtu"
        self._client = ModbusSerialClient(**client_arguments)

        try:
            # Polymetis pins PyModbus 2.5 while the standalone code uses 3.x,
            # their slave-address keywords differ. Resolve the keyword once.
            # This branch can disappear after both environments use one API.
            if PYMODBUS_V2:
                self._slave_kwarg = "unit"
            else:
                parameters = inspect.signature(self._client.write_registers).parameters
                if "device_id" in parameters:
                    self._slave_kwarg = "device_id"
                elif "slave" in parameters:
                    self._slave_kwarg = "slave"
                else:
                    raise RobotiqError("Unsupported PyModbus write_registers signature")

            try:
                connected = self._client.connect()
            except Exception as error:
                raise RobotiqError(
                    f"Robotiq connection failed on {self._port}: {error}"
                ) from error
            if not connected:
                raise RobotiqError(
                    f"Could not connect to Robotiq gripper on {self._port}"
                )

            self._activate()

        except BaseException:
            # __init__ failed, so the caller never receives an object on which
            # cleanup() could be called.
            self._closed = True
            try:
                self._client.close()
            except Exception:
                log.exception(
                    "Failed to close the Robotiq serial transport after "
                    "initialization failed"
                )
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
    def response_timeout(self) -> float:
        """Configured serial response timeout in seconds."""
        return self._response_timeout

    @property
    def activation_timeout(self) -> float:
        """Configured timeout for each activation stage in seconds."""
        return self._activation_timeout

    @property
    def poll_interval(self) -> float:
        """Configured delay between status polls in seconds."""
        return self._poll_interval

    def move(
        self,
        position_request: int,
        speed_request: int,
        force_request: int,
    ) -> None:
        """Send one raw go-to-position request.

        Args:
            position_request: Raw rPR value, 0 (open) to 255 (closed).
            speed_request: Raw rSP value, 0 (minimum) to 255 (maximum).
            force_request: Raw rFR value, 0 (minimum) to 255 (maximum).
        """
        if self._closed:
            raise RobotiqError("Cannot move a closed Robotiq serial transport")

        (
            position_request,
            speed_request,
            force_request,
        ) = self._validate_raw_motion_request(
            position_request,
            speed_request,
            force_request,
        )

        # rSP occupies the high byte and rFR occupies the low byte.
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
        if self._closed:
            raise RobotiqError("Cannot read from a closed Robotiq serial transport")

        # FC04 reads the verified input-register status block at 0x07D0.
        try:
            self._wait_for_request_interval()
            response = self._client.read_input_registers(
                address=ROBOTIQ_INPUT_REGISTER_ADDRESS,
                count=ROBOTIQ_REGISTER_COUNT,
                **{self._slave_kwarg: self._slave_id},
            )
            if response is None or response.isError():
                raise RobotiqError(
                    f"Robotiq FC04 read failed on {self._port}: {response}"
                )
        except RobotiqError:
            raise
        except Exception as error:
            raise RobotiqError(
                f"Robotiq FC04 read failed on {self._port}: {error}"
            ) from error

        return self._decode_status_registers(getattr(response, "registers", None))

    def cleanup(self) -> None:
        """Release the serial transport. This does not command a motion stop."""
        if self._closed:
            return
        self._closed = True
        try:
            self._client.close()
        except Exception as error:
            raise RobotiqError(
                f"Could not close Robotiq transport on {self._port}: {error}"
            ) from error

    # Private control-related methods

    def _wait_for_request_interval(self) -> None:
        """Respect Robotiq's minimum interval between Modbus requests."""
        now = time.monotonic()
        if self._last_request_started_at is not None:
            delay = MODBUS_MIN_REQUEST_INTERVAL_SECONDS - (
                now - self._last_request_started_at
            )
            if delay > 0.0:
                time.sleep(delay)
        self._last_request_started_at = time.monotonic()

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
        reset_error = None
        while time.monotonic() < reset_deadline:
            remaining = reset_deadline - time.monotonic()
            time.sleep(min(self._poll_interval, max(0.0, remaining)))
            try:
                reset_status = self.read_status()
                reset_error = None
            except RobotiqError as error:
                reset_status = None
                reset_error = error
                continue

            if (
                reset_status is not None
                and reset_status.gACT == 0
                and reset_status.gSTA == 0
            ):
                break
        else:
            details = reset_status if reset_status is not None else reset_error
            raise RobotiqError(
                f"Robotiq gripper on {self._port} did not acknowledge reset "
                f"within {self._activation_timeout:.1f} s; "
                f"last result: {details}"
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
        last_error = None
        while time.monotonic() < activation_deadline:
            remaining = activation_deadline - time.monotonic()
            time.sleep(min(self._poll_interval, max(0.0, remaining)))
            try:
                last_status = self.read_status()
                last_error = None
            except RobotiqError as error:
                last_status = None
                last_error = error
                continue

            if last_status is not None:
                fault = last_status.gFLT
                controller_fault = last_status.kFLT

                if controller_fault != 0:
                    raise RobotiqError(
                        f"Robotiq controller fault 0x{controller_fault:X} "
                        f"during activation on {self._port}: {last_status}"
                    )

                if fault >= 0x08:
                    raise RobotiqError(
                        f"Robotiq activation fault 0x{fault:02X} "
                        f"on {self._port}: {last_status}"
                    )

                if last_status.is_ready:
                    return

        details = last_status if last_status is not None else last_error
        raise RobotiqError(
            f"Robotiq gripper on {self._port} did not activate within "
            f"{self._activation_timeout:.1f} s; "
            f"last result: {details}"
        )

    @staticmethod
    def _validate_raw_motion_request(
        position_request,
        speed_request,
        force_request,
    ):
        """Validate one raw motion request before it can reach Modbus.

        Values are rejected instead of clipped so an invalid input cannot turn
        silently into an endpoint, maximum-speed, or maximum-force command.
        """
        requests = {
            "position_request": position_request,
            "speed_request": speed_request,
            "force_request": force_request,
        }
        for name, value in requests.items():
            if type(value) is not int:
                raise TypeError(
                    f"{name} must be int, received "
                    f"{type(value).__name__}: {value!r}"
                )
            if not 0 <= value <= 255:
                raise ValueError(f"{name} must be between 0 and 255, received {value}")

        return position_request, speed_request, force_request

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
            if type(value) is not int:
                raise TypeError(f"{name} must be a 16-bit integer, received {value!r}")

        if action_request not in ROBOTIQ_VALID_ACTION_REQUESTS:
            valid_values = ", ".join(
                f"0x{value:04X}" for value in sorted(ROBOTIQ_VALID_ACTION_REQUESTS)
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
        try:
            self._wait_for_request_interval()
            response = self._client.write_registers(
                address=ROBOTIQ_OUTPUT_REGISTER_ADDRESS,
                values=values,
                **{self._slave_kwarg: self._slave_id},
            )
            if response is None or response.isError():
                raise RobotiqError(
                    f"Robotiq FC16 write failed on {self._port}: "
                    f"address=0x{ROBOTIQ_OUTPUT_REGISTER_ADDRESS:04X}, "
                    f"values={[f'0x{value:04X}' for value in values]}, "
                    f"response={response}"
                )
        except RobotiqError:
            raise
        except Exception as error:
            raise RobotiqError(
                f"Robotiq FC16 write failed on {self._port}: {error}"
            ) from error

        response_address = getattr(response, "address", None)
        response_count = getattr(response, "count", None)
        if (
            response_address != ROBOTIQ_OUTPUT_REGISTER_ADDRESS
            or response_count != ROBOTIQ_REGISTER_COUNT
        ):
            raise RobotiqError(
                "Robotiq Modbus write returned an unexpected acknowledgement "
                f"on {self._port}: address={response_address}, "
                f"count={response_count}, response={response}"
            )

    def _decode_status_registers(
        self,
        registers,
    ) -> Optional[RobotiqStatus]:
        """Validate and decode one raw three-register status block."""
        try:
            register_count = len(registers)
        except TypeError:
            register_count = None
        if register_count != ROBOTIQ_REGISTER_COUNT:
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

        registers = tuple(int(value) for value in registers)

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
                "Reserved low byte of Robotiq status register is nonzero: " "0x%04X",
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

        return RobotiqStatus(
            # Register 0x07D0: gripper status byte
            gACT=gripper_status_byte & 0x01,
            gGTO=(gripper_status_byte >> 3) & 0x01,
            gSTA=gripper_state,
            gOBJ=(gripper_status_byte >> 6) & 0x03,
            # Register 0x07D1: fault status and position-request echo
            gFLT=gripper_fault,
            kFLT=controller_fault,
            gPR=fault_and_request_register & 0xFF,
            # Register 0x07D2: actual position and motor current
            gPO=(position_and_current_register >> 8) & 0xFF,
            gCU=position_and_current_register & 0xFF,
        )

    @staticmethod
    def _encode_action_request(
        activate: bool = False,
        go_to: bool = False,
    ) -> int:
        """Return the ACTION REQUEST value for Modbus address 0x03E8."""
        # See the module-linked manual: ACTION REQUEST bits on p. 32 and the
        # corresponding Modbus command examples on pp. 54-57.

        for name, value in (("activate", activate), ("go_to", go_to)):
            if type(value) is not bool:
                raise TypeError(
                    f"{name} must be bool, received "
                    f"{type(value).__name__}: {value!r}"
                )

        if go_to and not activate:
            raise ValueError(
                "A go-to request requires the gripper to remain activated, "
                f"received activate={activate} and go_to={go_to}."
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

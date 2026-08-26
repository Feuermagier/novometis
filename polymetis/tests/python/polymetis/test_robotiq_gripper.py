import math
import struct
from dataclasses import astuple, fields
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
from omegaconf import OmegaConf

import polymetis_pb2
from polymetis.robot_client.robotiq_gripper import robotiq_gripper_client
from polymetis.robot_client.robotiq_gripper.robotiq_gripper_client import (
    RobotiqClientConfig,
    RobotiqGripperClient,
)
from polymetis.robot_client.robotiq_gripper.third_party.robotiq_2finger_grippers import (
    robotiq_2f_gripper_refactored,
)

Robotiq2FingerGripper = robotiq_2f_gripper_refactored.Robotiq2FingerGripper
RobotiqDriverConfig = robotiq_2f_gripper_refactored.RobotiqDriverConfig
RobotiqStatus = robotiq_2f_gripper_refactored.RobotiqStatus
RobotiqError = robotiq_2f_gripper_refactored.RobotiqError


def adapter_values(**overrides):
    values = {
        "server_ip": "localhost",
        "server_port": 4322,
        "max_width": 0.085,
        "hz": 100,
        "model_min_speed_m_s": 0.020,
        "model_max_speed_m_s": 0.150,
        "model_min_force_n": 20.0,
        "model_max_force_n": 235.0,
        "max_command_speed_m_s": 0.150,
        "max_command_force_n": 235.0,
        "grpc_timeout": 1.0,
        "max_command_state_age": 0.25,
    }
    values.update(overrides)
    return values


def driver_values(**overrides):
    values = {
        "port": "/dev/serial/by-id/test-robotiq",
        "baudrate": 115200,
        "slave_id": 9,
        "response_timeout": 1.0,
        "activation_timeout": 5.0,
        "poll_interval": 0.1,
    }
    values.update(overrides)
    return values


def make_adapter(**overrides):
    adapter = object.__new__(RobotiqGripperClient)
    adapter._config = RobotiqClientConfig(**adapter_values(**overrides))
    return adapter


def make_status(**overrides):
    values = {
        "gACT": 1,
        "gGTO": 0,
        "gSTA": 3,
        "gOBJ": 0,
        "gFLT": 0,
        "kFLT": 0,
        "gPR": 0,
        "gPO": 0,
        "gCU": 0,
    }
    values.update(overrides)
    return RobotiqStatus(**values)


def make_driver(**config_overrides):
    driver = object.__new__(Robotiq2FingerGripper)
    config = RobotiqDriverConfig(**driver_values(**config_overrides))
    driver._port = config.port
    driver._baudrate = config.baudrate
    driver._slave_id = config.slave_id
    driver._response_timeout = config.response_timeout
    driver._activation_timeout = config.activation_timeout
    driver._poll_interval = config.poll_interval
    driver._closed = False
    driver._last_request_started_at = None
    driver._slave_kwarg = "unit"
    driver._client = MagicMock()
    return driver


def successful_write_response():
    response = MagicMock()
    response.isError.return_value = False
    response.address = robotiq_2f_gripper_refactored.ROBOTIQ_OUTPUT_REGISTER_ADDRESS
    response.count = robotiq_2f_gripper_refactored.ROBOTIQ_REGISTER_COUNT
    return response


def adjacent_float32(value, direction):
    bits = struct.unpack("!I", struct.pack("!f", value))[0]
    return struct.unpack("!f", struct.pack("!I", bits + direction))[0]


def configured_fields(config_class, source):
    return {
        field.name: source[field.name] for field in fields(config_class) if field.init
    }


def test_valid_configuration_is_preserved_and_protobuf_limits_are_derived():
    config = RobotiqClientConfig(**adapter_values())
    protobuf_limits = config.protobuf_limits

    assert config.max_width == 0.085
    assert (
        protobuf_limits["max_width"] == struct.unpack("!f", struct.pack("!f", 0.085))[0]
    )
    assert protobuf_limits["max_width"] != config.max_width
    assert RobotiqDriverConfig(**driver_values()).port == driver_values()["port"]
    with pytest.raises(TypeError):
        protobuf_limits["max_width"] = 0.1


def test_robotiq_yaml_configuration_satisfies_strict_types():
    config_dir = Path(__file__).resolve().parents[3] / "conf"
    config = OmegaConf.merge(
        OmegaConf.load(config_dir / "launch_gripper.yaml"),
        OmegaConf.load(config_dir / "gripper" / "robotiq_2f.yaml"),
    )
    config.ip = "localhost"
    config.port = 4322
    config.gripper.port = "/dev/serial/by-id/test-robotiq"

    assert set(config.gripper) == {
        field.name for field in fields(RobotiqClientConfig) if field.init
    } | {
        "_target_",
        "baudrate",
        "port",
        "slave_id",
        "response_timeout",
        "activation_timeout",
        "poll_interval",
    }

    RobotiqClientConfig(**configured_fields(RobotiqClientConfig, config.gripper))
    RobotiqDriverConfig(**configured_fields(RobotiqDriverConfig, config.gripper))


@pytest.mark.parametrize(
    "baudrate",
    sorted(robotiq_2f_gripper_refactored.ROBOTIQ_SUPPORTED_BAUDRATES),
)
def test_driver_config_accepts_documented_baudrates(baudrate):
    assert RobotiqDriverConfig(**driver_values(baudrate=baudrate)).baudrate == baudrate


def test_client_passes_transport_settings_and_float32_metadata_width():
    driver = MagicMock()
    channel = MagicMock()
    connection = MagicMock()

    with patch.object(
        robotiq_gripper_client,
        "Robotiq2FingerGripper",
        return_value=driver,
    ) as driver_class, patch.object(
        robotiq_gripper_client.grpc,
        "insecure_channel",
        return_value=channel,
    ), patch.object(
        robotiq_gripper_client.polymetis_pb2_grpc,
        "GripperServerStub",
        return_value=connection,
    ):
        client = RobotiqGripperClient(
            server_ip="localhost",
            server_port=4322,
            port="/dev/serial/by-id/test-robotiq",
            hz=100,
        )

    assert driver_class.call_args.kwargs["response_timeout"] == 1.0
    assert "max_width" not in driver_class.call_args.kwargs
    metadata = connection.InitRobotClient.call_args.args[0]
    assert metadata.max_width == client._config.protobuf_limits["max_width"]

    client.cleanup()
    channel.close.assert_called_once_with()
    driver.cleanup.assert_called_once_with()


def test_invalid_client_config_fails_before_hardware_construction():
    with patch.object(
        robotiq_gripper_client,
        "Robotiq2FingerGripper",
    ) as driver_class:
        with pytest.raises(TypeError, match="server_port"):
            RobotiqGripperClient(
                server_ip="localhost",
                server_port="4322",
                port="/dev/serial/by-id/test-robotiq",
            )

    driver_class.assert_not_called()


@pytest.mark.parametrize(
    ("config_class", "values", "field", "value"),
    [
        (RobotiqClientConfig, adapter_values, "server_ip", None),
        (RobotiqClientConfig, adapter_values, "server_port", "4322"),
        (RobotiqClientConfig, adapter_values, "hz", 100.0),
        (RobotiqClientConfig, adapter_values, "max_width", "0.085"),
        (RobotiqClientConfig, adapter_values, "model_min_speed_m_s", 0),
        (RobotiqClientConfig, adapter_values, "model_max_speed_m_s", 1),
        (RobotiqClientConfig, adapter_values, "model_min_force_n", 20),
        (RobotiqClientConfig, adapter_values, "model_max_force_n", 235),
        (RobotiqClientConfig, adapter_values, "max_command_speed_m_s", 1),
        (RobotiqClientConfig, adapter_values, "max_command_force_n", 235),
        (RobotiqClientConfig, adapter_values, "grpc_timeout", 1),
        (RobotiqClientConfig, adapter_values, "max_command_state_age", 1),
        (RobotiqDriverConfig, driver_values, "port", None),
        (RobotiqDriverConfig, driver_values, "baudrate", "115200"),
        (RobotiqDriverConfig, driver_values, "slave_id", 9.0),
        (RobotiqDriverConfig, driver_values, "response_timeout", 1),
        (RobotiqDriverConfig, driver_values, "activation_timeout", 5),
        (RobotiqDriverConfig, driver_values, "poll_interval", "0.1"),
    ],
)
def test_configuration_rejects_wrong_types(config_class, values, field, value):
    with pytest.raises(TypeError, match=field):
        config_class(**values(**{field: value}))


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"server_ip": ""}, "server_ip"),
        ({"server_ip": " "}, "server_ip"),
        ({"server_ip": " localhost"}, "server_ip"),
        ({"server_ip": "localhost "}, "server_ip"),
        ({"server_port": 0}, "server_port"),
        ({"server_port": 65536}, "server_port"),
        ({"hz": 0}, "hz"),
        ({"hz": 201}, "hz"),
        ({"max_width": 0.0}, "max_width"),
        ({"max_width": math.nan}, "max_width"),
        ({"model_min_speed_m_s": 0.0}, "model speed limits"),
        (
            {"model_min_speed_m_s": 0.150},
            "model speed limits",
        ),
        ({"max_command_speed_m_s": 0.151}, "max_command_speed_m_s"),
        ({"model_min_force_n": -1.0}, "model force limits"),
        ({"model_min_force_n": 235.0}, "model force limits"),
        ({"max_command_force_n": 236.0}, "max_command_force_n"),
        ({"grpc_timeout": 0.0}, "grpc_timeout"),
        ({"max_command_state_age": math.inf}, "max_command_state_age"),
    ],
)
def test_client_config_rejects_invalid_values(overrides, match):
    with pytest.raises(ValueError, match=match):
        RobotiqClientConfig(**adapter_values(**overrides))


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"port": ""}, "port"),
        ({"port": " /dev/ttyUSB0"}, "port"),
        ({"baudrate": 0}, "baudrate"),
        ({"baudrate": 115201}, "baudrate"),
        ({"slave_id": 0}, "slave_id"),
        ({"slave_id": 248}, "slave_id"),
        ({"response_timeout": 0.0}, "response_timeout"),
        ({"response_timeout": math.nan}, "response_timeout"),
        ({"activation_timeout": 0.0}, "activation_timeout"),
        ({"activation_timeout": math.inf}, "activation_timeout"),
        ({"poll_interval": 0.004}, "poll_interval"),
        ({"poll_interval": math.nan}, "poll_interval"),
        (
            {"activation_timeout": 0.05, "poll_interval": 0.1},
            "poll_interval",
        ),
    ],
)
def test_driver_config_rejects_invalid_values(overrides, match):
    with pytest.raises(ValueError, match=match):
        RobotiqDriverConfig(**driver_values(**overrides))


@pytest.mark.parametrize(
    ("arguments", "error_type", "match"),
    [
        (("0", 0, 0), TypeError, "position_request"),
        ((True, 0, 0), TypeError, "position_request"),
        ((0, 0.5, 0), TypeError, "speed_request"),
        ((0, True, 0), TypeError, "speed_request"),
        ((0, 0, 0.0), TypeError, "force_request"),
        ((0, 0, False), TypeError, "force_request"),
        ((-1, 0, 0), ValueError, "position_request"),
        ((0, -1, 0), ValueError, "speed_request"),
        ((0, 256, 0), ValueError, "speed_request"),
        ((0, 0, 256), ValueError, "force_request"),
    ],
)
def test_raw_motion_requests_are_strict(arguments, error_type, match):
    with pytest.raises(error_type, match=match):
        Robotiq2FingerGripper._validate_raw_motion_request(*arguments)


@pytest.mark.parametrize(
    "arguments",
    [
        (0, 0, 0),
        (255, 255, 255),
    ],
)
def test_raw_motion_request_boundaries_are_preserved(arguments):
    assert Robotiq2FingerGripper._validate_raw_motion_request(*arguments) == arguments


@pytest.mark.parametrize(
    ("activate", "go_to", "error_type"),
    [
        (False, False, None),
        (True, False, None),
        (True, True, None),
        (False, True, ValueError),
        (1, False, TypeError),
        (True, 1, TypeError),
    ],
)
def test_action_request_encoding_is_strict(activate, go_to, error_type):
    if error_type is not None:
        with pytest.raises(error_type):
            Robotiq2FingerGripper._encode_action_request(activate, go_to)
        return

    expected = {
        (False, False): 0x0000,
        (True, False): 0x0100,
        (True, True): 0x0900,
    }[(activate, go_to)]
    assert Robotiq2FingerGripper._encode_action_request(activate, go_to) == expected


def test_float32_command_boundaries_map_to_protocol_endpoints():
    adapter = make_adapter()

    open_command = polymetis_pb2.GripperCommand(
        width=0.085,
        speed=0.020,
        force=0.0,
    )
    close_command = polymetis_pb2.GripperCommand(
        width=0.0,
        speed=0.150,
        force=235.0,
    )

    assert adapter._command_to_robotiq_arguments(open_command) == {
        "position_request": 0,
        "speed_request": 0,
        "force_request": 0,
    }
    assert adapter._command_to_robotiq_arguments(close_command) == {
        "position_request": 255,
        "speed_request": 255,
        "force_request": 255,
    }


def test_policy_ceilings_do_not_rescale_model_conversion():
    adapter = make_adapter(
        max_command_speed_m_s=0.085,
        max_command_force_n=100.0,
    )
    command = polymetis_pb2.GripperCommand(
        width=0.0425,
        speed=0.085,
        force=100.0,
    )

    raw = adapter._command_to_robotiq_arguments(command)

    assert raw["position_request"] == 128
    assert 0 < raw["speed_request"] < 255
    assert raw["force_request"] == 95


def test_force_special_cases():
    adapter = make_adapter()
    minimum_force = polymetis_pb2.GripperCommand(
        width=0.0,
        speed=0.020,
        force=20.0,
    )
    subminimum_force = polymetis_pb2.GripperCommand(
        width=0.0,
        speed=0.020,
        force=10.0,
    )

    assert adapter._command_to_robotiq_arguments(minimum_force)["force_request"] == 0
    with pytest.raises(ValueError, match="force"):
        adapter._command_to_robotiq_arguments(subminimum_force)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (field, value)
        for field in ("width", "speed", "force")
        for value in (math.nan, math.inf, -math.inf)
    ]
    + [
        ("width", adjacent_float32(0.085, 1)),
        ("speed", adjacent_float32(0.020, -1)),
        ("speed", adjacent_float32(0.150, 1)),
        ("force", adjacent_float32(235.0, 1)),
    ],
)
def test_invalid_protobuf_commands_are_rejected(field, value):
    adapter = make_adapter()
    values = {"width": 0.085, "speed": 0.020, "force": 0.0}
    values[field] = value
    command = polymetis_pb2.GripperCommand(**values)

    with pytest.raises(ValueError, match=field):
        adapter._command_to_robotiq_arguments(command)


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_width": 3.5e38},
        {"max_width": 1.0e-50},
        {
            "model_min_speed_m_s": 0.1,
            "model_max_speed_m_s": math.nextafter(0.1, math.inf),
            "max_command_speed_m_s": 0.1,
        },
        {
            "model_min_force_n": 0.1,
            "model_max_force_n": math.nextafter(0.1, math.inf),
            "max_command_force_n": 0.1,
        },
    ],
)
def test_invalid_float32_protobuf_limits_are_rejected(overrides):
    with pytest.raises(ValueError):
        RobotiqClientConfig(**adapter_values(**overrides))


@pytest.mark.parametrize(
    ("position", "expected_width"),
    [
        (0, 0.085),
        (3, 0.085 * 252.0 / 255.0),
        (127, 0.085 * 128.0 / 255.0),
        (128, 0.085 * 127.0 / 255.0),
        (230, 0.085 * 25.0 / 255.0),
        (255, 0.0),
    ],
)
def test_feedback_maps_documented_raw_range(position, expected_width):
    adapter = make_adapter()

    assert adapter._status_to_polymetis_width(
        make_status(gPO=position)
    ) == pytest.approx(expected_width)


@pytest.mark.parametrize("position", ["3", 3.0, -1, 256])
def test_feedback_rejects_invalid_raw_position(position):
    adapter = make_adapter()
    error_type = TypeError if type(position) is not int else ValueError

    with pytest.raises(error_type, match="status position"):
        adapter._status_to_polymetis_width(make_status(gPO=position))


def test_decoded_status_uses_builtin_integers_across_adapter_boundary():
    driver = object.__new__(Robotiq2FingerGripper)
    status = driver._decode_status_registers(
        [
            np.int64(0x3100),
            np.int64(0x0000),
            np.int64(0x0300),
        ]
    )

    assert all(type(value) is int for value in astuple(status))
    assert make_adapter()._status_to_polymetis_width(status) == pytest.approx(
        0.085 * 252.0 / 255.0
    )


def test_status_decode_matches_manual_byte_layout():
    status = make_driver()._decode_status_registers([0xF900, 0xABCD, 0xE510])

    assert status == RobotiqStatus(
        gACT=1,
        gGTO=1,
        gSTA=3,
        gOBJ=3,
        gFLT=0xB,
        kFLT=0xA,
        gPR=0xCD,
        gPO=0xE5,
        gCU=0x10,
    )
    assert status.fault_code == 0xAB
    assert not status.is_ready
    assert make_status().is_ready


@pytest.mark.parametrize(
    "registers",
    [
        None,
        object(),
        [],
        [0, 0],
        [0, 0, 0, 0],
        [True, 0, 0],
        [-1, 0, 0],
        [0x10000, 0, 0],
        [0x0001, 0, 0],
        [0x0200, 0, 0],
        [0x2000, 0, 0],
    ],
)
def test_malformed_status_registers_are_rejected(registers):
    assert make_driver()._decode_status_registers(registers) is None


def test_move_packs_raw_request_bytes():
    driver = make_driver()
    driver._client.write_registers.return_value = successful_write_response()

    driver.move(
        position_request=255,
        speed_request=127,
        force_request=42,
    )

    driver._client.write_registers.assert_called_once_with(
        address=robotiq_2f_gripper_refactored.ROBOTIQ_OUTPUT_REGISTER_ADDRESS,
        values=[0x0900, 0x00FF, 0x7F2A],
        unit=9,
    )


def test_read_status_uses_fc04_and_returns_decoded_status():
    driver = make_driver()
    response = MagicMock()
    response.isError.return_value = False
    response.registers = [0xF900, 0x0000, 0x0300]
    driver._client.read_input_registers.return_value = response

    assert driver.read_status() == make_status(gGTO=1, gOBJ=3, gPO=3)
    driver._client.read_input_registers.assert_called_once_with(
        address=robotiq_2f_gripper_refactored.ROBOTIQ_INPUT_REGISTER_ADDRESS,
        count=robotiq_2f_gripper_refactored.ROBOTIQ_REGISTER_COUNT,
        unit=9,
    )


def test_modbus_failures_use_one_descriptive_error_type():
    driver = make_driver()
    driver._client.read_input_registers.side_effect = OSError("serial lost")
    with pytest.raises(RobotiqError, match="FC04.*serial lost"):
        driver.read_status()

    driver = make_driver()
    response = MagicMock()
    response.isError.return_value = True
    driver._client.write_registers.return_value = response
    with pytest.raises(RobotiqError, match="FC16 write failed"):
        driver.move(0, 0, 0)


def test_unexpected_fc16_acknowledgement_raises_descriptive_error():
    driver = make_driver()
    response = successful_write_response()
    response.count = 2
    driver._client.write_registers.return_value = response

    with pytest.raises(RobotiqError, match="unexpected acknowledgement"):
        driver.move(0, 0, 0)


def test_cleanup_is_idempotent_and_rejects_later_io():
    driver = make_driver()

    driver.cleanup()
    driver.cleanup()

    driver._client.close.assert_called_once_with()
    driver._client.write_registers.assert_not_called()
    with pytest.raises(RobotiqError, match="closed"):
        driver.read_status()
    with pytest.raises(RobotiqError, match="closed"):
        driver.move(0, 0, 0)


def test_cleanup_reports_serial_close_failure_once():
    driver = make_driver()
    driver._client.close.side_effect = OSError("close failed")

    with pytest.raises(RobotiqError, match="close failed"):
        driver.cleanup()
    driver.cleanup()

    driver._client.close.assert_called_once_with()


def test_modbus_requests_are_spaced_by_at_least_five_milliseconds():
    driver = make_driver()

    with patch.object(
        robotiq_2f_gripper_refactored.time,
        "monotonic",
        side_effect=[0.0, 0.0, 0.001, 0.005],
    ), patch.object(robotiq_2f_gripper_refactored.time, "sleep") as sleep:
        driver._wait_for_request_interval()
        driver._wait_for_request_interval()

    sleep.assert_called_once_with(pytest.approx(0.004))


def test_activation_accepts_reset_then_ready_states():
    driver = make_driver()
    driver._write_command_registers = MagicMock()
    driver.read_status = MagicMock(
        side_effect=[
            make_status(gACT=0, gSTA=0),
            make_status(gACT=1, gSTA=1),
            make_status(),
        ]
    )

    with patch.object(robotiq_2f_gripper_refactored.time, "sleep"):
        driver._activate()

    assert driver._write_command_registers.call_args_list == [
        call(
            action_request=0x0000,
            position_request=0,
            speed_force_request=0,
        ),
        call(
            action_request=0x0100,
            position_request=0,
            speed_force_request=0,
        ),
    ]
    assert driver.read_status.call_count == 3


def test_activation_retries_transient_read_errors():
    driver = make_driver()
    driver._write_command_registers = MagicMock()
    driver.read_status = MagicMock(
        side_effect=[
            RobotiqError("temporary reset read"),
            make_status(gACT=0, gSTA=0),
            RobotiqError("temporary activation read"),
            make_status(),
        ]
    )

    with patch.object(robotiq_2f_gripper_refactored.time, "sleep"):
        driver._activate()

    assert driver.read_status.call_count == 4


@pytest.mark.parametrize(
    ("status_overrides", "message"),
    [
        ({"gFLT": 0x0B}, "activation fault 0x0B"),
        ({"kFLT": 0x02}, "controller fault 0x2"),
    ],
)
def test_activation_reports_hardware_fault(status_overrides, message):
    driver = make_driver()
    driver._write_command_registers = MagicMock()
    driver.read_status = MagicMock(
        side_effect=[
            make_status(gACT=0, gSTA=0),
            make_status(**status_overrides),
        ]
    )

    with patch.object(robotiq_2f_gripper_refactored.time, "sleep"):
        with pytest.raises(RobotiqError, match=message):
            driver._activate()


def test_activation_timeout_reports_last_read_error():
    driver = make_driver(
        response_timeout=0.005,
        activation_timeout=0.005,
        poll_interval=0.005,
    )
    driver._write_command_registers = MagicMock()
    driver.read_status = MagicMock(side_effect=RobotiqError("no response"))

    with pytest.raises(RobotiqError, match="no response"):
        driver._activate()


def test_activation_stage_timeout_reports_last_read_error():
    driver = make_driver(
        response_timeout=0.005,
        activation_timeout=0.005,
        poll_interval=0.005,
    )
    driver._write_command_registers = MagicMock()
    driver.read_status = MagicMock(
        side_effect=[
            make_status(gACT=0, gSTA=0),
            RobotiqError("activation no response"),
        ]
    )

    with pytest.raises(RobotiqError, match="activation no response"):
        driver._activate()


class FakeModbusClient:
    def __init__(self, connect_result=True):
        self.connect_result = connect_result
        self.connected = False
        self.closed = False

    def connect(self):
        self.connected = True
        return self.connect_result

    def close(self):
        self.closed = True

    def write_registers(self, address, values, device_id=None):
        raise AssertionError("constructor compatibility test must not write")


class FakeSlaveKeywordModbusClient(FakeModbusClient):
    def write_registers(self, address, values, slave=None):
        raise AssertionError("constructor compatibility test must not write")


@pytest.mark.parametrize(
    ("pymodbus_v2", "client_class", "expected_keyword"),
    [
        (True, FakeModbusClient, "unit"),
        (False, FakeModbusClient, "device_id"),
        (False, FakeSlaveKeywordModbusClient, "slave"),
    ],
)
def test_constructor_selects_supported_pymodbus_slave_keyword(
    pymodbus_v2,
    client_class,
    expected_keyword,
):
    client = client_class()

    with patch.object(
        robotiq_2f_gripper_refactored,
        "PYMODBUS_V2",
        pymodbus_v2,
    ), patch.object(
        robotiq_2f_gripper_refactored,
        "ModbusSerialClient",
        return_value=client,
    ) as serial_client_class, patch.object(
        Robotiq2FingerGripper,
        "_activate",
    ):
        driver = Robotiq2FingerGripper(**driver_values())

    assert driver._slave_kwarg == expected_keyword
    assert client.connected
    assert serial_client_class.call_args.kwargs["timeout"] == 1.0
    assert ("method" in serial_client_class.call_args.kwargs) is pymodbus_v2
    driver.cleanup()
    assert client.closed


def test_failed_connection_closes_serial_client():
    client = FakeModbusClient(connect_result=False)

    with patch.object(
        robotiq_2f_gripper_refactored,
        "PYMODBUS_V2",
        True,
    ), patch.object(
        robotiq_2f_gripper_refactored,
        "ModbusSerialClient",
        return_value=client,
    ):
        with pytest.raises(RobotiqError, match="Could not connect"):
            Robotiq2FingerGripper(**driver_values())

    assert client.closed


def test_activation_failure_during_construction_closes_serial_client():
    client = FakeModbusClient()

    with patch.object(
        robotiq_2f_gripper_refactored,
        "PYMODBUS_V2",
        True,
    ), patch.object(
        robotiq_2f_gripper_refactored,
        "ModbusSerialClient",
        return_value=client,
    ), patch.object(
        Robotiq2FingerGripper,
        "_activate",
        side_effect=RobotiqError("activation failed"),
    ):
        with pytest.raises(RobotiqError, match="activation failed"):
            Robotiq2FingerGripper(**driver_values())

    assert client.closed


def test_client_state_and_command_use_the_raw_status_and_motion_contract():
    adapter = make_adapter()
    adapter.gripper = MagicMock()
    adapter.gripper.read_status.return_value = make_status(
        gGTO=1,
        gOBJ=0,
        gPR=128,
        gPO=128,
        gCU=10,
    )
    adapter._last_valid_state = None
    adapter._previous_command_successful = False
    adapter._last_command_failed = False
    adapter._gripper_ready = False
    adapter._last_status_monotonic = None
    adapter._status_failure_logged = False

    state = adapter.get_gripper_state()
    assert state.is_moving
    assert state.error_code == 0
    assert state.width == pytest.approx(0.085 * 127.0 / 255.0)

    command = polymetis_pb2.GripperCommand(
        width=0.0,
        speed=0.085,
        force=235.0,
    )
    adapter.apply_gripper_command(command)
    adapter.gripper.move.assert_called_once_with(
        position_request=255,
        speed_request=127,
        force_request=255,
    )

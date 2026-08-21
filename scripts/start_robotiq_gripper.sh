#!/usr/bin/env bash

# Launch one Robotiq hardware client and its Polymetis gRPC server.
set -euo pipefail

usage() {
    cat <<EOF
Usage: $(basename "$0") --serial-port DEVICE --server-port PORT [OPTIONS]

Required:
  -s, --serial-port DEVICE  Stable /dev/serial/by-id/... path for the gripper.
  -p, --server-port PORT    gRPC port exposed to GripperInterface.

Options:
  -i, --server-ip IP        Address used by the server (default: localhost).
      --hz HZ               Hardware-client target rate (default: 100).
  -c, --conda ENV           Conda environment (default: polymetis).
  -h, --help                Show this message.

Examples:
  # P1 endpoint on the application PC
  $(basename "$0") --server-port 1235 --serial-port /dev/serial/by-id/<P1-ROBOTIQ-ADAPTER>

  # P2 endpoint on the application PC
  $(basename "$0") --server-port 4322 --serial-port /dev/serial/by-id/<P2-ROBOTIQ-ADAPTER>
EOF
}

server_ip="localhost"
server_port=""
serial_port=""
control_hz="100"
conda_env="polymetis"

require_value() {
    if [[ $# -lt 2 ]]; then
        echo "$1 requires a value." >&2
        exit 2
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--serial-port)
            require_value "$@"
            serial_port="$2"
            shift 2
            ;;
        -p|--server-port)
            require_value "$@"
            server_port="$2"
            shift 2
            ;;
        -i|--server-ip)
            require_value "$@"
            server_ip="$2"
            shift 2
            ;;
        --hz)
            require_value "$@"
            control_hz="$2"
            shift 2
            ;;
        -c|--conda)
            require_value "$@"
            conda_env="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "$serial_port" || -z "$server_port" ]]; then
    echo "--serial-port and --server-port are required." >&2
    usage >&2
    exit 2
fi

if [[ ! "$server_port" =~ ^[0-9]+$ ]] \
    || (( server_port < 1 || server_port > 65535 )); then
    echo "--server-port must be an integer between 1 and 65535." >&2
    exit 2
fi

if [[ ! "$control_hz" =~ ^[0-9]+$ ]] \
    || (( control_hz < 1 || control_hz > 200 )); then
    echo "--hz must be an integer between 1 and 200." >&2
    exit 2
fi

if [[ "${CONDA_DEFAULT_ENV:-}" != "$conda_env" ]]; then
    if command -v mamba >/dev/null 2>&1; then
        eval "$(mamba shell hook --shell bash)"
        mamba activate "$conda_env"
    elif command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
        conda activate "$conda_env"
    else
        echo "Cannot activate '$conda_env': mamba and conda are unavailable." >&2
        exit 1
    fi
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
launcher="$repo_root/polymetis/python/scripts/launch_gripper.py"

if [[ ! -f "$launcher" ]]; then
    echo "Could not find launch_gripper.py at $launcher" >&2
    exit 1
fi

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "Launching Robotiq gripper:"
echo "  serial device: $serial_port"
echo "  gRPC endpoint: $server_ip:$server_port"
echo "  target rate:   $control_hz Hz"

exec python "$launcher" \
    gripper=robotiq_2f \
    ip="$server_ip" \
    port="$server_port" \
    gripper.port="$serial_port" \
    gripper.hz="$control_hz"

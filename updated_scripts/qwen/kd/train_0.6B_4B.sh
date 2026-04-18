#! /bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KD_TYPE=fkl UPDATED_METHOD=kd bash "${SCRIPT_DIR}/../run_0.6B_4B_updated.sh" "$@"

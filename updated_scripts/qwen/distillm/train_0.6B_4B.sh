#! /bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPDATED_METHOD=distillm bash "${SCRIPT_DIR}/../run_0.6B_4B_updated.sh" "$@"

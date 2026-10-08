#!/bin/sh
# Preserve WDL executors' shell commands while keeping the direct CLI convenient.
set -eu
if [ "$#" -eq 0 ]; then
  set -- --help
fi
case "$1" in
  -*) exec python -m tooling.cli.workbench "$@" ;;
  *) exec "$@" ;;
esac

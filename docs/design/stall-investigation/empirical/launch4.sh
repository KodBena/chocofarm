#!/bin/bash
export PYTHONFAULTHANDLER=1
exec /home/bork/w/vdc/venvs/generic/bin/python "/home/bork/w/vdc/chocobo/runs/stall-diag/standalone_server2.py" --endpoint "ipc:///tmp/choco-stall4-1781848958.ipc" --run stall-diag4 --version 0 --server-core 0

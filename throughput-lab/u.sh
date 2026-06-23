PYTHONPATH=throughput-lab /home/bork/w/vdc/venvs/generic/bin/python harness/run_lab.py

# a load-bearing sweep: more threads/rate, replicates (achieved-rate median), a persisted artifact:
PYTHONPATH=throughput-lab /home/bork/w/vdc/venvs/generic/bin/python harness/run_lab.py \
    --threads 4 --rate 5000 --seconds 5 --replicates 3 \
    --json-out ~/w/vdc/chocobo/runs/tlab/sweep.json

# a single cell (subset the matrix):
PYTHONPATH=throughput-lab /home/bork/w/vdc/venvs/generic/bin/python harness/run_lab.py \
    --topologies per-thread --modes coupled


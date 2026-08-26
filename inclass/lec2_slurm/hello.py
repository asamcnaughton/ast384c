import datetime
import os
import socket

import numpy as np

job_id = os.environ.get("SLURM_JOB_ID", "local")
hostname = socket.gethostname()
now = datetime.datetime.now()

x = np.random.default_rng(seed=0).normal(size=1_000_000)
mean = x.mean()

lines = [
    f"Hello from {hostname}",
    f"Job ID: {job_id}",
    f"Time: {now}",
    f"NumPy version: {np.__version__}",
    f"Mean of 1e6 random draws: {mean:.5f}",
]

for line in lines:
    print(line)

os.makedirs("results", exist_ok=True)
outfile = f"results/run_{job_id}.txt"
with open(outfile, "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"Wrote {outfile}")

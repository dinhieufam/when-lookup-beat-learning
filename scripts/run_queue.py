"""Run experiments.train_bundle jobs, one job per GPU, on at most --max-gpus GPUs.

The machine is shared: a GPU is used only if no other process holds memory on it
(memory.used below --idle-mb) when a job is launched. Finished runs (summary.json
present) are skipped, so the queue can be restarted at any time.

Grid mode (protocol x dataset x model x seed, plus --extra arguments for every job):

    python scripts/run_queue.py --protocols orig --datasets MealRec+H MealRec+L \
        --models LightGCN BGCN CrossCBR MultiCBR HyperMBR --seeds 0 1 2 3 4

Job-file mode (one line of train_bundle arguments per job; blank lines and # comments skipped):

    python scripts/run_queue.py --jobs-file results/tune_jobs.txt

To take over from a stopped queue whose jobs are still running, pass their PIDs with
--wait-pids: those jobs are not relaunched and count toward --max-gpus until they exit.
"""
from __future__ import annotations

import argparse
import itertools
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.train_bundle import out_dir, parse  # noqa: E402


def idle_gpus(idle_mb: int) -> list[int]:
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return [int(i) for i, mem in (line.split(",") for line in out.strip().splitlines()) if int(mem) < idle_mb]


def alive(pid: int) -> bool:
    status = Path(f"/proc/{pid}/status")
    try:
        return "zombie" not in status.read_text().lower()
    except FileNotFoundError:
        return False


def job_of(pid: int) -> list[str]:
    """train_bundle arguments of a running job, from its command line."""
    argv = Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")
    return [x for x in argv[argv.index("experiments.train_bundle") + 1:] if x]


def grid_jobs(a) -> list[list[str]]:
    # Protocol by protocol (in the order given); within one, slow models first so the tail is short jobs.
    combos = sorted(itertools.product(a.protocols, a.datasets, a.models, a.seeds),
                    key=lambda j: (a.protocols.index(j[0]), j[2] != "HyperMBR"))
    return [["--protocol", p, "--dataset", d, "--model", m, "--seed", str(s), *shlex.split(a.extra)]
            for p, d, m, s in combos]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--jobs-file", type=Path, help="one line of train_bundle arguments per job")
    p.add_argument("--protocols", nargs="+", default=["orig"])
    p.add_argument("--datasets", nargs="+", default=["MealRec+H", "MealRec+L"])
    p.add_argument("--models", nargs="+")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--max-gpus", type=int, default=3)
    p.add_argument("--idle-mb", type=int, default=500)
    p.add_argument("--extra", default="", help="extra arguments passed to every train_bundle job (grid mode)")
    p.add_argument("--wait-pids", nargs="*", type=int, default=[],
                   help="running train_bundle jobs from a stopped queue: skipped, and counted toward --max-gpus")
    a = p.parse_args()

    if a.jobs_file:
        lines = (line.split("#")[0].strip() for line in a.jobs_file.read_text().splitlines())
        jobs = [shlex.split(line) for line in lines if line]
    elif a.models:
        jobs = grid_jobs(a)
    else:
        p.error("give --jobs-file or --models")
    taken = {out_dir(parse(job_of(pid))) for pid in a.wait_pids if alive(pid)}
    jobs = [j for j in jobs if not (out_dir(parse(j)) / "summary.json").exists() and out_dir(parse(j)) not in taken]
    print(f"{len(jobs)} jobs to run; waiting on {len(taken)} running jobs", flush=True)

    running: dict[int, tuple[subprocess.Popen, str]] = {}
    failed = []
    while jobs or running or any(alive(pid) for pid in a.wait_pids):
        for gpu, (proc, name) in list(running.items()):
            if proc.poll() is not None:
                del running[gpu]
                status = "ok" if proc.returncode == 0 else f"FAILED ({proc.returncode})"
                if proc.returncode != 0:
                    failed.append(name)
                print(f"{time.strftime('%H:%M:%S')} {status}: {name} on GPU {gpu}; {len(jobs)} queued", flush=True)
        external = sum(alive(pid) for pid in a.wait_pids)
        if jobs and len(running) + external < a.max_gpus:
            for gpu in idle_gpus(a.idle_mb):
                if gpu in running or len(running) + external >= a.max_gpus or not jobs:
                    continue
                job = jobs.pop(0)
                out = out_dir(parse(job))
                out.mkdir(parents=True, exist_ok=True)
                name = str(out.relative_to(ROOT / "results"))
                cmd = [sys.executable, "-W", "ignore", "-m", "experiments.train_bundle", *job]
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
                log = open(out / "train.log", "w")
                running[gpu] = (subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT), name)
                print(f"{time.strftime('%H:%M:%S')} start: {name} on GPU {gpu}", flush=True)
                time.sleep(20)  # let the job allocate memory before the next idle check
        time.sleep(10)
    print(f"all done; {len(failed)} failed: {failed}", flush=True)


if __name__ == "__main__":
    main()

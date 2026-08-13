"""
RUN ALL GENERATOR MODELS - train -> eval -> plot, unattended.

Runs the full generator pipeline for every model in MODELS, in order:
    1. QLoRA fine-tune   (skipped if checkpoints/qlora-<tag> already exists)
    2. Generation eval   (skipped if results/generation_<tag>.json already exists)
Then regenerates all loss + generation figures once at the end.

Why a runner instead of one big script: each stage runs as a SEPARATE process, so
GPU memory is fully released between models. Training a 14B right after a 7B in the
same process would fragment VRAM and risk OOM on a 12 GB card.

Resumable: already-completed stages are detected and skipped, so if a run dies
overnight you just start it again and it picks up where it left off.

Usage:
    python run_all_models.py              # run everything that's missing
    python run_all_models.py --dry-run    # just show what it would do
"""
from __future__ import annotations
import subprocess, sys, time
from pathlib import Path

MODELS = [
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
]

CKPT = Path("checkpoints"); RESULTS = Path("results"); LOGS = Path("logs")
LOGS.mkdir(exist_ok=True)
DRY = "--dry-run" in sys.argv


def tag_of(model: str) -> str:
    return model.split("/")[-1].lower().replace(".", "-")


def run(cmd: list[str], logfile: Path) -> bool:
    """Run a stage as its own process; stream output to console AND a log file."""
    print(f"\n$ {' '.join(cmd)}")
    if DRY:
        return True
    t0 = time.time()
    with open(logfile, "w", encoding="utf-8") as lf:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace", bufsize=1)
        for line in p.stdout:
            print(line, end="")
            lf.write(line)
        p.wait()
    mins = (time.time() - t0) / 60
    ok = p.returncode == 0
    print(f"[{'OK' if ok else 'FAILED'}] {mins:.1f} min  (log: {logfile})")
    return ok


def main():
    py = sys.executable
    plan, done = [], []
    for m in MODELS:
        t = tag_of(m)
        need_train = not (CKPT / f"qlora-{t}").exists()
        need_eval = not (RESULTS / f"generation_{t}.json").exists()
        if need_train or need_eval:
            plan.append((m, need_train, need_eval))
        else:
            done.append(m)

    print("=" * 60)
    if done:
        print("Already complete (skipping):")
        for m in done:
            print(f"  - {m}")
    if not plan:
        print("Nothing to do. All models trained and evaluated.")
    else:
        print("\nWill run:")
        for m, tr, ev in plan:
            steps = " + ".join([s for s, f in (("train", tr), ("eval", ev)) if f])
            print(f"  - {m}  [{steps}]")
    print("=" * 60)

    failures = []
    for m, need_train, need_eval in plan:
        t = tag_of(m)
        print(f"\n{'#' * 60}\n# {m}\n{'#' * 60}")
        if need_train:
            if not run([py, "-u", "src/generate/finetune_qlora.py", m],
                       LOGS / f"train_{t}.log"):
                failures.append((m, "train")); continue      # skip eval if train failed
        else:
            print(f"[skip] adapter exists: checkpoints/qlora-{t}")
        if need_eval:
            if not run([py, "-u", "src/generate/run_generation_eval.py", m],
                       LOGS / f"eval_{t}.log"):
                failures.append((m, "eval"))
        else:
            print(f"[skip] results exist: results/generation_{t}.json")

    # regenerate every figure once, at the end
    if not DRY:
        print(f"\n{'#' * 60}\n# FIGURES\n{'#' * 60}")
        for script in ("results/plot_loss.py", "results/plot_generation.py"):
            if Path(script).exists():
                run([py, "-u", script], LOGS / f"{Path(script).stem}.log")

    print("\n" + "=" * 60)
    if failures:
        print("FAILED stages (rerun this script to retry them):")
        for m, s in failures:
            print(f"  - {m} [{s}]  see logs/{s}_{tag_of(m)}.log")
    else:
        print("ALL DONE.")
    print("=" * 60)


if __name__ == "__main__":
    main()
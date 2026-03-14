#!/usr/bin/env python3
"""
Benchmark HOSVD variants on MOSEv2 val.
Usage:
    srun python benchmark_mose.py
    srun python benchmark_mose.py --ranks 48 64 --skip_hosvd
    srun python benchmark_mose.py --only_baseline
    srun python benchmark_mose.py --no_clean --skip_baseline --ranks 48
"""

import argparse
import json
import logging
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════
#  Config
# ══════════════════════════════════════════════════════════════════════════

SAM2_CFG   = "configs/sam2.1/sam2.1_hiera_b+.yaml"
SAM2_CKPT  = "./checkpoints/sam2.1_hiera_base_plus.pt"

# MOSEv2 data paths
VIDEO_DIR  = "/home/ids/lnguyen-23/haianh/data/MOSEv2/valid/JPEGImages"
MASK_DIR   = "/home/ids/lnguyen-23/haianh/data/MOSEv2/valid/Annotations"
VIDEO_LIST = "/home/ids/lnguyen-23/haianh/data/MOSEv2/valid/val.txt"

# Evaluator — reuse davis2017 J&F toolkit
EVAL_CWD    = Path("/home/ids/lnguyen-23/haianh/evaluation/davis2017-evaluation")
EVAL_SCRIPT = str(EVAL_CWD / "evaluation_method.py")

# Output
RESULTS_BASE = Path("/home/ids/lnguyen-23/haianh/evaluation/mose/results")
LOG_FILE     = Path("/home/ids/lnguyen-23/haianh/evaluation/mose/benchmark_results.log")
RESULTS_JSON = Path("/home/ids/lnguyen-23/haianh/evaluation/mose/benchmark_results.json")

DEFAULT_SI_RANKS = [16, 32, 48, 64]

# ══════════════════════════════════════════════════════════════════════════
#  Logging
# ══════════════════════════════════════════════════════════════════════════

LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="w"),
    ],
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════

def _stream(proc: subprocess.Popen) -> list[str]:
    """Stream stdout live, filter tqdm noise, return captured lines."""
    lines = []
    for line in proc.stdout:
        line = line.rstrip()
        if "\r" in line or "it/s]" in line or "██" in line:
            continue
        if line:
            log.info(f"    {line}")
            lines.append(line)
    proc.wait()
    return lines


def run_inference(results_dir: Path, mode: str, extra_args: list[str]) -> bool:
    cmd = [
        "python3", "tools/vos_inference.py",
        "--sam2_cfg",        SAM2_CFG,
        "--sam2_checkpoint", SAM2_CKPT,
        "--base_video_dir",  VIDEO_DIR,
        "--input_mask_dir",  MASK_DIR,
        "--video_list_file", VIDEO_LIST,
        "--output_mask_dir", str(results_dir),
        "--mode",            mode,
        *extra_args,
    ]
    log.info(f"  CMD: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    _stream(proc)
    if proc.returncode != 0:
        log.error(f"  ✗ inference failed (exit {proc.returncode})")
        return False
    return True


def run_eval(results_dir: Path) -> tuple[bool, str]:
    """
    cd EVAL_CWD
    python evaluation_method.py
        --task semi-supervised
        --results_path {results_dir}
    """
    cmd = [
        "python3", EVAL_SCRIPT,
        "--task",         "semi-supervised",
        "--results_path", str(results_dir),
    ]
    log.info(f"  CMD (cwd={EVAL_CWD}): {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(EVAL_CWD),
    )
    lines = _stream(proc)
    if proc.returncode != 0:
        log.error(f"  ✗ evaluation failed (exit {proc.returncode})")
        return False, ""
    return True, "\n".join(lines)


def parse_scores(output: str) -> dict | None:
    """
    Parse J&F / J / F from davis2017 evaluator output.

    Handles multiple output formats:
      - "Global results: J&F-Mean: 75.32  J-Mean: 72.10  F-Mean: 78.54"
      - DataFrame row:  "Global     75.32   72.10   78.54"
      - Short form:     "J&F: 75.3  J: 72.1  F: 78.5"
    """
    # Pattern 1 — verbose with -Mean suffix
    m = re.search(
        r"J&F[- ]Mean[:\s]+([0-9.]+).*?J[- ]Mean[:\s]+([0-9.]+).*?F[- ]Mean[:\s]+([0-9.]+)",
        output, re.IGNORECASE | re.DOTALL,
    )
    if m:
        return {"jf": float(m.group(1)), "j": float(m.group(2)), "f": float(m.group(3))}

    # Pattern 2 — DataFrame "Global  75.32  72.10  78.54"
    m = re.search(r"Global\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)", output)
    if m:
        return {"jf": float(m.group(1)), "j": float(m.group(2)), "f": float(m.group(3))}

    # Pattern 3 — short form
    m = re.search(
        r"J&F[:\s]+([0-9.]+).*?J[:\s]+([0-9.]+).*?F[:\s]+([0-9.]+)",
        output,
    )
    if m:
        return {"jf": float(m.group(1)), "j": float(m.group(2)), "f": float(m.group(3))}

    log.warning("  ⚠ Could not parse scores — check log for raw evaluator output")
    return None


def save_result(record: dict):
    """Upsert record into RESULTS_JSON (overwrite same label if exists)."""
    if RESULTS_JSON.exists():
        data = json.loads(RESULTS_JSON.read_text())
    else:
        data = {"dataset": "mose_val", "runs": []}

    data["runs"] = [r for r in data["runs"] if r["label"] != record["label"]]
    data["runs"].append(record)

    RESULTS_JSON.write_text(json.dumps(data, indent=2))
    log.info(f"  💾 Saved → {RESULTS_JSON}")


def print_summary(data: dict):
    runs = data.get("runs", [])
    if not runs:
        return
    log.info("")
    log.info("┌──────────────────────────────────────────────────┐")
    log.info("│            MOSEv2 val — Results Summary           │")
    log.info("├─────────────────────────────────────┬────┬────┬────┤")
    log.info("│ Label                               │ J&F│  J │  F │")
    log.info("├─────────────────────────────────────┼────┼────┼────┤")
    for r in sorted(runs, key=lambda x: x.get("jf", 0), reverse=True):
        label   = r["label"][:35].ljust(35)
        jf      = f"{r['jf']:.1f}".rjust(4) if r.get("jf") is not None else "  - "
        j       = f"{r['j']:.1f}".rjust(4)  if r.get("j")  is not None else "  - "
        f_      = f"{r['f']:.1f}".rjust(4)  if r.get("f")  is not None else "  - "
        elapsed = f"  [{r['elapsed_s']:.0f}s]" if r.get("elapsed_s") else ""
        log.info(f"│ {label} │{jf}│{j} │{f_} │{elapsed}")
    log.info("└─────────────────────────────────────┴────┴────┴────┘")


def run_one(
    n: int,
    total: int,
    label: str,
    mode: str,
    extra_args: list[str],
    rank: int | None = None,
):
    log.info("")
    log.info(f"[{n}/{total}] {label}")

    results_dir = RESULTS_BASE / label
    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True)

    # ── inference ────────────────────────────────────────────────────────
    log.info("  → inference ...")
    t0 = time.time()
    ok = run_inference(results_dir, mode, extra_args)
    elapsed = time.time() - t0
    if not ok:
        return

    # ── evaluation ───────────────────────────────────────────────────────
    log.info("  → evaluating ...")
    ok, eval_output = run_eval(results_dir)
    if not ok:
        return

    # ── parse & store ─────────────────────────────────────────────────────
    scores = parse_scores(eval_output)
    record = {
        "label":     label,
        "mode":      mode,
        "rank":      rank,
        "elapsed_s": round(elapsed, 1),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **(scores if scores else {"jf": None, "j": None, "f": None}),
    }
    save_result(record)

    if scores:
        log.info(
            f"  ✓ J&F={scores['jf']:.1f}  J={scores['j']:.1f}  "
            f"F={scores['f']:.1f}  [{elapsed:.0f}s]"
        )
    else:
        log.info("  ✓ done  (scores not parsed — check log)")


# ══════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Benchmark HOSVD variants on MOSEv2 val")
    p.add_argument("--ranks", nargs="+", type=int, default=DEFAULT_SI_RANKS,
                   help="Ranks for hosvd_subspace_iteration (default: 16 32 48 64)")
    p.add_argument("--skip_baseline", action="store_true",
                   help="Skip baseline run")
    p.add_argument("--skip_hosvd", action="store_true",
                   help="Skip classic full-SVD HOSVD run")
    p.add_argument("--skip_si", action="store_true",
                   help="Skip all subspace iteration runs")
    p.add_argument("--only_baseline", action="store_true",
                   help="Run baseline only")
    p.add_argument("--var_threshold", type=float, default=0.99,
                   help="Variance threshold for classic HOSVD (default: 0.99)")
    p.add_argument("--no_clean", action="store_true",
                   help="Do not wipe RESULTS_BASE before starting")
    return p.parse_args()


def main():
    args = parse_args()

    if args.only_baseline:
        args.skip_hosvd = True
        args.skip_si    = True

    # ── build run list ───────────────────────────────────────────────────
    # (label, mode, extra_args, rank_or_None)
    runs: list[tuple[str, str, list[str], int | None]] = []

    if not args.skip_baseline:
        runs.append(("baseline", "baseline", [], None))

    if not args.skip_hosvd:
        runs.append((
            f"hosvd_var{args.var_threshold}",
            "hosvd",
            ["--var_threshold", str(args.var_threshold)],
            None,
        ))

    if not args.skip_si:
        for rank in args.ranks:
            runs.append((
                f"hosvd_subspace_iteration_rank{rank}",
                "hosvd_subspace_iteration",
                ["--rank", str(rank)],
                rank,
            ))

    total = len(runs)

    # ── clean results dir ────────────────────────────────────────────────
    if not args.no_clean and RESULTS_BASE.exists():
        log.info(f"🧹 Cleaning old results: {RESULTS_BASE}")
        shutil.rmtree(RESULTS_BASE)
    RESULTS_BASE.mkdir(parents=True, exist_ok=True)

    # ── header ───────────────────────────────────────────────────────────
    log.info("════════════════════════════════════════")
    log.info(f"  HOSVD Benchmark — MOSEv2 val")
    log.info(f"  Date       : {datetime.now():%Y-%m-%d %H:%M}")
    log.info(f"  Total runs : {total}")
    log.info(f"  Ranks (SI) : {args.ranks}")
    log.info(f"  Results    : {RESULTS_BASE}")
    log.info(f"  JSON       : {RESULTS_JSON}")
    log.info(f"  Log        : {LOG_FILE}")
    log.info("════════════════════════════════════════")

    # ── execute ──────────────────────────────────────────────────────────
    for i, (label, mode, extra, rank) in enumerate(runs, 1):
        run_one(i, total, label, mode, extra, rank=rank)

    # ── final summary ────────────────────────────────────────────────────
    if RESULTS_JSON.exists():
        print_summary(json.loads(RESULTS_JSON.read_text()))

    # ── footer ───────────────────────────────────────────────────────────
    log.info("")
    log.info("════════════════════════════════════════")
    log.info(f"  All {total} runs complete")
    log.info(f"  Results JSON → {RESULTS_JSON}")
    log.info(f"  Full log     → {LOG_FILE}")
    log.info("════════════════════════════════════════")


if __name__ == "__main__":
    main()

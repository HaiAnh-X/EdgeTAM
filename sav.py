#!/usr/bin/env python3
"""
Benchmark HOSVD variants on SA-V val.
Usage:
    srun python benchmark_sav.py
    srun python benchmark_sav.py --ranks 48 64 --skip_hosvd
    srun python benchmark_sav.py --only_baseline
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

SAM2_CFG      = "configs/sam2.1/sam2.1_hiera_b+.yaml"
SAM2_CKPT     = "./checkpoints/sam2.1_hiera_base_plus.pt"
VIDEO_DIR     = "/home/ids/lnguyen-23/haianh/data/SA-V/sav_val/JPEGImages_24fps"
MASK_DIR      = "/home/ids/lnguyen-23/haianh/data/SA-V/sav_val/Annotations_6fps"
VIDEO_LIST    = "/home/ids/lnguyen-23/haianh/data/SA-V/sav_val/sav_val.txt"
EVAL_CWD      = Path("/home/ids/lnguyen-23/haianh/evaluation/sav_dataset")
EVAL_SCRIPT   = str(EVAL_CWD / "sav_evaluator.py")
GT_MASK_DIR   = "/home/ids/lnguyen-23/haianh/data/SA-V/sav_val/Annotations_6fps"
RESULTS_BASE  = Path("/home/ids/lnguyen-23/haianh/evaluation/sav_dataset/results")
LOG_FILE      = Path("/home/ids/lnguyen-23/haianh/evaluation/sav_dataset/benchmark_results.log")

DEFAULT_SI_RANKS = [16, 32, 48, 64]

# JSON file that accumulates results across runs (for plotting later)
RESULTS_JSON = Path("/home/ids/lnguyen-23/haianh/evaluation/sav_dataset/benchmark_results.json")

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

def run(cmd: list[str], label: str) -> int:
    """Run a subprocess, stream output live, return exit code."""
    log.info(f"  CMD: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in proc.stdout:
        line = line.rstrip()
        # filter tqdm progress bars
        if "\r" in line or "it/s]" in line or "██" in line:
            continue
        if line:
            log.info(f"    {line}")
    proc.wait()
    return proc.returncode


def run_inference(results_dir: Path, mode: str, extra_args: list[str]) -> bool:
    cmd = [
        "python3", "tools/vos_inference.py",
        "--sam2_cfg",        SAM2_CFG,
        "--sam2_checkpoint", SAM2_CKPT,
        "--base_video_dir",  VIDEO_DIR,
        "--input_mask_dir",  MASK_DIR,
        "--video_list_file", VIDEO_LIST,
        "--output_mask_dir", str(results_dir),
        "--per_obj_png_file",
        "--mode",            mode,
        *extra_args,
    ]
    rc = run(cmd, mode)
    if rc != 0:
        log.error(f"  ✗ inference failed (exit {rc})")
        return False
    return True





def run_eval(results_dir: Path) -> tuple[bool, str]:
    """Run evaluator, return (success, raw_output)."""
    cmd = [
        "python3", EVAL_SCRIPT,
        "--gt_root",   GT_MASK_DIR,
        "--pred_root", str(results_dir),
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
    output_lines = []
    for line in proc.stdout:
        line = line.rstrip()
        if "\r" in line or "it/s]" in line or "██" in line:
            continue
        if line:
            log.info(f"    {line}")
            output_lines.append(line)
    proc.wait()
    if proc.returncode != 0:
        log.error(f"  ✗ evaluation failed (exit {proc.returncode})")
        return False, ""
    return True, "\n".join(output_lines)


def parse_scores(output: str) -> dict | None:
    """
    Extract J&F, J, F from evaluator output.
    Looks for lines like:
        Global score: J&F: 77.7 J: 74.3 F: 81.2
    or the SAV evaluator table format.
    """
    # Pattern 1: single summary line
    m = re.search(
        r"J&F[:\s]+([0-9.]+).*?J[:\s]+([0-9.]+).*?F[:\s]+([0-9.]+)",
        output,
    )
    if m:
        return {
            "jf": float(m.group(1)),
            "j":  float(m.group(2)),
            "f":  float(m.group(3)),
        }

    # Pattern 2: separate lines
    jf = re.search(r"J&F.*?([0-9]+\.[0-9]+)", output)
    j  = re.search(r"\bJ\b.*?([0-9]+\.[0-9]+)", output)
    f  = re.search(r"\bF\b.*?([0-9]+\.[0-9]+)", output)
    if jf and j and f:
        return {
            "jf": float(jf.group(1)),
            "j":  float(j.group(1)),
            "f":  float(f.group(1)),
        }

    log.warning("  ⚠ Could not parse scores from evaluator output")
    return None


def save_result(record: dict):
    """
    Append one result record to RESULTS_JSON.
    File structure:
    {
      "dataset": "sav_val",
      "runs": [
        {
          "label":     "baseline",
          "mode":      "baseline",
          "rank":      null,
          "jf":        77.7,
          "j":         74.3,
          "f":         81.2,
          "elapsed_s": 412.3,
          "timestamp": "2026-03-13T14:22:00"
        },
        ...
      ]
    }
    """
    if RESULTS_JSON.exists():
        data = json.loads(RESULTS_JSON.read_text())
    else:
        data = {"dataset": "sav_val", "runs": []}

    # overwrite entry with same label if it already exists
    data["runs"] = [r for r in data["runs"] if r["label"] != record["label"]]
    data["runs"].append(record)

    RESULTS_JSON.write_text(json.dumps(data, indent=2))
    log.info(f"  💾 Saved → {RESULTS_JSON}")


def print_summary(data: dict):
    """Print a nice ASCII table of all accumulated results."""
    runs = data.get("runs", [])
    if not runs:
        return

    log.info("")
    log.info("┌─────────────────────────────────────────────────┐")
    log.info("│              SA-V val — Results Summary          │")
    log.info("├────────────────────────────────────┬────┬────┬────┤")
    log.info("│ Label                              │ J&F│  J │  F │")
    log.info("├────────────────────────────────────┼────┼────┼────┤")
    for r in sorted(runs, key=lambda x: x.get("jf", 0), reverse=True):
        label = r["label"][:34].ljust(34)
        jf    = f"{r['jf']:.1f}".rjust(4) if r.get("jf") is not None else "  - "
        j     = f"{r['j']:.1f}".rjust(4)  if r.get("j")  is not None else "  - "
        f_    = f"{r['f']:.1f}".rjust(4)  if r.get("f")  is not None else "  - "
        elapsed = f"  [{r['elapsed_s']:.0f}s]" if r.get("elapsed_s") else ""
        log.info(f"│ {label} │{jf}│{j} │{f_} │{elapsed}")
    log.info("└────────────────────────────────────┴────┴────┴────┘")


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

    # ── parse & store ────────────────────────────────────────────────────
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
        log.info(f"  ✓ done (scores not parsed — check log)")


# ══════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Benchmark HOSVD variants on SA-V val")
    p.add_argument("--ranks", nargs="+", type=int, default=DEFAULT_SI_RANKS,
                   help="Ranks to test for hosvd_subspace_iteration (default: 16 32 48 64)")
    p.add_argument("--skip_baseline", action="store_true",
                   help="Skip the baseline run")
    p.add_argument("--skip_hosvd", action="store_true",
                   help="Skip the classic HOSVD (full SVD, var=0.99) run")
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
        args.skip_si = True

    # ── build run list ───────────────────────────────────────────────────
    # each entry: (label, mode, extra_args, rank_or_None)
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
    log.info(f"  HOSVD Benchmark — {datetime.now():%Y-%m-%d %H:%M}")
    log.info(f"  Total runs : {total}")
    log.info(f"  Ranks (SI) : {args.ranks}")
    log.info(f"  Log file   : {LOG_FILE}")
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

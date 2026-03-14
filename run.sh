#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════
#  Benchmark HOSVD variants on DAVIS-2017 val
#  baseline × 1 | hosvd × 1 | hosvd_subspace_iteration × 4 ranks
#  Total: 6 runs
# ══════════════════════════════════════════════════════════════════════════

set -e

SAM2_CFG="configs/sam2.1/sam2.1_hiera_b+.yaml"
SAM2_CKPT="./checkpoints/sam2.1_hiera_base_plus.pt"
VIDEO_DIR="/home/ids/lnguyen-23/haianh/data/DAVIS/JPEGImages/480p"
MASK_DIR="/home/ids/lnguyen-23/haianh/data/DAVIS/Annotations/480p"
VIDEO_LIST="/home/ids/lnguyen-23/haianh/data/DAVIS/ImageSets/2017/val.txt"
EVAL_SCRIPT="/home/ids/lnguyen-23/haianh/evaluation/davis2017-evaluation/evaluation_method.py"
LOG_FILE="/home/ids/lnguyen-23/haianh/evaluation/davis2017-evaluation/benchmark_results.log"

# Thư mục gốc chứa các kết quả benchmark
RESULTS_BASE_DIR="/home/ids/lnguyen-23/haianh/evaluation/davis2017-evaluation/results"

SI_RANKS=(16 32 48 64)
TOTAL=$(( 2 + ${#SI_RANKS[@]} ))   # baseline + hosvd + 4 ranks = 6

filter_tqdm() {
    grep -v $'\r' | grep -v "it/s]" | grep -v "|██"
}

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG_FILE; }

run_one() {
    local n="$1" label="$2" mode="$3" extra_args="$4"
    local results_dir="$RESULTS_BASE_DIR/$label"

    log ""
    log "[$n/$TOTAL] $label"
    log "  → inference ..."
    
    # Đảm bảo thư mục con sạch sẽ và tồn tại
    rm -rf "$results_dir"
    mkdir -p "$results_dir"

    TQDM_DISABLE=1 python3 tools/vos_inference.py \
        --sam2_cfg        "$SAM2_CFG" \
        --sam2_checkpoint "$SAM2_CKPT" \
        --base_video_dir  "$VIDEO_DIR" \
        --input_mask_dir  "$MASK_DIR" \
        --video_list_file "$VIDEO_LIST" \
        --output_mask_dir "$results_dir" \
        --mode "$mode" $extra_args \
        2>&1 | filter_tqdm | tee -a $LOG_FILE

    log "  → evaluating ..."
    TQDM_DISABLE=1 python3 "$EVAL_SCRIPT" \
        --task semi-supervised \
        --results_path "$results_dir" \
        2>&1 | filter_tqdm | tee -a $LOG_FILE

    log "  ✓ done"
}

# ── Header & Cleanup ───────────────────────────────────────────────────────
echo "" > $LOG_FILE
log "════════════════════════════════════════"
log "  HOSVD Benchmark — $(date +%Y-%m-%d)"
log "  Total runs: $TOTAL"
log "════════════════════════════════════════"

# Xóa toàn bộ thư mục results cũ để đảm bảo dữ liệu mới hoàn toàn
if [ -d "$RESULTS_BASE_DIR" ]; then
    log "🧹 Cleaning up old results directory: $RESULTS_BASE_DIR"
    rm -rf "$RESULTS_BASE_DIR"
fi
mkdir -p "$RESULTS_BASE_DIR"

# ── 1. Baseline ────────────────────────────────────────────────────────────
run_one 1 "baseline" "baseline" ""

# ── 2. HOSVD classic (var=0.99) ────────────────────────────────────────────
run_one 2 "hosvd_var0.99" "hosvd" "--var_threshold 0.99"

# ── 3-6. HOSVD Subspace Iteration × ranks ─────────────────────────────────
N=3
for RANK in "${SI_RANKS[@]}"; do
    run_one $N "hosvd_subspace_iteration_rank${RANK}" "hosvd_subspace_iteration" "--rank $RANK"
    N=$(( N + 1 ))
done

log ""
log "════════════════════════════════════════"
log "  All $TOTAL runs complete"
log "  Full log → $LOG_FILE"
log "════════════════════════════════════════"

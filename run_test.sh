#!/bin/bash
# 缓存一致性检查、完整测试集评估和报告；沿用指定 checkpoint 的配置。
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT="checkpoints/optimized/best_model.pt"
DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint)
            [[ $# -ge 2 ]] || { echo '--checkpoint 缺少路径' >&2; exit 2; }
            CHECKPOINT="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "未知参数: $1" >&2; exit 2 ;;
    esac
done
run_cmd() {
    printf '执行：'; printf ' %q' "$@"; printf '\n'
    if [[ "$DRY_RUN" != true ]]; then "$@"; fi
}
if [[ "$DRY_RUN" != true && ! -f "$CHECKPOINT" ]]; then
    echo "Checkpoint 不存在: $CHECKPOINT" >&2; exit 1
fi
EVAL_DIR="artifacts/evaluation/run_$(date +%Y%m%d_%H%M%S)"
run_cmd python submodules/check_cache.py --checkpoint "$CHECKPOINT"
run_cmd python submodules/evaluate.py --checkpoint "$CHECKPOINT" --output "$EVAL_DIR"
run_cmd python submodules/render.py "$EVAL_DIR"
echo "评估报告: $EVAL_DIR/index.html"

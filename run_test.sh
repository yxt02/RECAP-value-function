#!/bin/bash
# RECAP 价值函数 - 测试 + 报告流程
#
# 用法：
#     bash run_test.sh                           # 默认 checkpoint
#     bash run_test.sh --checkpoint <path>       # 指定 checkpoint
#     bash run_test.sh --dry-run                 # 只显示将执行的步骤

set -e  # 遇到错误立即退出

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
cd "$PROJECT_ROOT"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

log_step() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  步骤 $1: $2${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
}

# 解析参数
CHECKPOINT="checkpoints/optimized/best_model.pt"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            log_info "试运行模式，不会执行实际操作"
            shift
            ;;
        *)
            log_warn "未知参数: $1"
            shift
            ;;
    esac
done

run_cmd() {
    local cmd="$1"
    local desc="$2"

    if [ "$DRY_RUN" = true ]; then
        log_info "[试运行] $desc"
        log_info "  命令: $cmd"
        return 0
    fi

    log_info "$desc"
    log_info "  命令: $cmd"

    if eval "$cmd"; then
        log_success "$desc - 完成"
        return 0
    else
        log_error "$desc - 失败"
        return 1
    fi
}

echo ""
echo "============================================"
echo "  RECAP 价值函数 - 测试 + 报告"
echo "============================================"
echo ""
log_info "项目根目录: $PROJECT_ROOT"
log_info "Checkpoint: $CHECKPOINT"
log_info "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 检查 checkpoint
if [ ! -f "$CHECKPOINT" ]; then
    log_error "Checkpoint 不存在: $CHECKPOINT"
    log_error "请先运行 bash run_train.sh"
    exit 1
fi

# 步骤 1: 缓存检查
log_step "1" "缓存一致性检查"
run_cmd "python tests/check_cache.py" "验证缓存特征与在线编码的一致性"

# 步骤 2: 评估
log_step "2" "全测试集评估"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EVAL_DIR="artifacts/evaluation/run_${TIMESTAMP}"
run_cmd "python tests/evaluate.py --checkpoint $CHECKPOINT --output $EVAL_DIR" "评估模型"

# 步骤 3: 报告渲染
log_step "3" "渲染报告"
run_cmd "python tests/render.py $EVAL_DIR" "生成图表和 HTML 报告"

echo ""
echo "============================================"
echo "  测试 + 报告完成"
echo "============================================"
echo ""
log_info "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
log_info "评估报告: $EVAL_DIR/index.html"
log_info "查看报告: python -m http.server 8080 -d $EVAL_DIR"
echo ""

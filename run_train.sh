#!/bin/bash
# RECAP 价值函数 - 数据准备 + 训练流程
#
# 用法：
#     bash run_train.sh              # 标准流程
#     bash run_train.sh --smoke_test # 冒烟测试模式
#     bash run_train.sh --dry-run    # 只显示将执行的步骤

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
SMOKE_TEST=""
DRY_RUN=false

for arg in "$@"; do
    case $arg in
        --smoke_test)
            SMOKE_TEST="--smoke_test"
            log_info "冒烟测试模式已启用"
            ;;
        --dry-run)
            DRY_RUN=true
            log_info "试运行模式，不会执行实际操作"
            ;;
        *)
            log_warn "未知参数: $arg"
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
echo "  RECAP 价值函数 - 数据准备 + 训练"
echo "============================================"
echo ""
log_info "项目根目录: $PROJECT_ROOT"
log_info "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 步骤 1: 数据准备
log_step "1" "数据准备"
run_cmd "python scripts/prepare_data.py --analyze" "只读检查数据完整性"

if [ "$DRY_RUN" = false ]; then
    run_cmd "python scripts/prepare_data.py --all" "执行完整数据适配（审计 + 回报 + 划分）"
fi

# 步骤 2: 训练
log_step "2" "训练价值模型"
if [ -n "$SMOKE_TEST" ]; then
    run_cmd "python scripts/train.py $SMOKE_TEST" "冒烟测试训练"
else
    run_cmd "python scripts/train.py" "标准训练"
fi

echo ""
echo "============================================"
echo "  数据准备 + 训练完成"
echo "============================================"
echo ""
log_info "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
log_info "下一步: bash run_test.sh"
echo ""

#!/bin/bash
#==============================================================================
# 融合方式消融实验结果收集脚本
# 用途: 自动收集所有融合方式实验的结果并生成对比报告
#==============================================================================

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

#==============================================================================
# 配置
#==============================================================================

# 实验目录列表
EXPERIMENT_DIRS=(
    "hse_fusion_concat"
    "hse_fusion_gated"
    "hse_fusion_bilinear_r16"
    "hse_fusion_adaptive"
    "hse_fusion_tucker_r16"
)

# 融合方式名称
FUSION_NAMES=(
    "Concat (Baseline)"
    "Gated"
    "Bilinear (R=16)"
    "Adaptive"
    "Tucker (R=16)"
)

# 性质名称
PROPERTY="hse_bandgap-2"

# 输出文件
OUTPUT_FILE="fusion_ablation_results.txt"
CSV_FILE="fusion_ablation_results.csv"

#==============================================================================
# 函数定义
#==============================================================================

# 提取Best Val MAE
extract_val_mae() {
    local dir=$1
    local log_file="${dir}/${PROPERTY}/train_*.out"

    # 查找log文件
    local actual_log=$(ls ${log_file} 2>/dev/null | head -n 1)

    if [ -z "$actual_log" ]; then
        echo "N/A"
        return
    fi

    # 提取Best Validation MAE
    local val_mae=$(grep "Best Validation MAE:" "$actual_log" | tail -n 1 | grep -oP 'MAE: \K[0-9.]+')

    if [ -z "$val_mae" ]; then
        echo "N/A"
    else
        echo "$val_mae"
    fi
}

# 提取Best Test MAE
extract_test_mae() {
    local dir=$1
    local log_file="${dir}/${PROPERTY}/train_*.out"

    local actual_log=$(ls ${log_file} 2>/dev/null | head -n 1)

    if [ -z "$actual_log" ]; then
        echo "N/A"
        return
    fi

    local test_mae=$(grep "Best Test MAE:" "$actual_log" | tail -n 1 | grep -oP 'MAE: \K[0-9.]+')

    if [ -z "$test_mae" ]; then
        echo "N/A"
    else
        echo "$test_mae"
    fi
}

# 提取训练状态
extract_status() {
    local dir=$1
    local log_file="${dir}/${PROPERTY}/train_*.out"

    local actual_log=$(ls ${log_file} 2>/dev/null | head -n 1)

    if [ -z "$actual_log" ]; then
        echo "Not Started"
        return
    fi

    if grep -q "Training Complete" "$actual_log" 2>/dev/null; then
        echo "Completed"
    elif grep -q "Early stopping triggered" "$actual_log" 2>/dev/null; then
        echo "Early Stopped"
    else
        echo "Running"
    fi
}

# 提取训练epoch数
extract_epochs() {
    local dir=$1
    local log_file="${dir}/${PROPERTY}/train_*.out"

    local actual_log=$(ls ${log_file} 2>/dev/null | head -n 1)

    if [ -z "$actual_log" ]; then
        echo "N/A"
        return
    fi

    # 查找最后一个epoch
    local last_epoch=$(grep -oP 'Epoch \K[0-9]+' "$actual_log" | tail -n 1)

    if [ -z "$last_epoch" ]; then
        echo "0"
    else
        echo "$last_epoch"
    fi
}

#==============================================================================
# 主程序
#==============================================================================

echo -e "${BLUE}=========================================="
echo "融合方式消融实验结果收集"
echo -e "==========================================${NC}"
echo ""

# 创建结果文件
{
    echo "================================================================================"
    echo "融合方式消融实验结果汇总"
    echo "================================================================================"
    echo "生成时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "性质: $PROPERTY"
    echo "================================================================================"
    echo ""
} > "$OUTPUT_FILE"

# CSV表头
echo "Fusion_Type,Status,Epochs,Val_MAE,Test_MAE,Val_Improvement,Test_Improvement" > "$CSV_FILE"

# 存储基线结果
BASELINE_VAL_MAE=""
BASELINE_TEST_MAE=""

# 收集每个实验的结果
echo -e "${CYAN}收集实验结果...${NC}"
echo ""

for i in "${!EXPERIMENT_DIRS[@]}"; do
    dir="${EXPERIMENT_DIRS[$i]}"
    name="${FUSION_NAMES[$i]}"

    echo -e "${YELLOW}[$((i+1))/${#EXPERIMENT_DIRS[@]}] ${name}${NC}"

    # 提取指标
    status=$(extract_status "$dir")
    epochs=$(extract_epochs "$dir")
    val_mae=$(extract_val_mae "$dir")
    test_mae=$(extract_test_mae "$dir")

    # 如果是第一个（基线），保存数值
    if [ $i -eq 0 ] && [ "$val_mae" != "N/A" ]; then
        BASELINE_VAL_MAE=$val_mae
        BASELINE_TEST_MAE=$test_mae
    fi

    # 计算改进百分比
    val_improvement="N/A"
    test_improvement="N/A"

    if [ -n "$BASELINE_VAL_MAE" ] && [ "$val_mae" != "N/A" ]; then
        val_improvement=$(python3 -c "print(f'{((float('$BASELINE_VAL_MAE') - float('$val_mae')) / float('$BASELINE_VAL_MAE') * 100):.2f}%')" 2>/dev/null || echo "N/A")
    fi

    if [ -n "$BASELINE_TEST_MAE" ] && [ "$test_mae" != "N/A" ]; then
        test_improvement=$(python3 -c "print(f'{((float('$BASELINE_TEST_MAE') - float('$test_mae')) / float('$BASELINE_TEST_MAE') * 100):.2f}%')" 2>/dev/null || echo "N/A")
    fi

    # 打印到终端
    echo "  状态:        $status"
    echo "  Epochs:      $epochs"
    echo "  Val MAE:     $val_mae"
    echo "  Test MAE:    $test_mae"
    if [ "$val_improvement" != "N/A" ] && [ $i -ne 0 ]; then
        echo "  Val改进:     $val_improvement"
        echo "  Test改进:    $test_improvement"
    fi
    echo ""

    # 写入文本文件
    {
        echo "--------------------------------------------------------------------------------"
        echo "融合方式: $name"
        echo "--------------------------------------------------------------------------------"
        echo "状态:              $status"
        echo "训练Epochs:        $epochs"
        echo "Best Val MAE:      $val_mae"
        echo "Best Test MAE:     $test_mae"
        if [ "$val_improvement" != "N/A" ] && [ $i -ne 0 ]; then
            echo "Val MAE改进:       $val_improvement (相比Concat基线)"
            echo "Test MAE改进:      $test_improvement (相比Concat基线)"
        fi
        echo ""
    } >> "$OUTPUT_FILE"

    # 写入CSV
    echo "${name},${status},${epochs},${val_mae},${test_mae},${val_improvement},${test_improvement}" >> "$CSV_FILE"
done

# 生成对比表格
{
    echo "================================================================================"
    echo "性能对比表格"
    echo "================================================================================"
    echo ""
    printf "%-25s %-12s %-12s %-15s %-15s\n" "融合方式" "Val MAE" "Test MAE" "Val改进" "Test改进"
    echo "--------------------------------------------------------------------------------"
} >> "$OUTPUT_FILE"

# 读取CSV生成表格（跳过表头）
tail -n +2 "$CSV_FILE" | while IFS=',' read -r fusion status epochs val test val_imp test_imp; do
    printf "%-25s %-12s %-12s %-15s %-15s\n" "$fusion" "$val" "$test" "$val_imp" "$test_imp" >> "$OUTPUT_FILE"
done

{
    echo "================================================================================"
    echo ""
    echo "注:"
    echo "  - 改进百分比为正数表示性能提升（MAE下降）"
    echo "  - 改进百分比计算公式: (Baseline_MAE - Current_MAE) / Baseline_MAE * 100%"
    echo ""
    echo "================================================================================"
} >> "$OUTPUT_FILE"

#==============================================================================
# 显示结果
#==============================================================================

echo -e "${GREEN}=========================================="
echo "结果收集完成！"
echo -e "==========================================${NC}"
echo ""
echo -e "${CYAN}结果文件:${NC}"
echo "  文本报告:  $OUTPUT_FILE"
echo "  CSV数据:   $CSV_FILE"
echo ""
echo -e "${CYAN}快速查看:${NC}"
echo ""

# 显示对比表格
echo -e "${BOLD}性能对比表格:${NC}"
echo ""
printf "${BOLD}%-25s %-12s %-12s %-15s %-15s${NC}\n" "融合方式" "Val MAE" "Test MAE" "Val改进" "Test改进"
echo "--------------------------------------------------------------------------------"

tail -n +2 "$CSV_FILE" | while IFS=',' read -r fusion status epochs val test val_imp test_imp; do
    # 根据状态设置颜色
    if [ "$status" == "Completed" ] || [ "$status" == "Early Stopped" ]; then
        color=$GREEN
    elif [ "$status" == "Running" ]; then
        color=$YELLOW
    else
        color=$RED
    fi

    printf "${color}%-25s${NC} %-12s %-12s %-15s %-15s\n" "$fusion" "$val" "$test" "$val_imp" "$test_imp"
done

echo ""
echo -e "${YELLOW}查看完整报告:${NC}"
echo "  cat $OUTPUT_FILE"
echo ""
echo -e "${YELLOW}在Excel中打开CSV:${NC}"
echo "  使用Excel/LibreOffice打开 $CSV_FILE"
echo ""
echo -e "${GREEN}========================================${NC}"

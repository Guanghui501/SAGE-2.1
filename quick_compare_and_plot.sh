#!/bin/bash
"""
快速对比和作图脚本

自动完成：
1. 预生成遮挡数据
2. 评估两个模型
3. 生成对比图表

用法：
    ./quick_compare_and_plot.sh
"""

set -e

# ============================================================================
# 配置区域 - 修改这里的路径
# ============================================================================

# 测试集路径
TEST_DATA="corrected_test_set_mbj_seed42/test.pkl"

# 模型路径
MODEL1="/public/home/ghzhang/crysmmnet-main-2/src/coGN/band-shuangyanma/111my/SGA-V2.0/output_100epochs_42_bs64_sw_ju_fg_proj_crossmodal_nomiddle_mbj_bandgap_quantext/mbj_bandgap/best_test_model.pt"
MODEL2="/public/home/ghzhang/crysmmnet-main-2/src/coGN/band-shuangyanma/111my/SGA-V2.0/output_100epochs_42_bs64_sw_ju_fg_proj_crossmodal_middle_mbj_bandgap_quantext/mbj_bandgap/best_test_model.pt"

# 模型名称（用于图表标签）
MODEL1_NAME="model1+2"
MODEL2_NAME="SAGE-Net"

# 遮挡策略
#STRATEGIES="random_token sentence keep_keywords"
STRATEGIES="random_token random_word sentence"
# 遮挡率
RATIOS="0.0 0.2 0.4 0.6 0.8 1.0"

# 输出目录
MASKED_DATA_DIR="./deletion_datasets_comparison"
RESULTS_DIR="./comparison_results"
PLOTS_DIR="./comparison_plots"

# Batch size
BATCH_SIZE=64

# ===========================================================================
# 脚本开始
# ============================================================================

echo "========================================================================"
echo "快速对比和作图工具"
echo "========================================================================"
echo ""
echo "配置信息:"
echo "  模型1: $MODEL1_NAME"
echo "  模型2: $MODEL2_NAME"
echo "  策略: $STRATEGIES"
echo "  遮挡率: $RATIOS"
echo ""

# 创建输出目录
mkdir -p "$MASKED_DATA_DIR"
mkdir -p "$RESULTS_DIR"
mkdir -p "$PLOTS_DIR"

# ============================================================================
# 步骤1: 预生成遮挡数据
# ============================================================================

echo "========================================================================"
echo "步骤 1/3: 预生成遮挡数据"
echo "========================================================================"

if [ -f "$MASKED_DATA_DIR/index.json" ]; then
    echo "检测到已存在的遮挡数据，是否重新生成？(y/n)"
    read -r REGENERATE

    if [[ "$REGENERATE" =~ ^[Yy]$ ]]; then
        echo "删除旧数据..."
        rm -rf "$MASKED_DATA_DIR"/*

        echo "生成新的删除数据..."
        python pregenerate_deletion_dataset.py \
            --input_data "$TEST_DATA" \
            --output_dir "$MASKED_DATA_DIR" \
            --strategies $STRATEGIES \
            --ratios $RATIOS \
            --seed 42
    else
        echo "使用现有的删除数据"
    fi
else
    echo "生成遮挡数据..."
    python pregenerate_deletion_dataset.py \
        --input_data "$TEST_DATA" \
        --output_dir "$MASKED_DATA_DIR" \
        --strategies $STRATEGIES \
        --ratios $RATIOS \
        --seed 42
fi

echo ""

# ============================================================================
# 步骤2: 评估两个模型
# ============================================================================

echo "========================================================================"
echo "步骤 2/3: 评估模型"
echo "========================================================================"

# 评估模型1
echo ""
echo "评估模型1: $MODEL1_NAME"
echo "------------------------------------------------------------------------"

for strategy in $STRATEGIES; do
    for ratio in $RATIOS; do
        MASKED_FILE="$MASKED_DATA_DIR/${strategy}_${ratio}.pkl"
        OUTPUT_FILE="$RESULTS_DIR/model1_${strategy}_${ratio}.json"

        if [ -f "$OUTPUT_FILE" ]; then
            echo "  跳过（已存在）: ${strategy} ${ratio}"
        else
            echo "  评估: ${strategy} ${ratio}"
            python evaluate_with_premasked_data.py \
                --checkpoint "$MODEL1" \
                --masked_data "$MASKED_FILE" \
                --output_file "$OUTPUT_FILE" \
                --batch_size "$BATCH_SIZE" \
                --device cuda
        fi
    done
done

# 评估模型2
echo ""
echo "评估模型2: $MODEL2_NAME"
echo "------------------------------------------------------------------------"

for strategy in $STRATEGIES; do
    for ratio in $RATIOS; do
        MASKED_FILE="$MASKED_DATA_DIR/${strategy}_${ratio}.pkl"
        OUTPUT_FILE="$RESULTS_DIR/model2_${strategy}_${ratio}.json"

        if [ -f "$OUTPUT_FILE" ]; then
            echo "  跳过（已存在）: ${strategy} ${ratio}"
        else
            echo "  评估: ${strategy} ${ratio}"
            python evaluate_with_premasked_data.py \
                --checkpoint "$MODEL2" \
                --masked_data "$MASKED_FILE" \
                --output_file "$OUTPUT_FILE" \
                --batch_size "$BATCH_SIZE" \
                --device cuda
        fi
    done
done

echo ""

# ============================================================================
# 步骤3: 生成对比图表
# ============================================================================

echo "========================================================================"
echo "步骤 3/3: 生成对比图表"
echo "========================================================================"

python plot_model_comparison.py \
    --model1_results "$RESULTS_DIR/model1_*.json" \
    --model2_results "$RESULTS_DIR/model2_*.json" \
    --model1_name "$MODEL1_NAME" \
    --model2_name "$MODEL2_NAME" \
    --output_dir "$PLOTS_DIR"

echo ""

# ============================================================================
# 完成
# ============================================================================

echo "========================================================================"
echo "对比完成！"
echo "========================================================================"
echo ""
echo "结果文件:"
echo "  评估结果: $RESULTS_DIR/"
echo "  对比图表: $PLOTS_DIR/"
echo ""
echo "生成的图表:"
for strategy in $STRATEGIES; do
    echo "  - comparison_${strategy}.png"
done
echo "  - comprehensive_comparison.png (综合对比)"
echo ""
echo "查看图表:"
echo "  eog $PLOTS_DIR/comprehensive_comparison.png"
echo ""
echo "========================================================================"

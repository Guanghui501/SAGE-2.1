#!/bin/bash
#==============================================================================
# SLURM 融合方式消融实验批量提交脚本
# 用途: 测试5种后期融合方式对模型性能的影响
# 融合方式: concat, gated, bilinear, adaptive, tucker
#==============================================================================

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

#==============================================================================
# 配置部分
#==============================================================================

# 训练属性和随机种子
PROPERTIES=("hse_bandgap-2")
RANDOM_SEEDS=(42)

# SLURM资源配置
SLURM_PARTITION=""              # 留空则使用默认分区，或设置如 "gpu"
SLURM_GPUS=1                    # GPU数量
SLURM_NODES=1                   # 节点数
SLURM_NTASKS=1                  # 任务数

# Conda环境
CONDA_ENV="sganet"

# 数据集路径
DATA_ROOT="/public/home/ghzhang/crysmmnet-main/dataset"

# GPU设备
CUDA_DEVICE="3"

#==============================================================================
# 融合方式消融实验配置定义
#==============================================================================

# 配置1: Concat融合（基线）
CONFIG_1_NAME="concat"
CONFIG_1_DESC="Baseline: Concat Fusion"
CONFIG_1_SUFFIX="fusion_concat"
CONFIG_1_FUSION_TYPE="concat"
CONFIG_1_RANK="16"
CONFIG_1_OUTPUT_DIM="64"

# 配置2: Gated融合（门控）
CONFIG_2_NAME="gated"
CONFIG_2_DESC="Gated Fusion (Adaptive Weights)"
CONFIG_2_SUFFIX="fusion_gated"
CONFIG_2_FUSION_TYPE="gated"
CONFIG_2_RANK="16"
CONFIG_2_OUTPUT_DIM="64"

# 配置3: Bilinear融合（双线性）
CONFIG_3_NAME="bilinear"
CONFIG_3_DESC="Bilinear Fusion (Second-order Interaction, Rank=16)"
CONFIG_3_SUFFIX="fusion_bilinear_r16"
CONFIG_3_FUSION_TYPE="bilinear"
CONFIG_3_RANK="16"
CONFIG_3_OUTPUT_DIM="64"

# 配置4: Adaptive融合（自适应多策略）
CONFIG_4_NAME="adaptive"
CONFIG_4_DESC="Adaptive Fusion (Multi-strategy Combination)"
CONFIG_4_SUFFIX="fusion_adaptive"
CONFIG_4_FUSION_TYPE="adaptive"
CONFIG_4_RANK="16"
CONFIG_4_OUTPUT_DIM="64"

# 配置5: Tucker融合（张量分解，Rank=16）
CONFIG_5_NAME="tucker_r16"
CONFIG_5_DESC="Tucker Fusion (Tensor Decomposition, Rank=16)"
CONFIG_5_SUFFIX="fusion_tucker_r16"
CONFIG_5_FUSION_TYPE="tucker"
CONFIG_5_RANK="16"
CONFIG_5_OUTPUT_DIM="64"

#==============================================================================
# 函数定义
#==============================================================================

# 提交单个SLURM作业的函数
submit_job() {
    local job_name=$1
    local output_dir=$2
    local property=$3
    local seed=$4
    local fusion_type=$5
    local fusion_rank=$6
    local fusion_output_dim=$7
    local dependency_id=$8

    # 创建输出目录
    mkdir -p "$output_dir"

    # 构建依赖参数
    local dependency_flag=""
    if [ -n "$dependency_id" ]; then
        dependency_flag="--dependency=afterok:${dependency_id}"
    fi

    # 构建分区参数
    local partition_flag=""
    if [ -n "$SLURM_PARTITION" ]; then
        partition_flag="#SBATCH -p ${SLURM_PARTITION}"
    fi

    # 提交作业
    local job_submit=$(sbatch $dependency_flag <<EOF
#!/bin/bash
#SBATCH -J ${job_name}
#SBATCH -N ${SLURM_NODES}
#SBATCH --ntasks=${SLURM_NTASKS}
#SBATCH --gpus=${SLURM_GPUS}
#SBATCH -o ${output_dir}/%x-%j.out
#SBATCH -e ${output_dir}/%x-%j.err
${partition_flag}

#==============================================================================
# 作业执行脚本
#==============================================================================

# 清理模块环境
#module purge

# 激活Conda环境
#source ~/.bashrc
conda activate ${CONDA_ENV}

# 环境变量
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=${CUDA_DEVICE}

# 打印作业信息
echo "=========================================="
echo "SLURM 作业信息"
echo "=========================================="
echo "作业 ID:       \${SLURM_JOB_ID}"
echo "作业名称:      \${SLURM_JOB_NAME}"
echo "节点:          \${SLURM_NODELIST}"
echo "GPU:           \${CUDA_VISIBLE_DEVICES}"
echo "开始时间:      \$(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "训练配置:"
echo "  属性:                     ${property}"
echo "  随机种子:                 ${seed}"
echo "  融合方式:                 ${fusion_type}"
echo "  融合Rank:                 ${fusion_rank}"
echo "  融合输出维度:             ${fusion_output_dim}"
echo "  输出目录:                 ${output_dir}"
echo "=========================================="
echo ""

# 执行训练
python train_with_cross_modal_attention.py \
    --root_dir ${DATA_ROOT} \
    --dataset jarvis \
    --property ${property} \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1 \
    --batch_size 64 \
    --epochs 100 \
    --learning_rate 5e-4 \
    --weight_decay 1e-3 \
    --warmup_steps 2000 \
    --alignn_layers 4 \
    --gcn_layers 4 \
    --hidden_features 256 \
    --graph_dropout 0.15 \
    --late_fusion_type ${fusion_type} \
    --late_fusion_rank ${fusion_rank} \
    --late_fusion_output_dim ${fusion_output_dim} \
    --use_cross_modal False \
    --cross_modal_num_heads 2 \
    --use_middle_fusion True \
    --middle_fusion_layers 2 \
    --middle_fusion_dropout 0.35 \
    --middle_fusion_use_gate_norm True \
    --middle_fusion_use_learnable_scale True \
    --middle_fusion_initial_scale 12.0 \
    --use_fine_grained_attention False \
    --fine_grained_hidden_dim 256 \
    --fine_grained_num_heads 8 \
    --fine_grained_dropout 0.2 \
    --fine_grained_use_projection False \
    --early_stopping_patience 30 \
    --output_dir ${output_dir} \
    --num_workers 24 \
    --random_seed ${seed}

# 记录完成状态
EXIT_CODE=\$?
echo ""
echo "=========================================="
echo "作业完成信息"
echo "=========================================="
echo "结束时间:      \$(date '+%Y-%m-%d %H:%M:%S')"
echo "退出码:        \${EXIT_CODE}"
echo "=========================================="

exit \${EXIT_CODE}
EOF
)

    # 提取并返回作业ID
    local job_id=$(echo "$job_submit" | grep -oP 'Submitted batch job \K\d+')
    echo "$job_id"
}

# 打印配置信息的函数
print_config_info() {
    local config_num=$1
    local config_name=$2
    local config_desc=$3
    local fusion_type=$4
    local fusion_rank=$5
    local output_dim=$6
    local property=$7
    local seed=$8
    local output_dir=$9
    local dependency_id=${10}

    echo ""
    echo -e "${CYAN}=========================================="
    echo "提交训练任务 ${config_num}/5"
    echo -e "==========================================${NC}"
    echo -e "${GREEN}配置名称:${NC}       $config_name"
    echo -e "${GREEN}配置描述:${NC}       $config_desc"
    echo -e "${GREEN}融合方式:${NC}       $fusion_type"
    echo -e "${GREEN}Rank:${NC}           $fusion_rank"
    echo -e "${GREEN}输出维度:${NC}       $output_dim"
    echo -e "${GREEN}属性:${NC}           $property"
    echo -e "${GREEN}随机种子:${NC}       $seed"
    echo -e "${GREEN}输出目录:${NC}       $output_dir"
    if [ -n "$dependency_id" ]; then
        echo -e "${YELLOW}依赖作业:${NC}       $dependency_id (等待其完成)${NC}"
    else
        echo -e "${GREEN}依赖作业:${NC}       无（立即开始）"
    fi
}

#==============================================================================
# 主程序
#==============================================================================

echo -e "${BLUE}=========================================="
echo "SLURM 融合方式消融实验批量提交工具"
echo -e "==========================================${NC}"
echo ""
echo -e "${GREEN}实验配置:${NC}"
echo "  属性:           ${PROPERTIES[@]}"
echo "  随机种子:       ${RANDOM_SEEDS[@]}"
echo "  配置数量:       5种融合方式"
echo "  执行方式:       串行（依赖链）"
echo "  总任务数:       $((${#PROPERTIES[@]} * ${#RANDOM_SEEDS[@]} * 5))"
echo ""
echo -e "${GREEN}融合方式消融实验设计:${NC}"
echo "  1. ${CONFIG_1_DESC}"
echo "  2. ${CONFIG_2_DESC}"
echo "  3. ${CONFIG_3_DESC}"
echo "  4. ${CONFIG_4_DESC}"
echo "  5. ${CONFIG_5_DESC}"
echo ""
echo -e "${GREEN}资源配置:${NC}"
echo "  GPU:            ${SLURM_GPUS}"
echo "  GPU设备:        ${CUDA_DEVICE}"
echo "  Conda环境:      ${CONDA_ENV}"
echo -e "${BLUE}==========================================${NC}"

# 用于追踪作业链
PREV_JOB_ID=""
ALL_JOB_IDS=()
JOB_CONFIGS=()

# 循环遍历每个属性和种子
for PROPERTY in "${PROPERTIES[@]}"; do
    for SEED in "${RANDOM_SEEDS[@]}"; do

        # ===== 配置1: Concat融合（基线） =====
        OUTPUT_DIR="./hse_${CONFIG_1_SUFFIX}"
        JOB_NAME="train_${PROPERTY}_seed${SEED}_${CONFIG_1_NAME}"

        print_config_info "1" "$CONFIG_1_NAME" "$CONFIG_1_DESC" \
                         "$CONFIG_1_FUSION_TYPE" "$CONFIG_1_RANK" "$CONFIG_1_OUTPUT_DIM" \
                         "$PROPERTY" "$SEED" "$OUTPUT_DIR" "$PREV_JOB_ID"

        JOB_ID=$(submit_job "$JOB_NAME" "$OUTPUT_DIR" "$PROPERTY" "$SEED" \
                           "$CONFIG_1_FUSION_TYPE" "$CONFIG_1_RANK" "$CONFIG_1_OUTPUT_DIM" "$PREV_JOB_ID")

        if [ -n "$JOB_ID" ]; then
            echo -e "${GREEN}✓ 作业已提交: ID = ${JOB_ID}${NC}"
            PREV_JOB_ID=$JOB_ID
            ALL_JOB_IDS+=($JOB_ID)
            JOB_CONFIGS+=("${CONFIG_1_DESC}")
        else
            echo -e "${RED}✗ 作业提交失败！${NC}"
            exit 1
        fi
        sleep 1

        # ===== 配置2: Gated融合 =====
        OUTPUT_DIR="./hse_${CONFIG_2_SUFFIX}"
        JOB_NAME="train_${PROPERTY}_seed${SEED}_${CONFIG_2_NAME}"

        print_config_info "2" "$CONFIG_2_NAME" "$CONFIG_2_DESC" \
                         "$CONFIG_2_FUSION_TYPE" "$CONFIG_2_RANK" "$CONFIG_2_OUTPUT_DIM" \
                         "$PROPERTY" "$SEED" "$OUTPUT_DIR" "$PREV_JOB_ID"

        JOB_ID=$(submit_job "$JOB_NAME" "$OUTPUT_DIR" "$PROPERTY" "$SEED" \
                           "$CONFIG_2_FUSION_TYPE" "$CONFIG_2_RANK" "$CONFIG_2_OUTPUT_DIM" "$PREV_JOB_ID")

        if [ -n "$JOB_ID" ]; then
            echo -e "${GREEN}✓ 作业已提交: ID = ${JOB_ID}${NC}"
            PREV_JOB_ID=$JOB_ID
            ALL_JOB_IDS+=($JOB_ID)
            JOB_CONFIGS+=("${CONFIG_2_DESC}")
        else
            echo -e "${RED}✗ 作业提交失败！${NC}"
            exit 1
        fi
        sleep 1

        # ===== 配置3: Bilinear融合 =====
        OUTPUT_DIR="./hse_${CONFIG_3_SUFFIX}"
        JOB_NAME="train_${PROPERTY}_seed${SEED}_${CONFIG_3_NAME}"

        print_config_info "3" "$CONFIG_3_NAME" "$CONFIG_3_DESC" \
                         "$CONFIG_3_FUSION_TYPE" "$CONFIG_3_RANK" "$CONFIG_3_OUTPUT_DIM" \
                         "$PROPERTY" "$SEED" "$OUTPUT_DIR" "$PREV_JOB_ID"

        JOB_ID=$(submit_job "$JOB_NAME" "$OUTPUT_DIR" "$PROPERTY" "$SEED" \
                           "$CONFIG_3_FUSION_TYPE" "$CONFIG_3_RANK" "$CONFIG_3_OUTPUT_DIM" "$PREV_JOB_ID")

        if [ -n "$JOB_ID" ]; then
            echo -e "${GREEN}✓ 作业已提交: ID = ${JOB_ID}${NC}"
            PREV_JOB_ID=$JOB_ID
            ALL_JOB_IDS+=($JOB_ID)
            JOB_CONFIGS+=("${CONFIG_3_DESC}")
        else
            echo -e "${RED}✗ 作业提交失败！${NC}"
            exit 1
        fi
        sleep 1

        # ===== 配置4: Adaptive融合 =====
        OUTPUT_DIR="./hse_${CONFIG_4_SUFFIX}"
        JOB_NAME="train_${PROPERTY}_seed${SEED}_${CONFIG_4_NAME}"

        print_config_info "4" "$CONFIG_4_NAME" "$CONFIG_4_DESC" \
                         "$CONFIG_4_FUSION_TYPE" "$CONFIG_4_RANK" "$CONFIG_4_OUTPUT_DIM" \
                         "$PROPERTY" "$SEED" "$OUTPUT_DIR" "$PREV_JOB_ID"

        JOB_ID=$(submit_job "$JOB_NAME" "$OUTPUT_DIR" "$PROPERTY" "$SEED" \
                           "$CONFIG_4_FUSION_TYPE" "$CONFIG_4_RANK" "$CONFIG_4_OUTPUT_DIM" "$PREV_JOB_ID")

        if [ -n "$JOB_ID" ]; then
            echo -e "${GREEN}✓ 作业已提交: ID = ${JOB_ID}${NC}"
            PREV_JOB_ID=$JOB_ID
            ALL_JOB_IDS+=($JOB_ID)
            JOB_CONFIGS+=("${CONFIG_4_DESC}")
        else
            echo -e "${RED}✗ 作业提交失败！${NC}"
            exit 1
        fi
        sleep 1

        # ===== 配置5: Tucker融合 (Rank=16) =====
        OUTPUT_DIR="./hse_${CONFIG_5_SUFFIX}"
        JOB_NAME="train_${PROPERTY}_seed${SEED}_${CONFIG_5_NAME}"

        print_config_info "5" "$CONFIG_5_NAME" "$CONFIG_5_DESC" \
                         "$CONFIG_5_FUSION_TYPE" "$CONFIG_5_RANK" "$CONFIG_5_OUTPUT_DIM" \
                         "$PROPERTY" "$SEED" "$OUTPUT_DIR" "$PREV_JOB_ID"

        JOB_ID=$(submit_job "$JOB_NAME" "$OUTPUT_DIR" "$PROPERTY" "$SEED" \
                           "$CONFIG_5_FUSION_TYPE" "$CONFIG_5_RANK" "$CONFIG_5_OUTPUT_DIM" "$PREV_JOB_ID")

        if [ -n "$JOB_ID" ]; then
            echo -e "${GREEN}✓ 作业已提交: ID = ${JOB_ID}${NC}"
            PREV_JOB_ID=$JOB_ID
            ALL_JOB_IDS+=($JOB_ID)
            JOB_CONFIGS+=("${CONFIG_5_DESC}")
        else
            echo -e "${RED}✗ 作业提交失败！${NC}"
            exit 1
        fi
        sleep 1

    done
done

#==============================================================================
# 汇总信息
#==============================================================================

echo ""
echo -e "${GREEN}=========================================="
echo "所有作业提交完成！"
echo -e "==========================================${NC}"
echo ""
echo -e "${BLUE}提交的作业列表（串行执行顺序）:${NC}"
for i in "${!ALL_JOB_IDS[@]}"; do
    echo "  $((i+1)). 作业ID: ${ALL_JOB_IDS[$i]} - ${JOB_CONFIGS[$i]}"
done

echo ""
echo -e "${YELLOW}管理命令:${NC}"
echo "  查看所有作业:          squeue -u \$USER"
echo "  查看详细信息:          squeue -u \$USER -o '%.18i %.9P %.30j %.8u %.2t %.10M %.6D %R'"
echo "  查看依赖关系:          squeue -u \$USER -o '%.18i %.30j %.8T %.10r'"
echo "  取消所有作业:          scancel -u \$USER"
echo "  取消作业链:            scancel ${ALL_JOB_IDS[@]}"
echo ""
echo -e "${YELLOW}监控命令:${NC}"
echo "  实时监控:              watch -n 10 'squeue -u \$USER'"
echo "  查看作业日志示例:      tail -f ./hse_fusion_concat/train_*.out"
echo ""
echo -e "${CYAN}结果收集命令:${NC}"
echo "  收集所有Val MAE:       grep 'Best Validation MAE' ./hse_fusion_*/hse_bandgap-2/train_*.out"
echo ""
echo -e "${GREEN}========================================${NC}"

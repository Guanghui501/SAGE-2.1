#!/bin/bash
#==============================================================================
# SLURM 融合方式消融实验批量提交脚本 - 并行版本
# 用途: 并行测试5种后期融合方式对模型性能的影响
# 融合方式: concat, gated, bilinear, adaptive, tucker
# 注意: 需要足够的GPU资源支持并行执行
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
SLURM_GPUS=1                    # 每个任务的GPU数量
SLURM_NODES=1                   # 节点数
SLURM_NTASKS=1                  # 任务数

# Conda环境
CONDA_ENV="sganet"

# 数据集路径
DATA_ROOT="/public/home/ghzhang/crysmmnet-main/dataset"

# GPU设备分配（为每个任务分配不同的GPU）
# 如果有5张GPU，可以设置为 ("0" "1" "2" "3" "4")
# 如果只有1张GPU，所有任务会排队等待
CUDA_DEVICES=("3" "3" "3" "3" "3")  # 修改这里以使用不同GPU

#==============================================================================
# 融合方式消融实验配置定义
#==============================================================================

# 配置数组
FUSION_CONFIGS=(
    "concat:Baseline: Concat Fusion:fusion_concat:concat:16:64"
    "gated:Gated Fusion (Adaptive Weights):fusion_gated:gated:16:64"
    "bilinear:Bilinear Fusion (Second-order, Rank=16):fusion_bilinear_r16:bilinear:16:64"
    "adaptive:Adaptive Fusion (Multi-strategy):fusion_adaptive:adaptive:16:64"
    "tucker_r16:Tucker Fusion (Tensor, Rank=16):fusion_tucker_r16:tucker:16:64"
)

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
    local cuda_device=$8

    # 创建输出目录
    mkdir -p "$output_dir"

    # 构建分区参数
    local partition_flag=""
    if [ -n "$SLURM_PARTITION" ]; then
        partition_flag="#SBATCH -p ${SLURM_PARTITION}"
    fi

    # 提交作业（无依赖，并行执行）
    local job_submit=$(sbatch <<EOF
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

# 激活Conda环境
conda activate ${CONDA_ENV}

# 环境变量
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=${cuda_device}

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

#==============================================================================
# 主程序
#==============================================================================

echo -e "${BLUE}=========================================="
echo "SLURM 融合方式消融实验批量提交工具（并行版本）"
echo -e "==========================================${NC}"
echo ""
echo -e "${GREEN}实验配置:${NC}"
echo "  属性:           ${PROPERTIES[@]}"
echo "  随机种子:       ${RANDOM_SEEDS[@]}"
echo "  配置数量:       ${#FUSION_CONFIGS[@]}种融合方式"
echo "  执行方式:       并行（同时运行）"
echo "  总任务数:       $((${#PROPERTIES[@]} * ${#RANDOM_SEEDS[@]} * ${#FUSION_CONFIGS[@]}))"
echo ""
echo -e "${GREEN}融合方式消融实验设计:${NC}"
idx=1
for config in "${FUSION_CONFIGS[@]}"; do
    IFS=':' read -r name desc suffix type rank dim <<< "$config"
    echo "  ${idx}. ${desc}"
    ((idx++))
done
echo ""
echo -e "${GREEN}资源配置:${NC}"
echo "  每任务GPU:      ${SLURM_GPUS}"
echo "  GPU设备分配:    ${CUDA_DEVICES[@]}"
echo "  Conda环境:      ${CONDA_ENV}"
echo -e "${BLUE}==========================================${NC}"

# 用于追踪所有作业
ALL_JOB_IDS=()
JOB_CONFIGS=()

# 循环遍历每个属性和种子
config_idx=0
for PROPERTY in "${PROPERTIES[@]}"; do
    for SEED in "${RANDOM_SEEDS[@]}"; do

        # 遍历每种融合配置
        for config in "${FUSION_CONFIGS[@]}"; do
            # 解析配置
            IFS=':' read -r name desc suffix fusion_type fusion_rank fusion_output_dim <<< "$config"

            # 选择GPU设备（循环使用）
            cuda_device="${CUDA_DEVICES[$config_idx % ${#CUDA_DEVICES[@]}]}"

            OUTPUT_DIR="./hse_${suffix}"
            JOB_NAME="train_${PROPERTY}_seed${SEED}_${name}"

            echo ""
            echo -e "${CYAN}=========================================="
            echo "提交训练任务 $((config_idx + 1))/${#FUSION_CONFIGS[@]}"
            echo -e "==========================================${NC}"
            echo -e "${GREEN}配置名称:${NC}       $name"
            echo -e "${GREEN}配置描述:${NC}       $desc"
            echo -e "${GREEN}融合方式:${NC}       $fusion_type"
            echo -e "${GREEN}Rank:${NC}           $fusion_rank"
            echo -e "${GREEN}输出维度:${NC}       $fusion_output_dim"
            echo -e "${GREEN}属性:${NC}           $PROPERTY"
            echo -e "${GREEN}随机种子:${NC}       $SEED"
            echo -e "${GREEN}GPU设备:${NC}        $cuda_device"
            echo -e "${GREEN}输出目录:${NC}       $OUTPUT_DIR"

            JOB_ID=$(submit_job "$JOB_NAME" "$OUTPUT_DIR" "$PROPERTY" "$SEED" \
                               "$fusion_type" "$fusion_rank" "$fusion_output_dim" "$cuda_device")

            if [ -n "$JOB_ID" ]; then
                echo -e "${GREEN}✓ 作业已提交: ID = ${JOB_ID}${NC}"
                ALL_JOB_IDS+=($JOB_ID)
                JOB_CONFIGS+=("${desc}")
            else
                echo -e "${RED}✗ 作业提交失败！${NC}"
            fi

            ((config_idx++))
            sleep 0.5
        done

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
echo -e "${BLUE}提交的作业列表（并行执行）:${NC}"
for i in "${!ALL_JOB_IDS[@]}"; do
    echo "  $((i+1)). 作业ID: ${ALL_JOB_IDS[$i]} - ${JOB_CONFIGS[$i]}"
done

echo ""
echo -e "${YELLOW}管理命令:${NC}"
echo "  查看所有作业:          squeue -u \$USER"
echo "  查看详细信息:          squeue -u \$USER -o '%.18i %.9P %.30j %.8u %.2t %.10M %.6D %R'"
echo "  取消所有作业:          scancel -u \$USER"
echo "  取消这批作业:          scancel ${ALL_JOB_IDS[@]}"
echo ""
echo -e "${YELLOW}监控命令:${NC}"
echo "  实时监控:              watch -n 10 'squeue -u \$USER'"
echo "  查看作业日志示例:      tail -f ./hse_fusion_concat/train_*.out"
echo ""
echo -e "${CYAN}结果收集命令:${NC}"
echo "  收集所有Val MAE:       ./collect_fusion_results.sh"
echo ""
echo -e "${GREEN}========================================${NC}"

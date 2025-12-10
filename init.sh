#!/bin/bash
set -e

###############################################################################
# FlowRL Training Initialization Script
# 
# Usage: ./init.sh <model_name> <task_type> [options]
#
# Arguments:
#   model_name: Name of model to use (e.g., GLM-Z1-9B-0414, Qwen3-4B, Qwen3-30B-A3B)
#   task_type: Type of task - "code" or "math"
#
# Options:
#   --hf-model-id: Hugging Face model ID (default: derived from model_name)
#   --skip-download: Skip downloading model and datasets
#   --skip-convert: Skip converting model to Megatron format
#   --num-gpus: Number of GPUs per node (default: 8)
#   --num-nodes: Number of nodes (default: 1)
#   --node-rank: Current node rank, 0 for head node (default: 0)
#   --head-addr: Head node address for multi-node (default: 127.0.0.1)
#   --head-port: Head node port for Ray (default: 6379)
#   --flowrl-beta: FlowRL temperature parameter (default: 15.0)
#   --flowrl-partition-lr: Partition function learning rate (default: 1e-5)
#   --use-flowrl: Enable FlowRL objective and related hyperparameters
#   --datasets-path: Custom path for datasets (default: /workspace/datasets)
#   --models-path: Custom path for models (default: /workspace/models)
#   --checkpoints-path: Custom path for checkpoints (default: /workspace/checkpoints)
#   --tp-size: Tensor parallel size (default: auto-configured)
#   --pp-size: Pipeline parallel size (default: auto-configured)
#   --cp-size: Context parallel size (default: auto-configured)
#   --actor-gpus-per-node: GPUs per node for training (default: auto-configured)
#   --rollout-gpus: Total GPUs for rollout across all nodes (default: auto-configured)
#   --sandbox-url: SandboxFusion API URL for code evaluation (optional, for code tasks)
#
# Example:
#   # Single node:
#   ./init.sh Qwen3-4B code --num-gpus 4
#   
#   # Multi-node (on head node):
#   ./init.sh Qwen3-4B code --num-gpus 8 --num-nodes 4 --node-rank 0
#   
#   # Multi-node (on worker nodes):
#   ./init.sh Qwen3-4B code --num-gpus 8 --num-nodes 4 --node-rank 1 --head-addr 192.168.1.100 --skip-download --skip-convert
###############################################################################

# Parse arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <model_name> <task_type> [options]"
    echo "  task_type: code or math"
    exit 1
fi

MODEL_NAME=$1
TASK_TYPE=$2
shift 2

# Default values
HF_MODEL_ID=""
SKIP_DOWNLOAD=0
SKIP_CONVERT=0
NUM_GPUS=8
NUM_NODES=1
NODE_RANK=0
HEAD_ADDR="127.0.0.1"
HEAD_PORT=6379
FLOWRL_BETA=15.0
FLOWRL_PARTITION_LR=1e-5
USE_FLOWRL=0
USE_ASYNC=0
WORKSPACE="/workspace"
DATASETS_PATH=""
MODELS_PATH=""
CHECKPOINTS_PATH=""
TP_SIZE=""
PP_SIZE=""
CP_SIZE=""
ACTOR_GPUS_PER_NODE=""
TOTAL_ROLLOUT_GPUS=""
SANDBOX_URL=""

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --hf-model-id)
            HF_MODEL_ID="$2"
            shift 2
            ;;
        --skip-download)
            SKIP_DOWNLOAD=1
            shift
            ;;
        --skip-convert)
            SKIP_CONVERT=1
            shift
            ;;
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --num-nodes)
            NUM_NODES="$2"
            shift 2
            ;;
        --node-rank)
            NODE_RANK="$2"
            shift 2
            ;;
        --head-addr)
            HEAD_ADDR="$2"
            shift 2
            ;;
        --head-port)
            HEAD_PORT="$2"
            shift 2
            ;;
        --flowrl-beta)
            FLOWRL_BETA="$2"
            shift 2
            ;;
        --flowrl-partition-lr)
            FLOWRL_PARTITION_LR="$2"
            shift 2
            ;;
        --use-flowrl)
            USE_FLOWRL=1
            shift
            ;;
        --async)
            USE_ASYNC=1
            shift
            ;;
        --datasets-path)
            DATASETS_PATH="$2"
            shift 2
            ;;
        --models-path)
            MODELS_PATH="$2"
            shift 2
            ;;
        --checkpoints-path)
            CHECKPOINTS_PATH="$2"
            shift 2
            ;;
        --tp-size)
            TP_SIZE="$2"
            shift 2
            ;;
        --pp-size)
            PP_SIZE="$2"
            shift 2
            ;;
        --cp-size)
            CP_SIZE="$2"
            shift 2
            ;;
        --actor-gpus-per-node)
            ACTOR_GPUS_PER_NODE="$2"
            shift 2
            ;;
        --rollout-gpus)
            TOTAL_ROLLOUT_GPUS="$2"
            shift 2
            ;;
        --sandbox-url)
            SANDBOX_URL="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate task type
if [[ "$TASK_TYPE" != "code" && "$TASK_TYPE" != "math" ]]; then
    echo "Error: task_type must be 'code' or 'math'"
    exit 1
fi

# Set paths with defaults if not provided
[ -z "$DATASETS_PATH" ] && DATASETS_PATH="${WORKSPACE}/datasets"
[ -z "$MODELS_PATH" ] && MODELS_PATH="${WORKSPACE}/models"
[ -z "$CHECKPOINTS_PATH" ] && CHECKPOINTS_PATH="${WORKSPACE}/checkpoints"

MODEL_DIR="${MODELS_PATH}/${MODEL_NAME}"
MODEL_MEGATRON="${MODEL_DIR}_torch_dist"
CHECKPOINT_DIR="${CHECKPOINTS_PATH}/${MODEL_NAME}_flowrl_${TASK_TYPE}"
DATASET_DIR="${DATASETS_PATH}"

# Derive HF model ID and config if not provided
if [ -z "$HF_MODEL_ID" ]; then
    case "$MODEL_NAME" in
        # GLM Models
        GLM-Z1-9B-0414|GLM-Z1-9B)
            HF_MODEL_ID="zai-org/GLM-Z1-9B-0414"
            MODEL_CONFIG="glm4-9B"
            ;;
        GLM4-9B)
            HF_MODEL_ID="THUDM/glm-4-9b"
            MODEL_CONFIG="glm4-9B"
            ;;
        GLM4-32B)
            HF_MODEL_ID="THUDM/glm-4-32b"
            MODEL_CONFIG="glm4-32B"
            ;;
        GLM4.5-106B-A12B)
            HF_MODEL_ID="THUDM/glm-4.5-106b-a12b"
            MODEL_CONFIG="glm4.5-106B-A12B"
            ;;
        GLM4.5-355B-A32B)
            HF_MODEL_ID="THUDM/glm-4.5-355b-a32b"
            MODEL_CONFIG="glm4.5-355B-A32B"
            ;;
        # Qwen3 Models
        Qwen3-0.6B)
            HF_MODEL_ID="Qwen/Qwen3-0.6B"
            MODEL_CONFIG="qwen3-0.6B"
            ;;
        Qwen3-1.7B)
            HF_MODEL_ID="Qwen/Qwen3-1.7B"
            MODEL_CONFIG="qwen3-1.7B"
            ;;
        Qwen3-4B)
            HF_MODEL_ID="Qwen/Qwen3-4B"
            MODEL_CONFIG="qwen3-4B"
            ;;
        Qwen3-8B)
            HF_MODEL_ID="Qwen/Qwen3-8B"
            MODEL_CONFIG="qwen3-8B"
            ;;
        Qwen3-14B)
            HF_MODEL_ID="Qwen/Qwen3-14B"
            MODEL_CONFIG="qwen3-14B"
            ;;
        Qwen3-32B)
            HF_MODEL_ID="Qwen/Qwen3-32B"
            MODEL_CONFIG="qwen3-32B"
            ;;
        Qwen3-30B-A3B)
            HF_MODEL_ID="Qwen/Qwen3-30B-A3B"
            MODEL_CONFIG="qwen3-30B-A3B"
            ;;
        Qwen3-235B-A22B)
            HF_MODEL_ID="Qwen/Qwen3-235B-A22B"
            MODEL_CONFIG="qwen3-235B-A22B"
            ;;
        Qwen3-Next-80B-A3B)
            HF_MODEL_ID="Qwen/Qwen3-Next-80B-A3B"
            MODEL_CONFIG="qwen3-next-80B-A3B"
            ;;
        # Qwen2.5 Models
        Qwen2.5-0.5B)
            HF_MODEL_ID="Qwen/Qwen2.5-0.5B"
            MODEL_CONFIG="qwen2.5-0.5B"
            ;;
        Qwen2.5-1.5B)
            HF_MODEL_ID="Qwen/Qwen2.5-1.5B"
            MODEL_CONFIG="qwen2.5-1.5B"
            ;;
        Qwen2.5-3B)
            HF_MODEL_ID="Qwen/Qwen2.5-3B"
            MODEL_CONFIG="qwen2.5-3B"
            ;;
        Qwen2.5-7B)
            HF_MODEL_ID="Qwen/Qwen2.5-7B"
            MODEL_CONFIG="qwen2.5-7B"
            ;;
        Qwen2.5-32B)
            HF_MODEL_ID="Qwen/Qwen2.5-32B"
            MODEL_CONFIG="qwen2.5-32B"
            ;;
        # Llama Models
        Llama3.1-8B-Instruct)
            HF_MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
            MODEL_CONFIG="llama3.1-8B-Instruct"
            ;;
        Llama3.2-3B-Instruct)
            HF_MODEL_ID="meta-llama/Llama-3.2-3B-Instruct"
            MODEL_CONFIG="llama3.2-3B-Instruct"
            ;;
        # DeepSeek Models
        DeepSeek-V3)
            HF_MODEL_ID="deepseek-ai/DeepSeek-V3"
            MODEL_CONFIG="deepseek-v3"
            ;;
        DeepSeek-R1-Distill-Qwen-7B)
            HF_MODEL_ID="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
            MODEL_CONFIG="qwen2.5-7B"
            ;;
        DeepSeek-R1-Distill-Qwen-1.5B)
            HF_MODEL_ID="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
            MODEL_CONFIG="qwen2.5-1.5B"
            ;;
        # Kimi Models
        Kimi-K2)
            HF_MODEL_ID="Kimi/kimi-k2"
            MODEL_CONFIG="kimi-k2"
            ;;
        # Moonlight Models
        Moonlight-16B-A3B)
            HF_MODEL_ID="moonlight/moonlight-16b-a3b"
            MODEL_CONFIG="moonlight"
            ;;
        # MIMO Models
        MIMO-7B-RL)
            HF_MODEL_ID="mimo/mimo-7b-rl"
            MODEL_CONFIG="mimo-7B-rl"
            ;;
        *)
            echo "Error: Unknown model name. Please specify --hf-model-id and model config."
            echo "Supported models:"
            echo "  GLM: GLM-Z1-9B, GLM4-9B, GLM4-32B, GLM4.5-106B-A12B, GLM4.5-355B-A32B"
            echo "  Qwen3: Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B, Qwen3-8B, Qwen3-14B, Qwen3-32B, Qwen3-30B-A3B, Qwen3-235B-A22B, Qwen3-Next-80B-A3B"
            echo "  Qwen2.5: Qwen2.5-0.5B, Qwen2.5-1.5B, Qwen2.5-3B, Qwen2.5-7B, Qwen2.5-32B"
            echo "  Llama: Llama3.1-8B-Instruct, Llama3.2-3B-Instruct"
            echo "  DeepSeek: DeepSeek-V3, DeepSeek-R1-Distill-Qwen-7B"
            echo "  Kimi: Kimi-K2"
            echo "  Moonlight: Moonlight-16B-A3B"
            echo "  MIMO: MIMO-7B-RL"
            exit 1
            ;;
    esac
else
    # Try to derive config from model name
    MODEL_CONFIG=$(echo "$MODEL_NAME" | tr '[:upper:]' '[:lower:]' | sed 's/-0414//' | sed 's/glm-z1/glm4/')
fi

echo "============================================"
echo "FlowRL Training Setup"
echo "============================================"
echo "Model: $MODEL_NAME"
echo "HF Model ID: $HF_MODEL_ID"
echo "Model Config: $MODEL_CONFIG"
echo "Task Type: $TASK_TYPE"
echo "Number of Nodes: $NUM_NODES"
echo "Node Rank: $NODE_RANK ($([ $NODE_RANK -eq 0 ] && echo 'HEAD' || echo 'WORKER'))"
echo "GPUs per Node: $NUM_GPUS"
echo "Total GPUs: $((NUM_GPUS * NUM_NODES))"
echo "Ray Head: $HEAD_ADDR:$HEAD_PORT"
echo "FlowRL Beta: $FLOWRL_BETA"
echo "Partition LR: $FLOWRL_PARTITION_LR"
echo "FlowRL Enabled: $([ $USE_FLOWRL -eq 1 ] && echo 'yes' || echo 'no')"
echo "Async Rollout: $([ $USE_ASYNC -eq 1 ] && echo 'yes' || echo 'no')"
echo "============================================"

###############################################################################
# Step 1: Download Model and Datasets
###############################################################################

# Automatically skip download on worker nodes
if [ $NODE_RANK -ne 0 ]; then
    echo "Step 1: Skipping download (worker node)"
    SKIP_DOWNLOAD=1
fi

if [ $SKIP_DOWNLOAD -eq 0 ]; then
    echo "Step 1: Downloading model and datasets..."
    
    # Ensure directories exist
    mkdir -p "$MODELS_PATH" "$DATASETS_PATH"
    
    # Download model
    if [ ! -d "$MODEL_DIR" ]; then
        echo "Downloading model: $HF_MODEL_ID"
        hf download "$HF_MODEL_ID" --local-dir "$MODEL_DIR"
    else
        echo "Model already exists at $MODEL_DIR"
    fi
    
    # Download unified FlowRL dataset
    if [ ! -d "${DATASET_DIR}/flowrl-data" ]; then
        echo "Downloading FlowRL data collection..."
        hf download --repo-type dataset xuekai/flowrl-data-collection \
            --local-dir "${DATASET_DIR}/flowrl-data"
    else
        echo "FlowRL data collection already exists at ${DATASET_DIR}/flowrl-data"
    fi
    
    # Organize datasets into train/test folders for easier access
    if [ "$TASK_TYPE" == "code" ]; then
        echo "Organizing code datasets..."
        mkdir -p "${DATASET_DIR}/code-train" "${DATASET_DIR}/code-test"
        
        # Copy training parquet files to code-train directory
        echo "Copying code training parquet files..."
        if ls "${DATASET_DIR}/flowrl-data/code_data/deepcoder_train-"*.parquet 1> /dev/null 2>&1; then
            cp "${DATASET_DIR}/flowrl-data/code_data/deepcoder_train-"*.parquet "${DATASET_DIR}/code-train/" 2>/dev/null || true
            echo "Copied training parquet files to ${DATASET_DIR}/code-train/"
        else
            echo "No training parquet files found to copy"
        fi
        
        # Copy test files
        cp "${DATASET_DIR}/flowrl-data/code_data/test_"*.parquet "${DATASET_DIR}/code-test/" 2>/dev/null || true
        cp "${DATASET_DIR}/flowrl-data/code_data/test_"*.json "${DATASET_DIR}/code-test/" 2>/dev/null || true
        
    elif [ "$TASK_TYPE" == "math" ]; then
        echo "Organizing math datasets..."
        mkdir -p "${DATASET_DIR}/math-train" "${DATASET_DIR}/math-test"
        
        # Copy training file
        cp "${DATASET_DIR}/flowrl-data/math_data/dapo-math-17k.parquet" "${DATASET_DIR}/math-train/" 2>/dev/null || true
        
        # Copy test files
        cp "${DATASET_DIR}/flowrl-data/math_data/test.parquet" "${DATASET_DIR}/math-test/" 2>/dev/null || true
        cp "${DATASET_DIR}/flowrl-data/math_data/validation.parquet" "${DATASET_DIR}/math-test/" 2>/dev/null || true
    fi
    
    echo "Step 1: Complete ✓"
else
    echo "Step 1: Skipping download (--skip-download)"
fi

###############################################################################
# Step 2: Convert Model to Megatron Format
###############################################################################

# Automatically skip conversion on worker nodes
if [ $NODE_RANK -ne 0 ]; then
    echo "Step 2: Skipping conversion (worker node)"
    SKIP_CONVERT=1
fi

if [ $SKIP_CONVERT -eq 0 ]; then
    echo "Step 2: Converting model to Megatron format..."
    
    if [ ! -d "$MODEL_MEGATRON" ]; then
        cd /workspace/slime
        
        # Load model configuration
        if [ -f "scripts/models/${MODEL_CONFIG}.sh" ]; then
            source "scripts/models/${MODEL_CONFIG}.sh"
        else
            echo "Error: Model configuration file not found: scripts/models/${MODEL_CONFIG}.sh"
            echo "Available configurations:"
            ls scripts/models/*.sh
            exit 1
        fi
        
        # =========================================================================
        # Model Parameter Overrides
        # =========================================================================
        # Override model parameters here if needed. Examples:
        # 
        # MODEL_ARGS+=("--rotary-base" "10000")           # Override RoPE theta
        # MODEL_ARGS+=("--max-position-embeddings" "4096") # Override max position
        # MODEL_ARGS+=("--hidden-size" "4096")             # Override hidden size
        # MODEL_ARGS+=("--num-attention-heads" "32")       # Override attention heads
        # MODEL_ARGS+=("--num-layers" "32")                # Override number of layers
        #
        # Uncomment and modify the lines below for your specific needs:
        
        # For DeepSeek-R1-Distill-Qwen-7B (fix rope_theta mismatch):
        if [[ "$MODEL_NAME" == "DeepSeek-R1-Distill-Qwen-7B" ]]; then
            echo "Applying DeepSeek-R1-Distill-Qwen-7B specific overrides..."
            # Remove existing rotary-base from MODEL_ARGS and add correct value
            NEW_MODEL_ARGS=()
            skip_next=0
            for i in "${!MODEL_ARGS[@]}"; do
                if [ $skip_next -eq 1 ]; then
                    skip_next=0
                    continue
                fi
                if [[ "${MODEL_ARGS[$i]}" == "--rotary-base" ]]; then
                    skip_next=1
                    continue
                fi
                NEW_MODEL_ARGS+=("${MODEL_ARGS[$i]}")
            done
            MODEL_ARGS=("${NEW_MODEL_ARGS[@]}" "--rotary-base" "10000")
            echo "Updated rotary-base from 1000000 to 10000 to match HF config"
        fi
        
        # Add your custom overrides here:
        # MODEL_ARGS+=("--your-param" "value")
        
        # =========================================================================
        
        # Convert to Megatron format
        PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
            ${MODEL_ARGS[@]} \
            --hf-checkpoint "$MODEL_DIR" \
            --save "$MODEL_MEGATRON"
        
        echo "Model converted successfully to: $MODEL_MEGATRON"
    else
        echo "Megatron model already exists at $MODEL_MEGATRON"
    fi
    
    echo "Step 2: Complete ✓"
else
    echo "Step 2: Skipping conversion (--skip-convert)"
fi

###############################################################################
# Step 3: Initialize Ray and SGLang Settings
###############################################################################

echo "Step 3: Initializing Ray and SGLang settings..."

# Check if Ray is already running (e.g., in Kubernetes environment)
if ray status &>/dev/null; then
    echo "Ray is already running. Skipping Ray initialization."
    echo "Current Ray status:"
    ray status
else
    echo "Ray not detected. Starting Ray cluster..."
    
    # Clean up any existing Ray/SGLang processes
    pkill -9 sglang 2>/dev/null || true
    sleep 2
    ray stop --force 2>/dev/null || true
    pkill -9 ray 2>/dev/null || true
    pkill -9 python 2>/dev/null || true
    sleep 2

    # Start Ray based on node role
    if [ $NODE_RANK -eq 0 ]; then
        # Head node
        echo "Starting Ray head node at $HEAD_ADDR:$HEAD_PORT..."
        ray start --head \
            --node-ip-address=$HEAD_ADDR \
            --port=$HEAD_PORT \
            --num-gpus=$NUM_GPUS \
            --disable-usage-stats
        
        echo "Ray head node started with $NUM_GPUS GPUs"
        echo "Worker nodes can connect using: --head-addr $HEAD_ADDR --head-port $HEAD_PORT"
    else
        # Worker node
        echo "Connecting to Ray head node at $HEAD_ADDR:$HEAD_PORT..."
        ray start \
            --address=$HEAD_ADDR:$HEAD_PORT \
            --num-gpus=$NUM_GPUS \
            --disable-usage-stats
        
        echo "Ray worker node $NODE_RANK connected with $NUM_GPUS GPUs"
    fi
fi

echo "Step 3: Complete ✓"

###############################################################################
# Step 4: Generate Training Script (Head Node Only)
###############################################################################

if [ $NODE_RANK -eq 0 ]; then
    echo "Step 4: Generating FlowRL training script..."

    TRAIN_SCRIPT="${WORKSPACE}/slime/run_flowrl_${TASK_TYPE}.sh"

    cat > "$TRAIN_SCRIPT" << 'EOFSCRIPT'
#!/bin/bash
set -e

# Will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=1

# Detect NVLink
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
EOFSCRIPT

    # Add model configuration
    echo "source \"\${SCRIPT_DIR}/scripts/models/${MODEL_CONFIG}.sh\"" >> "$TRAIN_SCRIPT"
    
    # Add model-specific parameter overrides to training script
    if [[ "$MODEL_NAME" == "DeepSeek-R1-Distill-Qwen-7B" ]]; then
        cat >> "$TRAIN_SCRIPT" << 'EOFOVERRIDE'

# DeepSeek-R1-Distill-Qwen-7B override: fix rope_theta mismatch
echo "Applying DeepSeek-R1-Distill-Qwen-7B rotary-base override..."
NEW_MODEL_ARGS=()
skip_next=0
for i in "${!MODEL_ARGS[@]}"; do
    if [ $skip_next -eq 1 ]; then
        skip_next=0
        continue
    fi
    if [[ "${MODEL_ARGS[$i]}" == "--rotary-base" ]]; then
        skip_next=1
        continue
    fi
    NEW_MODEL_ARGS+=("${MODEL_ARGS[$i]}")
done
MODEL_ARGS=("${NEW_MODEL_ARGS[@]}" "--rotary-base" "10000")
echo "Updated rotary-base from 1000000 to 10000 to match HF config"
EOFOVERRIDE
    fi

# Add checkpoint arguments
cat >> "$TRAIN_SCRIPT" << EOFSCRIPT

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}
   --ref-load ${MODEL_MEGATRON}
   --load ${CHECKPOINT_DIR}/
   --save ${CHECKPOINT_DIR}/
   --save-interval 20
)
EOFSCRIPT

# Add rollout and eval arguments based on task type
if [ "$TASK_TYPE" == "math" ]; then
    cat >> "$TRAIN_SCRIPT" << EOFSCRIPT

ROLLOUT_ARGS=(
   --prompt-data ${DATASET_DIR}/math-train/dapo-math-17k.parquet
   --input-key prompt
   --label-key reward_model
   --apply-chat-template
   --rollout-shuffle

   --rm-type dapo
   --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
   --reward-key score
   --eval-reward-key score
EOFSCRIPT

    if [ $USE_ASYNC -eq 1 ]; then
        cat >> "$TRAIN_SCRIPT" <<'EOFSCRIPT'
   --rollout-function-path examples.fully_async.fully_async_rollout.generate_rollout_fully_async
EOFSCRIPT
    fi

    cat >> "$TRAIN_SCRIPT" << EOFSCRIPT

   --num-rollout 3000
   --rollout-batch-size 32
   --n-samples-per-prompt 16
   --rollout-max-response-len 2048
   --rollout-temperature 0.8

   --global-batch-size 256
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data math-test ${DATASET_DIR}/math-test/test.parquet
   --eval-prompt-data math-validation ${DATASET_DIR}/math-test/validation.parquet
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 2048
   --eval-top-p 0.7
)
EOFSCRIPT
elif [ "$TASK_TYPE" == "code" ]; then
    # Build the list of parquet files
    PARQUET_FILES=$(ls ${DATASET_DIR}/code-train/deepcoder_train-*.parquet 2>/dev/null | tr '\n' ' ')
    if [ -z "$PARQUET_FILES" ]; then
        echo "Warning: No training parquet files found in ${DATASET_DIR}/code-train/"
    fi
    
    cat >> "$TRAIN_SCRIPT" << EOFSCRIPT

ROLLOUT_ARGS=(
   --prompt-data ${PARQUET_FILES}
   --input-key prompt
   --label-key reward_model
   --apply-chat-template
   --rollout-shuffle
   --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
EOFSCRIPT

    if [ $USE_ASYNC -eq 1 ]; then
        cat >> "$TRAIN_SCRIPT" <<'EOFSCRIPT'
   --rollout-function-path examples.fully_async.fully_async_rollout.generate_rollout_fully_async
EOFSCRIPT
    fi

    if [ -n "$SANDBOX_URL" ]; then
        cat >> "$TRAIN_SCRIPT" << EOFSCRIPT
   --custom-rm-path slime.rollout.rm_hub.sandboxfusion_evaluator.async_rm
   --sandbox-url ${SANDBOX_URL}
   --sandbox-timeout 30
EOFSCRIPT
    else
        cat >> "$TRAIN_SCRIPT" <<'EOFSCRIPT'
   --custom-rm-path slime.rollout.rm_hub.code_evaluator.async_rm
EOFSCRIPT
    fi

    cat >> "$TRAIN_SCRIPT" <<'EOFSCRIPT'
   --rm-type code

   --num-rollout 3000
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 4096
   --rollout-temperature 0.8

   --global-batch-size 256
   --balance-data
)
EOFSCRIPT

     cat >> "$TRAIN_SCRIPT" << EOFSCRIPT

EVAL_ARGS=(
    --eval-interval 20
    --eval-prompt-data humanevalplus ${DATASET_DIR}/code-test/test_humanevalplus.parquet
    --eval-prompt-data livecodebench ${DATASET_DIR}/code-test/test_livecodebench.json
    --eval-prompt-data codeforces ${DATASET_DIR}/code-test/test_codeforces.parquet
    --n-samples-per-eval-prompt 16
    --eval-max-response-len 8192
    --eval-top-p 0.7
)
EOFSCRIPT
fi

# Calculate parallelism based on total number of GPUs (if not provided via CLI)
TOTAL_GPUS=$((NUM_GPUS * NUM_NODES))

# Determine if we should use colocate mode
USE_COLOCATE=0

# Only auto-configure if not explicitly set
if [ -z "$TP_SIZE" ] || [ -z "$PP_SIZE" ] || [ -z "$CP_SIZE" ] || [ -z "$ACTOR_GPUS_PER_NODE" ] || [ -z "$TOTAL_ROLLOUT_GPUS" ]; then
    if [ $NUM_GPUS -le 4 ]; then
        # Small single-node setup: use all GPUs for both (colocate required)
        USE_COLOCATE=1
        [ -z "$TP_SIZE" ] && TP_SIZE=2
        [ -z "$PP_SIZE" ] && PP_SIZE=1
        [ -z "$CP_SIZE" ] && CP_SIZE=1
        [ -z "$ACTOR_GPUS_PER_NODE" ] && ACTOR_GPUS_PER_NODE=$NUM_GPUS
        [ -z "$TOTAL_ROLLOUT_GPUS" ] && TOTAL_ROLLOUT_GPUS=$TOTAL_GPUS
    elif [ $NUM_GPUS -eq 8 ] && [ $NUM_NODES -eq 1 ]; then
        # Single node with 8 GPUs: split 4 for training, 4 for rollout
        USE_COLOCATE=0
        [ -z "$TP_SIZE" ] && TP_SIZE=2
        [ -z "$PP_SIZE" ] && PP_SIZE=2
        [ -z "$CP_SIZE" ] && CP_SIZE=1
        [ -z "$ACTOR_GPUS_PER_NODE" ] && ACTOR_GPUS_PER_NODE=4
        [ -z "$TOTAL_ROLLOUT_GPUS" ] && TOTAL_ROLLOUT_GPUS=4
    elif [ $NUM_GPUS -eq 8 ] && [ $NUM_NODES -gt 1 ]; then
        # Multi-node with 8 GPUs per node: split each node 4+4
        USE_COLOCATE=0
        [ -z "$TP_SIZE" ] && TP_SIZE=2
        [ -z "$PP_SIZE" ] && PP_SIZE=$((NUM_NODES * 2))  # PP across nodes
        [ -z "$CP_SIZE" ] && CP_SIZE=1
        [ -z "$ACTOR_GPUS_PER_NODE" ] && ACTOR_GPUS_PER_NODE=4
        [ -z "$TOTAL_ROLLOUT_GPUS" ] && TOTAL_ROLLOUT_GPUS=$((NUM_NODES * 4))  # 4 rollout GPUs per node
    else
        # For other configurations, adjust as needed
        USE_COLOCATE=0
        [ -z "$TP_SIZE" ] && TP_SIZE=2
        [ -z "$PP_SIZE" ] && PP_SIZE=$NUM_NODES
        [ -z "$CP_SIZE" ] && CP_SIZE=2
        [ -z "$ACTOR_GPUS_PER_NODE" ] && ACTOR_GPUS_PER_NODE=$NUM_GPUS
        [ -z "$TOTAL_ROLLOUT_GPUS" ] && TOTAL_ROLLOUT_GPUS=$TOTAL_GPUS
    fi
fi

echo "Parallelism configuration for $TOTAL_GPUS GPUs ($NUM_NODES nodes x $NUM_GPUS GPUs/node):"
echo "  Mode: $([ $USE_COLOCATE -eq 1 ] && echo 'COLOCATE (training and rollout share GPUs)' || echo 'SEPARATE (dedicated GPUs for training and rollout)')"
echo "  Training: $ACTOR_GPUS_PER_NODE GPUs/node × $NUM_NODES nodes = $((ACTOR_GPUS_PER_NODE * NUM_NODES)) total training GPUs"
echo "  Rollout: $TOTAL_ROLLOUT_GPUS total GPUs"
echo "  Tensor Parallel (TP): $TP_SIZE"
echo "  Pipeline Parallel (PP): $PP_SIZE"
echo "  Context Parallel (CP): $CP_SIZE"

# Add performance arguments
cat >> "$TRAIN_SCRIPT" << EOFSCRIPT

PERF_ARGS=(
   --tensor-model-parallel-size ${TP_SIZE}
   --sequence-parallel
   --pipeline-model-parallel-size ${PP_SIZE}
   --context-parallel-size ${CP_SIZE}
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 4608
)

# FlowRL Arguments
FLOWRL_ARGS=(
   --use-flowrl-objective
   --advantage-estimator grpo
   --flowrl-beta ${FLOWRL_BETA}
   --flowrl-partition-lr ${FLOWRL_PARTITION_LR}
   --flowrl-partition-num-layers 3
   --flowrl-partition-dropout 0.1
   
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.01
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)
WANDB_ARGS=(
   --use-wandb
   --wandb-mode online
   --wandb-host https://microsoft-research.wandb.io
   --wandb-key "\${WANDB_API_KEY}"
   --wandb-project slime-flowrl-${TASK_TYPE}
   --wandb-group ${MODEL_NAME}-flowrl-${TASK_TYPE}
   --wandb-run-id ${MODEL_NAME}-flowrl-${TASK_TYPE}
   --wandb-dir ${WORKSPACE}/slime/wandb
)
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine ${TP_SIZE}
   --sglang-tensor-parallel-size ${TP_SIZE}
   --sglang-mem-fraction-static 0.80
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
)

# Build colocate flag if needed
COLOCATE_FLAG=""
if [ $USE_COLOCATE -eq 1 ]; then
    COLOCATE_FLAG="--colocate"
fi

ENABLE_FLOWRL=${USE_FLOWRL}
ENABLE_ASYNC=${USE_ASYNC}
EOFSCRIPT

    cat >> "$TRAIN_SCRIPT" <<'EOFSCRIPT'

TRAIN_ARGS=()
if [ -n "$COLOCATE_FLAG" ]; then
    TRAIN_ARGS+=("$COLOCATE_FLAG")
fi
TRAIN_ARGS+=("${MODEL_ARGS[@]}")
TRAIN_ARGS+=("${CKPT_ARGS[@]}")
TRAIN_ARGS+=("${ROLLOUT_ARGS[@]}")
TRAIN_ARGS+=("${EVAL_ARGS[@]}")
TRAIN_ARGS+=("${PERF_ARGS[@]}")
TRAIN_ARGS+=("${OPTIMIZER_ARGS[@]}")
TRAIN_ARGS+=("${WANDB_ARGS[@]}")
TRAIN_ARGS+=("${SGLANG_ARGS[@]}")
TRAIN_ARGS+=("${MISC_ARGS[@]}")

if [ "$ENABLE_FLOWRL" -eq 1 ]; then
    TRAIN_ARGS+=("${FLOWRL_ARGS[@]}")
fi

# Select training script based on async flag
if [ "$ENABLE_ASYNC" -eq 1 ]; then
    TRAIN_SCRIPT_NAME="train_async.py"
else
    TRAIN_SCRIPT_NAME="train.py"
fi
EOFSCRIPT

     cat >> "$TRAIN_SCRIPT" << EOFSCRIPT

# Submit Ray job
export PYTHONPATH="/root/Megatron-LM/"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE=1
# export NVTE_FWD_LAYERNORM_SM_MARGIN=0
# export NVTE_BWD_LAYERNORM_SM_MARGIN=0
# export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
# export NVTE_FUSED_ATTN=1
# export NVTE_NORM_FWD_USE_CUDNN=1
# export NVTE_NORM_BWD_USE_CUDNN=1
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 \${TRAIN_SCRIPT_NAME} \\
    --actor-num-nodes ${NUM_NODES} \\
    --actor-num-gpus-per-node ${ACTOR_GPUS_PER_NODE} \\
    --rollout-num-gpus ${TOTAL_ROLLOUT_GPUS} \\
    "\${TRAIN_ARGS[@]}"
EOFSCRIPT

    echo "Training script generated: $TRAIN_SCRIPT"
    echo "Step 4: Complete ✓"

    ###########################################################################
    # Step 5: Start Training (Head Node Only)
    ###########################################################################

    echo ""
    echo "============================================"
    echo "Setup Complete!"
    echo "============================================"
    echo "Model: $MODEL_NAME"
    echo "Task: $TASK_TYPE"
    echo "Deployment: $NUM_NODES nodes x $NUM_GPUS GPUs = $TOTAL_GPUS total GPUs"
    echo "Training script: $TRAIN_SCRIPT"
    echo ""
    echo "To start training, run:"
    echo "  bash $TRAIN_SCRIPT"
    echo "============================================"

    # Uncomment to auto-start training
    bash "$TRAIN_SCRIPT"
else
    ###########################################################################
    # Worker Node - Setup Complete
    ###########################################################################
    
    echo ""
    echo "============================================"
    echo "Worker Node Setup Complete!"
    echo "============================================"
    echo "Node Rank: $NODE_RANK"
    echo "Connected to: $HEAD_ADDR:$HEAD_PORT"
    echo "GPUs: $NUM_GPUS"
    echo ""
    echo "Worker node is ready and waiting for training jobs from head node."
    echo "============================================"
fi

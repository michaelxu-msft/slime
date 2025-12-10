#!/bin/bash

# Training script for code generation with evaluation
# This script demonstrates how to use slime for RL training on code generation tasks

# Clean up any existing processes
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

# Prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

# Detect NVLink for NCCL optimization
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# ============================================================================
# Model Configuration
# ============================================================================
# You can source a model config file or define MODEL_ARGS here
# Example for Qwen3-4B:
MODEL_ARGS=(
   --model-size qwen2.5_3b
   --max-position-embeddings 32768
   --seq-length 8192
   --use-flash-attn
   --tokenizer-type AutoTokenizer
   --disable-bias-linear
   --swiglu
   --position-embedding-type rope
   --rotary-percent 1.0
   --untie-embeddings-and-output-weights
   --normalization RMSNorm
   --norm-epsilon 1e-06
)

# ============================================================================
# Checkpoint Configuration
# ============================================================================
CKPT_ARGS=(
   # Path to the base model checkpoint
   --hf-checkpoint /path/to/deepseek-r1-distill-qwen-7b
   
   # Reference model for KL divergence (optional, can be same as hf-checkpoint)
   --ref-load /path/to/deepseek-r1-distill-qwen-7b_torch_dist
   
   # Where to load/save training checkpoints
   --load /path/to/code_slime_checkpoints/
   --save /path/to/code_slime_checkpoints/
   --save-interval 25
)

# ============================================================================
# Rollout Configuration (Code Generation)
# ============================================================================
ROLLOUT_ARGS=(
   # Training data with code problems
   # Expected format: JSON/JSONL with fields:
   #   - prompt: the code generation problem
   #   - label: test cases in format {"inputs": [...], "outputs": [...]}
   #     OR {"ground_truth": {"inputs": [...], "outputs": [...]}}
   --prompt-data /path/to/code_train_data.jsonl
   --input-key prompt
   --label-key label
   
   # Apply chat template if using instruction-tuned model
   --apply-chat-template
   
   # Shuffle prompts during rollout
   --rollout-shuffle
   
   # *** KEY: Set reward model type to 'code' ***
   --rm-type code
   
   # Rollout settings
   --num-rollout 5000              # Number of rollout steps
   --rollout-batch-size 32         # Batch size for generation
   --n-samples-per-prompt 8        # Generate 8 solutions per problem
   --rollout-max-response-len 8192 # Max tokens per generation
   --rollout-temperature 0.6       # Temperature for diverse solutions
   
   # Training batch size
   --global-batch-size 64
   --balance-data                  # Balance successful/failed samples
)

# ============================================================================
# Evaluation Configuration
# ============================================================================
EVAL_ARGS=(
   --eval-interval 10
   
   # Evaluation dataset (separate from training)
   --eval-prompt-data code_test /path/to/code_test_data.jsonl
   
   # Generate more samples during evaluation for pass@k metrics
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 8192
   --eval-top-p 0.7
)

# ============================================================================
# Performance Configuration
# ============================================================================
PERF_ARGS=(
   # Model parallelism
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   
   # Activation checkpointing for memory efficiency
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   
   # Dynamic batching based on tokens
   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
)

# ============================================================================
# GRPO/RL Algorithm Configuration
# ============================================================================
GRPO_ARGS=(
   # Use GRPO (Group Relative Policy Optimization)
   --advantage-estimator grpo
   
   # KL divergence constraint
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   
   # Entropy regularization (encourage exploration)
   --entropy-coef 0.00
   
   # PPO clipping
   --eps-clip 0.2
   --eps-clip-high 0.28
)

# ============================================================================
# Optimizer Configuration
# ============================================================================
OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --lr-warmup-iters 10
)

# ============================================================================
# Weights & Biases Logging (Optional)
# ============================================================================
WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-code
   --wandb-group code-deepseek-r1-7b
   # --wandb-key ${WANDB_KEY}
)

# ============================================================================
# SGLang Configuration
# ============================================================================
SGLANG_ARGS=(
   # Number of GPUs per SGLang engine
   --rollout-num-gpus-per-engine 2
   
   # Memory allocation for SGLang (0.7 = 70% of GPU memory)
   --sglang-mem-fraction-static 0.7
)

# ============================================================================
# Miscellaneous Settings
# ============================================================================
MISC_ARGS=(
   # Disable dropout for stability
   --attention-dropout 0.0
   --hidden-dropout 0.0
   
   # Numerical stability improvements
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   
   # Flash attention backend
   --attention-backend flash
)

# ============================================================================
# Launch Ray and Training
# ============================================================================

# Start Ray head node
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# Build runtime environment
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

# Submit training job to Ray
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}

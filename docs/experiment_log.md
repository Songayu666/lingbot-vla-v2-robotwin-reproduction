# RoboTwin 50 Tasks Experiment Log

## EXP-001 — Single-GPU Smoke Test

### Date
2026-09-20

### Hardware
- GPU: NVIDIA RTX 4090
- VRAM: 47.36 GiB usable
- GPU count: 1

### Dataset
- RoboTwin clean 50
- Dataset list: `assets/training_data/robotwin_clean_50.txt`
- Norm stats: `assets/norm_stats/robotwin_clean_50.json`

### Smoke-test configuration
- micro_batch_size: 1
- global_batch_size: 1
- gradient_accumulation_steps: 1
- max_steps: 5
- gradient checkpointing: enabled
- mixed precision: enabled

### Progress reached
- LingBot-VLA checkpoint loaded
- Qwen3-VL tokenizer loaded
- Depth model loaded
- DINO Video model loaded
- 50 RoboTwin datasets initialized
- First batch loaded
- Forward pass reached
- Backward pass started

### Failure
CUDA OOM occurred during:

~~~~python
loss.backward()
~~~~

### Memory state

~~~~text
GPU total: 47.36 GiB
Process memory in use: 46.51 GiB
Free memory: 68.25 MiB
~~~~

### Conclusion

The full model can be loaded and the first forward pass can run on one GPU,
but backward propagation exceeds available VRAM.

### Next step

Test additional memory-saving strategies on single GPU.

## EXP-002 — Activation Offload Smoke Test

### Date
2026-09-20

### Changes from EXP-001
- enable_gradient_checkpointing: true
- enable_fp32: false
- use_compile: false
- enable_activation_offload: true
- PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

### Result

The model and all 50 RoboTwin datasets loaded successfully.

Training entered Step 0/5, but CUDA OOM still occurred during:

~~~~python
loss.backward()
~~~~

### Memory state

~~~~text
GPU total: 47.36 GiB
Process memory in use: 46.48 GiB
Free memory: 57.31 MiB
Failed allocation: 12.00 MiB
~~~~

### Conclusion

Activation offload reduced the size of the failed allocation compared with
EXP-001, but single-GPU VRAM is still insufficient to complete backward
propagation.

### Next step

Test a stronger memory-saving strategy, prioritizing FSDP/parameter offload.

## EXP-003 — FSDP Offload Attempt

### Date
2026-09-20

### Change
- enable_fsdp_offload: true

### Result

Training did not start.

The run failed during model parallelization with:

~~~~text
ValueError: Only FSDP training supports `enable_fsdp_offload`.
~~~~

### Conclusion

The current parallel configuration is not compatible with FSDP offload.
The required FSDP mode must be confirmed from the source code before retrying.

## EXP-005 — Expert-Only Single-GPU Smoke Test

### Date
2026-09-20

### Configuration
- GPU count: 1
- micro_batch_size: 1
- global_batch_size: 1
- gradient_accumulation_steps: 1
- max_steps: 5
- enable_gradient_checkpointing: true
- enable_activation_offload: true
- enable_fp32: false
- train_expert_only: true
- enable_fsdp_offload: false

### Key change

The Qwen/VLM backbone was frozen using:

~~~~text
train_expert_only = true
~~~~

The Action Expert and remaining trainable modules were optimized.

### Result

SUCCESS.

The single-GPU smoke test completed all 5 optimization steps:

~~~~text
Step 1/5
Step 2/5
Step 3/5
Step 4/5
Step 5/5
~~~~

Training stopped normally after reaching:

~~~~text
max_steps = 5
~~~~

### Losses

~~~~text
Step 1: Loss 0.2732
Step 2: Loss 0.3698
Step 3: Loss 0.3519
Step 4: Loss 0.4224
Step 5: Loss 0.2661
~~~~

### GPU Memory

~~~~text
VRAM current after epoch 1: 33.16 GB
VRAM peak: 40.60 GB
~~~~

### Conclusion

Freezing the Qwen/VLM backbone makes LingBot-VLA v2 RoboTwin post-training
feasible on the current single-GPU machine.

The previous full-training configuration exceeded GPU memory during backward,
while expert-only training completed forward, backward, optimization, and
five consecutive training steps successfully.

### Important limitation

This configuration is not full-parameter post-training.

The Qwen/VLM backbone is frozen and only the Action Expert and other remaining
trainable modules are updated.

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

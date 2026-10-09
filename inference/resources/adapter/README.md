# LoRA adapter

Download the adapter here before building the image:

```bash
huggingface-cli download srivatsan6923/surgledger-c2f-lora \
    adapter_config.json adapter_model.safetensors --local-dir .
```

`inference.py` merges it into the base weights at startup. If this folder has no
`adapter_config.json`, the base Qwen3-VL-8B-Instruct model runs unchanged.

LoRA rank 32, alpha 64, dropout 0.05 on q/k/v/o_proj and gate/up/down_proj of the
language model; the vision tower is frozen. See `training/README.md` for how it
was trained.

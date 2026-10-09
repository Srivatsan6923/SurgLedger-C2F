# LoRA adapter

The adapter is published at
[srivatsan6923/surgledger-c2f-lora](https://huggingface.co/srivatsan6923/surgledger-c2f-lora).
Put it in this folder before building the image:

```bash
huggingface-cli download srivatsan6923/surgledger-c2f-lora \
    adapter_config.json adapter_model.safetensors --local-dir .
```

`inference.py` merges it into the base weights at startup. If there is no
`adapter_config.json` here, the base Qwen3-VL-8B-Instruct model runs unchanged,
so the image still builds and runs.

The weights are the ones in the submitted container,
`adapter_model.safetensors` with sha256
`0d4a2831a3acb407fc3eb899b0e750a8d0d41ad60b6b6db2b79f258db8ff2c12`. The only
difference is `adapter_config.json`, where `base_model_name_or_path` now points
at the public base model instead of the local path it was trained from.

Settings: LoRA rank 32, alpha 64, dropout 0.05, applied to q/k/v/o_proj and
gate/up/down_proj of the language model, vision tower frozen.
`training/README.md` describes how it was trained.

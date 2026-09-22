# LoRA adapter

To reproduce the submitted container, put the adapter files in this folder
before building:

    adapter_config.json
    adapter_model.safetensors

The adapter is not part of this repository. It was trained on the ORena FOCUS
training data, which is covered by the challenge's data usage agreement.
`training/README.md` describes how to train it.

If there is no `adapter_config.json` here, `inference.py` runs the base
Qwen3-VL-8B-Instruct model without changes, so the image still builds and runs.

Adapter settings: LoRA rank 32, alpha 64, dropout 0.05, applied to
q/k/v/o_proj and gate/up/down_proj of the language model. The vision tower was
frozen. The adapter is merged into the base weights at startup
(`merge_and_unload`).

# Training

Code for the LoRA adapter used in the submission. It reuses the frame samplers,
prompts and format hints from `inference/resources/surgledger/`, so the model is
trained on the same kind of input it sees in the container.

| File | What it does |
|:--|:--|
| `extract_frames.py` | builds a JPEG frame cache (one frame every 2 s) from the challenge videos |
| `train_lora.py` | LoRA fine-tuning, single GPU or `torchrun` |
| `test_curriculum.py` | checks the data loading (window offsets, train/val split) without a GPU |

```bash
pip install -r requirements.txt
```

## 1. Data

Download the FRAME, SEGMENT and PROCEDURE training splits and the videos of
[HeiCo-FOCUS-VQA](https://huggingface.co/datasets/orena-dkfz/heico-focus-vqa) and
[LapChole-FOCUS-VQA](https://huggingface.co/datasets/orena-dkfz/lapchole-focus-vqa)
(access is granted by the organizers). Keep the Hugging Face layout, since
`train_lora.py` reads the track (frame / segment / procedure) from the path:

```
data/heico/data/{frame,segment,procedure}/train.parquet
data/lapchole/data/{frame,segment,procedure}/train.parquet
```

## 2. Frame cache

```bash
python extract_frames.py --roots data/heico/videos data/lapchole/videos --out frames/
```

Frames are named by their index in 5 fps clip units, so a file name refers to the
same moment as a frame index at inference time.

Optional sanity check of the data loading:

```bash
python test_curriculum.py --data data/
```

## 3. Training

The submitted adapter was trained in two stages with this script.

**Stage 1.** A LoRA trained from scratch on the PROCEDURE rows only
(`--parquet data/*/data/procedure/train.parquet`). We kept its checkpoint at
step 300.

**Stage 2.** Starting from that checkpoint, training on all three tracks of both
datasets, on 4 GPUs:

```bash
torchrun --standalone --nproc_per_node=4 train_lora.py \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --cache frames/ \
    --parquet data/*/data/*/train.parquet \
    --init-adapter runs/stage1/ckpt-300 \
    --n-frames 384 --lr 1e-4 --accum 4 --epochs 2 \
    --save-every 200 \
    --out runs/stage2
```

That is an effective batch size of 16 (4 GPUs x 4 accumulation steps), AdamW with
betas (0.9, 0.95) and no weight decay, 3% warm-up followed by cosine decay,
gradient clipping at 1.0, bf16 and gradient checkpointing. LoRA rank 32,
alpha 64, dropout 0.05 on q/k/v/o and gate/up/down; the vision tower is frozen.

The script holds out 15% of the videos for validation (`val_videos.json` in the
output folder) and balances the capability buckets by repeating the smaller ones,
at most 6 times.

We saved a checkpoint every 200 steps, scored them on the validation videos with
the official scorer, and submitted the one from **step 3200**. To use it, copy
`adapter_config.json` and `adapter_model.safetensors` from that checkpoint into
`inference/resources/adapter/`.

Before a long run, `--max-steps 2 --save-every 100000` does a quick probe that
prints memory use and sequence length per GPU without writing checkpoints.

## Hardware

We ran stage 2 on 4 A100 (80 GB) GPUs. At 384 frames a sample is around 43k
tokens; on smaller cards, lower `--n-frames`.

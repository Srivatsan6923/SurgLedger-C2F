FROM --platform=linux/amd64 pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime AS procedure-algorithm-amd64

# Base image from the challenge template. torch 2.11.0 with CUDA 12.8 runs on both
# evaluation GPUs without a rebuild: RTX PRO 6000 Blackwell (sm_120) natively and
# L40S (sm_89) through the sm_86 kernels. CUDA 12.4/12.6 wheels have no Blackwell
# kernels, and CUDA 13.0 wheels need a newer driver than the L40S machines run.

ENV PYTHONUNBUFFERED=1

# opencv-python (pulled in by orena-focus) needs these libraries at import time,
# and the runtime image does not include them. Installed as root, before USER.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libxcb1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

COPY --chown=user:user requirements.txt /opt/app/

# The image's Python is marked externally managed (PEP 668), hence
# --break-system-packages. With --user everything goes to ~/.local anyway.
RUN python -m pip install \
    --user \
    --break-system-packages \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

# Fail the build if pip replaced torch with a build that cannot run on both
# GPUs (e.g. the default CUDA 13.0 wheel from PyPI). There is no GPU during
# docker build, so the compiled arch flags are read directly.
RUN python -c "import torch; \
flags = torch._C._cuda_getArchFlags() or ''; \
cuda = torch.version.cuda or ''; \
print('torch', torch.__version__, '| cuda', cuda, '|', flags); \
assert cuda.startswith('12.8'), 'torch was replaced by a CUDA ' + cuda + ' build; the L40S driver cannot run it'; \
assert 'sm_120' in flags and 'sm_86' in flags, 'torch build misses a required GPU architecture: ' + repr(flags)"

# Base model weights (~17 GB). The container runs with --network none, so they
# are downloaded here at build time and baked into the image, in their own layer
# so that code changes do not invalidate it.
# Xet is disabled and the download limited to 2 workers because the parallel
# downloader kept crashing the Docker daemon on our build machine.
ARG MODEL_ID=Qwen/Qwen3-VL-8B-Instruct
ENV HF_HUB_DISABLE_XET=1 HF_HUB_DISABLE_PROGRESS_BARS=1
RUN python -m pip install --user --break-system-packages --no-cache-dir "huggingface_hub" && python -c "from huggingface_hub import snapshot_download; snapshot_download('${MODEL_ID}', local_dir='/opt/app/resources/model', max_workers=2, ignore_patterns=['*.pth','original/*','*.gguf','*.md'])" && rm -rf /opt/app/resources/model/.cache && du -sh /opt/app/resources/model

# LoRA adapter (~330 MB), merged into the base model at startup. Separate layer
# so that editing the code does not re-upload it. See resources/adapter/README.md.
COPY --chown=user:user resources/adapter/ /opt/app/resources/adapter/

COPY --chown=user:user resources/surgledger/ /opt/app/resources/surgledger/

COPY --chown=user:user inference.py /opt/app/

ENTRYPOINT ["python", "inference.py"]

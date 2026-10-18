# Audar MLX conversion tool

Unofficial tooling to convert [AudarAI's Audar-ASR-V1-Turbo](https://huggingface.co/audarai/Audar-ASR-V1-Turbo) to 8-bit MLX and run it on an Apple Silicon Mac.

**This repository contains conversion code only. No model weights, tokenizer files, or Audar configuration files are distributed here.** The converter downloads directly from Audar's official Hugging Face repository, then creates the MLX files on your computer.

AudarAI developed the original model and its training/adaptation. This project provides conversion and loading tooling; it is not affiliated with or endorsed by AudarAI.

## License first

Read the [AudarAI Community License v1.0](https://www.audarai.com/license/audarai-community-license-v1.0/) before downloading or using the model. The acceptance flag below records your explicit choice to proceed; it does not replace the license or any upstream access requirements.

The model permits research and qualifying limited commercial use subject to its conditions. The converted files remain subject to Audar's license. This repository grants no permission to rehost them, remove notices, or bypass access controls. Publishing this tool is not a grant of rights to the model.

The MIT license in this repository covers this project's original tooling only. MLX Audio and other dependencies retain their own licenses.

## Install

Requires an Apple Silicon Mac, Python 3.12, and Git. Leave several GB of memory and disk space available. The original checkpoint download is approximately 4.7 GB; conversion uses additional disk space temporarily.

```sh
git clone https://github.com/abdulrhmanG-alahmadi/audar-mlx.git
cd audar-mlx
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Convert on your Mac

Only pass the acceptance flag after reading and agreeing to Audar's license:

```sh
python audar_mlx.py convert --accept-audar-license --output models/audar-8bit
```

The source is pinned to revision `371428bea487c7aec82b27dc21f8d4324002e98e`. Downloads exclude nested serving builds and remote Python files. Conversion removes custom processor routing metadata so MLX Audio can use its built-in processor. An existing output directory is never overwritten.

## Transcribe

```sh
python audar_mlx.py transcribe --model models/audar-8bit clip.wav
```

Start with speech clips under 30 seconds. This example caps generation at 256 tokens and does not provide streaming or long-recording chunking. It has not been validated against Audar's published benchmarks; quantization and decoding differences may change accuracy.

## Why the loader patch is needed

The pinned MLX Audio implementation drops `lm_head.weight` during Qwen3-ASR sanitization. Audar uses an untied output layer, so losing that tensor causes incorrect output. The tool preserves the actual tensor during both conversion and loading. **Use this tool for loading too**; the converted model should not be assumed to work with an unpatched loader.

The patch wraps the upstream sanitizer instead of distributing a copy of its implementation. MLX Audio is pinned to the revision used by the existing local conversion. Upgrading requires checking whether the patch is still necessary.

## Checks

```sh
python -m unittest discover -s tests -v
```

These checks exercise output-layer preservation, repeated patching, output-directory protection, and license acceptance without loading model weights.

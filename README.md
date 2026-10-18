# Audar MLX conversion tool

Unofficial tooling to convert [AudarAI's Audar-ASR-V1-Turbo](https://huggingface.co/audarai/Audar-ASR-V1-Turbo) to 8-bit MLX and run it on an Apple Silicon Mac.

**This repository contains conversion code only. No model weights, tokenizer files, or Audar configuration files are distributed here.** The converter downloads directly from Audar's official Hugging Face repository, then creates the MLX files on your computer.

AudarAI developed the original model and its training/adaptation. This project provides conversion and loading tooling; it is not affiliated with or endorsed by AudarAI.

## What to download

1. **Get this tool:** [Download ZIP](https://github.com/abdulrhmanG-alahmadi/audar-mlx/archive/refs/heads/main.zip), unzip it, then open Terminal in the extracted `audar-mlx-main` folder. Alternatively, use the Git commands under Install.
2. **Original model:** [Audar-ASR-V1-Turbo on Hugging Face](https://huggingface.co/audarai/Audar-ASR-V1-Turbo). You do not need to manually download individual files: the conversion command below downloads the required checkpoint files directly from Audar.
3. **Your MLX model:** there is no ready-made model download in this repository. After conversion, your model is saved locally in `models/audar-8bit/` inside the tool's folder (about 2.6 GB).

The flow is: download this tool → install its dependencies → accept Audar's license → run conversion → transcribe your audio.

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

**If you downloaded the ZIP:** skip `git clone` and use `cd` to enter the extracted folder instead, then run the commands starting with `python3.12 -m venv .venv`. Git is still needed to install the pinned MLX Audio dependency.

If `python3.12` or `git` is not found, install Python 3.12 and Git before continuing. This tool requires Apple Silicon (M1 or newer), not an Intel Mac or a Windows PC.

## Convert on your Mac

Only pass the acceptance flag after reading and agreeing to Audar's license:

```sh
python audar_mlx.py convert --accept-audar-license --output models/audar-8bit
```

Keep Terminal open until it prints `Local model saved to ...`. The first run downloads approximately 4.7 GB and then converts it; duration depends on your internet connection and Mac. You only need to convert once. Keep the generated model folder on your own machine.

The source is pinned to revision `371428bea487c7aec82b27dc21f8d4324002e98e`. Downloads exclude nested serving builds and remote Python files. Conversion removes custom processor routing metadata so MLX Audio can use its built-in processor. An existing output directory is never overwritten.

## Transcribe

Put a short audio file named `clip.wav` in the tool's folder, then run:

```sh
python audar_mlx.py transcribe --model models/audar-8bit clip.wav
```

Your transcript prints in Terminal. You can also replace `clip.wav` with a full path; put paths containing spaces inside quotes:

```sh
python audar_mlx.py transcribe --model models/audar-8bit "/Users/yourname/Downloads/my recording.wav"
```

When you return later, open Terminal in the tool's folder and run `source .venv/bin/activate` before transcribing. No repeat download or conversion is needed.

Start with speech clips under 30 seconds. This example caps generation at 256 tokens and does not provide streaming or long-recording chunking. It has not been validated against Audar's published benchmarks; quantization and decoding differences may change accuracy.

## Why the loader patch is needed

The pinned MLX Audio implementation drops `lm_head.weight` during Qwen3-ASR sanitization. Audar uses an untied output layer, so losing that tensor causes incorrect output. The tool preserves the actual tensor during both conversion and loading. **Use this tool for loading too**; the converted model should not be assumed to work with an unpatched loader.

The patch wraps the upstream sanitizer instead of distributing a copy of its implementation. MLX Audio is pinned to the revision used by the existing local conversion. Upgrading requires checking whether the patch is still necessary.

## Checks

```sh
python -m unittest discover -s tests -v
```

These checks exercise output-layer preservation, repeated patching, output-directory protection, and license acceptance without loading model weights.

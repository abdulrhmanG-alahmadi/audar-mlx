"""Local conversion and transcription tooling; contains no Audar model artifacts."""

import argparse
import json
import tempfile
from pathlib import Path

SOURCE = "audarai/Audar-ASR-V1-Turbo"
REVISION = "371428bea487c7aec82b27dc21f8d4324002e98e"
LICENSE_URL = "https://www.audarai.com/license/audarai-community-license-v1.0/"


def patch_loader(model_class=None):
    if model_class is None:
        from mlx_audio.stt.models.qwen3_asr.qwen3_asr import Qwen3ASRModel

        model_class = Qwen3ASRModel
    original = model_class.sanitize
    if getattr(original, "keeps_audar_head", False):
        return

    def sanitize(weights):
        output = original(dict(weights))
        for name in ("thinker.lm_head.weight", "lm_head.weight"):
            if name in weights:
                output["lm_head.weight"] = weights[name]
        return output

    sanitize.keeps_audar_head = True
    model_class.sanitize = staticmethod(sanitize)


def convert_model(destination):
    destination = Path(destination).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}; choose a new directory.")
    from huggingface_hub import snapshot_download
    from mlx_audio.convert import convert
    from safetensors import safe_open

    patch_loader()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as staging:
        source = Path(staging) / "source"
        built = Path(staging) / "model"
        snapshot_download(
            SOURCE,
            revision=REVISION,
            local_dir=source,
            allow_patterns=["*.safetensors", "*.json", "*.txt", "*.jinja"],
            ignore_patterns=["*/*"],
        )
        convert(str(source), str(built), quantize=True, q_bits=8, model_domain="stt")
        for name in ("config.json", "preprocessor_config.json"):
            config_path = built / name
            config = json.loads(config_path.read_text())
            config.pop("auto_map", None)
            config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
        with safe_open(built / "model.safetensors", framework="numpy") as weights:
            if not {"lm_head.weight", "lm_head.scales"}.issubset(weights.keys()):
                raise RuntimeError("Converted model is missing its quantized output layer.")
        provenance = {"source": SOURCE, "revision": REVISION, "license_url": LICENSE_URL}
        (built / "conversion-source.json").write_text(json.dumps(provenance, indent=2) + "\n")
        built.rename(destination)
    print(f"Local model saved to {destination}. Audar's license still applies.")


def transcribe(model_path, audio_path):
    import numpy as np
    from mlx_audio.stt import load
    from mlx_audio.utils import load_audio

    patch_loader()
    model = load(str(Path(model_path).expanduser().resolve()))
    audio = np.asarray(load_audio(audio_path, sample_rate=16000))
    result = model.generate(audio, max_tokens=256)
    print(result.text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    conversion = commands.add_parser("convert", help="Download from Audar and convert locally")
    conversion.add_argument("--output", required=True)
    conversion.add_argument("--accept-audar-license", action="store_true")
    transcription = commands.add_parser("transcribe", help="Transcribe a short audio file locally")
    transcription.add_argument("--model", required=True)
    transcription.add_argument("audio")
    arguments = parser.parse_args()
    if arguments.command == "convert":
        if not arguments.accept_audar_license:
            parser.error(f"Read {LICENSE_URL}; pass --accept-audar-license only if you agree.")
        convert_model(arguments.output)
    else:
        transcribe(arguments.model, arguments.audio)


if __name__ == "__main__":
    main()

"""
generate_samples_binary.py
Piper-binary-based TTS clip generator for Python 3.12+ compatibility.

Drop-in replacement for piper-sample-generator's generate_samples when
piper-phonemize cannot be installed (no wheel for Python 3.12).

Auto-downloads the piper binary and a default English voice model on first use.
"""
import itertools
import logging
import os
import subprocess
import tarfile
import urllib.request
import uuid

logger = logging.getLogger(__name__)

_DEFAULT_PIPER_DIR = os.environ.get("PIPER_DIR", "/content/piper")
_DEFAULT_PIPER_BINARY = os.environ.get(
    "PIPER_BINARY", os.path.join(_DEFAULT_PIPER_DIR, "piper")
)
_DEFAULT_PIPER_MODEL = os.environ.get(
    "PIPER_VOICE_MODEL",
    os.path.join(_DEFAULT_PIPER_DIR, "en_US-lessac-medium.onnx"),
)

_PIPER_RELEASE_URL = (
    "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/"
    "piper_linux_x86_64.tar.gz"
)
_VOICE_MODEL_URL = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/"
    "en/en_US/lessac/medium/en_US-lessac-medium.onnx"
)
_VOICE_MODEL_JSON_URL = _VOICE_MODEL_URL + ".json"


def _download(url: str, dest: str) -> None:
    logger.info("Downloading %s -> %s", url, dest)
    print(f"  Downloading {os.path.basename(dest)} ...")
    urllib.request.urlretrieve(url, dest)


def _ensure_piper(piper_binary: str, piper_model: str):
    """Download piper binary and voice model if not already present."""
    piper_dir = os.path.dirname(piper_binary)
    os.makedirs(piper_dir, exist_ok=True)

    if not os.path.isfile(piper_binary):
        tar_path = os.path.join(piper_dir, "_piper.tar.gz")
        _download(_PIPER_RELEASE_URL, tar_path)
        with tarfile.open(tar_path) as tf:
            tf.extractall(os.path.dirname(piper_dir))
        os.remove(tar_path)
        if not os.path.isfile(piper_binary):
            raise RuntimeError(
                f"Piper binary not found at {piper_binary} after extraction."
            )
        os.chmod(piper_binary, 0o755)
        logger.info("Piper binary ready: %s", piper_binary)

    if not os.path.isfile(piper_model):
        _download(_VOICE_MODEL_URL, piper_model)
        _download(_VOICE_MODEL_JSON_URL, piper_model + ".json")
        logger.info("Piper voice model ready: %s", piper_model)

    return piper_binary, piper_model


def generate_samples(
    text,
    max_samples: int = 1000,
    batch_size: int = 50,
    noise_scales=None,
    noise_scale_ws=None,
    length_scales=None,
    output_dir: str = ".",
    auto_reduce_batch_size: bool = True,
    file_names=None,
    piper_binary: str = None,
    piper_model: str = None,
):
    """Generate TTS WAV clips using the piper binary.

    Matches the interface of piper-sample-generator's generate_samples so
    openwakeword/train.py can use it as a drop-in on Python 3.12+.
    """
    if noise_scales is None:
        noise_scales = [0.667]
    if noise_scale_ws is None:
        noise_scale_ws = [0.8]
    if length_scales is None:
        length_scales = [1.0]

    piper_binary, piper_model = _ensure_piper(
        piper_binary or _DEFAULT_PIPER_BINARY,
        piper_model or _DEFAULT_PIPER_MODEL,
    )

    os.makedirs(output_dir, exist_ok=True)

    texts = [text] if isinstance(text, str) else list(text)
    combos = list(itertools.product(texts, noise_scales, noise_scale_ws, length_scales))

    generated = 0
    idx = 0
    while generated < max_samples:
        txt, ns, nsw, ls = combos[idx % len(combos)]
        idx += 1

        out_name = (
            file_names[generated]
            if (file_names and generated < len(file_names))
            else uuid.uuid4().hex + ".wav"
        )
        out_path = os.path.join(output_dir, out_name)

        cmd = [
            piper_binary,
            "--model", piper_model,
            "--noise-scale", str(ns),
            "--noise-w", str(nsw),
            "--length-scale", str(ls),
            "--output-file", out_path,
        ]

        result = subprocess.run(cmd, input=txt.encode(), capture_output=True)
        if result.returncode == 0 and os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
            generated += 1
        else:
            logger.warning(
                "Piper failed for input %r (exit %d): %s",
                txt, result.returncode, result.stderr.decode()[:300],
            )
            if idx > max(max_samples * 5, 50) and generated == 0:
                raise RuntimeError(
                    f"Piper failed to generate any samples.\n"
                    f"Binary: {piper_binary}\nModel: {piper_model}\n"
                    f"stderr: {result.stderr.decode()}"
                )

    logger.info("Generated %d clips in %s", generated, output_dir)
    return generated

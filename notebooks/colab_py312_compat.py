"""
Python 3.12 / modern torchaudio compatibility patches for Google Colab.

Add this as a code cell immediately AFTER the pip install cell and BEFORE
the imports cell in automatic_model_training.ipynb, then run it once.
"""
import pathlib, sys, textwrap

def _patch(path_glob, old, new):
    for p in pathlib.Path(sys.prefix).rglob(path_glob):
        src = p.read_text()
        if old in src and new not in src:
            p.write_text(src.replace(old, new))
            print(f"Patched: {p}")
            return
        elif new in src:
            print(f"Already patched: {p}")
            return
    print(f"File not found for glob: {path_glob}")

# 1. torchaudio >= 2.1 removed set_audio_backend
_patch(
    "torch_audiomentations/utils/io.py",
    'torchaudio.set_audio_backend("soundfile")',
    'if hasattr(torchaudio, "set_audio_backend"): torchaudio.set_audio_backend("soundfile")',
)

# 2. scipy >= 1.15 renamed sph_harm → sph_harm_y
_patch(
    "acoustics/directivity.py",
    'from scipy.special import sph_harm  # pylint: disable=no-name-in-module',
    textwrap.dedent("""\
        try:
            from scipy.special import sph_harm  # pylint: disable=no-name-in-module
        except ImportError:
            from scipy.special import sph_harm_y as sph_harm  # scipy >= 1.15\
    """),
)

print("Done.")

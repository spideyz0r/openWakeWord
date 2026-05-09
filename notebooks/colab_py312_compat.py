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

# 3. torchaudio >= 2.6 removed torchaudio.info — patch torch_audiomentations to use soundfile
_torchaudio_info_patch_old = '''    info = torchaudio.info(file_path)
        # Deal with backwards-incompatible signature change.
        # See https://github.com/pytorch/audio/issues/903 for more information.
        if type(info) is tuple:
            si, ei = info
            num_samples = si.length
            sample_rate = si.rate
        else:
            num_samples = info.num_frames
            sample_rate = info.sample_rate
        return num_samples, sample_rate'''

_torchaudio_info_patch_new = '''    try:
            info = torchaudio.info(file_path)
            if type(info) is tuple:
                si, ei = info
                num_samples = si.length
                sample_rate = si.rate
            else:
                num_samples = info.num_frames
                sample_rate = info.sample_rate
        except AttributeError:
            import soundfile as _sf
            _info = _sf.info(str(file_path))
            num_samples = _info.frames
            sample_rate = _info.samplerate
        return num_samples, sample_rate'''

_patch(
    "torch_audiomentations/utils/io.py",
    _torchaudio_info_patch_old,
    _torchaudio_info_patch_new,
)

print("Done.")

import io, numpy as np, soundfile as sf
from parakeet_vllm.audio import decode_to_16k_mono

def _wav_bytes(sr, secs=0.5, ch=1):
    n = int(sr*secs)
    x = (0.1*np.sin(2*np.pi*220*np.arange(n)/sr)).astype("float32")
    if ch == 2:
        x = np.stack([x, x], axis=1)
    buf = io.BytesIO(); sf.write(buf, x, sr, format="WAV"); return buf.getvalue()

def test_resamples_to_16k():
    out = decode_to_16k_mono(_wav_bytes(8000))
    assert out.dtype == np.float32 and out.ndim == 1
    assert abs(len(out) - 8000) < 50   # 0.5s @16k

def test_downmixes_stereo():
    out = decode_to_16k_mono(_wav_bytes(16000, ch=2))
    assert out.ndim == 1

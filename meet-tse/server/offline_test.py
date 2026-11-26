import sys
from math import gcd
import numpy as np, soundfile as sf
from scipy.signal import resample_poly
from model_adapter import Runner

def load16k(p):
    w, sr = sf.read(p, dtype="float32", always_2d=True)
    w = w.mean(1)
    if sr != 16000:
        g = gcd(sr, 16000)
        w = resample_poly(w, 16000 // g, sr // g).astype("float32")
    return w

ck, en, mx = sys.argv[1:4]
r = Runner(ck)
r.set_enrollment(load16k(en)[:15 * 16000])
mix = load16k(mx)
out = r(mix)
sf.write("mix_16k.wav", mix, 16000)
sf.write("model_out.wav", out, 16000)
db = lambda x: 10 * np.log10(np.mean(x.astype(np.float64) ** 2) + 1e-12)
print(f"in {db(mix):.1f} dBFS, out {db(out):.1f} dBFS, {len(mix)/16000:.1f} s")
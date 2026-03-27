"""
ssm_s2p.py — Touchstone .s2p file I/O and forward simulation of dummy structures.

Forward simulators for Open and Short are here (not inside any model) because
they depend only on the pad/lead parameters, not on any intrinsic model.
"""
from __future__ import annotations
import numpy as np
from .ssm_core import open_elem_Y, short_lead_Z, y_to_s_single
from .ssm_deembedding import build_Y_pad, build_Z_ser


# ── Touchstone write ──────────────────────────────────────────────────────────

def write_s2p(freq_hz: np.ndarray, S: np.ndarray,
              title: str = "", params: dict | None = None) -> bytes:
    """
    Serialize S-parameter data to Touchstone .s2p format.

    Format: # Hz S DB R 50
    Each row: freq  S11_dB S11_ang  S21_dB S21_ang  S12_dB S12_ang  S22_dB S22_ang

    Header comment lines list every key/value from `params`.
    """
    lines = [f"! Forward-simulated: {title}"]
    if params:
        for k, v in params.items():
            lines.append(f"!   {k} = {v}")
    lines.append("# Hz S DB R 50")
    for i, f in enumerate(freq_hz):
        parts = [f"{f:.0f}"]
        for r, c in [(0,0), (1,0), (0,1), (1,1)]:
            s = S[i, r, c]
            db  = 20*np.log10(abs(s) + 1e-30)
            ang = np.degrees(np.angle(s))
            parts += [f"{db:.8f}", f"{ang:.8f}"]
        lines.append(" ".join(parts))
    return "\n".join(lines).encode("utf-8")


# ── Touchstone read ───────────────────────────────────────────────────────────

def parse_s2p_bytes(raw: bytes):
    """
    Parse a .s2p file from raw bytes.
    Returns (freq_hz, S[N,2,2], z0).
    Handles MA, DB, RI formats; Hz/kHz/MHz/GHz frequency units.
    """
    content = raw.decode("utf-8", errors="ignore")
    freq_unit, fmt, z0 = "hz", "ma", 50.0
    data_lines = []

    for line in content.splitlines():
        s = line.strip()
        if not s or s.startswith("!"): continue
        if s.startswith("#"):
            parts = s[1:].lower().split()
            for i, p in enumerate(parts):
                if p in ("hz","khz","mhz","ghz"):   freq_unit = p
                elif p in ("ma","db","ri"):           fmt = p
                elif p == "r" and i+1 < len(parts):
                    try: z0 = float(parts[i+1])
                    except: pass
            continue
        data_lines.append(s)

    vals = np.array([float(x) for x in " ".join(data_lines).split()])
    n = len(vals)//9
    vals = vals[:n*9].reshape(n, 9)
    scale = {"hz":1.0, "khz":1e3, "mhz":1e6, "ghz":1e9}[freq_unit]
    freq  = vals[:,0] * scale

    def to_c(ca, cb):
        a, b = vals[:,ca], vals[:,cb]
        if fmt == "db": return 10**(a/20.0) * np.exp(1j*np.deg2rad(b))
        if fmt == "ma": return a * np.exp(1j*np.deg2rad(b))
        return a + 1j*b   # ri

    S = np.zeros((n, 2, 2), dtype=complex)
    for (r,c),(ca,cb) in zip([(0,0),(1,0),(0,1),(1,1)], [(1,2),(3,4),(5,6),(7,8)]):
        S[:,r,c] = to_c(ca, cb)
    return freq, S, z0


def interpolate_s2f(f_src, S_src, f_tgt):
    """Interpolate S-parameter array from f_src grid to f_tgt grid."""
    S_out = np.zeros((len(f_tgt), 2, 2), dtype=complex)
    for r in range(2):
        for c in range(2):
            s = S_src[:,r,c]
            S_out[:,r,c] = (np.interp(f_tgt, f_src, s.real) +
                            1j*np.interp(f_tgt, f_src, s.imag))
    return S_out


# ── Forward simulators for dummy structures ───────────────────────────────────

def simulate_open(p: dict, freq: np.ndarray, z0: float = 50.0) -> np.ndarray:
    """
    Forward-simulate Open dummy S-parameters.
    Circuit: Y_pad only (no series leads, DUT replaced by open).

    p must contain: Cpbe/ce/bc and optional _mode/_extra.
    """
    N = len(freq)
    S = np.zeros((N, 2, 2), dtype=complex)
    for i, w in enumerate(2.0*np.pi*freq):
        Y = build_Y_pad(p, w)
        S[i] = y_to_s_single(Y, z0)
    return S


def simulate_short(p: dict, freq: np.ndarray, z0: float = 50.0) -> np.ndarray:
    """
    Forward-simulate Short dummy S-parameters.
    Circuit: Y_pad + inv(Z_ser)  (DUT terminals shorted → Z_DUT=0).

    p must contain all pad and lead parameters.
    """
    N = len(freq)
    S = np.zeros((N, 2, 2), dtype=complex)
    for i, w in enumerate(2.0*np.pi*freq):
        Y_pad = build_Y_pad(p, w)
        Z_ser = build_Z_ser(p, w)
        try:    Y_ser = np.linalg.inv(Z_ser)
        except: Y_ser = np.zeros((2,2), dtype=complex)
        S[i] = y_to_s_single(Y_pad + Y_ser, z0)
    return S

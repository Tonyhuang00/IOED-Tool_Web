"""
ssm_core.py — Shared RF math utilities and low-level element helpers.

All functions are pure numpy (no Streamlit). Import from every other SSM module.
"""
from __future__ import annotations
import hashlib, json
import numpy as np


# ── Y / Z / S conversions ─────────────────────────────────────────────────────

def s_to_y(S, z0=50.0):
    s11, s12, s21, s22 = S[:,0,0], S[:,0,1], S[:,1,0], S[:,1,1]
    d = (1+s11)*(1+s22) - s12*s21
    Y = np.zeros_like(S)
    Y[:,0,0] = ((1-s11)*(1+s22)+s12*s21) / (d*z0)
    Y[:,0,1] = -2*s12 / (d*z0)
    Y[:,1,0] = -2*s21 / (d*z0)
    Y[:,1,1] = ((1+s11)*(1-s22)+s12*s21) / (d*z0)
    return Y

def _inv2(M):
    out = np.zeros_like(M)
    for i in range(len(M)):
        try:    out[i] = np.linalg.inv(M[i])
        except: out[i] = np.full((2,2), np.nan+0j)
    return out

y_to_z = _inv2   # Z = Y⁻¹
z_to_y = _inv2   # Y = Z⁻¹

def y_to_s_single(Y, z0=50.0):
    Yn = Y * z0; I = np.eye(2)
    try:    return np.dot(I - Yn, np.linalg.inv(I + Yn))
    except: return np.full((2,2), np.nan+0j)

def y_to_s_batch(Y, z0=50.0):
    S = np.zeros_like(Y); I = np.eye(2)
    for i in range(len(Y)):
        yn = Y[i] * z0
        try:    S[i] = np.dot(I - yn, np.linalg.inv(I + yn))
        except: S[i] = np.full((2,2), np.nan+0j)
    return S


# ── Statistics helpers ─────────────────────────────────────────────────────────

def safe_median(arr, n=None):
    a = arr[:n] if n is not None else arr
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if len(a) > 0 else 0.0

def strict_freq_check(f_dut, f_dummy, label):
    if len(f_dut) != len(f_dummy) or not np.allclose(f_dut, f_dummy, rtol=1e-5):
        raise ValueError(f"DUT and {label} frequency grids differ.")


# ── Extended element admittance / impedance ───────────────────────────────────
# Used for Open (pad cap + optional secondary parasitic) and Short (lead + optional Cpar).

def open_elem_Y(C, mode, extra, w):
    """
    Admittance of one pad capacitor at angular frequency w.

    mode="None"       → Y = jωC
    mode="Parallel L" → Y = jωC + 1/(jωL)   [resonance at 1/√LC]
    mode="Series L"   → Y = jωC / (1 - ω²LC)  [series resonance]
    mode="Series R"   → Y = jωC / (1 + jωRC)   [lossy cap, adds Re(Y)]
    """
    if mode == "Parallel L" and extra > 0:
        return 1j*w*C + 1.0/(1j*w*extra + 1e-60)
    if mode == "Series L" and extra > 0:
        denom = 1.0 - w**2 * extra * C
        if abs(denom) < 1e-10: denom = 1e-10
        return 1j*w*C / denom
    if mode == "Series R" and extra > 0:
        return 1j*w*C / (1.0 + 1j*w*extra*C)
    return 1j*w*C   # pure cap (default)

def short_lead_Z(R, L, Cpar, w):
    """
    Impedance of one short-circuit lead at angular frequency w.

    Cpar=0  → Z = R + jωL
    Cpar>0  → Z = (R+jωL) ∥ (1/jωCpar)   [parallel tank]
    """
    Z = R + 1j*w*L
    if Cpar > 0:
        return 1.0 / (1.0/Z + 1j*w*Cpar)
    return Z


# ── Smith chart grid ──────────────────────────────────────────────────────────

def extended_smith_grid(max_r=1.0):
    """Return list of Plotly traces forming a Smith chart background."""
    import plotly.graph_objects as go
    traces = []; t = np.linspace(0, 2*np.pi, 500)
    sk = dict(mode="lines", showlegend=False, hoverinfo="skip")
    for ro in np.arange(1.0, max_r+0.5, 1.0):
        lw  = 1.6 if ro == 1.0 else 0.9
        col = "rgba(60,60,60,0.85)" if ro == 1.0 else "rgba(170,170,170,0.6)"
        traces.append(go.Scatter(x=np.cos(t)*ro, y=np.sin(t)*ro,
                                 line=dict(color=col, width=lw), **sk))
    traces.append(go.Scatter(x=[-max_r, max_r], y=[0., 0.],
                             line=dict(color="rgba(100,100,100,0.6)", width=0.8), **sk))
    gray = "rgba(155,155,155,0.5)"
    for r in [0.0, 0.2, 0.5, 1.0, 2.0, 5.0]:
        cx_ = r/(r+1); rad = 1.0/(r+1)
        xc = cx_ + rad*np.cos(t); yc = rad*np.sin(t)
        mg = np.sqrt(xc**2+yc**2); xc[mg>max_r]=np.nan; yc[mg>max_r]=np.nan
        traces.append(go.Scatter(x=xc, y=yc, line=dict(color=gray, width=0.8), **sk))
    for x in [0.2, 0.5, 1.0, 2.0, 5.0]:
        for sign in [1, -1]:
            xv = sign*x; rad_x = abs(1.0/xv)
            xc = 1.0 + rad_x*np.cos(t); yc = (1.0/xv) + rad_x*np.sin(t)
            mg = np.sqrt(xc**2+yc**2); xc[mg>max_r]=np.nan; yc[mg>max_r]=np.nan
            traces.append(go.Scatter(x=xc, y=yc, line=dict(color=gray, width=0.8), **sk))
    return traces


# ── Misc ──────────────────────────────────────────────────────────────────────

def params_hash(p: dict) -> str:
    try:
        return hashlib.md5(
            json.dumps({k: round(float(v), 15) for k, v in p.items()},
                       sort_keys=True).encode()
        ).hexdigest()
    except:
        return ""

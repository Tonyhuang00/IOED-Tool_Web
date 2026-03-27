"""
models/cheng.py — Cheng (2022) T-topology and π-topology HBT small-signal models.

Reference: Cheng et al., Microelectronics Journal vol. 121, 2022
           Eqs. 13, 16, 18–19, 21–22, 26–33

Both models share the same two-step extrinsic extraction (Step 2) that produces
Y_ex2, then diverge in Step 3.  Each is its own class so the registry and UI
treat them independently, but shared logic lives in module-level helpers below.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st

from ..ssm_core       import (y_to_z, z_to_y, y_to_s_single,
                               safe_median, params_hash,
                               extended_smith_grid)
from ..ssm_deembedding import build_Y_pad, build_Z_ser
from .base_ui         import (render_smith_chart, smith_scale_controls,
                               sync_pad_from_preov, PAD_SPECS, ssm_residual)
from . import AbstractSSMModel


# ════════════════════════════════════════════════════════════════════════════════
# Shared Step-2 helpers
# ════════════════════════════════════════════════════════════════════════════════

def _step2_T(Y_ex1, freq, n_low):
    """
    Cheng [Eqs. 13, 22] — Extract Cbex (T variant) and Cbcx.

    Cbex_T = Im(Y11 + Y12) / ω  [Eq. 13]

    Cbcx = −[Im(Yms)·Re(YL) − Re(Yms)·Im(YL)] / [ω · denominator]  [Eq. 22]
    where  Yms = Y12+Y22,  YL = det(Y_ex2),  Ytot = sum(Yij)
    """
    omega = 2.0*np.pi*freq

    # [Eq. 13] Cbex from low-frequency Im(Y11+Y12)/ω
    Cbex_arr = np.imag(Y_ex1[:,0,0] + Y_ex1[:,0,1]) / omega
    Cbex = safe_median(Cbex_arr, n_low)

    # Peel Cbex to get Y_ex2
    Y_ex2 = Y_ex1.copy()
    for i, w in enumerate(omega):
        Y_ex2[i,0,0] -= 1j*w*Cbex

    # [Eq. 22] Cbcx
    Yms   = Y_ex2[:,0,1] + Y_ex2[:,1,1]
    YL    = Y_ex2[:,0,0]*Y_ex2[:,1,1] - Y_ex2[:,0,1]*Y_ex2[:,1,0]
    Ytot  = Y_ex2[:,0,0] + Y_ex2[:,0,1] + Y_ex2[:,1,0] + Y_ex2[:,1,1]
    num   = np.imag(Yms)*np.real(YL) - np.real(Yms)*np.imag(YL)
    den   = np.real(Yms)*np.real(Ytot) + np.imag(Ytot)*np.imag(Yms)
    with np.errstate(divide="ignore", invalid="ignore"):
        Cbcx_arr = -np.where(np.abs(den) > 1e-40, num/(omega*den), np.nan)
    n0, n1 = len(freq)//4, 3*len(freq)//4
    Cbcx = safe_median(Cbcx_arr[n0:n1])

    return ({"Cbex": Cbex, "Cbcx": Cbcx},
            {"Cbex_arr": Cbex_arr, "Cbcx_arr": Cbcx_arr, "Y_ex2": Y_ex2})


def _step2_pi(Y_ex1, freq, n_low):
    """
    Cheng [Eqs. 26–28] — Extract Cbex (π variant) and Cbcx.

    B = Y12+Y22,  C = Y11+Y21
    Cbex_π = [Re(B)·Re(C) + Im(B)·Im(C)] / [ω·Im(B)]  [Eqs. 26–28]
    Cbcx same formula as T-topology [Eq. 22] applied on Y_ex2.
    """
    omega = 2.0*np.pi*freq
    B = Y_ex1[:,0,1] + Y_ex1[:,1,1]
    C = Y_ex1[:,0,0] + Y_ex1[:,1,0]

    # [Eqs. 26–28]
    with np.errstate(divide="ignore", invalid="ignore"):
        Cbex_arr = np.where(
            np.abs(np.imag(B)) > 1e-40,
            (np.real(B)*np.real(C) + np.imag(B)*np.imag(C)) / (omega*np.imag(B)),
            np.nan)
    Cbex = safe_median(Cbex_arr, n_low)

    Y_ex2 = Y_ex1.copy()
    for i, w in enumerate(omega):
        Y_ex2[i,0,0] -= 1j*w*Cbex

    # Cbcx [Eq. 22] — same as T
    Yms  = Y_ex2[:,0,1] + Y_ex2[:,1,1]
    YL   = Y_ex2[:,0,0]*Y_ex2[:,1,1] - Y_ex2[:,0,1]*Y_ex2[:,1,0]
    Ytot = Y_ex2[:,0,0] + Y_ex2[:,0,1] + Y_ex2[:,1,0] + Y_ex2[:,1,1]
    num  = np.imag(Yms)*np.real(YL) - np.real(Yms)*np.imag(YL)
    den  = np.real(Yms)*np.real(Ytot) + np.imag(Ytot)*np.imag(Yms)
    with np.errstate(divide="ignore", invalid="ignore"):
        Cbcx_arr = -np.where(np.abs(den) > 1e-40, num/(omega*den), np.nan)
    n0, n1 = len(freq)//4, 3*len(freq)//4
    Cbcx = safe_median(Cbcx_arr[n0:n1])

    return ({"Cbex": Cbex, "Cbcx": Cbcx},
            {"Cbex_arr": Cbex_arr, "Cbcx_arr": Cbcx_arr, "Y_ex2": Y_ex2})


def _step3_T(Y_ex2, freq, Cbcx, n_low):
    """
    Cheng [Eqs. 16, 29–33] — Extract intrinsic parameters for T-topology.

    Z_in = [Y_ex2 − Cbcx·shunt]⁻¹
    Zbe = Z12,   Zbc = Z22−Z21,   Zbi = Z11−Z12          [Eq. 16]
    α = (Z12−Z21) / (Z22−Z21),   α₀ = |α|ω→0             [Eq. 29]
    τB = √(U−1)/ω   where U = (α₀/|α|)²                  [Eq. 30]
    τC = −arctan(V/√(1−V²)) / (2ω)   where V=2ωτB/U      [Eq. 31]
    """
    omega = 2.0*np.pi*freq

    # Peel Cbcx to get intrinsic Y
    Y_in = Y_ex2.copy()
    for i, w in enumerate(omega):
        Ybcx = 1j*w*Cbcx
        Y_in[i,0,0] -= Ybcx; Y_in[i,0,1] += Ybcx
        Y_in[i,1,0] += Ybcx; Y_in[i,1,1] -= Ybcx

    Z_in = y_to_z(Y_in)  # [Eq. 16] ↓
    Zbe_arr = Z_in[:,0,1]
    Zbc_arr = Z_in[:,1,1] - Z_in[:,1,0]
    Zbi_arr = Z_in[:,0,0] - Z_in[:,0,1]

    with np.errstate(divide="ignore", invalid="ignore"):
        Ybe_arr = 1.0 / (Zbe_arr + 1e-40)
        Ybc_arr = 1.0 / (Zbc_arr + 1e-40)
        alpha_arr = (Z_in[:,0,1] - Z_in[:,1,0]) / (Zbc_arr + 1e-40)  # [Eq. 29]

    Rbe_a = 1.0 / np.real(Ybe_arr).clip(1e-6)
    Cbe_a = np.imag(Ybe_arr) / omega
    Rbc_a = 1.0 / np.real(Ybc_arr).clip(1e-12)
    Cbc_a = np.imag(Ybc_arr) / omega
    Rbi_a = np.real(Zbi_arr)

    Rbe  = safe_median(Rbe_a, n_low)
    Cbe  = safe_median(Cbe_a, n_low)
    Rbc  = safe_median(Rbc_a, n_low)
    Cbc  = safe_median(Cbc_a, n_low)
    Rbi  = safe_median(Rbi_a, n_low)
    alpha0 = safe_median(np.abs(alpha_arr), n_low)   # [Eq. 29]

    # [Eq. 30] τB
    U_arr    = (alpha0 / (np.abs(alpha_arr) + 1e-30))**2
    tauB_arr = np.sqrt(np.maximum(U_arr - 1.0, 0.0)) / omega
    tauB     = safe_median(tauB_arr[n_low:])

    # [Eq. 31] τC
    with np.errstate(divide="ignore", invalid="ignore"):
        V_arr    = 2.0*omega*tauB / (U_arr + 1e-30)
        tauC_arr = -np.arctan(V_arr / np.sqrt(np.maximum(1.0 - V_arr**2, 1e-30))) / (2.0*omega)
    tauC = safe_median(tauC_arr[n_low:])

    params = dict(Rbi=Rbi, Rbe=Rbe, Cbe=Cbe, Rbc=Rbc, Cbc=Cbc,
                  alpha0=alpha0, tauB=tauB, tauC=tauC)
    arrays = dict(Rbe=Rbe_a, Cbe=Cbe_a, Rbc=Rbc_a, Cbc=Cbc_a,
                  alpha=alpha_arr, tauB=tauB_arr, tauC=tauC_arr)
    return params, arrays


def _step3_pi(Y_ex2, freq, Cbcx, n_low):
    """
    Zhang et al. [4] — Extract intrinsic parameters for π-topology.

    Z_bc = Z22−Z21
    gm = (Z12−Z21) / (Zbc·Z12)  →  Gm0 = |gm|,  τ = −∠gm / ω
    Ybe = (Z22−Z12) / (Z12·Zbc)  →  Rbe, Cbe
    Rbi = Re(Z11−Z12)
    """
    omega = 2.0*np.pi*freq

    Y_in = Y_ex2.copy()
    for i, w in enumerate(omega):
        Ybcx = 1j*w*Cbcx
        Y_in[i,0,0] -= Ybcx; Y_in[i,0,1] += Ybcx
        Y_in[i,1,0] += Ybcx; Y_in[i,1,1] -= Ybcx

    Z_in = y_to_z(Y_in)
    Z12  = Z_in[:,0,1]; Z21 = Z_in[:,1,0]
    Z11  = Z_in[:,0,0]; Z22 = Z_in[:,1,1]
    Zbc  = Z22 - Z21  # [from Eq. 16 shared]

    with np.errstate(divide="ignore", invalid="ignore"):
        Ybc_arr = 1.0 / (Zbc + 1e-40)
        gm_arr  = (Z12 - Z21) / ((Zbc + 1e-40)*(Z12 + 1e-40))
        Ybe_arr = (Z22 - Z12) / ((Z12 + 1e-40)*(Zbc + 1e-40))
        Rbi_a   = np.real(Z11 - Z12)

    Rbc_a = 1.0 / np.real(Ybc_arr).clip(1e-12)
    Cbc_a = np.imag(Ybc_arr) / omega
    Rbe_a = 1.0 / np.real(Ybe_arr).clip(1e-6)
    Cbe_a = np.imag(Ybe_arr) / omega
    Gm0_a = np.abs(gm_arr)
    with np.errstate(divide="ignore", invalid="ignore"):
        tau_a = -np.angle(gm_arr) / omega

    Rbi  = safe_median(Rbi_a, n_low)
    Rbc  = safe_median(Rbc_a, n_low); Cbc = safe_median(Cbc_a, n_low)
    Rbe  = safe_median(Rbe_a, n_low); Cbe = safe_median(Cbe_a, n_low)
    Gm0  = safe_median(Gm0_a, n_low); tau = safe_median(tau_a,  n_low)

    params = dict(Rbi=Rbi, Rbe=Rbe, Cbe=Cbe, Rbc=Rbc, Cbc=Cbc, Gm0=Gm0, tau=tau)
    arrays = dict(Rbe=Rbe_a, Cbe=Cbe_a, Rbc=Rbc_a, Cbc=Cbc_a, Gm0=Gm0_a, tau=tau_a)
    return params, arrays


# ════════════════════════════════════════════════════════════════════════════════
# Shared forward-simulator helpers
# ════════════════════════════════════════════════════════════════════════════════

def _sim_wrap(Y_int_fn, p, freq, z0):
    """Add extrinsic caps + pad/lead parasitics around the intrinsic Y matrix."""
    omega = 2.0*np.pi*freq
    S = np.zeros((len(freq), 2, 2), dtype=complex)
    for i, w in enumerate(omega):
        Y_in  = Y_int_fn(p, w)
        Ybcx  = 1j*w*p["Cbcx"]
        Ybex  = 1j*w*p["Cbex"]
        Y_ex  = (Y_in
                 + Ybcx*np.array([[1,-1],[-1,1]])
                 + Ybex*np.array([[1, 0],[ 0,0]]))
        Z_ser = build_Z_ser(p, w)
        try:    Y_tot = np.linalg.inv(np.linalg.inv(Y_ex) + Z_ser)
        except: Y_tot = np.zeros((2,2), dtype=complex)
        Y_pad = build_Y_pad(p, w)
        S[i]  = y_to_s_single(Y_tot + Y_pad, z0)
    return S


# ── Override UI specs (used by render_override_and_smith) ─────────────────────

_EXT_SPECS = [
    ("Cbex","Cbex",1e15,"fF","%.4f",0.1),
    ("Cbcx","Cbcx",1e15,"fF","%.4f",0.1),
]
_INT_T_SPECS = [
    ("Rbi",   "Rbi", 1.0, "Ω",  "%.4f", 0.1),
    ("Rbe",   "Rbe", 1.0, "Ω",  "%.3f", 1.0),
    ("Cbe",   "Cbe", 1e15,"fF", "%.4f", 0.1),
    ("Rbc",   "Rbc", 1e-3,"kΩ", "%.4f", 0.01),
    ("Cbc",   "Cbc", 1e15,"fF", "%.4f", 0.01),
    ("alpha0","α₀",  1.0, "",   "%.5f", 0.001),
    ("tauB",  "τB",  1e12,"ps", "%.4f", 0.01),
    ("tauC",  "τC",  1e12,"ps", "%.4f", 0.01),
]
_INT_PI_SPECS = [
    ("Rbi","Rbi", 1.0, "Ω",  "%.4f", 0.1),
    ("Rbe","Rbe", 1.0, "Ω",  "%.3f", 1.0),
    ("Cbe","Cbe", 1e15,"fF", "%.4f", 0.1),
    ("Rbc","Rbc", 1e-3,"kΩ", "%.4f", 0.01),
    ("Cbc","Cbc", 1e15,"fF", "%.4f", 0.01),
    ("Gm0","Gm0", 1e3, "mS", "%.4f", 0.01),
    ("tau","τ",   1e12,"ps", "%.4f", 0.01),
]


def _override_ui(fname, tK, calc_vals, int_specs, label):
    """Render the override expander for one Cheng topology."""
    all_specs = PAD_SPECS + _EXT_SPECS + int_specs
    sync_pad_from_preov(fname, tK, calc_vals)

    for key, _, scale, *_ in _EXT_SPECS + int_specs:
        sk = f"sim_{tK}_{key}_{fname}"
        if sk not in st.session_state:
            st.session_state[sk] = float(calc_vals.get(key, 0.0)) * scale

    with st.expander(f"✏️ Fine-tune {label} intrinsic/extrinsic parameters", expanded=False):
        if st.button(f"↩️ Reset {label} to calculated", key=f"rst_sim_{tK}_{fname}"):
            for key, _, scale, *_ in all_specs:
                st.session_state[f"sim_{tK}_{key}_{fname}"] = float(calc_vals.get(key, 0.0)) * scale
            st.rerun()

        st.markdown("**Pad Parasitics** *(auto-synced from pre-extraction override)*")
        for row_start in range(0, len(PAD_SPECS), 3):
            row = PAD_SPECS[row_start:row_start+3]
            for col_w, (key, lbl, sc, unit, fmt, step) in zip(st.columns(len(row)), row):
                col_w.number_input(f"{lbl} ({unit})" if unit else lbl,
                                   key=f"sim_{tK}_{key}_{fname}", format=fmt, step=step)

        st.markdown("**Extrinsic Caps**")
        for col_w, (key, lbl, sc, unit, fmt, step) in zip(st.columns(2), _EXT_SPECS):
            col_w.number_input(f"{lbl} ({unit})", key=f"sim_{tK}_{key}_{fname}", format=fmt, step=step)

        st.markdown("**Intrinsic**")
        for row_start in range(0, len(int_specs), 4):
            row = int_specs[row_start:row_start+4]
            for col_w, (key, lbl, sc, unit, fmt, step) in zip(st.columns(len(row)), row):
                col_w.number_input(f"{lbl} ({unit})" if unit else lbl,
                                   key=f"sim_{tK}_{key}_{fname}", format=fmt, step=step)

    all_p = {key: st.session_state.get(f"sim_{tK}_{key}_{fname}",
                                        float(calc_vals.get(key, 0.0))*scale) / scale
             for key, _, scale, *_ in all_specs}
    # Propagate extended open/short params
    for ek in ["Cpbe_mode","Cpbe_extra","Cpce_mode","Cpce_extra",
               "Cpbc_mode","Cpbc_extra","Cpar_Lb","Cpar_Lc","Cpar_Le"]:
        all_p[ek] = calc_vals.get(ek, "None" if "mode" in ek else 0.0)
    return all_p


# ════════════════════════════════════════════════════════════════════════════════
# ChengT
# ════════════════════════════════════════════════════════════════════════════════

class ChengT(AbstractSSMModel):
    """
    Cheng (2022) T-topology.
    Two-step extraction:  Step 2 → Cbex, Cbcx  |  Step 3 → Rbi, Rbe, Cbe, …, α, τB, τC
    """
    NAME          = "T-topology (Cheng 2022)"
    SHORT         = "T"
    TOPOLOGY_CHAR = "T"

    @classmethod
    def extract(cls, Y_ex1, freq, n_low, **kwargs):
        """
        Full extraction: Step 2 (Cbex_T, Cbcx) → Step 3 (intrinsic T params).
        See _step2_T and _step3_T for formula references.
        """
        res_ext, arr_ext = _step2_T(Y_ex1, freq, n_low)
        res_int, arr_int = _step3_T(arr_ext["Y_ex2"], freq, res_ext["Cbcx"], n_low)
        params = {**res_ext, **res_int}
        arrays = {**arr_ext, **arr_int, "_res_ext": res_ext, "_res_int": res_int}
        return params, arrays

    @classmethod
    def simulate(cls, params, freq, z0=50.0):
        """
        Forward sim — inside-out:
          [Z_in]  →  add Cbcx/Cbex  →  [Y_ex]  →  add Z_ser  →  add Y_pad  →  S
        """
        def _Y_int(p, w):
            # T intrinsic Y matrix [Eq. 16 inverse]
            Zbe_v = p["Rbe"] / (1.0 + 1j*w*p["Rbe"]*p["Cbe"])
            Zbc_v = p["Rbc"] / (1.0 + 1j*w*p["Rbc"]*p["Cbc"])
            alpha = p["alpha0"] * np.exp(-1j*w*p["tauC"]) / (1.0 + 1j*w*p["tauB"])
            Z_in  = np.array([[p["Rbi"]+Zbe_v, Zbe_v],
                               [Zbe_v - alpha*Zbc_v, (1-alpha)*Zbc_v + Zbe_v]])
            return np.linalg.inv(Z_in)
        return _sim_wrap(_Y_int, params, freq, z0)

    @classmethod
    def render_step_formulas(cls):
        """Show Step 2 and Step 3 formulas (called before extract() in the UI)."""
        st.markdown("**T-topology — Step 2 [Eqs. 13, 22]:**")
        st.latex(r"C_{bex}^T = \frac{\mathrm{Im}(Y_{11}+Y_{12})}{\omega}\bigg|_{\omega\to 0}")
        st.latex(r"C_{bcx} = -\frac{\mathrm{Im}(Y_{ms})\mathrm{Re}(Y_L)"
                 r"- \mathrm{Re}(Y_{ms})\mathrm{Im}(Y_L)}{\omega \cdot \mathrm{denom}}")
        st.markdown("**T-topology — Step 3 [Eqs. 16, 29–33]:**")
        st.latex(r"\alpha = \frac{Z_{12}-Z_{21}}{Z_{22}-Z_{21}},\quad"
                 r"\alpha_0 = |\alpha|\big|_{\omega\to 0}")
        st.latex(r"\tau_B = \frac{\sqrt{U-1}}{\omega},\quad"
                 r"\tau_C = -\frac{\arctan\bigl[V(1-V^2)^{-1/2}\bigr]}{2\omega}")

    @classmethod
    def render_results_table(cls, params):
        ri = params
        rows = [
            ("Rbi",  f"{ri['Rbi']:.4f}",                                       "Ω"),
            ("Rbe",  f"{ri['Rbe']:.4f}" if ri['Rbe']<1000 else f"{ri['Rbe']*1e-3:.4f}k", "Ω"),
            ("Cbe",  f"{ri['Cbe']*1e15:.4f}" if ri['Cbe']<1e-12 else f"{ri['Cbe']*1e12:.4f}",
                     "fF" if ri['Cbe']<1e-12 else "pF"),
            ("Rbc",  f"{ri['Rbc']*1e-3:.4f}", "kΩ"),
            ("Cbc",  f"{ri['Cbc']*1e15:.4f}", "fF"),
            ("Cbex", f"{ri['Cbex']*1e15:.4f}","fF"),
            ("Cbcx", f"{ri['Cbcx']*1e15:.4f}","fF"),
            ("α₀",   f"{ri['alpha0']:.5f}",   ""),
            ("τB",   f"{ri['tauB']*1e12:.4f}", "ps"),
            ("τC",   f"{ri['tauC']*1e12:.4f}", "ps"),
        ]
        st.dataframe(pd.DataFrame(rows, columns=["Symbol","Value","Unit"]),
                     use_container_width=True, hide_index=True)

    @classmethod
    def render_formula_trace(cls):
        with st.expander("📐 Full formula trace — T-topology (Cheng 2022)", expanded=False):
            st.markdown("**Dependency chain:** Y_ex1 → peel Cbex → Y_ex2 → peel Cbcx → Z_in → intrinsic")
            st.markdown("**Step 2** *(input: Y_ex1)*")
            st.latex(r"[Eq.13]\;C_{bex}^T=\frac{\mathrm{Im}(Y_{11}+Y_{12})}{\omega}\big|_{\omega\to0}")
            st.latex(r"[Eq.22]\;C_{bcx}=-\frac{\mathrm{Im}(Y_{ms})\mathrm{Re}(Y_L)"
                     r"-\mathrm{Re}(Y_{ms})\mathrm{Im}(Y_L)}{\omega"
                     r"[\mathrm{Re}(Y_{ms})\mathrm{Re}(Y_{tot})+\mathrm{Im}(Y_{tot})\mathrm{Im}(Y_{ms})]}")
            st.markdown("**Step 3** *(input: Y_ex2, Cbcx)*")
            st.latex(r"[Eq.16]\;Z_{be}=Z_{12},\;Z_{bc}=Z_{22}-Z_{21},\;Z_{bi}=Z_{11}-Z_{12}")
            st.latex(r"[Eq.29]\;\alpha=\frac{Z_{12}-Z_{21}}{Z_{22}-Z_{21}},\;"
                     r"\alpha_0=|\alpha||_{\omega\to0}")
            st.latex(r"[Eq.30]\;\tau_B=\frac{\sqrt{U-1}}{\omega}")
            st.latex(r"[Eq.31]\;\tau_C=-\frac{\arctan[V(1-V^2)^{-1/2}]}{2\omega}")
            st.markdown("**Forward simulation** *(inside → outside)*")
            st.latex(r"Z_{be}^{sim}=\frac{R_{be}}{1+j\omega R_{be}C_{be}},\;"
                     r"\alpha=\alpha_0 e^{-j\omega\tau_C}/(1+j\omega\tau_B)")
            st.latex(r"[Z_{in}^{sim}]=\begin{bmatrix}R_{bi}+Z_{be}&Z_{be}\\"
                     r"Z_{be}-\alpha Z_{bc}&(1-\alpha)Z_{bc}+Z_{be}\end{bmatrix}")
            st.latex(r"[Y_{ex}]=[Z_{in}]^{-1}+j\omega C_{bcx}\begin{pmatrix}1&-1\\-1&1\end{pmatrix}"
                     r"+j\omega C_{bex}\begin{pmatrix}1&0\\0&0\end{pmatrix}")
            st.latex(r"[Y_{tot}]=([Y_{ex}]^{-1}+[Z_{ser}])^{-1}\;,\quad"
                     r"S=(I-Z_0[Y_{tot}+Y_{pad}])(I+Z_0[Y_{tot}+Y_{pad}])^{-1}")

    @classmethod
    def render_override_and_smith(cls, fname, S_raw, freq, z0,
                                  para_eff, extract_result, **kwargs):
        params, arrays = extract_result
        calc_vals = {**para_eff, **params}
        all_p = _override_ui(fname, cls.SHORT, calc_vals, _INT_T_SPECS, cls.NAME)

        # Cached simulation
        cache_key  = f"sim_result_{cls.SHORT}_{fname}"
        hash_key   = f"sim_phash_{cls.SHORT}_{fname}"
        cur_hash   = params_hash({k: str(v) for k, v in {**all_p, "__nf": len(freq)}.items()})
        if st.session_state.get(hash_key) != cur_hash:
            with st.spinner(f"Simulating {cls.NAME}…"):
                S_sim = cls.simulate(all_p, freq, z0)
            st.session_state[cache_key] = S_sim
            st.session_state[hash_key]  = cur_hash
        else:
            S_sim = st.session_state.get(cache_key)
            if S_sim is None or S_sim.shape[0] != len(freq):
                with st.spinner(f"Simulating {cls.NAME}…"):
                    S_sim = cls.simulate(all_p, freq, z0)
                st.session_state[cache_key] = S_sim
                st.session_state[hash_key]  = cur_hash

        sc  = smith_scale_controls(fname, cls.SHORT)
        err = ssm_residual(S_raw, S_sim)
        render_smith_chart(S_raw, S_sim, cls.NAME, err, sc,
                           key=f"smith_{cls.SHORT}_{fname}")
        return S_sim


# ════════════════════════════════════════════════════════════════════════════════
# ChengPi
# ════════════════════════════════════════════════════════════════════════════════

class ChengPi(AbstractSSMModel):
    """
    Cheng (2022) π-topology  (Step 3 after Zhang et al. 2015).
    Same Step 2 as T but with π variant of Cbex formula.
    """
    NAME          = "π-topology (Cheng 2022)"
    SHORT         = "pi"
    TOPOLOGY_CHAR = "pi"

    @classmethod
    def extract(cls, Y_ex1, freq, n_low, **kwargs):
        res_ext, arr_ext = _step2_pi(Y_ex1, freq, n_low)
        res_int, arr_int = _step3_pi(arr_ext["Y_ex2"], freq, res_ext["Cbcx"], n_low)
        params = {**res_ext, **res_int}
        arrays = {**arr_ext, **arr_int, "_res_ext": res_ext, "_res_int": res_int}
        return params, arrays

    @classmethod
    def simulate(cls, params, freq, z0=50.0):
        def _Y_int(p, w):
            Ybe_v = 1.0/p["Rbe"] + 1j*w*p["Cbe"]
            Ybc_v = 1.0/p["Rbc"] + 1j*w*p["Cbc"]
            gm_v  = p["Gm0"] * np.exp(-1j*w*p["tau"])
            Y_core = np.array([[Ybe_v+Ybc_v, -Ybc_v],
                                [gm_v-Ybc_v,   Ybc_v]])
            Z_core = np.linalg.inv(Y_core) + np.array([[p["Rbi"], 0],[0, 0]])
            return np.linalg.inv(Z_core)
        return _sim_wrap(_Y_int, params, freq, z0)

    @classmethod
    def render_step_formulas(cls):
        st.markdown("**π-topology — Step 2 [Eqs. 26–28]:**")
        st.latex(r"B=Y_{12}+Y_{22},\;C=Y_{11}+Y_{21}")
        st.latex(r"C_{bex}^\pi=\frac{\mathrm{Re}(B)\mathrm{Re}(C)"
                 r"+\mathrm{Im}(B)\mathrm{Im}(C)}{\omega\,\mathrm{Im}(B)}")
        st.markdown("**π-topology — Step 3 (Zhang et al. 2015):**")
        st.latex(r"g_m=\frac{Z_{12}-Z_{21}}{Z_{bc}Z_{12}}\;\Rightarrow\;"
                 r"G_{m0}=|g_m|,\;\tau=-\angle g_m/\omega")
        st.latex(r"Y_{be}=\frac{Z_{22}-Z_{12}}{Z_{12}Z_{bc}}\;\Rightarrow\;"
                 r"R_{be}=1/\mathrm{Re}(Y_{be}),\;C_{be}=\mathrm{Im}(Y_{be})/\omega")

    @classmethod
    def render_results_table(cls, params):
        ri = params
        rows = [
            ("Rbi",  f"{ri['Rbi']:.4f}", "Ω"),
            ("Rbe",  f"{ri['Rbe']:.4f}" if ri['Rbe']<1000 else f"{ri['Rbe']*1e-3:.4f}k", "Ω"),
            ("Cbe",  f"{ri['Cbe']*1e15:.4f}" if ri['Cbe']<1e-12 else f"{ri['Cbe']*1e12:.4f}",
                     "fF" if ri['Cbe']<1e-12 else "pF"),
            ("Rbc",  f"{ri['Rbc']*1e-3:.4f}", "kΩ"),
            ("Cbc",  f"{ri['Cbc']*1e15:.4f}", "fF"),
            ("Cbex", f"{ri['Cbex']*1e15:.4f}","fF"),
            ("Cbcx", f"{ri['Cbcx']*1e15:.4f}","fF"),
            ("Gm0",  f"{ri['Gm0']*1e3:.4f}",  "mS"),
            ("τ",    f"{ri['tau']*1e12:.4f}",  "ps"),
        ]
        st.dataframe(pd.DataFrame(rows, columns=["Symbol","Value","Unit"]),
                     use_container_width=True, hide_index=True)

    @classmethod
    def render_formula_trace(cls):
        with st.expander("📐 Full formula trace — π-topology (Cheng 2022 / Zhang 2015)", expanded=False):
            st.markdown("**Step 2** *(input: Y_ex1)*")
            st.latex(r"[Eqs.26–28]\;B=Y_{12}+Y_{22},\;C=Y_{11}+Y_{21},\;"
                     r"C_{bex}^\pi=\frac{\mathrm{Re}(B)\mathrm{Re}(C)+\mathrm{Im}(B)\mathrm{Im}(C)}{\omega\,\mathrm{Im}(B)}")
            st.markdown("**Step 3** *(input: Y_ex2, Cbcx — same Z-matrix peel as T)*")
            st.latex(r"g_m=(Z_{12}-Z_{21})/(Z_{bc}Z_{12})\;\Rightarrow\;G_{m0}=|g_m|,\;\tau=-\angle g_m/\omega")
            st.markdown("**Forward simulation** *(inside → outside)*")
            st.latex(r"[Y_{core}]=\begin{bmatrix}Y_{be}+Y_{bc}&-Y_{bc}\\g_m-Y_{bc}&Y_{bc}\end{bmatrix},\;"
                     r"[Z_{in}^{sim}]=[Y_{core}]^{-1}+\begin{bmatrix}R_{bi}&0\\0&0\end{bmatrix}")
            st.latex(r"[Y_{tot}]=([Y_{ex}]^{-1}+[Z_{ser}])^{-1},\;"
                     r"S=(I-Z_0[Y_{tot}+Y_{pad}])(I+Z_0[Y_{tot}+Y_{pad}])^{-1}")

    @classmethod
    def render_override_and_smith(cls, fname, S_raw, freq, z0,
                                  para_eff, extract_result, **kwargs):
        params, arrays = extract_result
        calc_vals = {**para_eff, **params}
        all_p = _override_ui(fname, cls.SHORT, calc_vals, _INT_PI_SPECS, cls.NAME)

        cache_key = f"sim_result_{cls.SHORT}_{fname}"
        hash_key  = f"sim_phash_{cls.SHORT}_{fname}"
        cur_hash  = params_hash({k: str(v) for k, v in {**all_p, "__nf": len(freq)}.items()})
        if st.session_state.get(hash_key) != cur_hash:
            with st.spinner(f"Simulating {cls.NAME}…"):
                S_sim = cls.simulate(all_p, freq, z0)
            st.session_state[cache_key] = S_sim
            st.session_state[hash_key]  = cur_hash
        else:
            S_sim = st.session_state.get(cache_key)
            if S_sim is None or S_sim.shape[0] != len(freq):
                with st.spinner(f"Simulating {cls.NAME}…"):
                    S_sim = cls.simulate(all_p, freq, z0)
                st.session_state[cache_key] = S_sim
                st.session_state[hash_key]  = cur_hash

        sc  = smith_scale_controls(fname, cls.SHORT)
        err = ssm_residual(S_raw, S_sim)
        render_smith_chart(S_raw, S_sim, cls.NAME, err, sc,
                           key=f"smith_{cls.SHORT}_{fname}")
        return S_sim

"""
models/degachi.py — Degachi & Ghannouchi (2008) augmented π HBT model.

Reference: Degachi & Ghannouchi, IEEE TED vol. 55 no. 4, 2008, Eqs. 3–26.

Key difference from Cheng: no Step-2 extrinsic-cap peeling.
Input is Y_ex1 directly.  Adds Cbi (intrinsic base capacitance) and Zcx (external BC network).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from ..ssm_core        import (y_to_z, y_to_s_single,
                                safe_median, params_hash,
                                extended_smith_grid)
from ..ssm_deembedding  import build_Y_pad, build_Z_ser
from .base_ui           import (render_smith_chart, smith_scale_controls,
                                 sync_pad_from_preov, PAD_SPECS, ssm_residual)
from . import AbstractSSMModel


# ════════════════════════════════════════════════════════════════════════════════
# Extraction
# ════════════════════════════════════════════════════════════════════════════════

def _extract(Y_ex1, freq, n_low):
    """
    Degachi [Eqs. 3–26] — full extraction from Y_ex1.

    Step-by-step dependency chain:
      Z1, Z3, Z4  [Eqs. 3–5]
      →  Fbi linear fit → Tbi  [Eqs. 8–12]
      →  Rbi/Rbc ratio, Rbi·Cbc  [Eqs. 13–14]
      →  F1 linear fit → Tbe  [Eqs. 19–20]
      →  R, RT  [Eqs. 23–24]
      →  Rbe, Rbi (2×2 linear system)  [Eq. 25]
      →  Rbc, Cbc, Cbe, Cbi
      →  Gm0, τ  (from residual core admittance)
      →  Ccx  (from high-frequency Im(Y12))
    """
    omega  = 2.0*np.pi*freq
    omega2 = omega**2
    N      = len(freq)
    n_fit  = max(4, 2*N//3)

    # ── [Eqs. 3–5]  Z1, Z3, Z4 ───────────────────────────────────────────────
    with np.errstate(divide="ignore", invalid="ignore"):
        Z1 = 1.0 / (Y_ex1[:,0,0] + Y_ex1[:,0,1])
        Z3 = (Y_ex1[:,1,0] + Y_ex1[:,0,0]) / \
             ((Y_ex1[:,0,0] + Y_ex1[:,0,1]) * (Y_ex1[:,1,1] + Y_ex1[:,0,1]))
        Z4 = -1.0 / Y_ex1[:,0,1]

    # ── [Eq. 8]  Fbi = ω / Im(Z1/Z3)  →  fit Fbi = A0 + ω²·B0 ─────────────
    with np.errstate(divide="ignore", invalid="ignore"):
        Fbi = omega / np.imag(Z1 / Z3)
    mask = np.isfinite(Fbi[:n_fit]) & (np.abs(Fbi[:n_fit]) < 1e15) & (Fbi[:n_fit] > 0)
    if mask.sum() >= 3:
        try:    coeffs = np.polyfit(omega2[:n_fit][mask], Fbi[:n_fit][mask], 1); B0, A0 = float(coeffs[0]), float(coeffs[1])
        except: A0, B0 = 1.0, 0.0
    else:
        A0, B0 = 1.0, 0.0

    # ── [Eq. 12]  Tbi = √(B0/A0) ─────────────────────────────────────────────
    Tbi = float(np.sqrt(max(B0/A0, 0.0))) if A0 > 1e-30 else 0.0

    # ── [Eqs. 13–14]  Rbi/Rbc, Rbi·Cbc ──────────────────────────────────────
    corr13 = (Z1 / Z3) * (1.0 + 1j*omega*Tbi)
    RbiOverRbc_arr = np.real(corr13)
    with np.errstate(divide="ignore", invalid="ignore"):
        RbiCbc_arr = np.where(omega > 0, np.imag(corr13)/omega, np.nan)
    RbiOverRbc = safe_median(RbiOverRbc_arr, n_low)
    RbiCbc     = safe_median(RbiCbc_arr,     n_low)

    # ── [Eqs. 19–20]  F1  →  Tbe = √(B1/A1) ─────────────────────────────────
    with np.errstate(divide="ignore", invalid="ignore"):
        Z1_mod = Z1 * (1.0 + 1j*omega*Tbi)
        F1 = omega / np.imag(Z1_mod)
    mask1 = np.isfinite(F1[:n_fit]) & (np.abs(F1[:n_fit]) < 1e15) & (F1[:n_fit] > 0)
    if mask1.sum() >= 3:
        try:    coeffs1 = np.polyfit(omega2[:n_fit][mask1], F1[:n_fit][mask1], 1); B1, A1 = float(coeffs1[0]), float(coeffs1[1])
        except: A1, B1 = 1.0, 0.0
    else:
        A1, B1 = 1.0, 0.0
    Tbe = float(np.sqrt(max(B1/A1, 0.0))) if A1 > 1e-30 else 0.0

    # ── [Eqs. 23–24]  R = Re(F2),  RT = Im(F2)/ω ─────────────────────────────
    F2 = Z1 * (1.0 + 1j*omega*Tbe) * (1.0 + 1j*omega*Tbi)
    R_arr  = np.real(F2)
    with np.errstate(divide="ignore", invalid="ignore"):
        RT_arr = np.where(omega > 0, np.imag(F2)/omega, np.nan)
    R  = safe_median(R_arr,  n_low)
    RT = safe_median(RT_arr, n_low)

    # ── [Eq. 25]  Solve 2×2 for Rbe, Rbi ─────────────────────────────────────
    mat = np.array([[1.0+RbiOverRbc, 1.0],
                    [Tbi+RbiCbc,     Tbe]])
    rhs = np.array([R, RT])
    try:
        sol = np.linalg.solve(mat, rhs)
        Rbe, Rbi = float(sol[0]), float(sol[1])
        if Rbe <= 0 or Rbi <= 0: raise ValueError("non-physical")
    except:
        Rbi = safe_median(np.real(Z1 - Z3), n_low)
        Rbe = max(R - Rbi, 1.0)

    # ── Derived parameters ────────────────────────────────────────────────────
    Rbc = float(Rbi/RbiOverRbc) if abs(RbiOverRbc) > 1e-30 else 1e6
    Cbc = float(RbiCbc/Rbi)     if abs(Rbi)        > 1e-30 else 0.0
    Cbe = float(Tbe/Rbe)        if abs(Rbe)        > 1e-30 else 0.0
    Cbi = float(Tbi/Rbi)        if abs(Rbi)        > 1e-30 else 0.0

    # ── Gm0, τ — from residual core after removing Zbi ───────────────────────
    Zbi_v = (Rbi/(1.0 + 1j*omega*Rbi*Cbi) if Cbi > 1e-40
             else Rbi*np.ones(N, dtype=complex))
    Z_in_full = y_to_z(Y_ex1)
    Z_in_core = Z_in_full.copy()
    Z_in_core[:,0,0] -= Zbi_v
    from ..ssm_core import z_to_y
    Y_core  = z_to_y(Z_in_core)
    gm_arr  = Y_core[:,1,0] - Y_core[:,0,1]
    Gm0_a   = np.abs(gm_arr)
    with np.errstate(divide="ignore", invalid="ignore"):
        tau_a = -np.angle(gm_arr) / omega
    Gm0 = safe_median(Gm0_a, n_low)
    tau = safe_median(tau_a,  n_low)

    # ── Ccx — from high-freq Im(Y12) ─────────────────────────────────────────
    n_hi = max(N//2, n_low+1)
    with np.errstate(divide="ignore", invalid="ignore"):
        Ccx_arr = -np.imag(Y_ex1[:,0,1]) / omega
    Ccx = max(safe_median(Ccx_arr[n_hi:]), 0.0)

    params = dict(
        Rbi=Rbi, Cbi=Cbi, Rbe=Rbe, Cbe=Cbe,
        Rbc=Rbc, Cbc=Cbc, Rcx=1e6,  Ccx=Ccx,
        Gm0=Gm0, tau=tau,
        # diagnostic scalars for the Fbi/F1 fit plots
        Tbi=Tbi, Tbe=Tbe, A0=A0, B0=B0, A1=A1, B1=B1,
        RbiOverRbc=RbiOverRbc, RbiCbc=RbiCbc, R=R, RT=RT,
    )
    arrays = dict(
        Z1=Z1, Z3=Z3, Z4=Z4, Fbi=Fbi, F1=F1, F2=F2,
        R_arr=R_arr, RT_arr=RT_arr,
        RbiOverRbc_arr=RbiOverRbc_arr, RbiCbc_arr=RbiCbc_arr,
        Gm0_a=Gm0_a, tau_a=tau_a, Ccx_arr=Ccx_arr, omega2=omega2,
    )
    return params, arrays


# ── Forward simulator ──────────────────────────────────────────────────────────

def _simulate(p, freq, z0=50.0):
    """
    Degachi forward sim — 5 layers inside-out:
      1. Zbi, Zbe, Zbc  (RC networks)
      2. [Y_core] intrinsic admittance with gm
      3. [Z_core] = [Y_core]⁻¹ + [[Zbi,0],[0,0]]
      4. [Y_int]  = [Z_core]⁻¹ + (1/Zcx)·shunt
      5. [Y_tot]  = ([Y_int]⁻¹ + Z_ser)⁻¹  then add Y_pad → S
    """
    omega = 2.0*np.pi*freq
    S = np.zeros((len(freq), 2, 2), dtype=complex)
    for i, w in enumerate(omega):
        Rbi, Cbi = p["Rbi"], p["Cbi"]
        Rbe, Cbe = p["Rbe"], p["Cbe"]
        Rbc, Cbc = p["Rbc"], p["Cbc"]
        Rcx, Ccx = p["Rcx"], p["Ccx"]
        Gm0, tau = p["Gm0"], p["tau"]

        # Step 1: element impedances
        Zbi_v = Rbi/(1.0 + 1j*w*Rbi*Cbi) if Cbi > 1e-40 else complex(Rbi)
        Zbe_v = Rbe/(1.0 + 1j*w*Rbe*Cbe)
        Zbc_v = Rbc/(1.0 + 1j*w*Rbc*Cbc)
        Zcx_v = (1.0/(1j*w*Ccx) if (Rcx > 1e4 and Ccx > 1e-40)
                 else Rcx/(1.0 + 1j*w*Rcx*Ccx) if Ccx > 1e-40
                 else complex(1e9))

        Ybe = 1.0/Zbe_v; Ybc = 1.0/Zbc_v; gm_v = Gm0*np.exp(-1j*w*tau)

        # Step 2: intrinsic π admittance matrix
        Y_core = np.array([[Ybe+Ybc, -Ybc],
                            [gm_v-Ybc,  Ybc]])

        # Step 3: add Zbi to base node
        Z_core = np.linalg.inv(Y_core) + np.array([[Zbi_v, 0.0],[0.0, 0.0]])

        # Step 4: add Zcx shunt
        Y_int = np.linalg.inv(Z_core) + (1.0/Zcx_v)*np.array([[1,-1],[-1,1]])

        # Step 5: add series leads and pad shunt
        Z_ser = build_Z_ser(p, w)
        try:    Y_tot = np.linalg.inv(np.linalg.inv(Y_int) + Z_ser)
        except: Y_tot = np.zeros((2,2), dtype=complex)
        Y_pad = build_Y_pad(p, w)
        S[i]  = y_to_s_single(Y_tot + Y_pad, z0)
    return S


# ── Override UI specs ──────────────────────────────────────────────────────────

_INT_D_SPECS = [
    ("Rbi","Rbi", 1.0, "Ω",  "%.4f", 0.1),
    ("Cbi","Cbi", 1e15,"fF", "%.4f", 0.1),
    ("Rbe","Rbe", 1.0, "Ω",  "%.3f", 1.0),
    ("Cbe","Cbe", 1e15,"fF", "%.4f", 0.1),
    ("Rbc","Rbc", 1e-3,"kΩ", "%.4f", 0.01),
    ("Cbc","Cbc", 1e15,"fF", "%.4f", 0.01),
    ("Rcx","Rcx", 1e-3,"kΩ", "%.2f", 10.0),
    ("Ccx","Ccx", 1e15,"fF", "%.4f", 0.1),
    ("Gm0","Gm0", 1e3, "mS", "%.4f", 0.01),
    ("tau","τ",   1e12,"ps", "%.4f", 0.01),
]


# ════════════════════════════════════════════════════════════════════════════════
# Degachi model class
# ════════════════════════════════════════════════════════════════════════════════

class Degachi(AbstractSSMModel):
    NAME          = "Degachi (2008) augmented π"
    SHORT         = "D"
    TOPOLOGY_CHAR = "D"

    @classmethod
    def extract(cls, Y_ex1, freq, n_low, **kwargs):
        return _extract(Y_ex1, freq, n_low)

    @classmethod
    def simulate(cls, params, freq, z0=50.0):
        return _simulate(params, freq, z0)

    @classmethod
    def render_step_formulas(cls):
        """Formulas shown immediately before extraction in the UI."""
        st.markdown("**Degachi (2008) — [Eqs. 3–26], input: Y_ex1 directly**")
        st.latex(r"Z_1=\frac{1}{Y_{11}+Y_{12}},\;"
                 r"Z_3=\frac{Y_{21}+Y_{11}}{(Y_{11}+Y_{12})(Y_{22}+Y_{12})}")
        st.latex(r"F_{bi}=\omega/\mathrm{Im}(Z_1/Z_3)=A_0+\omega^2 B_0"
                 r"\;\Rightarrow\;T_{bi}=\sqrt{B_0/A_0}")
        st.latex(r"\mathrm{Re}[Z_1/Z_3(1+j\omega T_{bi})]=R_{bi}/R_{bc},\;"
                 r"\mathrm{Im}[\cdot]/\omega=R_{bi}C_{bc}")
        st.latex(r"F_1=\omega/\mathrm{Im}[Z_1(1+j\omega T_{bi})]=A+\omega^2 B"
                 r"\;\Rightarrow\;T_{be}=\sqrt{B/A}")
        st.latex(r"\begin{bmatrix}R_{be}\\R_{bi}\end{bmatrix}="
                 r"\begin{bmatrix}1+R_{bi}/R_{bc}&1\\"
                 r"T_{bi}+R_{bi}C_{bc}&T_{be}\end{bmatrix}^{-1}"
                 r"\begin{bmatrix}R\\RT\end{bmatrix}")

    @classmethod
    def render_results_table(cls, params):
        ri = params
        rows = [
            ("Rbi",  f"{ri['Rbi']:.4f}",   "Ω"),
            ("Cbi",  f"{ri['Cbi']*1e15:.4f}", "fF"),
            ("Rbe",  f"{ri['Rbe']:.4f}" if ri['Rbe']<1000 else f"{ri['Rbe']*1e-3:.4f}k", "Ω"),
            ("Cbe",  f"{ri['Cbe']*1e15:.4f}" if ri['Cbe']<1e-12 else f"{ri['Cbe']*1e12:.4f}",
                     "fF" if ri['Cbe']<1e-12 else "pF"),
            ("Rbc",  f"{ri['Rbc']*1e-3:.4f}", "kΩ"),
            ("Cbc",  f"{ri['Cbc']*1e15:.4f}", "fF"),
            ("Rcx",  f"{ri['Rcx']*1e-3:.2f}", "kΩ"),
            ("Ccx",  f"{ri['Ccx']*1e15:.4f}", "fF"),
            ("Gm0",  f"{ri['Gm0']*1e3:.4f}",  "mS"),
            ("τ",    f"{ri['tau']*1e12:.4f}",  "ps"),
        ]
        st.dataframe(pd.DataFrame(rows, columns=["Symbol","Value","Unit"]),
                     use_container_width=True, hide_index=True)

    @classmethod
    def render_formula_trace(cls):
        with st.expander("📐 Full formula trace — Degachi (2008) augmented π", expanded=False):
            st.markdown("**Dependency chain:**  "
                        "Y_ex1 → Z1,Z3,Z4 → Tbi → Tbe → R,RT → Rbe,Rbi → all others")
            st.latex(r"[Eq.3]\;Z_1=\frac{1}{Y_{11}+Y_{12}},\;"
                     r"[Eq.4]\;Z_3=\frac{Y_{21}+Y_{11}}{(Y_{11}+Y_{12})(Y_{22}+Y_{12})},\;"
                     r"[Eq.5]\;Z_4=-\frac{1}{Y_{12}}")
            st.latex(r"[Eq.8]\;F_{bi}=\omega/\mathrm{Im}(Z_1/Z_3)=A_0+\omega^2 B_0"
                     r"\;\Rightarrow\;[Eq.12]\;T_{bi}=\sqrt{B_0/A_0}")
            st.latex(r"[Eq.13]\;\mathrm{Re}[Z_1/Z_3(1+j\omega T_{bi})]=R_{bi}/R_{bc}")
            st.latex(r"[Eq.14]\;\mathrm{Im}[\cdot]/\omega=R_{bi}C_{bc}")
            st.latex(r"[Eq.19]\;F_1=\omega/\mathrm{Im}[Z_1(1+j\omega T_{bi})]"
                     r"=A+\omega^2 B\;\Rightarrow\;T_{be}=\sqrt{B/A}")
            st.latex(r"[Eqs.23–24]\;R=\mathrm{Re}(F_2),\;RT=\mathrm{Im}(F_2)/\omega")
            st.latex(r"[Eq.25]\;\begin{bmatrix}R_{be}\\R_{bi}\end{bmatrix}="
                     r"\begin{bmatrix}1+R_{bi}/R_{bc}&1\\T_{bi}+R_{bi}C_{bc}&T_{be}\end{bmatrix}^{-1}"
                     r"\begin{bmatrix}R\\RT\end{bmatrix}")
            st.markdown("**Forward simulation** *(5 layers, inside → outside)*")
            st.latex(r"1.\;Z_{bi}=R_{bi}/(1+j\omega R_{bi}C_{bi}),\;"
                     r"Z_{be},Z_{bc}\text{ similarly}")
            st.latex(r"2.\;[Y_{core}]=[[Y_{be}+Y_{bc},-Y_{bc}],[g_m-Y_{bc},Y_{bc}]]")
            st.latex(r"3.\;[Z_{core}]=[Y_{core}]^{-1}+[[Z_{bi},0],[0,0]]")
            st.latex(r"4.\;[Y_{int}]=[Z_{core}]^{-1}+(1/Z_{cx})[[1,-1],[-1,1]]")
            st.latex(r"5.\;[Y_{tot}]=([Y_{int}]^{-1}+[Z_{ser}])^{-1},\;"
                     r"S=(I-Z_0[Y_{tot}+Y_{pad}])(I+Z_0[Y_{tot}+Y_{pad}])^{-1}")

    @classmethod
    def render_diagnostic_plots(cls, params, arrays, freq, fname):
        """Degachi-specific: Fbi and F1 linear-fit diagnostic plots."""
        ri = params; omega2 = arrays["omega2"]
        Fbi = arrays["Fbi"]; F1 = arrays["F1"]
        n_fit = max(4, 2*len(freq)//3)
        with st.expander("📊 Degachi — Fbi and F1 diagnostic fits", expanded=True):
            c1d, c2d = st.columns(2)
            for col_w, Farr, A, B, title in [
                (c1d, Fbi, ri["A0"], ri["B0"], "Fbi [Eq.8]"),
                (c2d, F1,  ri["A1"], ri["B1"], "F1 [Eq.19]"),
            ]:
                fig = go.Figure()
                mask = np.isfinite(Farr) & (np.abs(Farr) < 1e15) & (Farr > 0)
                fig.add_trace(go.Scatter(x=omega2[mask], y=Farr[mask], mode="markers",
                                         name="Meas.", marker=dict(size=5, color="#1f77b4")))
                if A > 0 and mask.any():
                    x_fit = np.linspace(0, np.max(omega2[mask])*1.05, 200)
                    fig.add_trace(go.Scatter(x=x_fit, y=A+B*x_fit, mode="lines",
                                             name=f"Fit {A:.3e}+{B:.3e}·ω²",
                                             line=dict(color="#d62728", dash="dash", width=2)))
                fig.update_layout(title=dict(text=title, font=dict(size=12)),
                                  xaxis_title="ω²", plot_bgcolor="white",
                                  paper_bgcolor="white", height=280,
                                  margin=dict(l=55,r=10,t=40,b=42))
                fig.update_xaxes(showgrid=True, gridcolor="#ebebeb")
                fig.update_yaxes(showgrid=True, gridcolor="#ebebeb")
                col_w.plotly_chart(fig, use_container_width=True,
                                   key=f"dfit_{title[:4]}_{fname}")

    @classmethod
    def render_override_and_smith(cls, fname, S_raw, freq, z0,
                                  para_eff, extract_result, **kwargs):
        params, arrays = extract_result
        calc_vals = {**para_eff, **params}
        all_specs  = PAD_SPECS + _INT_D_SPECS
        sync_pad_from_preov(fname, cls.SHORT, calc_vals)

        for key, _, scale, *_ in _INT_D_SPECS:
            sk = f"sim_{cls.SHORT}_{key}_{fname}"
            if sk not in st.session_state:
                st.session_state[sk] = float(calc_vals.get(key, 0.0)) * scale

        with st.expander(f"✏️ Fine-tune {cls.NAME} intrinsic parameters", expanded=False):
            if st.button(f"↩️ Reset {cls.NAME} to calculated", key=f"rst_sim_{cls.SHORT}_{fname}"):
                for key, _, scale, *_ in all_specs:
                    st.session_state[f"sim_{cls.SHORT}_{key}_{fname}"] = float(calc_vals.get(key, 0.0)) * scale
                st.rerun()

            st.markdown("**Pad Parasitics** *(auto-synced)*")
            for row_start in range(0, len(PAD_SPECS), 3):
                row = PAD_SPECS[row_start:row_start+3]
                for col_w, (key, lbl, sc, unit, fmt, step) in zip(st.columns(len(row)), row):
                    col_w.number_input(f"{lbl} ({unit})" if unit else lbl,
                                       key=f"sim_{cls.SHORT}_{key}_{fname}", format=fmt, step=step)

            st.markdown("**Intrinsic (Degachi)**")
            for row_start in range(0, len(_INT_D_SPECS), 4):
                row = _INT_D_SPECS[row_start:row_start+4]
                for col_w, (key, lbl, sc, unit, fmt, step) in zip(st.columns(len(row)), row):
                    col_w.number_input(f"{lbl} ({unit})" if unit else lbl,
                                       key=f"sim_{cls.SHORT}_{key}_{fname}", format=fmt, step=step)

        all_p = {key: st.session_state.get(f"sim_{cls.SHORT}_{key}_{fname}",
                                             float(calc_vals.get(key, 0.0))*scale) / scale
                 for key, _, scale, *_ in all_specs}
        for ek in ["Cpbe_mode","Cpbe_extra","Cpce_mode","Cpce_extra",
                   "Cpbc_mode","Cpbc_extra","Cpar_Lb","Cpar_Lc","Cpar_Le"]:
            all_p[ek] = calc_vals.get(ek, "None" if "mode" in ek else 0.0)

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

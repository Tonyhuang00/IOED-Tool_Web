"""
ssm_plots.py — Step 1 diagnostic plots and helper visualisations.

All functions render directly into Streamlit and return any UI-state values
needed by the caller (e.g. extended element modes, Cpar values).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from .ssm_core        import open_elem_Y, s_to_y, y_to_z
from .ssm_deembedding  import peel_parasitics
from .ssm_s2p          import simulate_open
from .models.base_ui   import render_smith_chart, ssm_residual


# ── Open element mode constants ───────────────────────────────────────────────

OPEN_MODES      = ["None", "Parallel L", "Series L", "Series R"]
_MODE_UNIT      = {"None": None, "Parallel L": "pH", "Series L": "pH", "Series R": "Ω"}
_MODE_SCALE     = {"None": 1,    "Parallel L": 1e12, "Series L": 1e12, "Series R": 1.0}


# ════════════════════════════════════════════════════════════════════════════════
# Open dummy plots
# ════════════════════════════════════════════════════════════════════════════════

def render_open_plots(open_data, para_caps, open_arr, fname=""):
    """
    Render four expanders for Open dummy diagnostics:
      1. Extra-element model controls (Parallel L / Series L / Series R per cap)
      2. Capacitance Im(Y)/ω vs frequency  [0–50 fF, fixed]
      3. Conductance Re(Y) vs frequency    [series R indicator]
      4. Im(Y)/ω vs 1/ω²                  [Parallel L linearisation]
      5. Smith chart: measured vs forward-simulated Open

    Returns
    -------
    dict  {cap: (mode_str, extra_SI)}  — propagated into para_eff and forward sims.
    """
    f_o, S_o, z0_o = open_data
    f_ghz = f_o * 1e-9
    omega  = 2.0*np.pi*f_o

    # ── 1. Per-cap extra element controls ─────────────────────────────────────
    with st.expander("🔧 Open Pad Element Model — extra parasitic options", expanded=False):
        st.markdown(
            "Each pad capacitor can include one secondary parasitic element.  \n"
            "**Parallel L** = inductor ∥ C → resonance at 1/√LC, affects Im(Y)/ω.  \n"
            "**Series L**   = inductor in series with C → increases effective C near resonance.  \n"
            "**Series R**   = resistor in series with C → adds Re(Y) that rises then saturates.  \n"
            "These choices propagate into de-embedding, Cold-HBT correction, and S2P downloads."
        )
        for cap, cap_lbl in [("Cpbe","Cpbe (B-E)"), ("Cpce","Cpce (C-E)"), ("Cpbc","Cpbc (B-C)")]:
            st.markdown(f"**{cap_lbl}**")
            c1, c2 = st.columns([2, 1])
            mode_sk  = f"open_mode_{cap}_{fname}"
            extra_sk = f"open_extra_{cap}_{fname}"
            if mode_sk  not in st.session_state: st.session_state[mode_sk]  = "None"
            if extra_sk not in st.session_state: st.session_state[extra_sk] = 0.0
            c1.radio("", OPEN_MODES, horizontal=True, key=mode_sk, label_visibility="collapsed")
            mode = st.session_state[mode_sk]
            if mode != "None":
                unit = _MODE_UNIT[mode]
                c2.number_input(f"Extra {unit}", min_value=0.0,
                                step=0.1 if unit == "pH" else 0.01,
                                format="%.3f" if unit == "pH" else "%.4f",
                                key=extra_sk)

    def _get_mode_extra(cap):
        """Return (mode_str, extra_in_SI) for one cap."""
        mode  = st.session_state.get(f"open_mode_{cap}_{fname}", "None")
        extra_disp = float(st.session_state.get(f"open_extra_{cap}_{fname}", 0.0))
        sc    = _MODE_SCALE.get(mode, 1)
        return mode, extra_disp / sc

    # ── 2. Capacitance plot ───────────────────────────────────────────────────
    with st.expander("📊 Open — Pad Capacitances vs Frequency", expanded=True):
        fig_cap = go.Figure()
        for key, lbl, col in [("Cpbe","Cpbe","#1f77b4"),
                               ("Cpce","Cpce","#ff7f0e"),
                               ("Cpbc","Cpbc","#2ca02c")]:
            arr_fF = open_arr[key] * 1e15
            val_fF = para_caps[key] * 1e15
            mode, extra = _get_mode_extra(key)
            # Measured trace
            fig_cap.add_trace(go.Scatter(x=f_ghz, y=arr_fF,
                name=f"{lbl} (meas.)", line=dict(color=col, width=2), mode="lines"))
            # Median dashed line
            fig_cap.add_trace(go.Scatter(x=[f_ghz[0], f_ghz[-1]], y=[val_fF, val_fF],
                name=f"{lbl}={val_fF:.3f} fF", line=dict(color=col, width=1.8, dash="dash"), mode="lines"))
            # Modelled effective C overlay (when extra element chosen)
            if mode != "None":
                # See ssm_core.open_elem_Y for the formula
                Y_mod_arr = np.array([open_elem_Y(para_caps[key], mode, extra, w) for w in omega])
                Ceff_fF   = np.imag(Y_mod_arr) / omega * 1e15
                fig_cap.add_trace(go.Scatter(x=f_ghz, y=Ceff_fF,
                    name=f"{lbl} model ({mode})", line=dict(color=col, width=2, dash="dot"), mode="lines"))
        fig_cap.update_layout(title="Pad Capacitances — Im(Y)/ω",
            xaxis_title="Frequency (GHz)", yaxis_title="Cap (fF)",
            plot_bgcolor="white", paper_bgcolor="white", height=360,
            legend=dict(x=1.02, y=1.0, xanchor="left", font=dict(size=9)),
            margin=dict(l=55,r=10,t=40,b=45), hovermode="x unified")
        fig_cap.update_xaxes(showgrid=True, gridcolor="#ebebeb")
        fig_cap.update_yaxes(showgrid=True, gridcolor="#ebebeb", range=[0, 50])
        st.plotly_chart(fig_cap, use_container_width=True, key=f"step1_cap_{fname}")
        st.caption("Flat line = pure C. Slope/resonance = inductive effect. Range fixed 0–50 fF.")

    # ── 3. Conductance plot (Series R indicator) ──────────────────────────────
    with st.expander("📊 Open — Pad Conductance vs Frequency (Re(Y) — series R indicator)",
                     expanded=False):
        fig_g = go.Figure()
        for key, lbl, col in [("Cpbe","Gpbe","#1f77b4"),
                               ("Cpce","Gpce","#ff7f0e"),
                               ("Cpbc","Gpbc","#2ca02c")]:
            arr_mS = open_arr[f"G{key[1:]}"] * 1e3   # Gpbe/Gpce/Gpbc keys
            mode, extra = _get_mode_extra(key)
            fig_g.add_trace(go.Scatter(x=f_ghz, y=arr_mS,
                name=f"{lbl} (meas.)", line=dict(color=col, width=2), mode="lines"))
            if mode == "Series R" and extra > 0:
                # Re[Y_series_R] = ω²RC² / (1+ω²R²C²)  → see ssm_core.open_elem_Y
                G_mod = np.array([np.real(open_elem_Y(para_caps[key], mode, extra, w))*1e3
                                  for w in omega])
                fig_g.add_trace(go.Scatter(x=f_ghz, y=G_mod,
                    name=f"{lbl} model (R={extra:.3f} Ω)",
                    line=dict(color=col, width=2, dash="dot"), mode="lines"))
        fig_g.add_hline(y=0, line_color="#aaa", line_width=1)
        fig_g.update_layout(
            title="Pad Conductance Re(Y) — nonzero = series R or parallel G loss",
            xaxis_title="Frequency (GHz)", yaxis_title="Conductance (mS)",
            plot_bgcolor="white", paper_bgcolor="white", height=320,
            legend=dict(x=1.02, y=1.0, xanchor="left", font=dict(size=9)),
            margin=dict(l=55,r=10,t=40,b=45), hovermode="x unified")
        fig_g.update_xaxes(showgrid=True, gridcolor="#ebebeb")
        fig_g.update_yaxes(showgrid=True, gridcolor="#ebebeb")
        st.plotly_chart(fig_g, use_container_width=True, key=f"step1_cond_{fname}")
        st.caption(
            "Pure C → Re(Y)=0.  "
            "Series R → Re(Y) = ω²RC² / (1+ω²R²C²) — rises then saturates.  \n"
            "Select 'Series R' above to overlay the modelled curve.")

    # ── 4. Im(Y)/ω vs 1/ω² (Parallel L linearisation) ───────────────────────
    with st.expander("📊 Open — Im(Y)/ω vs 1/ω²  (Parallel L linearisation)", expanded=False):
        fig_l = go.Figure()
        one_over_omega2 = 1.0 / (omega**2 + 1e-60)
        for key, lbl, col in [("Cpbe","Cpbe","#1f77b4"),
                               ("Cpce","Cpce","#ff7f0e"),
                               ("Cpbc","Cpbc","#2ca02c")]:
            Ceff = open_arr[key]  # Im(Y)/ω already stored per-frequency
            fig_l.add_trace(go.Scatter(
                x=one_over_omega2*1e-18, y=Ceff*1e15, name=lbl,
                line=dict(color=col, width=2), mode="lines",
                hovertemplate="1/ω²=%{x:.4f}×10¹⁸<br>Im(Y)/ω=%{y:.3f} fF<extra></extra>"))
        fig_l.update_layout(
            title="Im(Y)/ω vs 1/ω² — slope = −1/L if Parallel L present",
            xaxis_title="1/ω² (× 10¹⁸ rad⁻²s²)", yaxis_title="Im(Y)/ω  (fF equivalent)",
            plot_bgcolor="white", paper_bgcolor="white", height=320,
            legend=dict(x=1.02, y=1.0, xanchor="left", font=dict(size=9)),
            margin=dict(l=55,r=10,t=40,b=45), hovermode="x unified")
        fig_l.update_xaxes(showgrid=True, gridcolor="#ebebeb")
        fig_l.update_yaxes(showgrid=True, gridcolor="#ebebeb", range=[0, 50])
        st.plotly_chart(fig_l, use_container_width=True, key=f"step1_lind_{fname}")
        st.caption(
            "Parallel L model: Im(Y)/ω = C − 1/(ω²L).  "
            "A straight line with negative slope → L = −1/slope (SI).")

    # ── 5. Smith chart: measured vs modelled Open ─────────────────────────────
    with st.expander("📡 Open — Measured vs Modelled Smith Chart", expanded=False):
        _p_open = {}
        for cap in ["Cpbe", "Cpce", "Cpbc"]:
            mode, extra = _get_mode_extra(cap)
            _p_open[cap]            = para_caps[cap]
            _p_open[f"{cap}_mode"]  = mode
            _p_open[f"{cap}_extra"] = extra
        # Forward-simulate open (ssm_s2p.simulate_open)
        S_open_sim = simulate_open(_p_open, f_o, z0_o)
        err_open   = ssm_residual(S_o, S_open_sim)
        render_smith_chart(S_o, S_open_sim,
                           f"Open dummy",
                           err_open,
                           scales={"S11":1.0,"S12":1.0,"S21":1.0,"S22":1.0},
                           key=f"smith_open_{fname}")
        st.caption("Adjust the extra element controls above — the modelled curve updates live.")

    return {cap: _get_mode_extra(cap) for cap in ["Cpbe", "Cpce", "Cpbc"]}


# ════════════════════════════════════════════════════════════════════════════════
# Short dummy plots
# ════════════════════════════════════════════════════════════════════════════════

def render_short_plots(short_arr, para_short, fname=""):
    """
    Render three expanders for Short dummy diagnostics:
      1. Parallel C controls per lead
      2. Lead inductances vs frequency  [0–150 pH, fixed]
      3. Lead series resistances vs frequency

    Returns
    -------
    dict  {Cpar_Lb, Cpar_Lc, Cpar_Le}  in SI Farads.
    """

    # ── 1. Parallel C controls ────────────────────────────────────────────────
    with st.expander("🔧 Short Lead Model — optional parallel capacitance per lead",
                     expanded=False):
        st.markdown(
            "Adds a capacitance **in parallel** with each lead's R+jωL impedance.  \n"
            "Z_lead_eff = (R+jωL) ∥ (1/jωC_par) = (R+jωL) / (1 + jωC_par(R+jωL))  \n"
            "Causes extracted L to appear frequency-dependent (decreasing at high freq).  \n"
            "Default = 0 (disabled)."
        )
        cpar_cols = st.columns(3)
        for col_w, (key, lbl) in zip(cpar_cols, [("Cpar_Lb","Lb"),
                                                   ("Cpar_Lc","Lc"),
                                                   ("Cpar_Le","Le")]):
            ks = f"short_{key}_{fname}"
            if ks not in st.session_state: st.session_state[ks] = 0.0
            col_w.number_input(f"C_par_{lbl} (fF)", min_value=0.0,
                                step=0.1, format="%.3f", key=ks)

    def _get_cpar(lead_key):
        return float(st.session_state.get(f"short_{lead_key}_{fname}", 0.0)) * 1e-15

    cpar_Lb = _get_cpar("Cpar_Lb")
    cpar_Lc = _get_cpar("Cpar_Lc")
    cpar_Le = _get_cpar("Cpar_Le")

    # ── 2. Inductance plot ────────────────────────────────────────────────────
    with st.expander("📊 Short — Lead Inductances vs Frequency", expanded=True):
        fig_ind = go.Figure(); any_neg = False
        for key, lbl, col in [("Lb","Lb","#8e44ad"),
                               ("Lc","Lc","#e67e22"),
                               ("Le","Le","#16a085")]:
            arr_pH = short_arr[key] * 1e12
            val_pH = para_short[key] * 1e12
            if val_pH < 0: any_neg = True
            idx = np.arange(len(arr_pH))
            fig_ind.add_trace(go.Scatter(x=idx, y=arr_pH,
                name=f"{lbl} (per-freq)", line=dict(color=col, width=2), mode="lines"))
            fig_ind.add_trace(go.Scatter(x=[0, len(arr_pH)-1], y=[val_pH, val_pH],
                name=f"{lbl}={val_pH:.2f} pH",
                line=dict(color=col, width=1.8, dash="dash"), mode="lines"))
        fig_ind.add_hline(y=0, line_color="#333", line_width=1.2,
                           annotation_text="0 pH", annotation_position="left",
                           annotation_font=dict(size=9, color="#333"))
        fig_ind.update_layout(title="Lead Inductances",
            xaxis_title="Point index", yaxis_title="Inductance (pH)",
            plot_bgcolor="white", paper_bgcolor="white", height=360,
            legend=dict(x=1.02, y=1.0, xanchor="left", font=dict(size=9)),
            margin=dict(l=55,r=10,t=40,b=45), hovermode="x unified")
        fig_ind.update_xaxes(showgrid=True, gridcolor="#ebebeb")
        fig_ind.update_yaxes(showgrid=True, gridcolor="#ebebeb", range=[0, 150])
        st.plotly_chart(fig_ind, use_container_width=True, key=f"step1_ind_{fname}")
        if any_neg:
            st.warning("One or more lead inductances are negative. Use Short Override to correct.")
        st.caption("Range fixed 0–150 pH.")

    # ── 3. Series resistance plot ─────────────────────────────────────────────
    with st.expander("📊 Short — Lead Series Resistances vs Frequency", expanded=False):
        fig_r = go.Figure()
        for key, lbl, col in [("Rpb","Rb","#8e44ad"),
                               ("Rpc","Rc","#e67e22"),
                               ("Rpe","Re","#16a085")]:
            arr_O = short_arr[key]
            val_O = para_short[key]
            fig_r.add_trace(go.Scatter(x=np.arange(len(arr_O)), y=arr_O,
                name=f"{lbl} (per-freq)", line=dict(color=col, width=2), mode="lines"))
            fig_r.add_trace(go.Scatter(x=[0, len(arr_O)-1], y=[val_O, val_O],
                name=f"{lbl}={val_O:.4f} Ω",
                line=dict(color=col, width=1.8, dash="dash"), mode="lines"))
        fig_r.add_hline(y=0, line_color="#aaa", line_width=1)
        fig_r.update_layout(
            title="Lead Series Resistances — Re(Z terms from Short)",
            xaxis_title="Point index", yaxis_title="Resistance (Ω)",
            plot_bgcolor="white", paper_bgcolor="white", height=320,
            legend=dict(x=1.02, y=1.0, xanchor="left", font=dict(size=9)),
            margin=dict(l=55,r=10,t=40,b=45), hovermode="x unified")
        fig_r.update_xaxes(showgrid=True, gridcolor="#ebebeb")
        fig_r.update_yaxes(showgrid=True, gridcolor="#ebebeb")
        st.plotly_chart(fig_r, use_container_width=True, key=f"step1_res_{fname}")
        st.caption("Flat curve = clean extraction. Rising with frequency = skin effect or artefact.")

    return {"Cpar_Lb": cpar_Lb, "Cpar_Lc": cpar_Lc, "Cpar_Le": cpar_Le}


# ════════════════════════════════════════════════════════════════════════════════
# Helper: S-parameter comparison plot
# ════════════════════════════════════════════════════════════════════════════════

def render_sparams_comparison(S_raw, freq, z0, para_eff, fname):
    """Raw vs de-embedded S-parameters (4 side-by-side subplots)."""
    from .ssm_core import y_to_s_batch
    f_ghz   = freq * 1e-9
    Y_deemb = peel_parasitics(S_raw, freq, z0, para_eff)
    S_deemb = y_to_s_batch(Y_deemb, z0)
    with st.expander("📊 S-Parameters: Raw vs De-embedded (after Step 1)", expanded=True):
        st.caption("Solid = raw.  Dashed = after pad removal.")
        for col_w, (sname, (r,c), color) in zip(
                st.columns(4),
                [("S11",(0,0),"#1f77b4"),("S12",(0,1),"#d62728"),
                 ("S21",(1,0),"#2ca02c"),("S22",(1,1),"#ff7f0e")]):
            s_r = S_raw[:,r,c]; s_d = S_deemb[:,r,c]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=f_ghz, y=s_r.real, mode="lines",
                name="Re (raw)",   line=dict(color=color,  width=2.2)))
            fig.add_trace(go.Scatter(x=f_ghz, y=s_r.imag, mode="lines",
                name="Im (raw)",   line=dict(color=color,  width=2.2, dash="dot")))
            fig.add_trace(go.Scatter(x=f_ghz, y=s_d.real, mode="lines",
                name="Re (deemb)", line=dict(color="#888", width=1.6, dash="dash")))
            fig.add_trace(go.Scatter(x=f_ghz, y=s_d.imag, mode="lines",
                name="Im (deemb)", line=dict(color="#aaa", width=1.6, dash="longdash")))
            fig.update_layout(title=dict(text=sname, font=dict(size=12)),
                xaxis_title="GHz", plot_bgcolor="white", paper_bgcolor="white", height=300,
                legend=dict(x=0.01, y=0.99, xanchor="left", yanchor="top",
                            bgcolor="rgba(255,255,255,0.85)", bordercolor="#ccc",
                            borderwidth=1, font=dict(size=7.5)),
                margin=dict(l=42,r=6,t=32,b=42), hovermode="x unified")
            fig.update_xaxes(showgrid=True, gridcolor="#ebebeb")
            fig.update_yaxes(showgrid=True, gridcolor="#ebebeb")
            col_w.plotly_chart(fig, use_container_width=True, key=f"cmp_{sname}_{fname}")


# ════════════════════════════════════════════════════════════════════════════════
# Helper: fT / fmax Bode plot
# ════════════════════════════════════════════════════════════════════════════════

def _compute_h21_U(S):
    """Compute |h21|² (dB) and Mason's U (dB) from S-parameters."""
    Y = s_to_y(S, 50.0)
    y11, y12, y21, y22 = Y[:,0,0], Y[:,0,1], Y[:,1,0], Y[:,1,1]
    with np.errstate(divide="ignore", invalid="ignore"):
        h21     = -y21 / (y11 + 1e-30)
        h21_db  = 10.0*np.log10(np.abs(h21)**2 + 1e-30)
        num_u   = np.abs(y21 - y12)**2
        den_u   = 4.0*(y11.real*y22.real - y12.real*y21.real)
        U       = np.where(den_u > 0, num_u/den_u, np.nan)
        U_db    = 10.0*np.log10(np.abs(U) + 1e-30)
    return h21_db, U_db


def render_ft_fmax_overlay(S_raw, sim_results: dict[str, np.ndarray], freq, fname):
    """
    Bode plot: |h21|² and Mason U for measured + all simulated models.

    sim_results: {model_SHORT: S_sim_array}  (None values are skipped)
    """
    f_ghz      = freq * 1e-9
    h21_mea, U_mea = _compute_h21_U(S_raw)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=f_ghz, y=h21_mea, mode="lines",
        name="|h21|² Meas.", line=dict(color="#1f77b4", width=2.5)))
    fig.add_trace(go.Scatter(x=f_ghz, y=U_mea, mode="lines",
        name="Mason U Meas.", line=dict(color="#1f77b4", width=2.5, dash="dash")))

    palette = ["#d62728","#2ca02c","#9467bd","#8c564b","#e377c2"]
    for (short, S_sim), col in zip(sim_results.items(), palette):
        if S_sim is None: continue
        h21_s, U_s = _compute_h21_U(S_sim)
        fig.add_trace(go.Scatter(x=f_ghz, y=h21_s, mode="lines",
            name=f"|h21|² {short}", line=dict(color=col, width=2.0, dash="dash")))
        fig.add_trace(go.Scatter(x=f_ghz, y=U_s, mode="lines",
            name=f"Mason U {short}", line=dict(color=col, width=2.0, dash="longdash")))

    fig.add_hline(y=0, line_color="#333", line_width=1.2,
                   annotation_text="0 dB", annotation_position="right",
                   annotation_font=dict(size=9))
    fig.update_layout(
        title="Gain vs Frequency — Measured vs Modeled",
        xaxis=dict(title="Frequency (GHz)", type="log",
                   showgrid=True, gridcolor="#ebebeb"),
        yaxis=dict(title="Gain (dB)", range=[0, 50],
                   showgrid=True, gridcolor="#ebebeb"),
        plot_bgcolor="white", paper_bgcolor="white", height=450,
        legend=dict(x=1.01, y=1.0, xanchor="left", yanchor="top",
                    bgcolor="rgba(255,255,255,0.92)", bordercolor="#ccc",
                    borderwidth=1, font=dict(size=9)),
        hovermode="x unified", margin=dict(l=55,r=20,t=50,b=50))
    st.plotly_chart(fig, use_container_width=True, key=f"ftfmax_{fname}")
    st.caption("Y-axis fixed 0–50 dB.")


# ════════════════════════════════════════════════════════════════════════════════
# Helper: Re(Z12) vs 1/IE
# ════════════════════════════════════════════════════════════════════════════════

def render_rz12_section(all_data, para_eff, fname):
    """
    Multi-bias Re(Z₁₂) vs 1/IE fit for η and Re extraction.
    Gao [3] Ch. 5.5.1:  Re(Z₁₂) = (ηkT/q)·(1/IE) + Re
    """
    st.markdown("#### 📈 Re(Z₁₂) vs 1/IE")
    st.caption("Gao [3] Ch. 5.5.1.")
    st.latex(r"\mathrm{Re}(Z_{12})=\frac{\eta kT}{q}\cdot\frac{1}{I_E}+R_e")
    if not all_data:
        st.info("No DUT files loaded."); return

    for fn in all_data:
        for suf, dv in [("use", True), ("Ie", 0.0)]:
            gk = f"rz12_{suf}_{fn}"
            if gk not in st.session_state: st.session_state[gk] = dv

    ref_fn  = list(all_data.keys())[0]
    f_ref   = all_data[ref_fn]["freq"] * 1e-9
    f_min_v = float(f_ref[max(1, np.searchsorted(f_ref, 0.01))])
    f_max_v = float(f_ref[-1])
    col_fq, _ = st.columns([1, 2])
    f_extract = col_fq.number_input(
        "Z₁₂ freq (GHz)", min_value=f_min_v, max_value=f_max_v,
        value=min(1.0, f_max_v*0.05), step=0.5, format="%.2f",
        key=f"rz12_fext_{fname}")

    st.markdown("**Files:**")
    for h, t in zip(st.columns([0.3,2.3,1.2,1.4,1.4]),
                    ["","File","IE (mA)","Re(Z₁₂) (Ω)","Rbe*"]):
        h.markdown(f"<small><b>{t}</b></small>", unsafe_allow_html=True)

    points = []; Re_ref = para_eff.get("Rpe", 0.0)
    for fn, d in all_data.items():
        c0,c1,c2,c3,c4 = st.columns([0.3,2.3,1.2,1.4,1.4])
        use = c0.checkbox("", key=f"rz12_use_{fn}__{fname}", value=st.session_state[f"rz12_use_{fn}"],
                           label_visibility="collapsed")
        st.session_state[f"rz12_use_{fn}"] = use
        c1.markdown(f"<small>{Path(fn).stem}</small>", unsafe_allow_html=True)
        if not use: continue
        Ie = c2.number_input("", min_value=0.0, step=0.1, format="%.3f",
                              key=f"rz12_Ie_{fn}__{fname}",
                              value=float(st.session_state[f"rz12_Ie_{fn}"]),
                              label_visibility="collapsed")
        st.session_state[f"rz12_Ie_{fn}"] = Ie
        try:
            idx    = int(np.argmin(np.abs(d["freq"]*1e-9 - f_extract)))
            Y_ex1f = peel_parasitics(d["S_raw"], d["freq"], d["z0"], para_eff)
            ReZ12  = float(y_to_z(Y_ex1f)[idx,0,1].real)
            c3.markdown(f"**{ReZ12:.4f}**")
            c4.markdown(f"<small>{ReZ12-Re_ref:.4f}</small>", unsafe_allow_html=True)
            if Ie > 0: points.append((1.0/(Ie*1e-3), ReZ12, Path(fn).stem))
        except Exception as ex:
            c3.markdown(f"*err:{ex}*")

    if len(points) < 2:
        st.caption("Need ≥ 2 files with IE for fit."); return

    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
    lbl = [p[2] for p in points]
    try:
        slope, Re_fit = np.polyfit(x, y, 1)
        eta = slope / (1.381e-23 * 300 / 1.602e-19)
        x_fit = np.linspace(0, max(x)*1.08, 200)
        y_fit = slope*x_fit + Re_fit
    except Exception as ex:
        st.error(f"Fit failed: {ex}"); return

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers+text", text=lbl,
        textposition="top center", name="Re(Z₁₂)",
        marker=dict(size=11, color="#1f77b4", line=dict(color="#0d4a7a", width=1.5))))
    fig.add_trace(go.Scatter(x=x_fit, y=y_fit, mode="lines",
        name=f"Fit Re={Re_fit:.4f} Ω  η={eta:.3f}",
        line=dict(color="#d62728", width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=[0], y=[Re_fit], mode="markers",
        name=f"Re={Re_fit:.4f} Ω",
        marker=dict(size=14, symbol="star", color="#d62728")))
    fig.update_layout(
        title=f"Re(Z₁₂) vs 1/IE @ {f_extract:.2f} GHz",
        xaxis=dict(title="1/IE (A⁻¹)", rangemode="tozero",
                   showgrid=True, gridcolor="#ebebeb"),
        yaxis=dict(title="Re(Z₁₂) (Ω)", showgrid=True, gridcolor="#ebebeb"),
        plot_bgcolor="white", paper_bgcolor="white", height=380,
        legend=dict(x=0.01, y=0.99, xanchor="left", yanchor="top",
                    bgcolor="rgba(255,255,255,0.9)", bordercolor="#ccc",
                    borderwidth=1, font=dict(size=9)),
        margin=dict(l=55,r=20,t=50,b=50))
    st.plotly_chart(fig, use_container_width=True, key=f"rz12_{fname}")

    current_stem  = Path(fname).stem
    current_idx   = next((i for i, l in enumerate(lbl) if l == current_stem), None)
    if current_idx is not None:
        st.session_state[f"rz12_Rbe_{fname}"] = y[current_idx] - Re_fit
    else:
        st.session_state.pop(f"rz12_Rbe_{fname}", None)
    st.session_state[f"rz12_Re_{fname}"] = Re_fit

    mc1, mc2 = st.columns(2)
    mc1.metric("Re (intercept)", f"{Re_fit:.4f} Ω",
               delta=f"{Re_fit - para_eff.get('Rpe',0):+.4f} vs open-short")
    if current_idx is not None:
        mc2.metric(f"Rbe ({current_stem})", f"{y[current_idx]-Re_fit:.4f} Ω")
    else:
        mc2.info("Current file not in fit.")

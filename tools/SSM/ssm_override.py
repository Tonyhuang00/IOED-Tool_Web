"""
ssm_override.py — Unified pre-extraction parameter override UI.

Also contains make_topology_fig (circuit schematic), kept here so this module
covers "everything the user sees before model-specific extraction".
"""
from __future__ import annotations
import numpy as np
import streamlit as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

from .models.base_ui import PAD_SPECS


# ════════════════════════════════════════════════════════════════════════════════
# Unified pre-extraction override
# ════════════════════════════════════════════════════════════════════════════════

def render_unified_pre_override(fname, para_step1, cold_res, rz12_Re):
    """
    Show and return the unified pad parameter set used by all models.

    Priority for Rb/Rc/Re: highest value among available sources
    (Short Step 1b, Cold-HBT, Z-parameter method) is pre-selected.

    Returns
    -------
    para_eff : dict   Effective pad parameters (SI units).
    """
    st.markdown("---")
    st.markdown("### ⚙️ Pre-Extraction Parameter Review")
    st.caption(
        "All pad parameters feeding every model extraction.  \n"
        "For Rb/Rc/Re the source with the highest value is pre-selected.")
    para_eff = para_step1.copy()

    # Collect all available resistance sources
    src_Rb = {"Short (Step 1b)": para_step1.get("Rpb", 0.0)}
    src_Rc = {"Short (Step 1b)": para_step1.get("Rpc", 0.0)}
    src_Re = {"Short (Step 1b)": para_step1.get("Rpe", 0.0)}
    if cold_res is not None:
        src_Rb["Cold-HBT"] = float(cold_res.get("Rb_cold", 0.0))
        src_Rc["Cold-HBT"] = float(cold_res.get("Rc_cold", 0.0))
        src_Re["Cold-HBT"] = float(cold_res.get("Re_cold", 0.0))
    if rz12_Re is not None:
        src_Re["Z-parameter method"] = float(rz12_Re)

    def _best(d): return max(d, key=lambda k: d[k])
    def _src_lbl(k, v): return f"{k}: {v:.4f} Ω"

    _cap_keys = ["Cpbe","Cpce","Cpbc"]
    _ind_keys = ["Lb","Lc","Le"]

    # Sync preov_ caps/inds whenever upstream ov_ (Open/Short) values change
    _step1_hash = tuple(
        round(para_step1.get(k, 0.0) * 1e18)
        for k in _cap_keys + _ind_keys
    )
    if st.session_state.get(f"preov_step1_hash_{fname}") != _step1_hash:
        for k in _cap_keys:
            st.session_state[f"preov_{k}_{fname}"] = para_step1.get(k, 0.0) * 1e15
        for k in _ind_keys:
            st.session_state[f"preov_{k}_{fname}"] = para_step1.get(k, 0.0) * 1e12
        st.session_state[f"preov_step1_hash_{fname}"] = _step1_hash
    for var, sources in [("Rb",src_Rb),("Rc",src_Rc),("Re",src_Re)]:
        sk_src = f"preov_src_{var}_{fname}"
        sk_val = f"preov_{var}_{fname}"
        if sk_src not in st.session_state:
            st.session_state[sk_src] = _best(sources)
        if sk_val not in st.session_state:
            st.session_state[sk_val] = sources.get(st.session_state[sk_src],
                                                    list(sources.values())[0])

    with st.expander("✏️ Inspect / override pad parameters", expanded=True):
        if st.button("↩️ Reset all to defaults (highest source)",
                     key=f"preov_reset_{fname}"):
            for k in _cap_keys:
                st.session_state[f"preov_{k}_{fname}"] = para_step1.get(k,0.0) * 1e15
            for k in _ind_keys:
                st.session_state[f"preov_{k}_{fname}"] = para_step1.get(k,0.0) * 1e12
            for var, sources in [("Rb",src_Rb),("Rc",src_Rc),("Re",src_Re)]:
                bk = _best(sources)
                st.session_state[f"preov_src_{var}_{fname}"] = bk
                st.session_state[f"preov_{var}_{fname}"]     = sources[bk]
            st.rerun()

        st.markdown("**Pad capacitances** *(from Open, single source)*")
        for col_w, (k, lbl) in zip(st.columns(3),
                                    [("Cpbe","Cpbe (fF)"),
                                     ("Cpce","Cpce (fF)"),
                                     ("Cpbc","Cpbc (fF)")]):
            col_w.number_input(lbl, key=f"preov_{k}_{fname}", format="%.4f", step=0.1)

        st.markdown("**Lead inductances** *(from Short, single source)*")
        for col_w, (k, lbl) in zip(st.columns(3),
                                    [("Lb","Lb (pH)"),
                                     ("Lc","Lc (pH)"),
                                     ("Le","Le (pH)")]):
            col_w.number_input(lbl, key=f"preov_{k}_{fname}", format="%.3f", step=0.1)

        st.markdown("**Series resistances** *(choose source — default = highest)*")
        for var, sources, label in [
            ("Rb", src_Rb, "**Rb = Rpb** — base"),
            ("Rc", src_Rc, "**Rc = Rpc** — collector"),
            ("Re", src_Re, "**Re = Rpe** — emitter"),
        ]:
            src_opts  = list(sources.keys()) + ["Custom"]
            sk_src    = f"preov_src_{var}_{fname}"
            sk_val    = f"preov_{var}_{fname}"
            cur_src   = st.session_state.get(sk_src, _best(sources))
            if cur_src not in src_opts: cur_src = src_opts[0]
            radio_lbls = [_src_lbl(k, v) for k, v in sources.items()] + ["Custom"]
            cur_idx    = src_opts.index(cur_src)
            st.markdown(label)
            sc1, sc2 = st.columns([3, 1])
            chosen_lbl = sc1.radio("", radio_lbls, index=cur_idx,
                                    key=f"preov_radio_{var}_{fname}",
                                    horizontal=True, label_visibility="collapsed")
            chosen_src = src_opts[radio_lbls.index(chosen_lbl)]
            st.session_state[sk_src] = chosen_src
            if chosen_src != "Custom":
                resolved = sources[chosen_src]
                st.session_state[sk_val] = resolved
                sc2.metric(var, f"{resolved:.4f} Ω")
            else:
                sc2.number_input(f"{var} (Ω)", key=sk_val, format="%.4f", step=0.01)

    # Assemble para_eff from session state
    for k in _cap_keys:
        para_eff[k] = st.session_state.get(f"preov_{k}_{fname}",
                                            para_step1.get(k,0.0)*1e15) / 1e15
    for k in _ind_keys:
        para_eff[k] = st.session_state.get(f"preov_{k}_{fname}",
                                            para_step1.get(k,0.0)*1e12) / 1e12
    para_eff["Rpb"] = st.session_state.get(f"preov_Rb_{fname}", para_step1.get("Rpb",0.0))
    para_eff["Rpc"] = st.session_state.get(f"preov_Rc_{fname}", para_step1.get("Rpc",0.0))
    para_eff["Rpe"] = st.session_state.get(f"preov_Re_{fname}", para_step1.get("Rpe",0.0))
    return para_eff


# ════════════════════════════════════════════════════════════════════════════════
# Circuit schematic
# ════════════════════════════════════════════════════════════════════════════════

def make_topology_fig(params, topology="T"):
    """
    Draw the HBT SSM circuit schematic for topology in {"T","pi","D"}.
    Uses matplotlib; returns figure (caller closes with plt.close).
    """
    p = params or {}
    C_PAD="#E67E22"; C_EXT="#27AE60"; C_INT="#2980B9"; C_SRC="#C0392B"; C_WIR="#2C3E50"
    fig, ax = plt.subplots(figsize=(17,7))
    ax.set_xlim(-0.5,17); ax.set_ylim(-1.0,8.5); ax.axis("off")
    ax.set_facecolor("white"); fig.patch.set_facecolor("white"); lw=1.8

    def wl(x1,y1,x2,y2,c=C_WIR,lw_=None):
        ax.plot([x1,x2],[y1,y2],color=c,lw=lw_ or lw,solid_capstyle="round",zorder=2)
    def dot(x,y): ax.plot(x,y,"o",color=C_WIR,ms=5.5,zorder=6)
    def _fv(key,scale,unit,d=2):
        v=p.get(key)
        if v is None: return ""
        try: fv=float(v); return "" if not np.isfinite(fv) else f"{fv*scale:.{d}f} {unit}"
        except: return ""
    def box(cx,cy,sym,lbl,val,color,w=0.88,h=0.42):
        rect=FancyBboxPatch((cx-w/2,cy-h/2),w,h,boxstyle="round,pad=0.04",
                             fc=color+"28",ec=color,lw=1.9,zorder=4); ax.add_patch(rect)
        ax.text(cx,cy+0.04,sym,ha="center",va="center",fontsize=9,fontweight="bold",color=color,zorder=5)
        ax.text(cx,cy+h/2+0.12,lbl,ha="center",va="bottom",fontsize=8,color="#222",style="italic",zorder=5)
        if val: ax.text(cx,cy-h/2-0.10,val,ha="center",va="top",fontsize=7,color="#555",zorder=5)
    def rc_block(cx,cy,rlbl,rval,clbl,cval,color,w=0.62,h=1.10):
        rect=FancyBboxPatch((cx-w/2,cy-h/2),w,h,boxstyle="round,pad=0.04",
                             fc=color+"18",ec=color,lw=1.7,ls="dashed",zorder=4); ax.add_patch(rect)
        ax.text(cx,cy+h*0.22,rlbl,ha="center",va="center",fontsize=8.5,color=color,style="italic",zorder=5)
        if rval: ax.text(cx,cy+h*0.05,rval,ha="center",va="center",fontsize=7,color="#555",zorder=5)
        ax.plot([cx-w*0.3,cx+w*0.3],[cy-h*0.02]*2,color=color,lw=1,zorder=5)
        ax.text(cx,cy-h*0.20,clbl,ha="center",va="center",fontsize=8.5,color=color,style="italic",zorder=5)
        if cval: ax.text(cx,cy-h*0.37,cval,ha="center",va="center",fontsize=7,color="#555",zorder=5)
    def cap_v(cx,y_top,y_bot,lbl,val,color,pw=0.28):
        ctr=(y_top+y_bot)/2; wl(cx,y_top,cx,ctr+0.07); wl(cx,ctr-0.07,cx,y_bot)
        ax.plot([cx-pw,cx+pw],[ctr+0.07]*2,color=color,lw=2.8,zorder=5)
        ax.plot([cx-pw,cx+pw],[ctr-0.07]*2,color=color,lw=2.8,zorder=5)
        ax.text(cx+pw+0.13,ctr,lbl,ha="left",va="center",fontsize=8,color="#222",style="italic",zorder=5)
        if val: ax.text(cx+pw+0.13,ctr-0.28,val,ha="left",va="center",fontsize=7,color="#555",zorder=5)
    def cur_src(cx,y_top,y_bot,lbl):
        ctr=(y_top+y_bot)/2; r=0.34
        circ=plt.Circle((cx,ctr),r,fc="white",ec=C_SRC,lw=2.0,zorder=4); ax.add_patch(circ)
        ax.annotate("",xy=(cx,ctr+r*0.55),xytext=(cx,ctr-r*0.55),
                     arrowprops=dict(arrowstyle="->",color=C_SRC,lw=2.0),zorder=5)
        wl(cx,y_top,cx,ctr+r); wl(cx,ctr-r,cx,y_bot)
        ax.text(cx+r+0.15,ctr,lbl,ha="left",va="center",fontsize=8,color=C_SRC,zorder=5)
    def port(x,y,lbl,color=C_INT):
        ax.plot(x,y,"o",ms=18,color=color,alpha=0.2,zorder=3)
        ax.plot(x,y,"o",ms=9,color=color,zorder=4)
        ax.text(x,y,lbl,ha="center",va="center",fontsize=10,fontweight="bold",color="white",zorder=5)
    def layer_box(x0,y0,x1,y1,label,color):
        rect=FancyBboxPatch((x0,y0),x1-x0,y1-y0,boxstyle="round,pad=0.08",
                             fc=color+"08",ec=color,lw=1.5,ls="dotted",zorder=0); ax.add_patch(rect)
        ax.text(x0+0.15,y1-0.12,label,ha="left",va="top",fontsize=8,color=color,style="italic",zorder=1)

    y_top=5.5; y_emi=1.5; y_gnd=0.2
    xB=0.7; xLb=1.85; xRb=3.0; xn1=3.85; xRbi=5.0; xn2=5.9; xZbc=8.4
    xn3=10.8; xRc=11.9; xLc=13.1; xC=14.2
    xCbex=xn1; xZbe=xn2; xCbcx=xn3; xSrc=xn3+1.3; xRe=7.2; xLe=9.0

    # GND bus
    wl(xB-0.5,y_gnd,xC+0.5,y_gnd,C_WIR,2.0)
    wl(xB-0.5,y_emi,xB-0.5,y_gnd); wl(xC+0.5,y_emi,xC+0.5,y_gnd)
    for xg in [1.0,xC+0.5]:
        for k in range(3): ax.plot([xg-0.32+k*0.08,xg+0.32-k*0.08],[y_gnd-k*0.13]*2,color=C_WIR,lw=2)
    # Emitter rail
    wl(xB-0.5,y_emi,xCbex-0.2,y_emi); wl(xLe+0.47,y_emi,xSrc+0.4,y_emi)
    wl(xSrc+0.4,y_emi,xC+0.5,y_emi);  wl(xCbex-0.2,y_emi,xRe-0.44,y_emi)
    box(xRe,y_emi,"R","Re",_fv("Rpe",1,"Ω"),C_PAD)
    wl(xRe+0.44,y_emi,xLe-0.44,y_emi)
    box(xLe,y_emi,"L","Le",_fv("Le",1e12,"pH"),C_PAD)
    wl(xLe+0.44,y_emi,xLe+0.6,y_emi)
    port(xLe+0.85,y_emi,"E","#7F8C8D")
    # Top rail
    port(xB,y_top,"B"); wl(xB,y_top,xLb-0.44,y_top)
    box(xLb,y_top,"L","Lb",_fv("Lb",1e12,"pH"),C_PAD); wl(xLb+0.44,y_top,xRb-0.44,y_top)
    box(xRb,y_top,"R","Rb",_fv("Rpb",1,"Ω"),C_PAD);   wl(xRb+0.44,y_top,xn1,y_top); dot(xn1,y_top)
    wl(xn1,y_top,xRbi-0.44,y_top)
    box(xRbi,y_top,"R","Zbi" if topology=="D" else "Rbi",_fv("Rbi",1,"Ω"),C_INT)
    wl(xRbi+0.44,y_top,xn2,y_top); dot(xn2,y_top)
    wl(xn2,y_top,xn3,y_top); dot(xn3,y_top); wl(xn3,y_top,xRc-0.44,y_top)
    box(xRc,y_top,"R","Rc",_fv("Rpc",1,"Ω"),C_PAD); wl(xRc+0.44,y_top,xLc-0.44,y_top)
    box(xLc,y_top,"L","Lc",_fv("Lc",1e12,"pH"),C_PAD); wl(xLc+0.44,y_top,xC,y_top)
    port(xC,y_top,"C")
    # Rbc/Cbc branch
    y_zbc=y_top-1.35; wl(xn2,y_top,xn2,y_zbc); wl(xn3,y_top,xn3,y_zbc)
    wl(xn2,y_zbc,xZbc-0.32,y_zbc); wl(xZbc+0.32,y_zbc,xn3,y_zbc)
    rval_bc=(_fv("Rbc",1e-3,"kΩ") if (p.get("Rbc") or 0)>1000 else _fv("Rbc",1,"Ω"))
    rc_block(xZbc,y_zbc,"Rbc",rval_bc,"Cbc",_fv("Cbc",1e15,"fF"),C_INT)
    # Degachi Ccx or Cheng Cbex
    if topology=="D" and p.get("Ccx") is not None:
        cap_v(xCbex,y_top,y_emi,"Ccx",_fv("Ccx",1e15,"fF"),C_EXT)
        ax.text(xRbi,y_top+0.65,"Cbi:"+_fv("Cbi",1e15,"fF"),ha="center",fontsize=7,color=C_INT,zorder=5)
    else:
        cap_v(xCbex,y_top,y_emi,"Cbex",_fv("Cbex",1e15,"fF"),C_EXT)
    # Rbe/Cbe
    wl(xZbe,y_top,xZbe,y_top-0.25)
    rval_be=(_fv("Rbe",1e-3,"kΩ") if (p.get("Rbe") or 0)>1000 else _fv("Rbe",1,"Ω"))
    cval_be=(_fv("Cbe",1e12,"pF") if (p.get("Cbe") or 0)>1e-12 else _fv("Cbe",1e15,"fF"))
    rc_block(xZbe,(y_top+y_emi)/2-0.15,"Rbe",rval_be,"Cbe",cval_be,C_INT,h=1.2)
    wl(xZbe,y_top-0.25,xZbe,(y_top+y_emi)/2-0.15+0.62)
    wl(xZbe,(y_top+y_emi)/2-0.15-0.62,xZbe,y_emi)
    # Cbcx (not Degachi)
    if topology != "D":
        cap_v(xCbcx,y_top,y_emi,"Cbcx",_fv("Cbcx",1e15,"fF"),C_EXT)
    # Current source
    wl(xn3,y_top,xSrc,y_top); wl(xSrc,y_emi+0.36,xSrc,y_emi)
    cur_src(xSrc,y_top,y_emi+0.36,"α·IE" if topology=="T" else "gm·Vbe")
    # Pad caps B/C
    xCpbe=xB-0.5; wl(xB,y_top,xCpbe,y_top)
    cap_v(xCpbe,y_top,y_emi,"Cpbe",_fv("Cpbe",1e15,"fF"),C_PAD)
    xCpce=xC+0.5; wl(xC,y_top,xCpce,y_top)
    cap_v(xCpce,y_top,y_emi,"Cpce",_fv("Cpce",1e15,"fF"),C_PAD)
    # Cpbc across top
    y_cpbc=7.3; xm=(xB+xC)/2
    wl(xB,y_top,xB,y_cpbc); wl(xC,y_top,xC,y_cpbc)
    wl(xB,y_cpbc,xm-0.18,y_cpbc)
    ax.plot([xm-0.18]*2,[y_cpbc-0.22,y_cpbc+0.22],color=C_PAD,lw=2.8,zorder=5)
    ax.plot([xm+0.18]*2,[y_cpbc-0.22,y_cpbc+0.22],color=C_PAD,lw=2.8,zorder=5)
    wl(xm+0.18,y_cpbc,xC,y_cpbc)
    ax.text(xm,y_cpbc+0.32,"Cpbc",ha="center",va="bottom",fontsize=8,color="#222",style="italic")
    ax.text(xm,y_cpbc-0.35,_fv("Cpbc",1e15,"fF"),ha="center",va="top",fontsize=7,color="#555")
    # Layer annotation boxes
    layer_box(xn2-0.2,y_emi-0.4,xn3+0.2,y_top+0.55,"Intrinsic Model",C_INT)
    if topology != "D":
        layer_box(xn1-0.2,y_emi-0.6,xn3+0.2,y_top+0.75,"Extrinsic Distributed Caps",C_EXT)
    # Legend
    handles=[
        mpatches.Patch(fc=C_PAD+"40",ec=C_PAD,lw=1.5,label="Pad Parasitics"),
        mpatches.Patch(fc=C_EXT+"40",ec=C_EXT,lw=1.5,label="Extrinsic Caps"),
        mpatches.Patch(fc=C_INT+"40",ec=C_INT,lw=1.5,label="Intrinsic Model"),
        mpatches.Patch(fc="white",   ec=C_SRC,lw=1.5,label="Current Source"),
    ]
    ax.legend(handles=handles,loc="lower left",fontsize=8.5,framealpha=0.95,
              edgecolor="#ccc",ncol=2)
    tname = {"T":"T-topology (Cheng 2022)","pi":"π-topology (Cheng 2022)",
             "D":"Degachi (2008) augmented π"}.get(topology,"")
    ax.set_title(f"HBT Small-Signal Model — {tname}",fontsize=13,fontweight="bold",pad=10)
    plt.tight_layout(pad=0.4)
    return fig

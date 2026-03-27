"""
hbt_rf_tool.py  — Beta v3.1
============================
Main Streamlit application for HBT RF extraction.

SSM Extraction functionality lives entirely in ssm_extraction.py.
The only SSM-related line in this file is:

    from cheng_extraction import render_ssm_tab

which is then called once inside the "SSM Extraction" sub-tab.
"""
import io, re, zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

from tools.SSM.ssm_extraction import render_ssm_tab   # ← SSM module (Cheng 2022)

# ─────────────────────────────────────────────────────────────────────────────
if "rf_uploader_key" not in st.session_state:
    st.session_state["rf_uploader_key"] = 0

st.title("📡 IOED HBT RF Extraction Tool (Beta v3.1)")
st.caption(
    "**v3.1**: SSM extraction lives in `ssm_extraction.py` — "
    "T & π topology, analytical peeling, schematic, measured-vs-modeled overlay. "
    "Pad resistances correctly named R_pb / R_pc / R_pe (Gao, Wiley 2015 §4.2)."
)

# ═════════════════════════════════════════════════════════════════════════════
#  CORE RF UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def parse_s2p(content: str):
    freq_unit, fmt, z0, data_lines = "hz", "ma", 50.0, []
    for line in content.splitlines():
        s = line.strip()
        if not s or s.startswith("!"): continue
        if s.startswith("#"):
            parts = s[1:].lower().split()
            for i, p in enumerate(parts):
                if p in ("hz","khz","mhz","ghz"): freq_unit = p
                elif p in ("ma","db","ri"): fmt = p
                elif p == "r" and i+1 < len(parts):
                    try: z0 = float(parts[i+1])
                    except: pass
            continue
        data_lines.append(s)
    vals = np.array([float(x) for x in " ".join(data_lines).split()])
    n = len(vals) // 9
    vals = vals[:n*9].reshape(n, 9)
    freq = vals[:,0]*{"hz":1.0,"khz":1e3,"mhz":1e6,"ghz":1e9}[freq_unit]
    def to_c(ca, cb):
        a, b = vals[:,ca], vals[:,cb]
        if fmt == "db": return 10**(a/20.0)*np.exp(1j*np.deg2rad(b))
        if fmt == "ma": return a*np.exp(1j*np.deg2rad(b))
        return a + 1j*b
    S = np.zeros((n,2,2), dtype=complex)
    for (r,c),(ca,cb) in zip([(0,0),(1,0),(0,1),(1,1)],[(1,2),(3,4),(5,6),(7,8)]):
        S[:,r,c] = to_c(ca, cb)
    return freq, S, z0


def s_to_y(S, z0=50.0):
    s11,s12,s21,s22 = S[:,0,0],S[:,0,1],S[:,1,0],S[:,1,1]
    d = (1+s11)*(1+s22)-s12*s21
    Y = np.zeros_like(S)
    Y[:,0,0] = ((1-s11)*(1+s22)+s12*s21)/(d*z0)
    Y[:,0,1] = -2.0*s12/(d*z0)
    Y[:,1,0] = -2.0*s21/(d*z0)
    Y[:,1,1] = ((1+s11)*(1-s22)+s12*s21)/(d*z0)
    return Y


def y_to_s(Y, z0=50.0):
    S = np.zeros_like(Y)
    I = np.eye(2)
    for i in range(len(Y)):
        yn = Y[i]*z0
        try: S[i] = np.dot(I-yn, np.linalg.inv(I+yn))
        except: S[i] = np.full((2,2), np.nan+0j)
    return S


def _inv2(M):
    out = np.zeros_like(M)
    for i in range(len(M)):
        try: out[i] = np.linalg.inv(M[i])
        except: out[i] = np.full((2,2), np.nan+0j)
    return out

y_to_z = z_to_y = _inv2


def deembed_open_short(Y_dut, Y_open, Y_short):
    """Standard open-short de-embedding (Gao 2015 §4.2). Verified correct."""
    return z_to_y(y_to_z(Y_dut - Y_open) - y_to_z(Y_short - Y_open))


def deembed_thru_half(Y_dut, Y_thru_deemb):
    """THRU/2 half-impedance subtraction. Verified correct."""
    return z_to_y(y_to_z(Y_dut) - 0.5*y_to_z(Y_thru_deemb))


def strict_freq_check(f_dut, f_dummy, dummy_name):
    if len(f_dut) != len(f_dummy) or not np.allclose(f_dut, f_dummy, rtol=1e-5):
        raise ValueError(f"DUT and {dummy_name} frequency grids differ.")


def compute_metrics(Y, freq_hz):
    f = freq_hz*1e-9
    y11,y12,y21,y22 = Y[:,0,0],Y[:,0,1],Y[:,1,0],Y[:,1,1]
    with np.errstate(divide="ignore", invalid="ignore"):
        h21 = -y21/y11
        num_u = np.abs(y21-y12)**2
        den_u = 4.0*(y11.real*y22.real - y12.real*y21.real)
        U = np.where(den_u>0, num_u/den_u, np.nan)
        num_k = 2.0*y11.real*y22.real-(y12*y21).real
        K = num_k/(np.abs(y12*y21)+1e-60)
        MSG = np.abs(y21)/(np.abs(y12)+1e-30)
        MAG = MSG*(K-np.sqrt(np.clip(K**2-1.0,0,None)))
        MAG_MSG = np.where(K>1.0, MAG, MSG)
    return pd.DataFrame({
        "Freq (GHz)":f, "|h21|² (dB)":10*np.log10(np.abs(h21)**2+1e-30),
        "Mason U (dB)":10*np.log10(np.abs(U)+1e-30),
        "MAG/MSG (dB)":10*np.log10(np.abs(MAG_MSG)+1e-30),
        "K Factor":K, "fT Plateau (GHz)":f*np.abs(h21),
        "fmax U Plateau (GHz)":f*np.sqrt(np.abs(U)),
        "fmax MAG Plateau (GHz)":f*np.sqrt(np.abs(MAG_MSG)),
    })


def extract_limit(freq_ghz, gain_db, plateau_arr, n_pts, f_min, f_max):
    vm = (freq_ghz>=f_min)&(freq_ghz<=f_max)&~np.isnan(gain_db)
    if not np.any(vm): return np.nan, np.nan, "No Data"
    f_v,g_v,p_v,N = freq_ghz[vm],gain_db[vm],plateau_arr[vm],vm.sum()
    if np.nanmax(g_v)<=0: return np.nan, np.nan, "No Gain"
    above = g_v>=0
    crossings = np.where(above[:-1]&~above[1:])[0]
    genuine_idx = None
    for idx in crossings[::-1]:
        cnt=0
        for j in range(idx,-1,-1):
            if above[j]: cnt+=1
            else: break
        if cnt<10: continue
        if max(0,idx-cnt+1)>int(0.80*N) and cnt<20: continue
        genuine_idx=idx; break
    if genuine_idx is None:
        if np.median(g_v)>0:
            v_plat=np.nanmax(p_v) if not np.isnan(p_v).all() else np.nan
            n_use,v_extrap=min(n_pts,len(f_v)),np.nan
            if len(f_v[-n_use:])>=2:
                with np.errstate(all="ignore"):
                    m,c=np.polyfit(np.log10(f_v[-n_use:]),g_v[-n_use:],1)
                    if m<0: v_extrap=10**(-c/m)
            return v_extrap,v_plat,"Extrap & Plat."
        return np.nan,np.nan,"No Gain"
    idx=genuine_idx
    s,e=max(0,idx-n_pts//2+1),min(N,idx+n_pts//2+1+(n_pts%2))
    if (e-s)<2: s,e=max(0,idx),min(N,idx+2)
    with np.errstate(all="ignore"):
        v_cross=np.polyval(np.polyfit(g_v[s:e],f_v[s:e],min(2,e-s-1)),0.0)
        if v_cross<=0 or v_cross<f_v[s] or v_cross>f_v[e-1]:
            v_cross=f_v[idx]+(0-g_v[idx])*(f_v[idx+1]-f_v[idx])/(g_v[idx+1]-g_v[idx])
    return v_cross,np.nan,"0dB Cross"


def process_dut(content, filename, s1_o, s1_s, s2_o, s2_s, s3_t, n_pts, f_min, f_max):
    """Parse, de-embed, compute metrics, extract fT/fmax for one DUT file."""
    freq,S_raw,z0 = parse_s2p(content)
    Y_raw = s_to_y(S_raw,z0)
    df_raw = compute_metrics(Y_raw,freq)
    Y_fin,stages,d1_o,d1_s = Y_raw,[],None,None
    if s1_o and s1_s:
        f1o,S1o,z1o=s1_o; f1s,S1s,z1s=s1_s
        strict_freq_check(freq,f1o,"Probe Open")
        d1_o,d1_s=s_to_y(S1o,z1o),s_to_y(S1s,z1s)
        Y_fin=deembed_open_short(Y_fin,d1_o,d1_s); stages.append("Probe")
    Y2o=Y2s=None
    if s2_o and s2_s:
        f2o,S2o,z2o=s2_o; f2s,S2s,z2s=s2_s
        strict_freq_check(freq,f2o,"Dev Open")
        Y2o,Y2s=s_to_y(S2o,z2o),s_to_y(S2s,z2s)
        if d1_o is not None:
            Y2o=deembed_open_short(Y2o,d1_o,d1_s)
            Y2s=deembed_open_short(Y2s,d1_o,d1_s)
        Y_fin=deembed_open_short(Y_fin,Y2o,Y2s); stages.append("Dev(O/S)")
    if s3_t:
        f3t,S3t,z3t=s3_t; strict_freq_check(freq,f3t,"Dev Thru")
        Y3t=s_to_y(S3t,z3t)
        if d1_o is not None: Y3t=deembed_open_short(Y3t,d1_o,d1_s)
        if "Dev(O/S)" in stages:
            Y3t_r=s_to_y(S3t,z3t)
            if d1_o is not None: Y3t_r=deembed_open_short(Y3t_r,d1_o,d1_s)
            Y3t=deembed_open_short(Y3t_r,Y2o,Y2s)
        Y_fin=deembed_thru_half(Y_fin,Y3t); stages.append("Dev(Thru)")
    note=" + ".join(stages) if stages else "None"
    df_fin=compute_metrics(Y_fin,freq) if stages else None
    S_fin=y_to_s(Y_fin,z0) if stages else S_raw
    df_e=df_fin if df_fin is not None else df_raw
    f_arr=df_e["Freq (GHz)"].values
    fT_cr,fT_pl,ft_m    = extract_limit(f_arr,df_e["|h21|² (dB)"].values,  df_e["fT Plateau (GHz)"].values,  n_pts,f_min,f_max)
    fmU_cr,fmU_pl,fmU_m = extract_limit(f_arr,df_e["Mason U (dB)"].values,  df_e["fmax U Plateau (GHz)"].values,n_pts,f_min,f_max)
    fmM_cr,fmM_pl,fmM_m = extract_limit(f_arr,df_e["MAG/MSG (dB)"].values,  df_e["fmax MAG Plateau (GHz)"].values,n_pts,f_min,f_max)
    stem=re.sub(r"\.s2p$","",filename,flags=re.IGNORECASE)
    m=re.search(r"[Vv][Cc][Ee][_\-]?([\d]+(?:p\d+)?)\s*[Vv]",stem)
    vce=float(m.group(1).replace("p",".")) if m else None
    m=re.search(r"[Ii][Bb][_\-]?([\d]+(?:p\d+)?)\s*([pnuUmM]?)[Aa]?",stem)
    ib=float(m.group(1).replace("p","."))*{"p":1e-12,"n":1e-9,"u":1e-6,"U":1e-6,"m":1e-3,"M":1e-3,"":1.0}.get(m.group(2) if m else "",1.0) if m else None
    return df_raw,df_fin,S_fin,S_raw,freq,z0,{
        "Label":stem,"Vce (V)":vce,"Ib (A)":ib,"De-embedding":note,
        "fT Cross/Extrap (GHz)":fT_cr,"fT Plateau (GHz)":fT_pl,"fT Method":ft_m,
        "fmax U Cross/Extrap (GHz)":fmU_cr,"fmax U Plateau (GHz)":fmU_pl,"fmax U Method":fmU_m,
        "fmax MAG Cross/Extrap (GHz)":fmM_cr,"fmax MAG Plateau (GHz)":fmM_pl,"fmax MAG Method":fmM_m,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  PLOTTING UTILITIES
# ═════════════════════════════════════════════════════════════════════════════
PALETTE=["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
         "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]

def _darken(c):
    try:
        h=c.lstrip("#"); r,g,b=int(h[0:2],16),int(h[2:4],16),int(h[4:6],16)
        return f"#{max(0,r-45):02x}{max(0,g-45):02x}{max(0,b-45):02x}"
    except: return c

def _layout(title,ytitle,yr,xr):
    return dict(
        title=dict(text=title,font=dict(size=13)),
        xaxis=dict(title="Frequency (GHz)",type="log",
                   range=[np.log10(max(xr[0],1e-4)),np.log10(xr[1])],
                   showgrid=True,gridcolor="#ebebeb",minor_showgrid=True),
        yaxis=dict(title=ytitle,range=list(yr),showgrid=True,gridcolor="#ebebeb"),
        legend=dict(x=1.0,y=1.0,xanchor="left",yanchor="top",
                    bgcolor="rgba(255,255,255,0.88)",bordercolor="#ccc",borderwidth=1),
        plot_bgcolor="white",paper_bgcolor="white",height=500,
        margin=dict(l=55,r=25,t=45,b=50),hovermode="x unified",template="plotly_white")

def _extended_smith_grid(max_r=1.0):
    traces=[]; t=np.linspace(0,2*np.pi,500)
    sk=dict(mode="lines",showlegend=False,hoverinfo="skip")
    for ro in np.arange(1.0,max_r+0.5,1.0):
        lw_=1.6 if ro==1.0 else 0.9
        col="rgba(60,60,60,0.85)" if ro==1.0 else "rgba(170,170,170,0.6)"
        traces.append(go.Scatter(x=np.cos(t)*ro,y=np.sin(t)*ro,
                                 line=dict(color=col,width=lw_),**sk))
    traces.append(go.Scatter(x=[-max_r,max_r],y=[0.,0.],
                             line=dict(color="rgba(100,100,100,0.6)",width=0.8),**sk))
    gray="rgba(155,155,155,0.5)"
    for r in [0.0,0.2,0.5,1.0,2.0,5.0]:
        cx_=r/(r+1); rad=1.0/(r+1)
        xc=cx_+rad*np.cos(t); yc=rad*np.sin(t)
        mg=np.sqrt(xc**2+yc**2); xc[mg>max_r]=np.nan; yc[mg>max_r]=np.nan
        traces.append(go.Scatter(x=xc,y=yc,line=dict(color=gray,width=0.8),**sk))
    for x in [0.2,0.5,1.0,2.0,5.0]:
        for sign in [1,-1]:
            xv=sign*x; rad_x=abs(1.0/xv)
            xc=1.0+rad_x*np.cos(t); yc=(1.0/xv)+rad_x*np.sin(t)
            mg=np.sqrt(xc**2+yc**2); xc[mg>max_r]=np.nan; yc[mg>max_r]=np.nan
            traces.append(go.Scatter(x=xc,y=yc,line=dict(color=gray,width=0.8),**sk))
    return traces

def make_smith(S,f_array,f_min,f_max,toggles,scales,title,max_r=1.0):
    mask=(f_array>=f_min)&(f_array<=f_max)
    S_p=S[mask].copy(); f_p=f_array[mask]
    fig=go.Figure()
    for tr in _extended_smith_grid(max_r): fig.add_trace(tr)
    for key,(r,c),color,dash in [("S11",(0,0),"#1f77b4","solid"),("S22",(1,1),"#ff7f0e","dash"),
                                  ("S21",(1,0),"#2ca02c","dot"),("S12",(0,1),"#d62728","dashdot")]:
        if not toggles.get(key,False): continue
        sv=S_p[:,r,c]*scales.get(key,1.0); sv[np.abs(sv)>max_r]=np.nan+1j*np.nan
        sc=scales.get(key,1.0)
        lbl=f"{key} ({f_min:.2g}–{f_max:.2g} GHz)" if sc==1 else f"{key} ×{sc:g}"
        hov=[f"f={fv:.3f} GHz<br>Re={rv:.4f}<br>Im={iv:.4f}"
             for fv,rv,iv in zip(f_p,sv.real,sv.imag)]
        fig.add_trace(go.Scatter(x=sv.real,y=sv.imag,mode="lines",
                                 line=dict(color=color,width=2.2,dash=dash),
                                 name=lbl,text=hov,hoverinfo="text"))
    lim=max_r*1.05
    fig.update_layout(
        title=dict(text=f"Smith Chart — {title}",font=dict(size=13)),
        xaxis=dict(title="Re(Γ)",range=[-lim,lim],showgrid=False,zeroline=False,
                   scaleanchor="y",scaleratio=1),
        yaxis=dict(title="Im(Γ)",range=[-lim,lim],showgrid=False,zeroline=False),
        plot_bgcolor="white",paper_bgcolor="white",height=540,
        margin=dict(l=50,r=30,t=50,b=50),
        legend=dict(x=1.02,y=1.0,xanchor="left",yanchor="top",
                    bgcolor="rgba(255,255,255,0.92)",bordercolor="#ccc",borderwidth=1),
        hovermode="closest")
    return fig

def make_bode(df,title,xr,yr,sh21,su,smag,color):
    fig=go.Figure(); f=df["Freq (GHz)"]
    hov="Freq:%{x:.4f}GHz<br>Gain:%{y:.4f}dB<extra></extra>"
    if sh21: fig.add_trace(go.Scatter(x=f,y=df["|h21|² (dB)"],name="|h21|²",line=dict(color=color,width=2.5),hovertemplate=hov))
    if su:   fig.add_trace(go.Scatter(x=f,y=df["Mason U (dB)"],name="Mason U",line=dict(color=_darken(color),width=2.5,dash="dash"),hovertemplate=hov))
    if smag: fig.add_trace(go.Scatter(x=f,y=df["MAG/MSG (dB)"],name="MAG/MSG",line=dict(color="#2ca02c",width=2.5,dash="dot"),hovertemplate=hov))
    fig.add_hline(y=0,line_dash="dash",line_color="black")
    fig.update_layout(**_layout(f"Bode — {title}","Gain (dB)",yr,xr)); return fig

def make_plateau(df,res,title,xr,sh21,su,smag,color):
    cols=[]
    if sh21: cols+=df["fT Plateau (GHz)"].tolist()
    if su:   cols+=df["fmax U Plateau (GHz)"].tolist()
    arr=np.array([v for v in cols if np.isfinite(v) and v>0])
    ym=float(np.quantile(arr,0.97))*1.3 if len(arr) else 100
    hov="Freq:%{x:.4f}GHz<br>GBP:%{y:.4f}GHz<extra></extra>"
    fig=go.Figure()
    if sh21: fig.add_trace(go.Scatter(x=df["Freq (GHz)"],y=df["fT Plateau (GHz)"],name="fT",line=dict(color=color,width=2.5),hovertemplate=hov))
    if su:   fig.add_trace(go.Scatter(x=df["Freq (GHz)"],y=df["fmax U Plateau (GHz)"],name="fmax(U)",line=dict(color=_darken(color),width=2.5,dash="dash"),hovertemplate=hov))
    if smag: fig.add_trace(go.Scatter(x=df["Freq (GHz)"],y=df["fmax MAG Plateau (GHz)"],name="fmax(MAG)",line=dict(color="#2ca02c",width=2,dash="dot"),hovertemplate=hov))
    fig.update_layout(**_layout(f"Plateau — {title}","GBP (GHz)",[0,ym],xr)); return fig

def _card(col,title,val,sub,color="#4A90D9"):
    col.markdown(
        f'<div style="padding:10px 14px;border-radius:8px;border-left:4px solid {color};'
        f'background:#f7f9fc;min-height:70px;margin-bottom:10px;">'
        f'<div style="font-size:0.74rem;color:#666;">{title}</div>'
        f'<div style="font-size:1.15rem;font-weight:700;color:#1a2e4a;">{val}</div>'
        f'<div style="font-size:0.70rem;color:#888;margin-top:1px;">{sub}</div></div>',
        unsafe_allow_html=True)

def build_excel(summary_df,all_data):
    buf=io.BytesIO()
    with pd.ExcelWriter(buf,engine="openpyxl") as w:
        summary_df.to_excel(w,sheet_name="Summary",index=False)
        for k,v in all_data.items():
            df_p=v["df_fin"] if v["df_fin"] is not None else v["df_raw"]
            base=re.sub(r"[:\\/*?\[\]]","_",Path(k).stem)[:28]
            df_p.to_excel(w,sheet_name=base,index=False)
    return buf.getvalue()

def _load_cal(fobj):
    if fobj is None: return None
    try: return parse_s2p(fobj.getvalue().decode("utf-8",errors="ignore"))
    except Exception as e: st.sidebar.error(f"Parse failed {fobj.name}: {e}"); return None


# ═════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("#### 🔧 3-Step De-embedding")
    sw1=st.toggle("① Probe (Open-Short)",value=False)
    f1o=st.file_uploader("Probe Open",  type=["s2p"],key="p1o") if sw1 else None
    f1s=st.file_uploader("Probe Short", type=["s2p"],key="p1s") if sw1 else None
    st.divider()
    sw2=st.toggle("② Device Dummy (Open-Short)",value=False,help="Required for SSM extraction")
    f2o=st.file_uploader("Dev Open",  type=["s2p"],key="d2o") if sw2 else None
    f2s=st.file_uploader("Dev Short", type=["s2p"],key="d2s") if sw2 else None
    if sw2: st.caption("✅ SSM Extraction tab enabled.")
    st.divider()
    sw3=st.toggle("③ Device Thru (Half-Z)",value=False)
    f3t=st.file_uploader("Dev Thru",  type=["s2p"],key="d3t") if sw3 else None
    st.divider()
    st.markdown("#### 📊 Chart Control")
    freq_min=st.number_input("Freq Min (GHz)",value=0.01,min_value=0.01,format="%.4f")
    freq_max=st.number_input("Freq Max (GHz)",value=50.0,min_value=1.0)
    db_min=st.number_input("Bode Y Min (dB)",value=0.0)
    db_max=st.number_input("Bode Y Max (dB)",value=50.0)
    n_pts=st.slider("Interpolation pts",2,20,2)
    show_raw=st.checkbox("Overlay Raw",value=True,disabled=not(sw1 or sw2 or sw3))
    st.markdown("##### Trace Selection")
    sh21=st.checkbox("|h21|² → fT",value=True,key="sh21")
    su=st.checkbox("Mason U → fmax(U)",value=True,key="su")
    smag=st.checkbox("MAG/MSG → fmax",value=True,key="smag")
    st.divider()
    st.markdown("#### 🍩 Smith Chart")
    smith_f_min=st.number_input("Smith Freq Min (GHz)",value=freq_min,min_value=0.01,format="%.4f")
    smith_f_max=st.number_input("Smith Freq Max (GHz)",value=freq_max,min_value=1.0)
    smith_max_r=st.slider("|Γ| Max Radius",1.0,5.0,1.0,0.5)
    st.markdown("##### Display & Scale")
    ca,cb=st.columns(2); show_s11=ca.checkbox("S11",value=True,key="ss11"); scale_s11=cb.number_input("S11 ×",value=1.0,step=0.1,key="sc11")
    ca,cb=st.columns(2); show_s22=ca.checkbox("S22",value=True,key="ss22"); scale_s22=cb.number_input("S22 ×",value=1.0,step=0.1,key="sc22")
    ca,cb=st.columns(2); show_s21=ca.checkbox("S21",value=True,key="ss21"); scale_s21=cb.number_input("S21 ×",value=1.0,step=0.1,key="sc21")
    ca,cb=st.columns(2); show_s12=ca.checkbox("S12",value=True,key="ss12"); scale_s12=cb.number_input("S12 ×",value=1.0,step=0.1,key="sc12")

# ═════════════════════════════════════════════════════════════════════════════
#  FILE UPLOADER & PROCESSING
# ═════════════════════════════════════════════════════════════════════════════
col_up1,col_up2=st.columns([4,1])
with col_up1:
    dut_files=st.file_uploader("Upload DUT .s2p files",type=["s2p"],
                               accept_multiple_files=True,
                               key=st.session_state["rf_uploader_key"])
with col_up2:
    st.write(""); st.write("")
    if st.button("🗑️ Clear uploads",use_container_width=True):
        st.session_state["rf_uploader_key"]+=1
        st.session_state.pop("rf_ms_files",None)
        st.session_state.pop("rf_prev_uploaded",None)
        st.rerun()

s1o=_load_cal(f1o) if sw1 else None
s1s=_load_cal(f1s) if sw1 else None
s2o=_load_cal(f2o) if sw2 else None   # passed to render_ssm_tab
s2s=_load_cal(f2s) if sw2 else None   # passed to render_ssm_tab
s3t=_load_cal(f3t) if sw3 else None

all_data,errors={},{}
if dut_files:
    for f in dut_files:
        try:
            df_raw,df_fin,S_fin,S_raw,freq,z0_dut,res=process_dut(
                f.getvalue().decode("utf-8",errors="ignore"),
                f.name,s1o,s1s,s2o,s2s,s3t,n_pts,freq_min,freq_max)
            all_data[f.name]={
                "df_raw":df_raw,"df_fin":df_fin,
                "S_fin":S_fin,"S_raw":S_raw,
                "freq":freq,"z0":z0_dut,**res}
        except Exception as e:
            errors[f.name]=str(e)

for fname,err in errors.items():
    st.error(f"**{fname}**: {err}")

if all_data:
    file_options=list(all_data.keys())
    cur=set(file_options); prev=st.session_state.get("rf_prev_uploaded",set())
    new_=cur-prev
    sel=[f for f in st.session_state.get("rf_ms_files",[]) if f in cur]
    for nf in new_:
        if nf not in sel: sel.append(nf)
    st.session_state["rf_ms_files"]=sel
    st.session_state["rf_prev_uploaded"]=cur

    c1,c2,_=st.columns([1.5,1.5,7])
    if c1.button("✅ Select All"):  st.session_state["rf_ms_files"]=file_options
    if c2.button("❌ Clear"):       st.session_state["rf_ms_files"]=[]
    selected_files=st.multiselect("📂 Files to analyse:",options=file_options,
                                  key="rf_ms_files",format_func=lambda x:Path(x).stem)
else:
    selected_files=[]
    st.info("Upload DUT .s2p files above to begin.")

xr,yr=(freq_min,freq_max),(db_min,db_max)

# ═════════════════════════════════════════════════════════════════════════════
#  TABS
# ═════════════════════════════════════════════════════════════════════════════
tab_ov,tab_ind,tab_sum=st.tabs(["📊 Overlay","📁 Individual","📋 Summary"])

with tab_ov:
    st.markdown("### 📊 Bode Plot Overlay")
    f_bode=go.Figure()
    if all_data and selected_files:
        for i,n in enumerate(selected_files):
            d,c,lbl=all_data[n],PALETTE[i%len(PALETTE)],Path(n).stem
            df_p=d["df_fin"] if d["df_fin"] is not None else d["df_raw"]
            hov="Freq:%{x:.4f}GHz<br>%{y:.4f}dB<extra></extra>"
            if show_raw and d["df_fin"] is not None and sh21:
                f_bode.add_trace(go.Scatter(x=d["df_raw"]["Freq (GHz)"],y=d["df_raw"]["|h21|² (dB)"],
                                            name=f"|h21|² raw–{lbl}",line=dict(color=c,width=1.2,dash="dot"),
                                            opacity=0.35,hovertemplate=hov))
            if sh21: f_bode.add_trace(go.Scatter(x=df_p["Freq (GHz)"],y=df_p["|h21|² (dB)"],name=f"|h21|²–{lbl}",line=dict(color=c,width=2.5),hovertemplate=hov))
            if su:   f_bode.add_trace(go.Scatter(x=df_p["Freq (GHz)"],y=df_p["Mason U (dB)"],name=f"U–{lbl}",line=dict(color=_darken(c),width=2.5,dash="dash"),hovertemplate=hov))
            if smag: f_bode.add_trace(go.Scatter(x=df_p["Freq (GHz)"],y=df_p["MAG/MSG (dB)"],name=f"MAG–{lbl}",line=dict(color=c,width=2,dash="dot"),opacity=0.7,hovertemplate=hov))
    f_bode.add_hline(y=0,line_dash="dash",line_color="black")
    f_bode.update_layout(**_layout("Overlay — Bode Plot","Gain (dB)",yr,xr)); f_bode.update_layout(height=550)
    st.plotly_chart(f_bode,use_container_width=True)

    st.markdown("### 📊 Plateau Plot Overlay")
    f_plat=go.Figure(); all_v=[]
    if all_data and selected_files:
        for i,n in enumerate(selected_files):
            d,c,lbl=all_data[n],PALETTE[i%len(PALETTE)],Path(n).stem
            df_p=d["df_fin"] if d["df_fin"] is not None else d["df_raw"]
            hov="Freq:%{x:.4f}GHz<br>GBP:%{y:.4f}GHz<extra></extra>"
            if sh21:
                f_plat.add_trace(go.Scatter(x=df_p["Freq (GHz)"],y=df_p["fT Plateau (GHz)"],name=f"fT–{lbl}",line=dict(color=c,width=2.5),hovertemplate=hov))
                all_v+=df_p["fT Plateau (GHz)"].dropna().tolist()
            if su:
                f_plat.add_trace(go.Scatter(x=df_p["Freq (GHz)"],y=df_p["fmax U Plateau (GHz)"],name=f"fmax(U)–{lbl}",line=dict(color=_darken(c),width=2.5,dash="dash"),hovertemplate=hov))
                all_v+=df_p["fmax U Plateau (GHz)"].dropna().tolist()
    arr=np.array([v for v in all_v if np.isfinite(v) and v>0])
    ym=float(np.quantile(arr,0.97))*1.3 if len(arr) else 100
    f_plat.update_layout(**_layout("Overlay — Plateau","GBP (GHz)",[0,ym],xr)); f_plat.update_layout(height=550)
    st.plotly_chart(f_plat,use_container_width=True)

with tab_ind:
    if not all_data or not selected_files:
        st.info("Upload and select files to view individual analysis.")
    else:
        stabs=st.tabs([Path(n).stem for n in selected_files])
        file_names=list(all_data.keys())
        for stab,n in zip(stabs,selected_files):
            c=PALETTE[file_names.index(n)%len(PALETTE)]
            d=all_data[n]
            df_p=d["df_fin"] if d["df_fin"] is not None else d["df_raw"]

            def _fc(v_cr,v_pl,method):
                if method in ["No Gain","No Data"]: return method
                if method=="0dB Cross":      return f"{v_cr:.3f} GHz" if np.isfinite(v_cr) else "N/A"
                if method=="Extrap & Plat.": return f"{v_pl:.3f} GHz" if np.isfinite(v_pl) else "N/A"
                return "N/A"

            with stab:
                c1,c2,c3,c4,c5=st.columns(5)
                _card(c1,"De-embedding",d["De-embedding"],"mode","#888")
                _card(c2,"fT (GHz)",_fc(d["fT Cross/Extrap (GHz)"],d["fT Plateau (GHz)"],d["fT Method"]),d["fT Method"])
                _card(c3,"fmax U",_fc(d["fmax U Cross/Extrap (GHz)"],d["fmax U Plateau (GHz)"],d["fmax U Method"]),d["fmax U Method"],"#d62728")
                _card(c4,"fmax MAG",_fc(d["fmax MAG Cross/Extrap (GHz)"],d["fmax MAG Plateau (GHz)"],d["fmax MAG Method"]),d["fmax MAG Method"],"#2ca02c")
                if d["Vce (V)"] is not None:  _card(c5,"Vce",f"{d['Vce (V)']} V","bias","#9467bd")
                elif d["Ib (A)"] is not None: _card(c5,"Ib",f"{d['Ib (A)']*1e6:.1f} µA","bias","#9467bd")

                toggles={"S11":show_s11,"S22":show_s22,"S21":show_s21,"S12":show_s12}
                scales ={"S11":scale_s11,"S22":scale_s22,"S21":scale_s21,"S12":scale_s12}

                ta,tb,tc,td=st.tabs(["Bode Plot","Plateau Plot","Smith Chart","🔬 SSM Extraction"])
                with ta: st.plotly_chart(make_bode(df_p,Path(n).stem,xr,yr,sh21,su,smag,c),use_container_width=True)
                with tb: st.plotly_chart(make_plateau(df_p,d,Path(n).stem,xr,sh21,su,smag,c),use_container_width=True)
                with tc: st.plotly_chart(make_smith(d["S_fin"],df_p["Freq (GHz)"].values,
                                                    smith_f_min,smith_f_max,toggles,scales,
                                                    Path(n).stem,max_r=smith_max_r),use_container_width=True)
                with td:
                    # Gate key unique per file
                    run_key = f"ssm_run_{n}"

                    if not st.session_state.get(run_key, False):
                        st.markdown(" ")
                        col_ctr, _, _ = st.columns([1, 2, 2])
                        if col_ctr.button(
                            "▶ Run SSM Extraction",
                            key=f"ssm_btn_{n}",
                            use_container_width=True,
                            type="primary",
                        ):
                            st.session_state[run_key] = True
                            st.rerun()
                        st.caption(
                            "SSM extraction is skipped until activated to keep the app fast. "
                            "Click above to run it for this file."
                        )
                    else:
                        # Optional: allow the user to reset / clear the results
                        if st.button(
                            "✕ Clear SSM results",
                            key=f"ssm_clear_{n}",
                            help="Frees cached computation for this file.",
                        ):
                            st.session_state[run_key] = False
                            # Also clear any downstream caches for this file
                            for k in list(st.session_state.keys()):
                                if k.endswith(f"_{n}") and k != run_key:
                                    del st.session_state[k]
                            st.rerun()

                        render_ssm_tab(
                            n, d["S_raw"], d["freq"], d["z0"],
                            s2o, s2s, all_data=all_data,
                        )

                with st.expander("📋 Data Table"):
                    if d["df_fin"] is not None:
                        ta2,tb2=st.tabs(["De-embedded","Raw"])
                        with ta2: st.dataframe(df_p.round(4),use_container_width=True,hide_index=True)
                        with tb2: st.dataframe(d["df_raw"].round(4),use_container_width=True,hide_index=True)
                    else: st.dataframe(df_p.round(4),use_container_width=True,hide_index=True)

with tab_sum:
    if not all_data:
        st.info("Upload files to generate summary.")
    else:
        rows=[{"File":k,"De-embedding":d["De-embedding"],"Vce (V)":d["Vce (V)"],
               "Ib (µA)":round(d["Ib (A)"]*1e6,1) if d["Ib (A)"] else None,
               "fT Cross":d["fT Cross/Extrap (GHz)"],"fT Plat":d["fT Plateau (GHz)"],
               "fmax U Cross":d["fmax U Cross/Extrap (GHz)"],"fmax U Plat":d["fmax U Plateau (GHz)"]}
              for k,d in all_data.items()]
        sum_df=pd.DataFrame(rows)
        fmt={c:"{:.4f}" for c in sum_df.columns if "Cross" in c or "Plat" in c}
        fmt["Vce (V)"]="{:.3f}"; fmt["Ib (µA)"]="{:.1f}"
        st.dataframe(sum_df.style.format(fmt,na_rep="—"),use_container_width=True,hide_index=True)
        date=datetime.now().strftime("%Y-%m-%d")
        d1,d2=st.columns(2)
        with d1:
            st.download_button("📥 Excel",data=build_excel(sum_df,all_data),
                               file_name=f"RF_Extraction_{date}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               use_container_width=True)
        with d2:
            zbuf=io.BytesIO()
            with zipfile.ZipFile(zbuf,"w",zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("Summary.csv",sum_df.to_csv(index=False).encode())
                for k,d in all_data.items():
                    dp=d["df_fin"] if d["df_fin"] is not None else d["df_raw"]
                    zf.writestr(f"{Path(k).stem}.csv",dp.to_csv(index=False).encode())
            st.download_button("📦 ZIP (CSV)",data=zbuf.getvalue(),
                               file_name=f"RF_Extraction_{date}.zip",
                               mime="application/zip",use_container_width=True)
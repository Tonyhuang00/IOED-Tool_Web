"""
ssm_extraction.py  —  HBT Small-Signal Model Parameter Extraction
References  [1] Cheng et al. MicroJ 2022  [2] Degachi & Ghannouchi IEEE TED 2008
            [3] Gao HBT Circuit Design Wiley 2015  [4] Zhang et al. AICSP 2015
Rb=Rpb  Rc=Rpc  Re=Rpe  (same physical quantity, different name contexts)
"""
from __future__ import annotations
import hashlib, io, json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ─── SECTION A ── CORE RF UTILITIES ──────────────────────────────────────────
def _s_to_y(S,z0=50.0):
    s11,s12,s21,s22=S[:,0,0],S[:,0,1],S[:,1,0],S[:,1,1]
    d=(1+s11)*(1+s22)-s12*s21; Y=np.zeros_like(S)
    Y[:,0,0]=((1-s11)*(1+s22)+s12*s21)/(d*z0); Y[:,0,1]=-2*s12/(d*z0)
    Y[:,1,0]=-2*s21/(d*z0); Y[:,1,1]=((1+s11)*(1-s22)+s12*s21)/(d*z0)
    return Y
def _y_to_s_batch(Y,z0=50.0):
    S=np.zeros_like(Y); I=np.eye(2)
    for i in range(len(Y)):
        yn=Y[i]*z0
        try: S[i]=np.dot(I-yn,np.linalg.inv(I+yn))
        except: S[i]=np.full((2,2),np.nan+0j)
    return S
def _inv2(M):
    out=np.zeros_like(M)
    for i in range(len(M)):
        try: out[i]=np.linalg.inv(M[i])
        except: out[i]=np.full((2,2),np.nan+0j)
    return out
_y_to_z=_z_to_y=_inv2
def _strict_freq_check(f_dut,f_dummy,label):
    if len(f_dut)!=len(f_dummy) or not np.allclose(f_dut,f_dummy,rtol=1e-5):
        raise ValueError(f"DUT and {label} frequency grids differ.")
def _safe_median(arr,n=None):
    a=arr[:n] if n is not None else arr; a=np.asarray(a,dtype=float); a=a[np.isfinite(a)]
    return float(np.median(a)) if len(a)>0 else 0.0
def _y_to_s_single(Y,z0=50.0):
    Yn=Y*z0; I=np.eye(2)
    try: return np.dot(I-Yn,np.linalg.inv(I+Yn))
    except: return np.full((2,2),np.nan+0j)
def _extended_smith_grid(max_r=1.0):
    traces=[]; t=np.linspace(0,2*np.pi,500); sk=dict(mode="lines",showlegend=False,hoverinfo="skip")
    for ro in np.arange(1.0,max_r+0.5,1.0):
        lw=1.6 if ro==1.0 else 0.9; col="rgba(60,60,60,0.85)" if ro==1.0 else "rgba(170,170,170,0.6)"
        traces.append(go.Scatter(x=np.cos(t)*ro,y=np.sin(t)*ro,line=dict(color=col,width=lw),**sk))
    traces.append(go.Scatter(x=[-max_r,max_r],y=[0.,0.],line=dict(color="rgba(100,100,100,0.6)",width=0.8),**sk))
    gray="rgba(155,155,155,0.5)"
    for r in [0.0,0.2,0.5,1.0,2.0,5.0]:
        cx_=r/(r+1); rad=1.0/(r+1); xc=cx_+rad*np.cos(t); yc=rad*np.sin(t)
        mg=np.sqrt(xc**2+yc**2); xc[mg>max_r]=np.nan; yc[mg>max_r]=np.nan
        traces.append(go.Scatter(x=xc,y=yc,line=dict(color=gray,width=0.8),**sk))
    for x in [0.2,0.5,1.0,2.0,5.0]:
        for sign in [1,-1]:
            xv=sign*x; rad_x=abs(1.0/xv); xc=1.0+rad_x*np.cos(t); yc=(1.0/xv)+rad_x*np.sin(t)
            mg=np.sqrt(xc**2+yc**2); xc[mg>max_r]=np.nan; yc[mg>max_r]=np.nan
            traces.append(go.Scatter(x=xc,y=yc,line=dict(color=gray,width=0.8),**sk))
    return traces

# ─── NEW: extended element helpers ───────────────────────────────────────────
def _open_elem_Y_sc(C, mode, extra, w):
    """Single-freq admittance for one pad cap with optional extra element."""
    if mode=="Parallel L" and extra>0:
        return 1j*w*C + 1.0/(1j*w*extra + 1e-60)
    if mode=="Series L" and extra>0:
        denom=1.0-w**2*extra*C
        if abs(denom)<1e-10: denom=1e-10
        return 1j*w*C/denom
    if mode=="Series R" and extra>0:
        return 1j*w*C/(1.0+1j*w*extra*C)
    return 1j*w*C

def _short_lead_Z_sc(R, L, Cpar, w):
    """Single-freq impedance for one short lead with optional parallel C."""
    Z=R+1j*w*L
    if Cpar>0: return 1.0/(1.0/Z+1j*w*Cpar)
    return Z

def _write_s2p(freq_hz, S, title="", params=None):
    """Generate Touchstone .s2p bytes in dB/angle format."""
    lines=[f"! Forward-simulated: {title}"]
    if params:
        for k,v in params.items(): lines.append(f"!   {k} = {v}")
    lines.append("# Hz S DB R 50")
    for i,f in enumerate(freq_hz):
        parts=[f"{f:.0f}"]
        for r,c in [(0,0),(1,0),(0,1),(1,1)]:
            s=S[i,r,c]; db=20*np.log10(abs(s)+1e-30); ang=np.degrees(np.angle(s))
            parts+=[f"{db:.8f}",f"{ang:.8f}"]
        lines.append(" ".join(parts))
    return "\n".join(lines).encode("utf-8")

# ─── SECTION B ── STEP 1: PAD PARASITICS ─────────────────────────────────────
def ssm_step1a_open(open_data,n_low_frac=0.20):
    """Gao [3] §4.2. Also extracts per-cap conductance for R/L diagnostics."""
    f,S_o,z0=open_data; omega=2.0*np.pi*f; n_low=max(3,int(len(f)*n_low_frac))
    Y_o=_s_to_y(S_o,z0)
    # capacitances (imaginary)
    Cpbe_arr=np.imag(Y_o[:,0,0]+Y_o[:,0,1])/omega
    Cpce_arr=np.imag(Y_o[:,1,1]+Y_o[:,0,1])/omega
    Cpbc_arr=-np.imag(Y_o[:,0,1])/omega
    # conductances (real) — detect series R
    Gpbe_arr=np.real(Y_o[:,0,0]+Y_o[:,0,1])
    Gpce_arr=np.real(Y_o[:,1,1]+Y_o[:,0,1])
    Gpbc_arr=-np.real(Y_o[:,0,1])
    params=dict(Cpbe=_safe_median(Cpbe_arr,n_low),Cpce=_safe_median(Cpce_arr,n_low),Cpbc=_safe_median(Cpbc_arr,n_low))
    arrays=dict(Cpbe=Cpbe_arr,Cpce=Cpce_arr,Cpbc=Cpbc_arr,
                Gpbe=Gpbe_arr,Gpce=Gpce_arr,Gpbc=Gpbc_arr,omega=omega)
    return params,arrays

def ssm_step1b_short(short_data,freq,Cpbe,Cpce,Cpbc,open_data=None,n_low_frac=0.20,measured=True,
                     Cpbe_mode="None",Cpbe_extra=0.0,Cpce_mode="None",Cpce_extra=0.0,
                     Cpbc_mode="None",Cpbc_extra=0.0):
    """Gao [3] §4.2. Uses extended open model for de-embedding."""
    _,S_s,z0=short_data; omega=2.0*np.pi*freq; N=len(freq); n_low=max(3,int(N*n_low_frac))
    Y_s=_s_to_y(S_s,z0)
    if measured and open_data is not None:
        _,S_o,z0_o=open_data; Y_open_eff=_s_to_y(S_o,z0_o)
    else:
        Y_open_eff=np.zeros((N,2,2),dtype=complex)
        for i,w in enumerate(omega):
            Ypbe=_open_elem_Y_sc(Cpbe,Cpbe_mode,Cpbe_extra,w)
            Ypce=_open_elem_Y_sc(Cpce,Cpce_mode,Cpce_extra,w)
            Ypbc=_open_elem_Y_sc(Cpbc,Cpbc_mode,Cpbc_extra,w)
            Y_open_eff[i]=np.array([[Ypbe+Ypbc,-Ypbc],[-Ypbc,Ypce+Ypbc]])
    Z_corr=_y_to_z(Y_s-Y_open_eff)
    Rpe_arr=np.real(Z_corr[:,0,1]); Rpb_arr=np.real(Z_corr[:,0,0]-Z_corr[:,0,1]); Rpc_arr=np.real(Z_corr[:,1,1]-Z_corr[:,1,0])
    Le_arr=np.imag(Z_corr[:,0,1])/omega; Lb_arr=np.imag(Z_corr[:,0,0]-Z_corr[:,0,1])/omega; Lc_arr=np.imag(Z_corr[:,1,1]-Z_corr[:,1,0])/omega
    Le_raw=_safe_median(Le_arr,n_low); Lb_raw=_safe_median(Lb_arr,n_low); Lc_raw=_safe_median(Lc_arr,n_low)
    Rpe_raw=_safe_median(Rpe_arr,n_low); Rpb_raw=_safe_median(Rpb_arr,n_low); Rpc_raw=_safe_median(Rpc_arr,n_low)
    warnings_list=[]
    NOISE_THRESH=3e-12
    if Le_raw<-NOISE_THRESH: warnings_list.append("Le significantly negative — Open caps may over-correct.")
    if Lb_raw<-NOISE_THRESH: warnings_list.append(f"⚠️ Lb negative ({Lb_raw*1e12:.1f} pH).")
    if Lc_raw<-NOISE_THRESH: warnings_list.append(f"⚠️ Lc negative ({Lc_raw*1e12:.1f} pH).")
    params=dict(Le=Le_raw,Lb=Lb_raw,Lc=Lc_raw,Rpe=Rpe_raw,Rpb=Rpb_raw,Rpc=Rpc_raw)
    arrays=dict(Le=Le_arr,Lb=Lb_arr,Lc=Lc_arr,Rpe=Rpe_arr,Rpb=Rpb_arr,Rpc=Rpc_arr,warnings=warnings_list)
    return params,arrays

# ─── SECTION B2 ── S2P / INTERPOLATION ───────────────────────────────────────
def _parse_s2p_bytes(raw:bytes):
    content=raw.decode("utf-8",errors="ignore"); freq_unit,fmt,z0,data_lines="hz","ma",50.0,[]
    for line in content.splitlines():
        s=line.strip()
        if not s or s.startswith("!"): continue
        if s.startswith("#"):
            parts=s[1:].lower().split()
            for i,p in enumerate(parts):
                if p in ("hz","khz","mhz","ghz"): freq_unit=p
                elif p in ("ma","db","ri"): fmt=p
                elif p=="r" and i+1<len(parts):
                    try: z0=float(parts[i+1])
                    except: pass
            continue
        data_lines.append(s)
    vals=np.array([float(x) for x in " ".join(data_lines).split()])
    n=len(vals)//9; vals=vals[:n*9].reshape(n,9)
    freq=vals[:,0]*{"hz":1.0,"khz":1e3,"mhz":1e6,"ghz":1e9}[freq_unit]
    def to_c(ca,cb):
        a,b=vals[:,ca],vals[:,cb]
        if fmt=="db": return 10**(a/20.0)*np.exp(1j*np.deg2rad(b))
        if fmt=="ma": return a*np.exp(1j*np.deg2rad(b))
        return a+1j*b
    S=np.zeros((n,2,2),dtype=complex)
    for (r,c),(ca,cb) in zip([(0,0),(1,0),(0,1),(1,1)],[(1,2),(3,4),(5,6),(7,8)]): S[:,r,c]=to_c(ca,cb)
    return freq,S,z0

def _interpolate_s2f(f_src,S_src,f_tgt):
    S_out=np.zeros((len(f_tgt),2,2),dtype=complex)
    for r in range(2):
        for c in range(2):
            s=S_src[:,r,c]; S_out[:,r,c]=np.interp(f_tgt,f_src,s.real)+1j*np.interp(f_tgt,f_src,s.imag)
    return S_out

# ─── SECTION C ── PEEL PARASITICS (extended) ──────────────────────────────────
def _peel_para_from_Y(Y_dut,freq,p):
    """Extended: uses open elem mode/extra and short lead Cpar from p dict."""
    omega=2.0*np.pi*freq
    Y_pad=np.zeros((len(freq),2,2),dtype=complex)
    for i,w in enumerate(omega):
        Ypbe=_open_elem_Y_sc(p["Cpbe"],p.get("Cpbe_mode","None"),p.get("Cpbe_extra",0.0),w)
        Ypce=_open_elem_Y_sc(p["Cpce"],p.get("Cpce_mode","None"),p.get("Cpce_extra",0.0),w)
        Ypbc=_open_elem_Y_sc(p["Cpbc"],p.get("Cpbc_mode","None"),p.get("Cpbc_extra",0.0),w)
        Y_pad[i]=np.array([[Ypbe+Ypbc,-Ypbc],[-Ypbc,Ypce+Ypbc]])
    Z1=_y_to_z(Y_dut-Y_pad)
    Z_ser=np.zeros((len(freq),2,2),dtype=complex)
    for i,w in enumerate(omega):
        Zb=_short_lead_Z_sc(p["Rpb"],p["Lb"],p.get("Cpar_Lb",0.0),w)
        Zc=_short_lead_Z_sc(p["Rpc"],p["Lc"],p.get("Cpar_Lc",0.0),w)
        Ze=_short_lead_Z_sc(p["Rpe"],p["Le"],p.get("Cpar_Le",0.0),w)
        Z_ser[i]=np.array([[Zb+Ze,Ze],[Ze,Zc+Ze]])
    return _z_to_y(Z1-Z_ser)

def ssm_peel_parasitics(S_raw,freq,z0,p):
    return _peel_para_from_Y(_s_to_y(S_raw,z0),freq,p)

# ─── Open/Short forward simulation for download ───────────────────────────────
def _simulate_open_Sparams(p,freq,z0=50.0):
    """Forward-simulate Open dummy S-params from pad cap model (p dict)."""
    N=len(freq); S=np.zeros((N,2,2),dtype=complex)
    for i,w in enumerate(2.0*np.pi*freq):
        Ypbe=_open_elem_Y_sc(p["Cpbe"],p.get("Cpbe_mode","None"),p.get("Cpbe_extra",0.0),w)
        Ypce=_open_elem_Y_sc(p["Cpce"],p.get("Cpce_mode","None"),p.get("Cpce_extra",0.0),w)
        Ypbc=_open_elem_Y_sc(p["Cpbc"],p.get("Cpbc_mode","None"),p.get("Cpbc_extra",0.0),w)
        Y=np.array([[Ypbe+Ypbc,-Ypbc],[-Ypbc,Ypce+Ypbc]]); S[i]=_y_to_s_single(Y,z0)
    return S

def _simulate_short_Sparams(p,freq,z0=50.0):
    """Forward-simulate Short dummy S-params from pad cap + lead model."""
    N=len(freq); S=np.zeros((N,2,2),dtype=complex)
    for i,w in enumerate(2.0*np.pi*freq):
        Ypbe=_open_elem_Y_sc(p["Cpbe"],p.get("Cpbe_mode","None"),p.get("Cpbe_extra",0.0),w)
        Ypce=_open_elem_Y_sc(p["Cpce"],p.get("Cpce_mode","None"),p.get("Cpce_extra",0.0),w)
        Ypbc=_open_elem_Y_sc(p["Cpbc"],p.get("Cpbc_mode","None"),p.get("Cpbc_extra",0.0),w)
        Y_pad=np.array([[Ypbe+Ypbc,-Ypbc],[-Ypbc,Ypce+Ypbc]])
        Zb=_short_lead_Z_sc(p["Rpb"],p["Lb"],p.get("Cpar_Lb",0.0),w)
        Zc=_short_lead_Z_sc(p["Rpc"],p["Lc"],p.get("Cpar_Lc",0.0),w)
        Ze=_short_lead_Z_sc(p["Rpe"],p["Le"],p.get("Cpar_Le",0.0),w)
        Z_ser=np.array([[Zb+Ze,Ze],[Ze,Zc+Ze]])
        # Short: DUT terminals shorted → Z_DUT=0, so Y_total = Y_pad + inv(Z_ser)
        try: Y_ser=np.linalg.inv(Z_ser)
        except: Y_ser=np.zeros((2,2),dtype=complex)
        S[i]=_y_to_s_single(Y_pad+Y_ser,z0)
    return S


# ─── SECTION D ── STEP 2 (CHENG): EXTRINSIC CAPS ─────────────────────────────
def ssm_step2_extrinsic_T(Y_ex1,freq,n_low):
    """Cheng [1] Eqs. 13, 18–19, 21–22."""
    omega=2.0*np.pi*freq
    Cbex_arr=np.imag(Y_ex1[:,0,0]+Y_ex1[:,0,1])/omega
    Cbex=_safe_median(Cbex_arr,n_low)
    Y_ex2=Y_ex1.copy()
    for i,w in enumerate(omega): Y_ex2[i,0,0]-=1j*w*Cbex
    Yms=Y_ex2[:,0,1]+Y_ex2[:,1,1]; YL=Y_ex2[:,0,0]*Y_ex2[:,1,1]-Y_ex2[:,0,1]*Y_ex2[:,1,0]
    Ytotal=Y_ex2[:,0,0]+Y_ex2[:,0,1]+Y_ex2[:,1,0]+Y_ex2[:,1,1]
    num=np.imag(Yms)*np.real(YL)-np.real(Yms)*np.imag(YL)
    den=np.real(Yms)*np.real(Ytotal)+np.imag(Ytotal)*np.imag(Yms)
    with np.errstate(divide="ignore",invalid="ignore"):
        Cbcx_arr=-np.where(np.abs(den)>1e-40,num/(omega*den),np.nan)
    n0,n1=len(freq)//4,3*len(freq)//4; Cbcx=_safe_median(Cbcx_arr[n0:n1])
    return ({"Cbex":Cbex,"Cbcx":Cbcx},{"Cbex_arr":Cbex_arr,"Cbcx_arr":Cbcx_arr,"Y_ex2":Y_ex2})

def ssm_step2_extrinsic_pi(Y_ex1,freq,n_low):
    """Cheng [1] Eqs. 22, 26–28."""
    omega=2.0*np.pi*freq; B=Y_ex1[:,0,1]+Y_ex1[:,1,1]; C=Y_ex1[:,0,0]+Y_ex1[:,1,0]
    with np.errstate(divide="ignore",invalid="ignore"):
        Cbex_arr=np.where(np.abs(np.imag(B))>1e-40,(np.real(B)*np.real(C)+np.imag(B)*np.imag(C))/(omega*np.imag(B)),np.nan)
    Cbex=_safe_median(Cbex_arr,n_low)
    Y_ex2=Y_ex1.copy()
    for i,w in enumerate(omega): Y_ex2[i,0,0]-=1j*w*Cbex
    Yms=Y_ex2[:,0,1]+Y_ex2[:,1,1]; YL=Y_ex2[:,0,0]*Y_ex2[:,1,1]-Y_ex2[:,0,1]*Y_ex2[:,1,0]
    Ytotal=Y_ex2[:,0,0]+Y_ex2[:,0,1]+Y_ex2[:,1,0]+Y_ex2[:,1,1]
    num=np.imag(Yms)*np.real(YL)-np.real(Yms)*np.imag(YL)
    den=np.real(Yms)*np.real(Ytotal)+np.imag(Ytotal)*np.imag(Yms)
    with np.errstate(divide="ignore",invalid="ignore"):
        Cbcx_arr=-np.where(np.abs(den)>1e-40,num/(omega*den),np.nan)
    n0,n1=len(freq)//4,3*len(freq)//4; Cbcx=_safe_median(Cbcx_arr[n0:n1])
    return ({"Cbex":Cbex,"Cbcx":Cbcx},{"Cbex_arr":Cbex_arr,"Cbcx_arr":Cbcx_arr,"Y_ex2":Y_ex2})

# ─── SECTION E ── STEP 3 (CHENG): INTRINSIC ──────────────────────────────────
def ssm_step3_intrinsic_T(Y_ex2,freq,Cbcx,n_low):
    """Cheng [1] Eqs. 16, 29–33."""
    omega=2.0*np.pi*freq; Y_in=Y_ex2.copy()
    for i,w in enumerate(omega):
        Ybcx=1j*w*Cbcx; Y_in[i,0,0]-=Ybcx; Y_in[i,0,1]+=Ybcx; Y_in[i,1,0]+=Ybcx; Y_in[i,1,1]-=Ybcx
    Z_in=_y_to_z(Y_in)
    Zbe_arr=Z_in[:,0,1]; Zbc_arr=Z_in[:,1,1]-Z_in[:,1,0]; Zbi_arr=Z_in[:,0,0]-Z_in[:,0,1]
    with np.errstate(divide="ignore",invalid="ignore"):
        Ybe_arr=1.0/(Zbe_arr+1e-40); Ybc_arr=1.0/(Zbc_arr+1e-40)
        alpha_arr=(Z_in[:,0,1]-Z_in[:,1,0])/(Zbc_arr+1e-40)
    Rbe_a=1.0/np.real(Ybe_arr).clip(1e-6); Cbe_a=np.imag(Ybe_arr)/omega
    Rbc_a=1.0/np.real(Ybc_arr).clip(1e-12); Cbc_a=np.imag(Ybc_arr)/omega; Rbi_a=np.real(Zbi_arr)
    Rbe=_safe_median(Rbe_a,n_low); Cbe=_safe_median(Cbe_a,n_low)
    Rbc=_safe_median(Rbc_a,n_low); Cbc=_safe_median(Cbc_a,n_low); Rbi=_safe_median(Rbi_a,n_low)
    alpha0=_safe_median(np.abs(alpha_arr),n_low)
    U_arr=(alpha0/(np.abs(alpha_arr)+1e-30))**2
    tauB_arr=np.sqrt(np.maximum(U_arr-1.0,0.0))/omega; tauB=_safe_median(tauB_arr[n_low:])
    with np.errstate(divide="ignore",invalid="ignore"):
        V_arr=2.0*omega*tauB/(U_arr+1e-30)
        tauC_arr=-np.arctan(V_arr/np.sqrt(np.maximum(1.0-V_arr**2,1e-30)))/(2.0*omega)
    tauC=_safe_median(tauC_arr[n_low:])
    return dict(Rbi=Rbi,Rbe=Rbe,Cbe=Cbe,Rbc=Rbc,Cbc=Cbc,alpha0=alpha0,tauB=tauB,tauC=tauC),\
           dict(Rbe=Rbe_a,Cbe=Cbe_a,Rbc=Rbc_a,Cbc=Cbc_a,alpha=alpha_arr,tauB=tauB_arr,tauC=tauC_arr)

def ssm_step3_intrinsic_pi(Y_ex2,freq,Cbcx,n_low):
    """Zhang et al. [4]."""
    omega=2.0*np.pi*freq; Y_in=Y_ex2.copy()
    for i,w in enumerate(omega):
        Ybcx=1j*w*Cbcx; Y_in[i,0,0]-=Ybcx; Y_in[i,0,1]+=Ybcx; Y_in[i,1,0]+=Ybcx; Y_in[i,1,1]-=Ybcx
    Z_in=_y_to_z(Y_in); Z12=Z_in[:,0,1]; Z21=Z_in[:,1,0]; Z11=Z_in[:,0,0]; Z22=Z_in[:,1,1]; Zbc=Z22-Z21
    with np.errstate(divide="ignore",invalid="ignore"):
        Ybc_arr=1.0/(Zbc+1e-40); gm_arr=(Z12-Z21)/((Zbc+1e-40)*(Z12+1e-40))
        Ybe_arr=(Z22-Z12)/((Z12+1e-40)*(Zbc+1e-40)); Rbi_a=np.real(Z11-Z12)
    Rbc_a=1.0/np.real(Ybc_arr).clip(1e-12); Cbc_a=np.imag(Ybc_arr)/omega
    Rbe_a=1.0/np.real(Ybe_arr).clip(1e-6); Cbe_a=np.imag(Ybe_arr)/omega
    Gm0_a=np.abs(gm_arr)
    with np.errstate(divide="ignore",invalid="ignore"): tau_a=-np.angle(gm_arr)/omega
    Rbi=_safe_median(Rbi_a,n_low); Rbc=_safe_median(Rbc_a,n_low); Cbc=_safe_median(Cbc_a,n_low)
    Rbe=_safe_median(Rbe_a,n_low); Cbe=_safe_median(Cbe_a,n_low); Gm0=_safe_median(Gm0_a,n_low); tau=_safe_median(tau_a,n_low)
    return dict(Rbi=Rbi,Rbe=Rbe,Cbe=Cbe,Rbc=Rbc,Cbc=Cbc,Gm0=Gm0,tau=tau),\
           dict(Rbe=Rbe_a,Cbe=Cbe_a,Rbc=Rbc_a,Cbc=Cbc_a,Gm0=Gm0_a,tau=tau_a)

# ─── SECTION E2 ── DEGACHI (2008) ─────────────────────────────────────────────
def ssm_step3_intrinsic_degachi(Y_ex1,freq,n_low):
    """Degachi & Ghannouchi [2], Eqs. 3–26."""
    omega=2.0*np.pi*freq; omega2=omega**2; N=len(freq)
    with np.errstate(divide="ignore",invalid="ignore"):
        Z1=1.0/(Y_ex1[:,0,0]+Y_ex1[:,0,1])
        Z3=(Y_ex1[:,1,0]+Y_ex1[:,0,0])/((Y_ex1[:,0,0]+Y_ex1[:,0,1])*(Y_ex1[:,1,1]+Y_ex1[:,0,1]))
        Z4=-1.0/Y_ex1[:,0,1]
    with np.errstate(divide="ignore",invalid="ignore"): Fbi=omega/np.imag(Z1/Z3)
    n_fit=max(4,2*N//3)
    mask=np.isfinite(Fbi[:n_fit])&(np.abs(Fbi[:n_fit])<1e15)&(Fbi[:n_fit]>0)
    if mask.sum()>=3:
        try: coeffs=np.polyfit(omega2[:n_fit][mask],Fbi[:n_fit][mask],1); B0,A0=float(coeffs[0]),float(coeffs[1])
        except: A0,B0=1.0,0.0
    else: A0,B0=1.0,0.0
    Tbi=float(np.sqrt(max(B0/A0,0.0))) if A0>1e-30 else 0.0
    corr13=(Z1/Z3)*(1.0+1j*omega*Tbi)
    RbiOverRbc_arr=np.real(corr13)
    with np.errstate(divide="ignore",invalid="ignore"):
        RbiCbc_arr=np.where(omega>0,np.imag(corr13)/omega,np.nan)
    RbiOverRbc=_safe_median(RbiOverRbc_arr,n_low); RbiCbc=_safe_median(RbiCbc_arr,n_low)
    with np.errstate(divide="ignore",invalid="ignore"):
        Z1_mod=Z1*(1.0+1j*omega*Tbi); F1=omega/np.imag(Z1_mod)
    mask1=np.isfinite(F1[:n_fit])&(np.abs(F1[:n_fit])<1e15)&(F1[:n_fit]>0)
    if mask1.sum()>=3:
        try: coeffs1=np.polyfit(omega2[:n_fit][mask1],F1[:n_fit][mask1],1); B1,A1=float(coeffs1[0]),float(coeffs1[1])
        except: A1,B1=1.0,0.0
    else: A1,B1=1.0,0.0
    Tbe=float(np.sqrt(max(B1/A1,0.0))) if A1>1e-30 else 0.0
    F2=Z1*(1.0+1j*omega*Tbe)*(1.0+1j*omega*Tbi)
    R_arr=np.real(F2)
    with np.errstate(divide="ignore",invalid="ignore"):
        RT_arr=np.where(omega>0,np.imag(F2)/omega,np.nan)
    R=_safe_median(R_arr,n_low); RT=_safe_median(RT_arr,n_low)
    mat=np.array([[1.0+RbiOverRbc,1.0],[Tbi+RbiCbc,Tbe]]); rhs=np.array([R,RT])
    try:
        sol=np.linalg.solve(mat,rhs); Rbe,Rbi=float(sol[0]),float(sol[1])
        if Rbe<=0 or Rbi<=0: raise ValueError
    except: Rbi=_safe_median(np.real(Z1-Z3),n_low); Rbe=max(R-Rbi,1.0)
    Rbc=float(Rbi/RbiOverRbc) if abs(RbiOverRbc)>1e-30 else 1e6
    Cbc=float(RbiCbc/Rbi) if abs(Rbi)>1e-30 else 0.0
    Cbe=float(Tbe/Rbe) if abs(Rbe)>1e-30 else 0.0
    Cbi=float(Tbi/Rbi) if abs(Rbi)>1e-30 else 0.0
    Zbi_v=Rbi/(1.0+1j*omega*Rbi*Cbi) if Cbi>0 else Rbi*np.ones(N,dtype=complex)
    Z_in_full=_y_to_z(Y_ex1); Z_in_core=Z_in_full.copy(); Z_in_core[:,0,0]-=Zbi_v
    Y_core=_z_to_y(Z_in_core); gm_arr=Y_core[:,1,0]-Y_core[:,0,1]; Gm0_a=np.abs(gm_arr)
    with np.errstate(divide="ignore",invalid="ignore"): tau_a=-np.angle(gm_arr)/omega
    Gm0=_safe_median(Gm0_a,n_low); tau=_safe_median(tau_a,n_low)
    n_hi=max(N//2,n_low+1)
    with np.errstate(divide="ignore",invalid="ignore"): Ccx_arr=-np.imag(Y_ex1[:,0,1])/omega
    Ccx=max(_safe_median(Ccx_arr[n_hi:]),0.0); Rcx=1e6
    params=dict(Rbi=Rbi,Cbi=Cbi,Rbe=Rbe,Cbe=Cbe,Rbc=Rbc,Cbc=Cbc,Rcx=Rcx,Ccx=Ccx,Gm0=Gm0,tau=tau,
                Tbi=Tbi,Tbe=Tbe,A0=A0,B0=B0,A1=A1,B1=B1,RbiOverRbc=RbiOverRbc,RbiCbc=RbiCbc,R=R,RT=RT)
    arrays=dict(Z1=Z1,Z3=Z3,Z4=Z4,Fbi=Fbi,F1=F1,F2=F2,R_arr=R_arr,RT_arr=RT_arr,
                RbiOverRbc_arr=RbiOverRbc_arr,RbiCbc_arr=RbiCbc_arr,Gm0_a=Gm0_a,tau_a=tau_a,
                Ccx_arr=Ccx_arr,omega2=omega2)
    return params,arrays

# ─── SECTION F ── FORWARD SIMULATION (updated: extended open/short model) ─────
def _build_Y_pad(p,w):
    """Build Y_pad matrix at single frequency using extended open model."""
    Ypbe=_open_elem_Y_sc(p["Cpbe"],p.get("Cpbe_mode","None"),p.get("Cpbe_extra",0.0),w)
    Ypce=_open_elem_Y_sc(p["Cpce"],p.get("Cpce_mode","None"),p.get("Cpce_extra",0.0),w)
    Ypbc=_open_elem_Y_sc(p["Cpbc"],p.get("Cpbc_mode","None"),p.get("Cpbc_extra",0.0),w)
    return np.array([[Ypbe+Ypbc,-Ypbc],[-Ypbc,Ypce+Ypbc]])

def _build_Z_ser(p,w):
    """Build Z_ser matrix at single frequency using extended short model."""
    Zb=_short_lead_Z_sc(p["Rpb"],p["Lb"],p.get("Cpar_Lb",0.0),w)
    Zc=_short_lead_Z_sc(p["Rpc"],p["Lc"],p.get("Cpar_Lc",0.0),w)
    Ze=_short_lead_Z_sc(p["Rpe"],p["Le"],p.get("Cpar_Le",0.0),w)
    return np.array([[Zb+Ze,Ze],[Ze,Zc+Ze]])

def ssm_simulate_T(p,freq,z0=50.0):
    omega=2.0*np.pi*freq; S=np.zeros((len(freq),2,2),dtype=complex)
    for i,w in enumerate(omega):
        Zbe_v=p["Rbe"]/(1.0+1j*w*p["Rbe"]*p["Cbe"]); Zbc_v=p["Rbc"]/(1.0+1j*w*p["Rbc"]*p["Cbc"])
        alpha=p["alpha0"]*np.exp(-1j*w*p["tauC"])/(1.0+1j*w*p["tauB"])
        Z_in=np.array([[p["Rbi"]+Zbe_v,Zbe_v],[Zbe_v-alpha*Zbc_v,(1-alpha)*Zbc_v+Zbe_v]])
        Y_in=np.linalg.inv(Z_in)
        Ybcx=1j*w*p["Cbcx"]; Ybex=1j*w*p["Cbex"]
        Y_ex=Y_in+Ybcx*np.array([[1,-1],[-1,1]])+Ybex*np.array([[1,0],[0,0]])
        Y_tot=np.linalg.inv(np.linalg.inv(Y_ex)+_build_Z_ser(p,w))
        S[i]=_y_to_s_single(Y_tot+_build_Y_pad(p,w),z0)
    return S

def ssm_simulate_pi(p,freq,z0=50.0):
    omega=2.0*np.pi*freq; S=np.zeros((len(freq),2,2),dtype=complex)
    for i,w in enumerate(omega):
        Ybe_v=1.0/p["Rbe"]+1j*w*p["Cbe"]; Ybc_v=1.0/p["Rbc"]+1j*w*p["Cbc"]
        gm_v=p["Gm0"]*np.exp(-1j*w*p["tau"])
        Y_core=np.array([[Ybe_v+Ybc_v,-Ybc_v],[gm_v-Ybc_v,Ybc_v]])
        Z_in=np.linalg.inv(Y_core)+np.array([[p["Rbi"],0],[0,0]])
        Y_in=np.linalg.inv(Z_in)
        Ybcx=1j*w*p["Cbcx"]; Ybex=1j*w*p["Cbex"]
        Y_ex=Y_in+Ybcx*np.array([[1,-1],[-1,1]])+Ybex*np.array([[1,0],[0,0]])
        Y_tot=np.linalg.inv(np.linalg.inv(Y_ex)+_build_Z_ser(p,w))
        S[i]=_y_to_s_single(Y_tot+_build_Y_pad(p,w),z0)
    return S

def ssm_simulate_degachi(p,freq,z0=50.0):
    omega=2.0*np.pi*freq; S=np.zeros((len(freq),2,2),dtype=complex)
    for i,w in enumerate(omega):
        Rbi,Cbi=p["Rbi"],p["Cbi"]; Rbe,Cbe=p["Rbe"],p["Cbe"]; Rbc,Cbc=p["Rbc"],p["Cbc"]
        Rcx,Ccx=p["Rcx"],p["Ccx"]; Gm0,tau=p["Gm0"],p["tau"]
        Zbi_v=Rbi/(1.0+1j*w*Rbi*Cbi) if Cbi>1e-40 else complex(Rbi)
        Zbe_v=Rbe/(1.0+1j*w*Rbe*Cbe); Zbc_v=Rbc/(1.0+1j*w*Rbc*Cbc)
        Zcx_v=(1.0/(1j*w*Ccx) if Rcx>1e4 and Ccx>1e-40 else Rcx/(1.0+1j*w*Rcx*Ccx) if Ccx>1e-40 else complex(1e9))
        gm_v=Gm0*np.exp(-1j*w*tau)
        Ybe=1.0/Zbe_v; Ybc=1.0/Zbc_v
        Y_core=np.array([[Ybe+Ybc,-Ybc],[gm_v-Ybc,Ybc]])
        Z_core=np.linalg.inv(Y_core)+np.array([[Zbi_v,0.0],[0.0,0.0]])
        Y_int=np.linalg.inv(Z_core)+(1.0/Zcx_v)*np.array([[1,-1],[-1,1]])
        Y_tot=np.linalg.inv(np.linalg.inv(Y_int)+_build_Z_ser(p,w))
        S[i]=_y_to_s_single(Y_tot+_build_Y_pad(p,w),z0)
    return S

def ssm_residual(S_mea,S_mod):
    total=0.0
    for r in range(2):
        for c in range(2):
            sm=S_mea[:,r,c]; sk=S_mod[:,r,c]; denom=np.sum(np.abs(sm)**2)
            if denom>0: total+=np.sqrt(np.sum(np.abs(sm-sk)**2)/denom)
    return total/4.0*100.0


# ─── SECTION F2 ── FORMULA TRACES ────────────────────────────────────────────
def _render_formula_trace_T():
    with st.expander("📐 Extraction & simulation formula trace — T-topology (Cheng 2022)",expanded=False):
        st.markdown("**Dependencies:** Y_ex1 → peel Cbex → Y_ex2 → peel Cbcx → Z_in → intrinsic params")
        st.markdown("**Step 2** *(depends on: Y_ex1)*")
        st.latex(r"\text{[Eq.13]}\;C_{bex}^T=\frac{\mathrm{Im}(Y_{11}+Y_{12})}{\omega}\big|_{\omega\to0}")
        st.latex(r"\text{[Eq.22]}\;C_{bcx}=-\frac{\mathrm{Im}(Y_{ms})\mathrm{Re}(Y_L)-\mathrm{Re}(Y_{ms})\mathrm{Im}(Y_L)}{\omega[\mathrm{Re}(Y_{ms})\mathrm{Re}(Y_{tot})+\mathrm{Im}(Y_{tot})\mathrm{Im}(Y_{ms})]}")
        st.markdown("**Step 3** *(depends on: Y_ex2, Cbcx)*")
        st.latex(r"\text{[Eq.16]}\;Z_{be}=Z_{12},\;Z_{bc}=Z_{22}-Z_{21},\;Z_{bi}=Z_{11}-Z_{12}")
        st.latex(r"\text{[Eq.29]}\;\alpha=(Z_{12}-Z_{21})/(Z_{22}-Z_{21}),\;\alpha_0=|\alpha||_{\omega\to0}")
        st.latex(r"\text{[Eq.30]}\;\tau_B=\sqrt{U-1}/\omega,\;\text{[Eq.31]}\;\tau_C=-\arctan[V(1-V^2)^{-1/2}]/(2\omega)")
        st.markdown("**Forward simulation** *(inside→outside)*")
        st.latex(r"Z_{be}^{sim}=R_{be}/(1+j\omega R_{be}C_{be}),\;\alpha=\alpha_0 e^{-j\omega\tau_C}/(1+j\omega\tau_B)")
        st.latex(r"[Z_{in}^{sim}]=\begin{bmatrix}R_{bi}+Z_{be}&Z_{be}\\Z_{be}-\alpha Z_{bc}&(1-\alpha)Z_{bc}+Z_{be}\end{bmatrix}")
        st.latex(r"[Y_{ex}]=[Z_{in}]^{-1}+j\omega C_{bcx}\begin{pmatrix}1&-1\\-1&1\end{pmatrix}+j\omega C_{bex}\begin{pmatrix}1&0\\0&0\end{pmatrix}")
        st.latex(r"[Y_{tot}]=([Y_{ex}]^{-1}+[Z_{ser}])^{-1}\;,\quad S=(I-Z_0[Y_{tot}+Y_{pad}])(I+Z_0[Y_{tot}+Y_{pad}])^{-1}")

def _render_formula_trace_pi():
    with st.expander("📐 Extraction & simulation formula trace — π-topology (Cheng 2022 / Zhang 2015)",expanded=False):
        st.markdown("**Step 2** *(depends on: Y_ex1)*")
        st.latex(r"\text{[Eqs.26–28]}\;B=Y_{12}+Y_{22},C=Y_{11}+Y_{21},\;C_{bex}^\pi=\frac{\mathrm{Re}(B)\mathrm{Re}(C)+\mathrm{Im}(B)\mathrm{Im}(C)}{\omega\,\mathrm{Im}(B)}")
        st.markdown("**Step 3** *(depends on: Y_ex2, Cbcx → Z_in same peel as T)*")
        st.latex(r"g_m=(Z_{12}-Z_{21})/(Z_{bc}Z_{12})\Rightarrow G_{m0}=|g_m|,\;\tau=-\angle g_m/\omega")
        st.latex(r"Y_{be}=(Z_{22}-Z_{12})/(Z_{12}Z_{bc})\Rightarrow R_{be}=1/\mathrm{Re}(Y_{be}),\;C_{be}=\mathrm{Im}(Y_{be})/\omega")
        st.markdown("**Forward simulation** *(inside→outside)*")
        st.latex(r"[Y_{core}]=\begin{bmatrix}Y_{be}+Y_{bc}&-Y_{bc}\\g_m-Y_{bc}&Y_{bc}\end{bmatrix},\;[Z_{in}^{sim}]=[Y_{core}]^{-1}+\begin{bmatrix}R_{bi}&0\\0&0\end{bmatrix}")
        st.latex(r"[Y_{tot}]=([Y_{ex}]^{-1}+[Z_{ser}])^{-1},\;S=(I-Z_0[Y_{tot}+Y_{pad}])(I+Z_0[Y_{tot}+Y_{pad}])^{-1}")

def _render_formula_trace_degachi():
    with st.expander("📐 Extraction & simulation formula trace — Degachi (2008) augmented π",expanded=False):
        st.markdown("**Dependency chain:** Y_ex1 → Z1,Z3,Z4 → Tbi → Tbe → R,RT → Rbe,Rbi → all others")
        st.latex(r"\text{[Eq.3]}\;Z_1=\frac{1}{Y_{11}+Y_{12}},\;\text{[Eq.4]}\;Z_3=\frac{Y_{21}+Y_{11}}{(Y_{11}+Y_{12})(Y_{22}+Y_{12})},\;\text{[Eq.5]}\;Z_4=-\frac{1}{Y_{12}}")
        st.latex(r"\text{[Eq.8]}\;F_{bi}=\omega/\mathrm{Im}(Z_1/Z_3)=A_0+\omega^2 B_0\;\Rightarrow\;\text{[Eq.12]}\;T_{bi}=\sqrt{B_0/A_0}")
        st.latex(r"\text{[Eq.13]}\;\mathrm{Re}[Z_1/Z_3(1+j\omega T_{bi})]=R_{bi}/R_{bc},\;\text{[Eq.14]}\;\mathrm{Im}[\cdot]/\omega=R_{bi}C_{bc}")
        st.latex(r"\text{[Eq.19]}\;F_1=\omega/\mathrm{Im}[Z_1(1+j\omega T_{bi})]=A+\omega^2 B\;\Rightarrow\;T_{be}=\sqrt{B/A}")
        st.latex(r"\text{[Eqs.23–24]}\;R=\mathrm{Re}(F_2),\;RT=\mathrm{Im}(F_2)/\omega")
        st.latex(r"\text{[Eq.25]}\;\begin{bmatrix}R_{be}\\R_{bi}\end{bmatrix}=\begin{bmatrix}1+R_{bi}/R_{bc}&1\\T_{bi}+R_{bi}C_{bc}&T_{be}\end{bmatrix}^{-1}\begin{bmatrix}R\\RT\end{bmatrix}")
        st.markdown("**Forward simulation** *(inside→outside, 5 steps)*")
        st.latex(r"\text{1. }Z_{bi}=R_{bi}/(1+j\omega R_{bi}C_{bi}),\;Z_{be},Z_{bc}\text{ similarly}")
        st.latex(r"\text{2. }[Y_{core}]=[[Y_{be}+Y_{bc},-Y_{bc}],[g_m-Y_{bc},Y_{bc}]]")
        st.latex(r"\text{3. }[Z_{core}]=[Y_{core}]^{-1}+[[Z_{bi},0],[0,0]]")
        st.latex(r"\text{4. }[Y_{int}]=[Z_{core}]^{-1}+(1/Z_{cx})[[1,-1],[-1,1]]")
        st.latex(r"\text{5. }[Y_{tot}]=([Y_{int}]^{-1}+[Z_{ser}])^{-1},\;S=(I-Z_0[Y_{tot}+Y_{pad}])(I+Z_0[Y_{tot}+Y_{pad}])^{-1}")

# ─── SECTION G ── SCHEMATIC ───────────────────────────────────────────────────
def _fv(p,key,scale,unit,d=2):
    v=p.get(key)
    if v is None: return ""
    try:
        fv=float(v); return "" if not np.isfinite(fv) else f"{fv*scale:.{d}f} {unit}"
    except: return ""

def make_topology_fig(params,topology="T"):
    p=params or {}
    C_PAD="#E67E22"; C_EXT="#27AE60"; C_INT="#2980B9"; C_SRC="#C0392B"; C_WIR="#2C3E50"
    fig,ax=plt.subplots(figsize=(17,7))
    ax.set_xlim(-0.5,17); ax.set_ylim(-1.0,8.5); ax.axis("off")
    ax.set_facecolor("white"); fig.patch.set_facecolor("white"); lw=1.8
    def wl(x1,y1,x2,y2,c=C_WIR,lw_=None): ax.plot([x1,x2],[y1,y2],color=c,lw=lw_ or lw,solid_capstyle="round",zorder=2)
    def dot(x,y): ax.plot(x,y,"o",color=C_WIR,ms=5.5,zorder=6)
    def box(cx,cy,sym,lbl,val,color,w=0.88,h=0.42):
        rect=FancyBboxPatch((cx-w/2,cy-h/2),w,h,boxstyle="round,pad=0.04",fc=color+"28",ec=color,lw=1.9,zorder=4); ax.add_patch(rect)
        ax.text(cx,cy+0.04,sym,ha="center",va="center",fontsize=9,fontweight="bold",color=color,zorder=5)
        ax.text(cx,cy+h/2+0.12,lbl,ha="center",va="bottom",fontsize=8,color="#222",style="italic",zorder=5)
        if val: ax.text(cx,cy-h/2-0.10,val,ha="center",va="top",fontsize=7,color="#555",zorder=5)
    def rc_block(cx,cy,rlbl,rval,clbl,cval,color,w=0.62,h=1.10):
        rect=FancyBboxPatch((cx-w/2,cy-h/2),w,h,boxstyle="round,pad=0.04",fc=color+"18",ec=color,lw=1.7,ls="dashed",zorder=4); ax.add_patch(rect)
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
        ctr=(y_top+y_bot)/2; r=0.34; circ=plt.Circle((cx,ctr),r,fc="white",ec=C_SRC,lw=2.0,zorder=4); ax.add_patch(circ)
        ax.annotate("",xy=(cx,ctr+r*0.55),xytext=(cx,ctr-r*0.55),arrowprops=dict(arrowstyle="->",color=C_SRC,lw=2.0),zorder=5)
        wl(cx,y_top,cx,ctr+r); wl(cx,ctr-r,cx,y_bot)
        ax.text(cx+r+0.15,ctr,lbl,ha="left",va="center",fontsize=8,color=C_SRC,zorder=5)
    def port(x,y,lbl,color=C_INT):
        ax.plot(x,y,"o",ms=18,color=color,alpha=0.2,zorder=3); ax.plot(x,y,"o",ms=9,color=color,zorder=4)
        ax.text(x,y,lbl,ha="center",va="center",fontsize=10,fontweight="bold",color="white",zorder=5)
    def layer_box(x0,y0,x1,y1,label,color):
        rect=FancyBboxPatch((x0,y0),x1-x0,y1-y0,boxstyle="round,pad=0.08",fc=color+"08",ec=color,lw=1.5,ls="dotted",zorder=0); ax.add_patch(rect)
        ax.text(x0+0.15,y1-0.12,label,ha="left",va="top",fontsize=8,color=color,style="italic",zorder=1)
    y_top=5.5; y_emi=1.5; y_gnd=0.2
    xB=0.7; xLb=1.85; xRb=3.0; xn1=3.85; xRbi=5.0; xn2=5.9; xZbc=8.4
    xn3=10.8; xRc=11.9; xLc=13.1; xC=14.2
    xCbex=xn1; xZbe=xn2; xCbcx=xn3; xSrc=xn3+1.3; xRe=7.2; xLe=9.0
    wl(xB-0.5,y_gnd,xC+0.5,y_gnd,C_WIR,2.0)
    wl(xB-0.5,y_emi,xB-0.5,y_gnd); wl(xC+0.5,y_emi,xC+0.5,y_gnd)
    for xg in [1.0,xC+0.5]:
        for k in range(3): ax.plot([xg-0.32+k*0.08,xg+0.32-k*0.08],[y_gnd-k*0.13]*2,color=C_WIR,lw=2)
    wl(xB-0.5,y_emi,xCbex-0.2,y_emi); wl(xLe+0.47,y_emi,xSrc+0.4,y_emi)
    wl(xSrc+0.4,y_emi,xC+0.5,y_emi); wl(xCbex-0.2,y_emi,xRe-0.44,y_emi)
    box(xRe,y_emi,"R","Re",_fv(p,"Rpe",1,"Ω"),C_PAD); wl(xRe+0.44,y_emi,xLe-0.44,y_emi)
    box(xLe,y_emi,"L","Le",_fv(p,"Le",1e12,"pH"),C_PAD); wl(xLe+0.44,y_emi,xLe+0.6,y_emi)
    port(xLe+0.85,y_emi,"E","#7F8C8D")
    port(xB,y_top,"B"); wl(xB,y_top,xLb-0.44,y_top)
    box(xLb,y_top,"L","Lb",_fv(p,"Lb",1e12,"pH"),C_PAD); wl(xLb+0.44,y_top,xRb-0.44,y_top)
    box(xRb,y_top,"R","Rb",_fv(p,"Rpb",1,"Ω"),C_PAD); wl(xRb+0.44,y_top,xn1,y_top); dot(xn1,y_top)
    wl(xn1,y_top,xRbi-0.44,y_top)
    box(xRbi,y_top,"R","Zbi" if topology=="D" else "Rbi",_fv(p,"Rbi",1,"Ω"),C_INT)
    wl(xRbi+0.44,y_top,xn2,y_top); dot(xn2,y_top)
    wl(xn2,y_top,xn3,y_top); dot(xn3,y_top); wl(xn3,y_top,xRc-0.44,y_top)
    box(xRc,y_top,"R","Rc",_fv(p,"Rpc",1,"Ω"),C_PAD); wl(xRc+0.44,y_top,xLc-0.44,y_top)
    box(xLc,y_top,"L","Lc",_fv(p,"Lc",1e12,"pH"),C_PAD); wl(xLc+0.44,y_top,xC,y_top)
    port(xC,y_top,"C")
    y_zbc=y_top-1.35; wl(xn2,y_top,xn2,y_zbc); wl(xn3,y_top,xn3,y_zbc)
    wl(xn2,y_zbc,xZbc-0.32,y_zbc); wl(xZbc+0.32,y_zbc,xn3,y_zbc)
    rval_bc=(_fv(p,"Rbc",1e-3,"kΩ") if (p.get("Rbc") or 0)>1000 else _fv(p,"Rbc",1,"Ω"))
    rc_block(xZbc,y_zbc,"Rbc",rval_bc,"Cbc",_fv(p,"Cbc",1e15,"fF"),C_INT)
    if topology=="D" and p.get("Ccx") is not None:
        cap_v(xCbex,y_top,y_emi,"Ccx",_fv(p,"Ccx",1e15,"fF"),C_EXT)
        ax.text(xRbi,y_top+0.65,"Cbi:"+_fv(p,"Cbi",1e15,"fF"),ha="center",fontsize=7,color=C_INT,zorder=5)
    else: cap_v(xCbex,y_top,y_emi,"Cbex",_fv(p,"Cbex",1e15,"fF"),C_EXT)
    wl(xZbe,y_top,xZbe,y_top-0.25)
    rval_be=(_fv(p,"Rbe",1e-3,"kΩ") if (p.get("Rbe") or 0)>1000 else _fv(p,"Rbe",1,"Ω"))
    cval_be=(_fv(p,"Cbe",1e12,"pF") if (p.get("Cbe") or 0)>1e-12 else _fv(p,"Cbe",1e15,"fF"))
    rc_block(xZbe,(y_top+y_emi)/2-0.15,"Rbe",rval_be,"Cbe",cval_be,C_INT,h=1.2)
    wl(xZbe,y_top-0.25,xZbe,(y_top+y_emi)/2-0.15+0.62)
    wl(xZbe,(y_top+y_emi)/2-0.15-0.62,xZbe,y_emi)
    if topology!="D": cap_v(xCbcx,y_top,y_emi,"Cbcx",_fv(p,"Cbcx",1e15,"fF"),C_EXT)
    wl(xn3,y_top,xSrc,y_top); wl(xSrc,y_emi+0.36,xSrc,y_emi)
    cur_src(xSrc,y_top,y_emi+0.36,"α·IE" if topology=="T" else "gm·Vbe")
    xCpbe=xB-0.5; wl(xB,y_top,xCpbe,y_top)
    cap_v(xCpbe,y_top,y_emi,"Cpbe",_fv(p,"Cpbe",1e15,"fF"),C_PAD)
    xCpce=xC+0.5; wl(xC,y_top,xCpce,y_top)
    cap_v(xCpce,y_top,y_emi,"Cpce",_fv(p,"Cpce",1e15,"fF"),C_PAD)
    y_cpbc=7.3; xm=(xB+xC)/2
    wl(xB,y_top,xB,y_cpbc); wl(xC,y_top,xC,y_cpbc); wl(xB,y_cpbc,xm-0.18,y_cpbc)
    ax.plot([xm-0.18]*2,[y_cpbc-0.22,y_cpbc+0.22],color=C_PAD,lw=2.8,zorder=5)
    ax.plot([xm+0.18]*2,[y_cpbc-0.22,y_cpbc+0.22],color=C_PAD,lw=2.8,zorder=5)
    wl(xm+0.18,y_cpbc,xC,y_cpbc)
    ax.text(xm,y_cpbc+0.32,"Cpbc",ha="center",va="bottom",fontsize=8,color="#222",style="italic")
    ax.text(xm,y_cpbc-0.35,_fv(p,"Cpbc",1e15,"fF"),ha="center",va="top",fontsize=7,color="#555")
    layer_box(xn2-0.2,y_emi-0.4,xn3+0.2,y_top+0.55,"Intrinsic Model",C_INT)
    if topology!="D": layer_box(xn1-0.2,y_emi-0.6,xn3+0.2,y_top+0.75,"Extrinsic Distributed Caps",C_EXT)
    handles=[mpatches.Patch(fc=C_PAD+"40",ec=C_PAD,lw=1.5,label="Pad Parasitics"),
             mpatches.Patch(fc=C_EXT+"40",ec=C_EXT,lw=1.5,label="Extrinsic Caps"),
             mpatches.Patch(fc=C_INT+"40",ec=C_INT,lw=1.5,label="Intrinsic Model"),
             mpatches.Patch(fc="white",ec=C_SRC,lw=1.5,label="Current Source")]
    ax.legend(handles=handles,loc="lower left",fontsize=8.5,framealpha=0.95,edgecolor="#ccc",ncol=2)
    tname={"T":"T-topology (Cheng 2022)","pi":"π-topology (Cheng 2022)","D":"Degachi (2008) augmented π"}.get(topology,"")
    ax.set_title(f"HBT Small-Signal Model — {tname}",fontsize=13,fontweight="bold",pad=10)
    plt.tight_layout(pad=0.4); return fig


# ─── SECTION H ── SMITH CHART ────────────────────────────────────────────────
_SMITH_COLORS={"S11":"#1f77b4","S22":"#ff7f0e","S21":"#2ca02c","S12":"#d62728"}

def _make_ssm_smith(S_mea,S_sim,model_name,error_pct,scales=None):
    if scales is None: scales={"S11":1.0,"S12":1.0,"S21":1.0,"S22":1.0}
    fig=go.Figure()
    for tr in _extended_smith_grid(1.0): fig.add_trace(tr)
    for name,(r,c) in [("S11",(0,0)),("S22",(1,1)),("S21",(1,0)),("S12",(0,1))]:
        col=_SMITH_COLORS[name]; sc=scales.get(name,1.0)
        sm=S_mea[:,r,c]*sc; sk=S_sim[:,r,c]*sc
        sc_lbl="" if abs(sc-1.0)<1e-9 else (f" ×{sc:.2g}" if sc>=1 else f" ÷{1/sc:.2g}")
        fig.add_trace(go.Scatter(x=sm.real,y=sm.imag,mode="markers",name=f"{name}{sc_lbl} Meas.",
            marker=dict(color=col,size=5,symbol="circle",line=dict(color=col,width=0.5)),
            hovertemplate=f"{name} Meas.<br>Re=%{{x:.4f}}<br>Im=%{{y:.4f}}<extra></extra>"))
        fig.add_trace(go.Scatter(x=sk.real,y=sk.imag,mode="lines",name=f"{name}{sc_lbl} Model",
            line=dict(color=col,width=2.0,dash="dash"),
            hovertemplate=f"{name} Model<br>Re=%{{x:.4f}}<br>Im=%{{y:.4f}}<extra></extra>"))
    fig.update_layout(
        title=f"Measured vs Modeled — {model_name}   (Residual: {error_pct:.2f}%)",
        xaxis=dict(title="Re(Γ)",range=[-1.1,1.1],scaleanchor="y",scaleratio=1,showgrid=False,zeroline=False),
        yaxis=dict(title="Im(Γ)",range=[-1.1,1.1],showgrid=False,zeroline=False),
        plot_bgcolor="white",paper_bgcolor="white",height=560,
        margin=dict(l=50,r=30,t=55,b=50),legend=dict(x=1.02,y=1.0,xanchor="left"),hovermode="closest",
        annotations=[dict(x=0.5,y=-0.08,xref="paper",yref="paper",showarrow=False,
                          text="● Measured (markers)  |  - - Modeled (dashed)",font=dict(size=10,color="gray"),align="center")])
    return fig

def _smith_scale_controls(fname,topo_key):
    st.markdown("<small>**S-param display scale** — multiply before plotting (does not affect residual)</small>",unsafe_allow_html=True)
    c1,c2,c3,c4=st.columns(4); sc={}
    for col_w,name,default in [(c1,"S11",1.0),(c2,"S12",1.0),(c3,"S21",1.0),(c4,"S22",1.0)]:
        sk=f"smith_scale_{topo_key}_{name}_{fname}"
        if sk not in st.session_state: st.session_state[sk]=default
        sc[name]=col_w.number_input(f"{name} ×",min_value=0.01,max_value=1000.0,value=float(st.session_state[sk]),step=0.5,format="%.2f",key=sk)
    return sc

# ─── SECTION H2 ── STEP 1 DIAGNOSTIC PLOTS (enhanced) ───────────────────────
_OPEN_MODES=["None","Parallel L","Series L","Series R"]
_OPEN_MODE_UNIT={"None":None,"Parallel L":"pH","Series L":"pH","Series R":"Ω"}
_OPEN_MODE_SCALE={"None":1,"Parallel L":1e12,"Series L":1e12,"Series R":1.0}

def _render_open_plots(open_data,para_caps,open_arr,fname=""):
    """Enhanced open plots: C, conductance (R indication), extra-element model selector."""
    f_o,S_o,z0_o=open_data; f_ghz=f_o*1e-9; omega=2.0*np.pi*f_o

    # ── Per-cap extra element controls ───────────────────────────────────────
    with st.expander("🔧 Open Pad Element Model — extra parasitic options",expanded=False):
        st.markdown(
            "Each pad capacitor can include a secondary parasitic element.  \n"
            "**Parallel L** = inductor in parallel with C (resonance at 1/√LC).  \n"
            "**Series L** = inductor in series with C (increases effective C near resonance).  \n"
            "**Series R** = resistance in series with C (adds loss, flattens conductance peak).  \n"
            "These affect the *modeled* overlay below, the de-embedding, and the s2p forward sim."
        )
        _cap_defs=[("Cpbe","Cpbe (B-E)"),("Cpce","Cpce (C-E)"),("Cpbc","Cpbc (B-C)")]
        for cap,cap_lbl in _cap_defs:
            st.markdown(f"**{cap_lbl}**")
            c1,c2=st.columns([2,1])
            mode_sk=f"open_mode_{cap}_{fname}"
            extra_sk=f"open_extra_{cap}_{fname}"
            if mode_sk not in st.session_state: st.session_state[mode_sk]="None"
            if extra_sk not in st.session_state: st.session_state[extra_sk]=0.0
            mode=c1.radio(f"",_OPEN_MODES,horizontal=True,key=mode_sk,label_visibility="collapsed")
            if mode!="None":
                unit=_OPEN_MODE_UNIT[mode]
                sc=_OPEN_MODE_SCALE[mode]
                c2.number_input(f"Extra {unit}",min_value=0.0,step=0.1 if unit=="pH" else 0.01,
                                format="%.3f" if unit=="pH" else "%.4f",key=extra_sk)

    def _get_mode_extra(cap):
        mode=st.session_state.get(f"open_mode_{cap}_{fname}","None")
        extra_disp=float(st.session_state.get(f"open_extra_{cap}_{fname}",0.0))
        sc=_OPEN_MODE_SCALE.get(mode,1)
        return mode, extra_disp/sc  # convert to SI

    # ── Plot 1: Capacitance vs frequency ─────────────────────────────────────
    with st.expander("📊 Open — Pad Capacitances vs Frequency",expanded=True):
        fig_cap=go.Figure()
        for key,lbl,col in [("Cpbe","Cpbe","#1f77b4"),("Cpce","Cpce","#ff7f0e"),("Cpbc","Cpbc","#2ca02c")]:
            arr_fF=open_arr[key]*1e15; val_fF=para_caps[key]*1e15
            mode,extra=_get_mode_extra(key)
            fig_cap.add_trace(go.Scatter(x=f_ghz,y=arr_fF,name=f"{lbl} (meas.)",line=dict(color=col,width=2),mode="lines"))
            fig_cap.add_trace(go.Scatter(x=[f_ghz[0],f_ghz[-1]],y=[val_fF,val_fF],
                name=f"{lbl}={val_fF:.3f} fF",line=dict(color=col,width=1.8,dash="dash"),mode="lines"))
            # Modeled effective C = Im(Y_elem)/ω
            if mode!="None":
                Y_mod_arr=np.array([_open_elem_Y_sc(para_caps[key],mode,extra,w) for w in omega])
                Ceff_fF=np.imag(Y_mod_arr)/omega*1e15
                fig_cap.add_trace(go.Scatter(x=f_ghz,y=Ceff_fF,
                    name=f"{lbl} model ({mode})",line=dict(color=col,width=2,dash="dot"),mode="lines"))
        fig_cap.update_layout(title="Pad Capacitances — Im(Y)/ω",xaxis_title="Frequency (GHz)",yaxis_title="Cap (fF)",
            plot_bgcolor="white",paper_bgcolor="white",height=360,
            legend=dict(x=1.02,y=1.0,xanchor="left",font=dict(size=9)),margin=dict(l=55,r=10,t=40,b=45),hovermode="x unified")
        fig_cap.update_xaxes(showgrid=True,gridcolor="#ebebeb")
        fig_cap.update_yaxes(showgrid=True,gridcolor="#ebebeb",range=[0,50])
        st.plotly_chart(fig_cap,use_container_width=True,key=f"step1_cap_{fname}")
        st.caption("Flat line = pure C. Slope/resonance = inductance effect. Range fixed 0–50 fF.")

    # ── Plot 2: Conductance (Re(Y)) — indicates series R ─────────────────────
    with st.expander("📊 Open — Pad Conductance vs Frequency (Re(Y) — series R indicator)",expanded=False):
        fig_g=go.Figure()
        for key,lbl,col in [("Cpbe","Gpbe","#1f77b4"),("Cpce","Gpce","#ff7f0e"),("Cpbc","Gpbc","#2ca02c")]:
            arr_mS=open_arr[f"G{key[1:]}"] * 1e3
            mode,extra=_get_mode_extra(key)
            fig_g.add_trace(go.Scatter(x=f_ghz,y=arr_mS,name=f"{lbl} (meas.)",line=dict(color=col,width=2),mode="lines"))
            if mode=="Series R" and extra>0:
                G_mod=np.array([np.real(_open_elem_Y_sc(para_caps[key],mode,extra,w))*1e3 for w in omega])
                fig_g.add_trace(go.Scatter(x=f_ghz,y=G_mod,name=f"{lbl} model (Series R={extra:.3f} Ω)",
                    line=dict(color=col,width=2,dash="dot"),mode="lines"))
        fig_g.add_hline(y=0,line_color="#aaa",line_width=1)
        fig_g.update_layout(title="Pad Conductance Re(Y) — nonzero = series R or parallel G",
            xaxis_title="Frequency (GHz)",yaxis_title="Conductance (mS)",
            plot_bgcolor="white",paper_bgcolor="white",height=320,
            legend=dict(x=1.02,y=1.0,xanchor="left",font=dict(size=9)),margin=dict(l=55,r=10,t=40,b=45),hovermode="x unified")
        fig_g.update_xaxes(showgrid=True,gridcolor="#ebebeb"); fig_g.update_yaxes(showgrid=True,gridcolor="#ebebeb")
        st.plotly_chart(fig_g,use_container_width=True,key=f"step1_cond_{fname}")
        st.caption(
            "Pure C → Re(Y)=0. Series R → Re(Y) = ω²RC² / (1+ω²R²C²) — rises then saturates.  \n"
            "Parallel L → shifts the Im(Y)/ω plot but leaves Re(Y)=0.  \n"
            "Select 'Series R' above to overlay the modeled conductance.")

    # ── Plot 3: Im(Y)/ω vs 1/ω² — linearises Parallel L effect ─────────────
    with st.expander("📊 Open — Im(Y)/ω vs 1/ω²  (Parallel L linearisation)",expanded=False):
        fig_l=go.Figure()
        one_over_omega2=1.0/(omega**2+1e-60)
        for key,lbl,col in [("Cpbe","Cpbe","#1f77b4"),("Cpce","Cpce","#ff7f0e"),("Cpbc","Cpbc","#2ca02c")]:
            Ceff=open_arr[key]
            fig_l.add_trace(go.Scatter(x=one_over_omega2*1e-18,y=Ceff*1e15,name=f"{lbl}",line=dict(color=col,width=2),mode="lines",
                hovertemplate="1/ω²=%{x:.4f}×10¹⁸<br>Im(Y)/ω=%{y:.3f} fF<extra></extra>"))
        fig_l.update_layout(title="Im(Y)/ω vs 1/ω² — slope = −1/L (if Parallel L present)",
            xaxis_title="1/ω² (× 10¹⁸ rad⁻²s²)",yaxis_title="Im(Y)/ω  (fF equivalent)",
            plot_bgcolor="white",paper_bgcolor="white",height=320,
            legend=dict(x=1.02,y=1.0,xanchor="left",font=dict(size=9)),margin=dict(l=55,r=10,t=40,b=45),hovermode="x unified")
        fig_l.update_xaxes(showgrid=True,gridcolor="#ebebeb"); fig_l.update_yaxes(showgrid=True,gridcolor="#ebebeb",range=[0,50])
        st.plotly_chart(fig_l,use_container_width=True,key=f"step1_lind_{fname}")
        st.caption("Parallel L model: Im(Y)/ω = C − 1/(ω²L). A straight line with negative slope → L = −1/slope.")
    
    # ── Smith chart: measured Open vs forward-simulated Open ─────────────────
    with st.expander("📡 Open — Measured vs Modelled Smith Chart", expanded=False):
        # Build full para dict for _simulate_open_Sparams
        _p_open = {}
        for cap in ["Cpbe", "Cpce", "Cpbc"]:
            mode, extra = _get_mode_extra(cap)
            _p_open[cap]            = para_caps[cap]
            _p_open[f"{cap}_mode"]  = mode
            _p_open[f"{cap}_extra"] = extra
        S_open_mea = open_data[1]          # measured open S-params
        S_open_sim = _simulate_open_Sparams(_p_open, f_o, z0_o)

        fig_os = go.Figure()
        for tr in _extended_smith_grid(1.0):
            fig_os.add_trace(tr)

        _oc = {"S11": "#1f77b4", "S22": "#ff7f0e", "S21": "#2ca02c", "S12": "#d62728"}
        for name, (r, c) in [("S11",(0,0)), ("S22",(1,1)), ("S21",(1,0)), ("S12",(0,1))]:
            col = _oc[name]
            sm  = S_open_mea[:, r, c]
            sk  = S_open_sim[:, r, c]
            fig_os.add_trace(go.Scatter(
                x=sm.real, y=sm.imag, mode="markers", name=f"{name} Meas.",
                marker=dict(color=col, size=5, symbol="circle"),
                hovertemplate=f"{name} Meas.<br>Re=%{{x:.4f}}<br>Im=%{{y:.4f}}<extra></extra>"))
            fig_os.add_trace(go.Scatter(
                x=sk.real, y=sk.imag, mode="lines",   name=f"{name} Model",
                line=dict(color=col, width=2, dash="dash"),
                hovertemplate=f"{name} Model<br>Re=%{{x:.4f}}<br>Im=%{{y:.4f}}<extra></extra>"))

        err_open = ssm_residual(S_open_mea, S_open_sim)
        fig_os.update_layout(
            title=f"Open dummy — Measured vs Modelled   (Residual: {err_open:.2f}%)",
            xaxis=dict(title="Re(Γ)", range=[-1.1,1.1], scaleanchor="y", scaleratio=1,
                       showgrid=False, zeroline=False),
            yaxis=dict(title="Im(Γ)", range=[-1.1,1.1], showgrid=False, zeroline=False),
            plot_bgcolor="white", paper_bgcolor="white", height=520,
            margin=dict(l=50,r=30,t=55,b=50),
            legend=dict(x=1.02, y=1.0, xanchor="left"),
            hovermode="closest")
        st.plotly_chart(fig_os, use_container_width=True, key=f"smith_open_{fname}")
        st.caption("Adjust the extra element controls above — the modelled curve updates live.")
        
    return {cap: _get_mode_extra(cap) for cap,_ in [("Cpbe",""),("Cpce",""),("Cpbc","")]}


def _render_short_plots(short_arr,para_short,fname=""):
    """Enhanced short plots: inductance, resistance, and parallel C option per lead."""

    # ── Parallel C controls ───────────────────────────────────────────────────
    with st.expander("🔧 Short Lead Model — optional parallel capacitance per lead",expanded=False):
        st.markdown(
            "Adds a capacitance **in parallel** with each lead's R+jωL impedance.  \n"
            "Z_lead_eff = (R+jωL) ∥ (1/jωC_par) = (R+jωL) / (1 + jωC_par(R+jωL))  \n"
            "This causes Lb/Lc/Le to appear frequency-dependent (decreasing at high freq).  \n"
            "Default = 0 (disabled)."
        )
        cpar_cols=st.columns(3)
        for col_w,(key,lbl) in zip(cpar_cols,[("Cpar_Lb","Lb"),("Cpar_Lc","Lc"),("Cpar_Le","Le")]):
            ks=f"short_{key}_{fname}"
            if ks not in st.session_state: st.session_state[ks]=0.0
            col_w.number_input(f"C_par_{lbl} (fF)",min_value=0.0,step=0.1,format="%.3f",key=ks)

    def _get_cpar(lead_key):
        return float(st.session_state.get(f"short_{lead_key}_{fname}",0.0))*1e-15

    cpar_Lb=_get_cpar("Cpar_Lb"); cpar_Lc=_get_cpar("Cpar_Lc"); cpar_Le=_get_cpar("Cpar_Le")

    # ── Plot 1: Inductances ───────────────────────────────────────────────────
    with st.expander("📊 Short — Lead Inductances vs Frequency",expanded=True):
        fig_ind=go.Figure(); any_neg=False
        omega_arr=2.0*np.pi*np.arange(len(short_arr["Lb"]))*1e9  # placeholder; no actual freq stored here
        for key,lbl,col,cpar in [("Lb","Lb","#8e44ad",cpar_Lb),("Lc","Lc","#e67e22",cpar_Lc),("Le","Le","#16a085",cpar_Le)]:
            arr_pH=short_arr[key]*1e12; val_pH=para_short[key]*1e12
            if val_pH<0: any_neg=True
            idx=np.arange(len(arr_pH))
            fig_ind.add_trace(go.Scatter(x=idx,y=arr_pH,name=f"{lbl} (per-freq)",line=dict(color=col,width=2),mode="lines"))
            fig_ind.add_trace(go.Scatter(x=[0,len(arr_pH)-1],y=[val_pH,val_pH],
                name=f"{lbl}={val_pH:.2f} pH",line=dict(color=col,width=1.8,dash="dash"),mode="lines"))
        fig_ind.add_hline(y=0,line_color="#333",line_width=1.2,annotation_text="0 pH",annotation_position="left",annotation_font=dict(size=9,color="#333"))
        fig_ind.update_layout(title="Lead Inductances",xaxis_title="Point index",yaxis_title="Inductance (pH)",
            plot_bgcolor="white",paper_bgcolor="white",height=360,
            legend=dict(x=1.02,y=1.0,xanchor="left",font=dict(size=9)),margin=dict(l=55,r=10,t=40,b=45),hovermode="x unified")
        fig_ind.update_xaxes(showgrid=True,gridcolor="#ebebeb")
        fig_ind.update_yaxes(showgrid=True,gridcolor="#ebebeb",range=[0,150])
        st.plotly_chart(fig_ind,use_container_width=True,key=f"step1_ind_{fname}")
        if any_neg: st.warning("One or more lead inductances are negative. Use Short Override to correct.")
        st.caption("Range fixed 0–150 pH. Parallel C (if set above) will make the extracted L appear frequency-dependent.")

    # ── Plot 2: Series resistances ────────────────────────────────────────────
    with st.expander("📊 Short — Lead Series Resistances vs Frequency",expanded=False):
        fig_r=go.Figure()
        for key,lbl,col in [("Rpb","Rb","#8e44ad"),("Rpc","Rc","#e67e22"),("Rpe","Re","#16a085")]:
            arr_O=short_arr[key]
            val_O=para_short[key]
            fig_r.add_trace(go.Scatter(x=np.arange(len(arr_O)),y=arr_O,name=f"{lbl} (per-freq)",line=dict(color=col,width=2),mode="lines"))
            fig_r.add_trace(go.Scatter(x=[0,len(arr_O)-1],y=[val_O,val_O],
                name=f"{lbl}={val_O:.4f} Ω",line=dict(color=col,width=1.8,dash="dash"),mode="lines"))
        fig_r.add_hline(y=0,line_color="#aaa",line_width=1)
        fig_r.update_layout(title="Lead Series Resistances — Re(Z terms from Short)",
            xaxis_title="Point index",yaxis_title="Resistance (Ω)",
            plot_bgcolor="white",paper_bgcolor="white",height=320,
            legend=dict(x=1.02,y=1.0,xanchor="left",font=dict(size=9)),margin=dict(l=55,r=10,t=40,b=45),hovermode="x unified")
        fig_r.update_xaxes(showgrid=True,gridcolor="#ebebeb"); fig_r.update_yaxes(showgrid=True,gridcolor="#ebebeb")
        st.plotly_chart(fig_r,use_container_width=True,key=f"step1_res_{fname}")
        st.caption("Flat curve = clean R extraction. Rising with frequency = skin effect or de-embedding artefact.")

    return {"Cpar_Lb":cpar_Lb,"Cpar_Lc":cpar_Lc,"Cpar_Le":cpar_Le}


# ─── SECTION H3 ── HELPER PLOTS ──────────────────────────────────────────────
def _compute_h21_U(S):
    Y=_s_to_y(S,50.0); y11,y12,y21,y22=Y[:,0,0],Y[:,0,1],Y[:,1,0],Y[:,1,1]
    with np.errstate(divide="ignore",invalid="ignore"):
        h21=-y21/(y11+1e-30); h21_db=10.0*np.log10(np.abs(h21)**2+1e-30)
        num_u=np.abs(y21-y12)**2; den_u=4.0*(y11.real*y22.real-y12.real*y21.real)
        U=np.where(den_u>0,num_u/den_u,np.nan); U_db=10.0*np.log10(np.abs(U)+1e-30)
    return h21_db,U_db

def _render_sparams_comparison(S_raw,freq,z0,para_eff,fname):
    f_ghz=freq*1e-9; Y_deemb=_peel_para_from_Y(_s_to_y(S_raw,z0),freq,para_eff); S_deemb=_y_to_s_batch(Y_deemb,z0)
    with st.expander("📊 S-Parameters: Raw vs De-embedded (after Step 1)",expanded=True):
        st.caption("Solid = raw.  Dashed = after pad removal.")
        cols4=st.columns(4)
        for col_w,(sname,(r,c),color) in zip(cols4,[("S11",(0,0),"#1f77b4"),("S12",(0,1),"#d62728"),("S21",(1,0),"#2ca02c"),("S22",(1,1),"#ff7f0e")]):
            s_raw_v=S_raw[:,r,c]; s_deemb_v=S_deemb[:,r,c]; fig=go.Figure()
            fig.add_trace(go.Scatter(x=f_ghz,y=s_raw_v.real,mode="lines",name="Re (raw)",line=dict(color=color,width=2.2)))
            fig.add_trace(go.Scatter(x=f_ghz,y=s_raw_v.imag,mode="lines",name="Im (raw)",line=dict(color=color,width=2.2,dash="dot")))
            fig.add_trace(go.Scatter(x=f_ghz,y=s_deemb_v.real,mode="lines",name="Re (deemb)",line=dict(color="#888",width=1.6,dash="dash")))
            fig.add_trace(go.Scatter(x=f_ghz,y=s_deemb_v.imag,mode="lines",name="Im (deemb)",line=dict(color="#aaa",width=1.6,dash="longdash")))
            fig.update_layout(title=dict(text=sname,font=dict(size=12)),xaxis_title="GHz",plot_bgcolor="white",paper_bgcolor="white",height=300,
                legend=dict(x=0.01,y=0.99,xanchor="left",yanchor="top",bgcolor="rgba(255,255,255,0.85)",bordercolor="#ccc",borderwidth=1,font=dict(size=7.5)),
                margin=dict(l=42,r=6,t=32,b=42),hovermode="x unified")
            fig.update_xaxes(showgrid=True,gridcolor="#ebebeb"); fig.update_yaxes(showgrid=True,gridcolor="#ebebeb")
            col_w.plotly_chart(fig,use_container_width=True,key=f"cmp_{sname}_{fname}")

def _render_rz12_section(all_data,para_eff,fname):
    st.markdown("#### 📈 Re(Z₁₂) vs 1/IE")
    st.caption("Gao [3] Ch. 5.5.1."); st.latex(r"\mathrm{Re}(Z_{12})=\frac{\eta kT}{q}\cdot\frac{1}{I_E}+R_e")
    if not all_data: st.info("No DUT files loaded."); return
    for fn in all_data:
        for suf,dv in [("use",True),("Ie",0.0)]:
            gk=f"rz12_{suf}_{fn}"
            if gk not in st.session_state: st.session_state[gk]=dv
    ref_fn=list(all_data.keys())[0]; f_ref=all_data[ref_fn]["freq"]*1e-9
    f_min_v=float(f_ref[max(1,np.searchsorted(f_ref,0.01))]); f_max_v=float(f_ref[-1])
    col_fq,_=st.columns([1,2])
    f_extract=col_fq.number_input("Z₁₂ freq (GHz)",min_value=f_min_v,max_value=f_max_v,value=min(1.0,f_max_v*0.05),step=0.5,format="%.2f",key=f"rz12_fext_{fname}")
    st.markdown("**Files:**")
    for h,t in zip(st.columns([0.3,2.3,1.2,1.4,1.4]),["","File","IE (mA)","Re(Z₁₂) (Ω)","Rbe*"]):
        h.markdown(f"<small><b>{t}</b></small>",unsafe_allow_html=True)
    points=[]; Re_ref=para_eff.get("Rpe",0.0)
    for fn,d in all_data.items():
        c0,c1,c2,c3,c4=st.columns([0.3,2.3,1.2,1.4,1.4])
        use=c0.checkbox("",key=f"rz12_use_{fn}__{fname}",value=st.session_state[f"rz12_use_{fn}"],label_visibility="collapsed")
        st.session_state[f"rz12_use_{fn}"]=use; c1.markdown(f"<small>{Path(fn).stem}</small>",unsafe_allow_html=True)
        if not use: continue
        Ie=c2.number_input("",min_value=0.0,step=0.1,format="%.3f",key=f"rz12_Ie_{fn}__{fname}",value=float(st.session_state[f"rz12_Ie_{fn}"]),label_visibility="collapsed")
        st.session_state[f"rz12_Ie_{fn}"]=Ie
        try:
            idx=int(np.argmin(np.abs(d["freq"]*1e-9-f_extract)))
            Y_ex1f=_peel_para_from_Y(_s_to_y(d["S_raw"],d["z0"]),d["freq"],para_eff)
            ReZ12=float(_y_to_z(Y_ex1f)[idx,0,1].real); c3.markdown(f"**{ReZ12:.4f}**"); c4.markdown(f"<small>{ReZ12-Re_ref:.4f}</small>",unsafe_allow_html=True)
            if Ie>0: points.append((1.0/(Ie*1e-3),ReZ12,Path(fn).stem))
        except Exception as ex: c3.markdown(f"*err:{ex}*")
    if len(points)<2: st.caption("Need ≥ 2 files with IE for fit."); return
    x=np.array([p[0] for p in points]); y=np.array([p[1] for p in points]); lbl=[p[2] for p in points]
    try:
        slope,Re_fit=np.polyfit(x,y,1); eta=slope/(1.381e-23*300/1.602e-19)
        x_fit=np.linspace(0,max(x)*1.08,200); y_fit=slope*x_fit+Re_fit
    except Exception as ex: st.error(f"Fit failed: {ex}"); return
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=x,y=y,mode="markers+text",text=lbl,textposition="top center",name="Re(Z₁₂)",marker=dict(size=11,color="#1f77b4",line=dict(color="#0d4a7a",width=1.5))))
    fig.add_trace(go.Scatter(x=x_fit,y=y_fit,mode="lines",name=f"Fit Re={Re_fit:.4f}Ω η={eta:.3f}",line=dict(color="#d62728",width=2,dash="dash")))
    fig.add_trace(go.Scatter(x=[0],y=[Re_fit],mode="markers",name=f"Re={Re_fit:.4f}Ω",marker=dict(size=14,symbol="star",color="#d62728")))
    fig.update_layout(title=f"Re(Z₁₂) vs 1/IE @ {f_extract:.2f} GHz",xaxis_title="1/IE (A⁻¹)",yaxis_title="Re(Z₁₂) (Ω)",
        xaxis=dict(rangemode="tozero",showgrid=True,gridcolor="#ebebeb"),yaxis=dict(showgrid=True,gridcolor="#ebebeb"),
        plot_bgcolor="white",paper_bgcolor="white",height=380,
        legend=dict(x=0.01,y=0.99,xanchor="left",yanchor="top",bgcolor="rgba(255,255,255,0.9)",bordercolor="#ccc",borderwidth=1,font=dict(size=9)),
        margin=dict(l=55,r=20,t=50,b=50))
    st.plotly_chart(fig,use_container_width=True,key=f"rz12_{fname}")
    current_stem=Path(fname).stem; current_idx=next((i for i,l in enumerate(lbl) if l==current_stem),None)
    if current_idx is not None: st.session_state[f"rz12_Rbe_{fname}"]=y[current_idx]-Re_fit
    else: st.session_state.pop(f"rz12_Rbe_{fname}",None)
    st.session_state[f"rz12_Re_{fname}"]=Re_fit
    mc1,mc2=st.columns(2)
    mc1.metric("Re (intercept)",f"{Re_fit:.4f} Ω",delta=f"{Re_fit-para_eff.get('Rpe',0):+.4f} vs open-short")
    if current_idx is not None: mc2.metric(f"Rbe ({current_stem})",f"{y[current_idx]-Re_fit:.4f} Ω")
    else: mc2.info("Current file not in fit.")

def _render_ft_fmax_overlay(S_raw,S_sim_T,S_sim_pi,S_sim_D,freq,fname):
    f_ghz=freq*1e-9; h21_mea,U_mea=_compute_h21_U(S_raw); fig=go.Figure()
    fig.add_trace(go.Scatter(x=f_ghz,y=h21_mea,mode="lines",name="|h21|² Meas.",line=dict(color="#1f77b4",width=2.5)))
    fig.add_trace(go.Scatter(x=f_ghz,y=U_mea,mode="lines",name="Mason U Meas.",line=dict(color="#1f77b4",width=2.5,dash="dash")))
    for S_sim,label,col in [(S_sim_T,"T","#d62728"),(S_sim_pi,"π","#2ca02c"),(S_sim_D,"Degachi","#9467bd")]:
        if S_sim is not None:
            h21_s,U_s=_compute_h21_U(S_sim)
            fig.add_trace(go.Scatter(x=f_ghz,y=h21_s,mode="lines",name=f"|h21|² {label}",line=dict(color=col,width=2.0,dash="dash")))
            fig.add_trace(go.Scatter(x=f_ghz,y=U_s,mode="lines",name=f"Mason U {label}",line=dict(color=col,width=2.0,dash="longdash")))
    fig.add_hline(y=0,line_color="#333",line_width=1.2,annotation_text="0 dB",annotation_position="right",annotation_font=dict(size=9))
    fig.update_layout(title="Gain vs Frequency — Measured vs Modeled",
        xaxis=dict(title="Frequency (GHz)",type="log",showgrid=True,gridcolor="#ebebeb"),
        yaxis=dict(title="Gain (dB)",range=[0,50],showgrid=True,gridcolor="#ebebeb"),
        plot_bgcolor="white",paper_bgcolor="white",height=450,
        legend=dict(x=1.01,y=1.0,xanchor="left",yanchor="top",bgcolor="rgba(255,255,255,0.92)",bordercolor="#ccc",borderwidth=1,font=dict(size=9)),
        hovermode="x unified",margin=dict(l=55,r=20,t=50,b=50))
    st.plotly_chart(fig,use_container_width=True,key=f"ftfmax_{fname}")
    st.caption("Y-axis fixed 0–50 dB.")

# ─── SECTION H4 ── UNIFIED PRE-EXTRACTION OVERRIDE ───────────────────────────
def _render_unified_pre_override(fname,para_step1,cold_res,rz12_Re):
    st.markdown("---"); st.markdown("### ⚙️ Pre-Extraction Parameter Review")
    st.caption("All pad parameters feeding every model extraction. For Rb/Rc/Re, highest source is pre-selected.")
    para_eff=para_step1.copy()
    src_Rb={"Short (Step 1b)":para_step1.get("Rpb",0.0)}
    src_Rc={"Short (Step 1b)":para_step1.get("Rpc",0.0)}
    src_Re={"Short (Step 1b)":para_step1.get("Rpe",0.0)}
    if cold_res is not None:
        src_Rb["Cold-HBT"]=float(cold_res.get("Rb_cold",0.0))
        src_Rc["Cold-HBT"]=float(cold_res.get("Rc_cold",0.0))
        src_Re["Cold-HBT"]=float(cold_res.get("Re_cold",0.0))
    if rz12_Re is not None: src_Re["Z-parameter method"]=float(rz12_Re)
    def _best(d): return max(d,key=lambda k:d[k])
    def _src_lbl(k,v): return f"{k}: {v:.4f} Ω"
    _cap_keys=["Cpbe","Cpce","Cpbc"]; _ind_keys=["Lb","Lc","Le"]
    for k in _cap_keys:
        sk=f"preov_{k}_{fname}"
        if sk not in st.session_state: st.session_state[sk]=para_step1.get(k,0.0)*1e15
    for k in _ind_keys:
        sk=f"preov_{k}_{fname}"
        if sk not in st.session_state: st.session_state[sk]=para_step1.get(k,0.0)*1e12
    for var,sources in [("Rb",src_Rb),("Rc",src_Rc),("Re",src_Re)]:
        sk_src=f"preov_src_{var}_{fname}"; sk_val=f"preov_{var}_{fname}"
        if sk_src not in st.session_state: st.session_state[sk_src]=_best(sources)
        if sk_val not in st.session_state: st.session_state[sk_val]=sources.get(st.session_state[sk_src],list(sources.values())[0])
    with st.expander("✏️ Inspect / override pad parameters",expanded=True):
        if st.button("↩️ Reset all to defaults (highest source)",key=f"preov_reset_{fname}"):
            for k in _cap_keys: st.session_state[f"preov_{k}_{fname}"]=para_step1.get(k,0.0)*1e15
            for k in _ind_keys: st.session_state[f"preov_{k}_{fname}"]=para_step1.get(k,0.0)*1e12
            for var,sources in [("Rb",src_Rb),("Rc",src_Rc),("Re",src_Re)]:
                bk=_best(sources); st.session_state[f"preov_src_{var}_{fname}"]=bk; st.session_state[f"preov_{var}_{fname}"]=sources[bk]
            st.rerun()
        st.markdown("**Pad capacitances** *(from Open, single source)*")
        cap_cols=st.columns(3)
        for col_w,(k,lbl) in zip(cap_cols,[("Cpbe","Cpbe (fF)"),("Cpce","Cpce (fF)"),("Cpbc","Cpbc (fF)")]):
            col_w.number_input(lbl,key=f"preov_{k}_{fname}",format="%.4f",step=0.1)
        st.markdown("**Lead inductances** *(from Short, single source)*")
        ind_cols=st.columns(3)
        for col_w,(k,lbl) in zip(ind_cols,[("Lb","Lb (pH)"),("Lc","Lc (pH)"),("Le","Le (pH)")]):
            col_w.number_input(lbl,key=f"preov_{k}_{fname}",format="%.3f",step=0.1)
        st.markdown("**Series resistances** *(choose source — default = highest)*")
        for var,sources,label in [("Rb",src_Rb,"**Rb = Rpb** — base"),("Rc",src_Rc,"**Rc = Rpc** — collector"),("Re",src_Re,"**Re = Rpe** — emitter")]:
            src_opts=list(sources.keys())+["Custom"]; sk_src=f"preov_src_{var}_{fname}"; sk_val=f"preov_{var}_{fname}"
            cur_src=st.session_state.get(sk_src,_best(sources))
            if cur_src not in src_opts: cur_src=src_opts[0]
            radio_lbls=[_src_lbl(k,v) for k,v in sources.items()]+["Custom"]
            cur_idx=src_opts.index(cur_src); st.markdown(label)
            sc1,sc2=st.columns([3,1])
            chosen_lbl=sc1.radio("",radio_lbls,index=cur_idx,key=f"preov_radio_{var}_{fname}",horizontal=True,label_visibility="collapsed")
            chosen_src=src_opts[radio_lbls.index(chosen_lbl)]; st.session_state[sk_src]=chosen_src
            if chosen_src!="Custom":
                resolved=sources[chosen_src]; st.session_state[sk_val]=resolved; sc2.metric(f"{var}",f"{resolved:.4f} Ω")
            else: sc2.number_input(f"{var} (Ω)",key=sk_val,format="%.4f",step=0.01)
    for k in _cap_keys: para_eff[k]=st.session_state.get(f"preov_{k}_{fname}",para_step1.get(k,0.0)*1e15)/1e15
    for k in _ind_keys: para_eff[k]=st.session_state.get(f"preov_{k}_{fname}",para_step1.get(k,0.0)*1e12)/1e12
    para_eff["Rpb"]=st.session_state.get(f"preov_Rb_{fname}",para_step1.get("Rpb",0.0))
    para_eff["Rpc"]=st.session_state.get(f"preov_Rc_{fname}",para_step1.get("Rpc",0.0))
    para_eff["Rpe"]=st.session_state.get(f"preov_Re_{fname}",para_step1.get("Rpe",0.0))
    return para_eff

# ─── SECTION H5 ── PER-TOPOLOGY OVERRIDE + SMITH CHART ───────────────────────
def _params_hash(p:dict)->str:
    try: return hashlib.md5(json.dumps({k:round(float(v),15) for k,v in p.items()},sort_keys=True).encode()).hexdigest()
    except: return ""

_PAD_SPECS=[("Cpbe","Cpbe",1e15,"fF","%.4f",0.1),("Cpce","Cpce",1e15,"fF","%.4f",0.1),("Cpbc","Cpbc",1e15,"fF","%.4f",0.01),
            ("Lb","Lb",1e12,"pH","%.3f",0.1),("Lc","Lc",1e12,"pH","%.3f",0.1),("Le","Le",1e12,"pH","%.3f",0.01),
            ("Rpb","Rb",1.0,"Ω","%.4f",0.01),("Rpc","Rc",1.0,"Ω","%.4f",0.01),("Rpe","Re",1.0,"Ω","%.4f",0.01)]
_EXT_SPECS=[("Cbex","Cbex",1e15,"fF","%.4f",0.1),("Cbcx","Cbcx",1e15,"fF","%.4f",0.1)]
_INT_T_SPECS=[("Rbi","Rbi",1.0,"Ω","%.4f",0.1),("Rbe","Rbe",1.0,"Ω","%.3f",1.0),("Cbe","Cbe",1e15,"fF","%.4f",0.1),
              ("Rbc","Rbc",1e-3,"kΩ","%.4f",0.01),("Cbc","Cbc",1e15,"fF","%.4f",0.01),
              ("alpha0","α₀",1.0,"","%.5f",0.001),("tauB","τB",1e12,"ps","%.4f",0.01),("tauC","τC",1e12,"ps","%.4f",0.01)]
_INT_PI_SPECS=[("Rbi","Rbi",1.0,"Ω","%.4f",0.1),("Rbe","Rbe",1.0,"Ω","%.3f",1.0),("Cbe","Cbe",1e15,"fF","%.4f",0.1),
               ("Rbc","Rbc",1e-3,"kΩ","%.4f",0.01),("Cbc","Cbc",1e15,"fF","%.4f",0.01),
               ("Gm0","Gm0",1e3,"mS","%.4f",0.01),("tau","τ",1e12,"ps","%.4f",0.01)]
_INT_D_SPECS=[("Rbi","Rbi",1.0,"Ω","%.4f",0.1),("Cbi","Cbi",1e15,"fF","%.4f",0.1),
              ("Rbe","Rbe",1.0,"Ω","%.3f",1.0),("Cbe","Cbe",1e15,"fF","%.4f",0.1),
              ("Rbc","Rbc",1e-3,"kΩ","%.4f",0.01),("Cbc","Cbc",1e15,"fF","%.4f",0.01),
              ("Rcx","Rcx",1e-3,"kΩ","%.2f",10.0),("Ccx","Ccx",1e15,"fF","%.4f",0.1),
              ("Gm0","Gm0",1e3,"mS","%.4f",0.01),("tau","τ",1e12,"ps","%.4f",0.01)]
_PAD_KEYS=[k for k,*_ in _PAD_SPECS]

def _sync_pad_from_preov(fname,tK,para_eff):
    preov_hash=_params_hash({k:para_eff.get(k,0.0) for k in _PAD_KEYS})
    sync_key=f"smith_pad_synced_{tK}_{fname}"
    if st.session_state.get(sync_key)!=preov_hash:
        for key,_,scale,*_ in _PAD_SPECS: st.session_state[f"sim_{tK}_{key}_{fname}"]=float(para_eff.get(key,0.0))*scale
        st.session_state[sync_key]=preov_hash

def _render_topo_override_and_smith(fname,topo_char,S_raw,freq,z0,para_eff,res_ext,res_int):
    if res_ext is None or res_int is None: return None
    topo_name={"T":"T-topology (Cheng)","pi":"π-topology (Cheng)"}[topo_char]; tK=topo_char
    calc_vals={**para_eff,**res_ext,**res_int}
    int_specs=_INT_T_SPECS if topo_char=="T" else _INT_PI_SPECS
    all_specs=_PAD_SPECS+_EXT_SPECS+int_specs
    _sync_pad_from_preov(fname,tK,para_eff)
    for key,_,scale,*_ in _EXT_SPECS+int_specs:
        sk=f"sim_{tK}_{key}_{fname}"
        if sk not in st.session_state: st.session_state[sk]=float(calc_vals.get(key,0.0))*scale
    with st.expander(f"✏️ Fine-tune {topo_name} intrinsic/extrinsic parameters",expanded=False):
        if st.button(f"↩️ Reset {topo_name} to calculated",key=f"rst_sim_{tK}_{fname}"):
            for key,_,scale,*_ in all_specs: st.session_state[f"sim_{tK}_{key}_{fname}"]=float(calc_vals.get(key,0.0))*scale
            st.rerun()
        st.markdown("**Pad Parasitics** *(auto-synced from pre-extraction override)*")
        for row_start in range(0,len(_PAD_SPECS),3):
            row=_PAD_SPECS[row_start:row_start+3]; cols=st.columns(len(row))
            for col_w,(key,lbl,sc,unit,fmt,step) in zip(cols,row):
                col_w.number_input(f"{lbl} ({unit})" if unit else lbl,key=f"sim_{tK}_{key}_{fname}",format=fmt,step=step)
        st.markdown("**Extrinsic Caps**"); cols=st.columns(2)
        for col_w,(key,lbl,sc,unit,fmt,step) in zip(cols,_EXT_SPECS):
            col_w.number_input(f"{lbl} ({unit})",key=f"sim_{tK}_{key}_{fname}",format=fmt,step=step)
        st.markdown("**Intrinsic**")
        for row_start in range(0,len(int_specs),4):
            row=int_specs[row_start:row_start+4]; cols=st.columns(len(row))
            for col_w,(key,lbl,sc,unit,fmt,step) in zip(cols,row):
                col_w.number_input(f"{lbl} ({unit})" if unit else lbl,key=f"sim_{tK}_{key}_{fname}",format=fmt,step=step)
    all_p={key:st.session_state.get(f"sim_{tK}_{key}_{fname}",float(calc_vals.get(key,0.0))*scale)/scale for key,_,scale,*_ in all_specs}
    # carry extended open/short params into sim
    for ek in ["Cpbe_mode","Cpbe_extra","Cpce_mode","Cpce_extra","Cpbc_mode","Cpbc_extra","Cpar_Lb","Cpar_Lc","Cpar_Le"]:
        all_p[ek]=para_eff.get(ek,"None" if "mode" in ek else 0.0)
    cache_p_key=f"sim_result_{tK}_{fname}"; cache_hash_key=f"sim_phash_{tK}_{fname}"
    current_hash=_params_hash({k:str(v) for k,v in {**all_p,"__nf":len(freq)}.items()})
    if st.session_state.get(cache_hash_key)!=current_hash:
        with st.spinner(f"Simulating {topo_name}…"):
            S_sim=ssm_simulate_T(all_p,freq,z0) if topo_char=="T" else ssm_simulate_pi(all_p,freq,z0)
        st.session_state[cache_p_key]=S_sim; st.session_state[cache_hash_key]=current_hash
    else:
        S_sim=st.session_state.get(cache_p_key)
        if S_sim is None or S_sim.shape[0]!=len(freq):
            with st.spinner(f"Simulating {topo_name}…"):
                S_sim=ssm_simulate_T(all_p,freq,z0) if topo_char=="T" else ssm_simulate_pi(all_p,freq,z0)
            st.session_state[cache_p_key]=S_sim; st.session_state[cache_hash_key]=current_hash
    sc=_smith_scale_controls(fname,tK); err=ssm_residual(S_raw,S_sim)
    st.plotly_chart(_make_ssm_smith(S_raw,S_sim,topo_name,err,sc),use_container_width=True,key=f"smith_{tK}_{fname}")
    return S_sim

def _render_degachi_override_and_smith(fname,S_raw,freq,z0,para_eff,res_int_D):
    if res_int_D is None: return None
    topo_name="Degachi (2008)"; tK="D"; calc_vals={**para_eff,**res_int_D}; all_specs=_PAD_SPECS+_INT_D_SPECS
    _sync_pad_from_preov(fname,tK,para_eff)
    for key,_,scale,*_ in _INT_D_SPECS:
        sk=f"sim_{tK}_{key}_{fname}"
        if sk not in st.session_state: st.session_state[sk]=float(calc_vals.get(key,0.0))*scale
    with st.expander(f"✏️ Fine-tune {topo_name} intrinsic parameters",expanded=False):
        if st.button("↩️ Reset Degachi to calculated",key=f"rst_sim_{tK}_{fname}"):
            for key,_,scale,*_ in all_specs: st.session_state[f"sim_{tK}_{key}_{fname}"]=float(calc_vals.get(key,0.0))*scale
            st.rerun()
        st.markdown("**Pad Parasitics** *(auto-synced)*")
        for row_start in range(0,len(_PAD_SPECS),3):
            row=_PAD_SPECS[row_start:row_start+3]; cols=st.columns(len(row))
            for col_w,(key,lbl,sc,unit,fmt,step) in zip(cols,row):
                col_w.number_input(f"{lbl} ({unit})" if unit else lbl,key=f"sim_{tK}_{key}_{fname}",format=fmt,step=step)
        st.markdown("**Intrinsic (Degachi)**")
        for row_start in range(0,len(_INT_D_SPECS),4):
            row=_INT_D_SPECS[row_start:row_start+4]; cols=st.columns(len(row))
            for col_w,(key,lbl,sc,unit,fmt,step) in zip(cols,row):
                col_w.number_input(f"{lbl} ({unit})" if unit else lbl,key=f"sim_{tK}_{key}_{fname}",format=fmt,step=step)
    all_p={key:st.session_state.get(f"sim_{tK}_{key}_{fname}",float(calc_vals.get(key,0.0))*scale)/scale for key,_,scale,*_ in all_specs}
    for ek in ["Cpbe_mode","Cpbe_extra","Cpce_mode","Cpce_extra","Cpbc_mode","Cpbc_extra","Cpar_Lb","Cpar_Lc","Cpar_Le"]:
        all_p[ek]=para_eff.get(ek,"None" if "mode" in ek else 0.0)
    cache_p_key=f"sim_result_{tK}_{fname}"; cache_hash_key=f"sim_phash_{tK}_{fname}"
    current_hash=_params_hash({k:str(v) for k,v in {**all_p,"__nf":len(freq)}.items()})
    if st.session_state.get(cache_hash_key)!=current_hash:
        with st.spinner(f"Simulating {topo_name}…"):
            S_sim=ssm_simulate_degachi(all_p,freq,z0)
        st.session_state[cache_p_key]=S_sim; st.session_state[cache_hash_key]=current_hash
    else:
        S_sim=st.session_state.get(cache_p_key)
        if S_sim is None or S_sim.shape[0]!=len(freq):
            with st.spinner(f"Simulating {topo_name}…"):
                S_sim=ssm_simulate_degachi(all_p,freq,z0)
            st.session_state[cache_p_key]=S_sim; st.session_state[cache_hash_key]=current_hash
    sc=_smith_scale_controls(fname,tK); err=ssm_residual(S_raw,S_sim)
    st.plotly_chart(_make_ssm_smith(S_raw,S_sim,topo_name,err,sc),use_container_width=True,key=f"smith_{tK}_{fname}")
    return S_sim


# ─── SECTION I ── MAIN ENTRY POINT ───────────────────────────────────────────
def render_ssm_tab_cheng(fname,S_raw,freq,z0,open_data,short_data,all_data=None):
    if open_data is None or short_data is None:
        st.warning("⚠️ SSM Extraction requires Device Open & Short dummy files."); return
    try:
        _strict_freq_check(freq,open_data[0],"Device Open")
        _strict_freq_check(freq,short_data[0],"Device Short")
    except ValueError as e:
        st.error(f"Frequency grid mismatch: {e}"); return

    original_points=len(freq); freq_original=freq.copy()
    col_info,col_dec=st.columns([2,1]); col_info.markdown(f"**Total data points:** {original_points}")
    decimate_factor=col_dec.selectbox("Decimate by:",options=[1,2,4,8,16,32],index=0,key=f"decimate_{fname}")
    if decimate_factor>1:
        freq=freq[::decimate_factor]; S_raw=S_raw[::decimate_factor]
        f_open,S_open,z0_open=open_data; f_short,S_short,z0_short=short_data
        open_data=(f_open[::decimate_factor],S_open[::decimate_factor],z0_open)
        short_data=(f_short[::decimate_factor],S_short[::decimate_factor],z0_short)
        st.success(f"✓ Using {len(freq)} points (every {decimate_factor}th from {original_points})")

    st.markdown("---"); st.markdown("## 🔬 Small-Signal Model (SSM) Parameter Extraction")
    with st.expander("ℹ️ Model notation",expanded=False):
        st.markdown("Rb=Rpb, Rc=Rpc, Re=Rpe — same pad resistance, different naming contexts.  \n"
                    "Open extra elements (Parallel L / Series L / Series R) affect de-embedding and all forward simulations.")

    col_nl,_=st.columns([1,3])
    n_low=col_nl.slider("Low-freq pts",3,40,10,key=f"nlow_{fname}",help="Points used for median extraction.")

    # ── Step 1a ───────────────────────────────────────────────────────────────
    st.markdown("---"); st.markdown("### 📌 Step 1a — Open Dummy: Pad Capacitances")
    st.caption("Gao [3] §4.2. Bias-independent.")
    st.image(image="tools/SSM/de_embedding_illus.png")
    c1,c2,c3=st.columns(3)
    with c1: st.latex(r"C_{pbe}=\mathrm{Im}(Y_{11}^{open}+Y_{12}^{open})/\omega")
    with c2: st.latex(r"C_{pce}=\mathrm{Im}(Y_{22}^{open}+Y_{12}^{open})/\omega")
    with c3: st.latex(r"C_{pbc}=-\mathrm{Im}(Y_{12}^{open})/\omega")

    para_open_calc,open_arr=ssm_step1a_open(open_data)
    st.dataframe(pd.DataFrame([{"Parameter":k,"Value":f"{para_open_calc[k]*1e15:.4f}","Unit":"fF","Description":d}
        for k,d in [("Cpbe","Pad B-E shunt cap"),("Cpce","Pad C-E shunt cap"),("Cpbc","Pad B-C shunt cap")]]),use_container_width=True,hide_index=True)
    _OPEN_OV=[("Cpbe","Cpbe",1e15,"fF","%.4f",0.01),("Cpce","Cpce",1e15,"fF","%.4f",0.01),("Cpbc","Cpbc",1e15,"fF","%.4f",0.01)]
    for dk,_,sc,_,_,_ in _OPEN_OV:
        sk=f"ov_{dk}_{fname}"
        if sk not in st.session_state: st.session_state[sk]=para_open_calc[dk]*sc
    with st.expander("✏️ Override Open Capacitances",expanded=False):
        if st.button("↩️ Reset Caps",key=f"rst_caps_{fname}"):
            for dk,_,sc,_,_,_ in _OPEN_OV: st.session_state[f"ov_{dk}_{fname}"]=para_open_calc[dk]*sc
            st.rerun()
        ov_cols=st.columns(3)
        for col_w,(dk,lbl,sc,unit,fmt,step) in zip(ov_cols,_OPEN_OV):
            col_w.number_input(f"{lbl} ({unit})",key=f"ov_{dk}_{fname}",format=fmt,step=step)
    para_caps_ov={dk:st.session_state[f"ov_{dk}_{fname}"]/sc for dk,_,sc,_,_,_ in _OPEN_OV}

    # Render enhanced open plots (returns mode+extra per cap)
    open_mode_extra=_render_open_plots(open_data,para_caps_ov,open_arr,fname)
    # Store modes/extras into para_caps_ov for propagation
    for cap,(mode,extra) in open_mode_extra.items():
        para_caps_ov[f"{cap}_mode"]=mode; para_caps_ov[f"{cap}_extra"]=extra

    # ── Step 1b ───────────────────────────────────────────────────────────────
    st.markdown("---"); st.markdown("### 📌 Step 1b — Short Dummy: Lead Inductances & Series Resistances")
    st.caption("Gao [3] §4.2.")
    col_m2,_=st.columns([1,1])
    open_sel=col_m2.radio("Use open from:",["measured","modeled"],horizontal=True,key=f"osl_{fname}")
    do_measured=(open_sel=="measured")
    c1,c2,c3=st.columns(3)
    with c1: st.latex(r"R_e=\mathrm{Re}(Z_{12}^{corr})")
    with c2: st.latex(r"R_b=\mathrm{Re}(Z_{11}^{corr}-Z_{12}^{corr})")
    with c3: st.latex(r"R_c=\mathrm{Re}(Z_{22}^{corr}-Z_{21}^{corr})")

    para_short_calc,short_arr=ssm_step1b_short(
        short_data,open_data[0],para_caps_ov["Cpbe"],para_caps_ov["Cpce"],para_caps_ov["Cpbc"],
        open_data,measured=do_measured,
        Cpbe_mode=para_caps_ov.get("Cpbe_mode","None"),Cpbe_extra=para_caps_ov.get("Cpbe_extra",0.0),
        Cpce_mode=para_caps_ov.get("Cpce_mode","None"),Cpce_extra=para_caps_ov.get("Cpce_extra",0.0),
        Cpbc_mode=para_caps_ov.get("Cpbc_mode","None"),Cpbc_extra=para_caps_ov.get("Cpbc_extra",0.0))
    for w in short_arr.get("warnings",[]):
        if w.startswith("⚠️"): st.warning(w)
        else: st.info(w)
    st.dataframe(pd.DataFrame([{"Parameter":lbl,"Value":f"{para_short_calc[dk]*sc:.4f}","Unit":unit,"Description":desc}
        for dk,lbl,sc,unit,desc in [("Lb","Lb",1e12,"pH","Base lead L"),("Lc","Lc",1e12,"pH","Collector lead L"),
            ("Le","Le",1e12,"pH","Emitter lead L"),("Rpb","Rb (Short)",1.0,"Ω",""),("Rpc","Rc (Short)",1.0,"Ω",""),("Rpe","Re (Short)",1.0,"Ω","")]]),
        use_container_width=True,hide_index=True)

    _SHORT_OV=[("Lb","Lb",1e12,"pH","%.3f",0.1),("Lc","Lc",1e12,"pH","%.3f",0.1),("Le","Le",1e12,"pH","%.3f",0.1),
               ("Rpb","Rb",1.0,"Ω","%.4f",0.01),("Rpc","Rc",1.0,"Ω","%.4f",0.01),("Rpe","Re",1.0,"Ω","%.4f",0.01)]
    cap_hash=(round(para_caps_ov["Cpbe"]*1e18),round(para_caps_ov["Cpce"]*1e18),round(para_caps_ov["Cpbc"]*1e18))
    prev_hash_key=f"ov_cap_hash_{fname}"; cap_changed=st.session_state.get(prev_hash_key)!=cap_hash
    for dk,_,sc,_,_,_ in _SHORT_OV:
        sk=f"ov_{dk}_{fname}"
        if sk not in st.session_state or cap_changed: st.session_state[sk]=para_short_calc[dk]*sc
    st.session_state[prev_hash_key]=cap_hash
    with st.expander("✏️ Override Short Lead Values",expanded=False):
        if st.button("↩️ Reset Short",key=f"rst_short_{fname}"):
            for dk,_,sc,_,_,_ in _SHORT_OV: st.session_state[f"ov_{dk}_{fname}"]=para_short_calc[dk]*sc
            st.rerun()
        for row_params in [_SHORT_OV[:3],_SHORT_OV[3:]]:
            row_cols=st.columns(3)
            for col_w,(dk,lbl,sc,unit,fmt,step) in zip(row_cols,row_params):
                col_w.number_input(f"{lbl} ({unit})",key=f"ov_{dk}_{fname}",format=fmt,step=step)
    para_short_ov={dk:st.session_state[f"ov_{dk}_{fname}"]/sc for dk,_,sc,_,_,_ in _SHORT_OV}

    # Render enhanced short plots (returns Cpar values)
    short_cpar=_render_short_plots(short_arr,para_short_ov,fname)
    para_short_ov.update(short_cpar)  # adds Cpar_Lb, Cpar_Lc, Cpar_Le

    para_step1={**para_caps_ov,**para_short_ov}; cold_res=None

    # ── Z Parameter Method ────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("📈 Z Parameter Method — Extract Rbe and Re  *(Gao [3] Ch. 5.5.1)*",expanded=False):
        _render_rz12_section(all_data or {},para_step1,fname)
    rz12_Re=st.session_state.get(f"rz12_Re_{fname}")
    rz12_Rbe=st.session_state.get(f"rz12_Rbe_{fname}")

    # ── Cold-HBT ──────────────────────────────────────────────────────────────
    st.markdown("---"); st.markdown("#### 🧊 Cold-HBT Extraction  *(Gao [3] Ch. 5.5.2)*")
    st.caption("Upload cut-off bias (Vce=0, Vbe≤0) S2P.")
    with st.expander("📐 Cold-HBT Formulas",expanded=False):
        st.latex(r"[Z_{cor}]=[Y_{cold}-Y_{open}]^{-1}")
        st.latex(r"D=\frac{AB+\sqrt{A^2B^2+4ABC^2}}{2C^2},\;C_{ex}=-\frac{(C/B)^2}{\omega A[(1+1/D)^2+(C/B)^2]}")
    cold_file=st.file_uploader("Cold HBT S2P",type=["s2p"],key=f"cold_upload_{fname}")
    if cold_file is not None:
        try:
            f_c_raw,S_c_raw,z0_c=_parse_s2p_bytes(cold_file.getvalue()); f_o,S_o,z0_o=open_data
            if np.allclose(f_c_raw,freq_original,rtol=1e-4):
                f_c_use=f_c_raw[::decimate_factor] if decimate_factor>1 else f_c_raw
                S_c_use=S_c_raw[::decimate_factor] if decimate_factor>1 else S_c_raw
            elif not np.allclose(f_c_raw,f_o,rtol=1e-4):
                f_c_use=f_o; S_c_use=_interpolate_s2f(f_c_raw,S_c_raw,f_o); st.info("Cold S2P interpolated.")
            else: f_c_use=f_c_raw; S_c_use=S_c_raw
            omega_c=2.0*np.pi*f_o; N_c=len(f_o); Y_cold=_s_to_y(S_c_use,z0_c)
            if do_measured: Y_open_eff=_s_to_y(S_o,z0_o)
            else:
                Y_open_eff=np.zeros((N_c,2,2),dtype=complex)
                for i,w in enumerate(omega_c):
                    Ypbe=_open_elem_Y_sc(para_caps_ov["Cpbe"],para_caps_ov.get("Cpbe_mode","None"),para_caps_ov.get("Cpbe_extra",0.0),w)
                    Ypce=_open_elem_Y_sc(para_caps_ov["Cpce"],para_caps_ov.get("Cpce_mode","None"),para_caps_ov.get("Cpce_extra",0.0),w)
                    Ypbc=_open_elem_Y_sc(para_caps_ov["Cpbc"],para_caps_ov.get("Cpbc_mode","None"),para_caps_ov.get("Cpbc_extra",0.0),w)
                    Y_open_eff[i]=np.array([[Ypbe+Ypbc,-Ypbc],[-Ypbc,Ypce+Ypbc]])
            Z_cor=_y_to_z(Y_cold-Y_open_eff)
            A=np.imag(Z_cor[:,0,0]-Z_cor[:,0,1]); B=np.imag(Z_cor[:,1,1]-Z_cor[:,0,1]); C=np.real(Z_cor[:,0,1])
            with np.errstate(divide="ignore",invalid="ignore"):
                disc=A**2*B**2+4.0*A*B*C**2
                D_arr=np.where(np.abs(C)>1e-30,(A*B+np.sqrt(np.maximum(disc,0.0)))/(2.0*C**2),np.nan)
                Cex_arr=np.where(np.isfinite(D_arr),-((C/B)**2)/(omega_c*A*((1.0+1.0/D_arr)**2+(C/B)**2)),np.nan)
                CbcCex_arr=np.where(np.isfinite(D_arr),-1.0/(omega_c*B*(1.0+A**2/(C**2*D_arr**2))),np.nan)
                Cbc_arr=CbcCex_arr-Cex_arr
                Rbi_arr=np.where(np.abs(omega_c*Cex_arr)>1e-40,-D_arr/(omega_c*Cex_arr),np.nan)
                num_cbe=Rbi_arr*Cex_arr; den_cbe=Cex_arr+Cbc_arr+1j*omega_c*Rbi_arr*Cbc_arr*Cex_arr
                Cbe_arr=np.where(np.abs(den_cbe)>1e-40,1.0/(omega_c*np.imag(Z_cor[:,0,1]-num_cbe/den_cbe)),np.nan)
                Zex_arr=np.where(np.abs(Cex_arr)>1e-40,1.0/(1j*omega_c*Cex_arr),np.nan+0j)
                Zbc_z=np.where(np.abs(Cbc_arr)>1e-40,1.0/(1j*omega_c*Cbc_arr),np.nan+0j)
                Zbe_arr=np.where(np.abs(Cbe_arr)>1e-40,1.0/(1j*omega_c*Cbe_arr),np.nan+0j)
                denom_back=Zbc_z+Zex_arr+Rbi_arr
                Zb_arr=(Z_cor[:,0,0]-Z_cor[:,0,1])-np.where(np.abs(denom_back)>1e-40,Zex_arr*Rbi_arr/denom_back,np.nan+0j)
                Rb_arr=np.real(Zb_arr)
                Zc_arr=(Z_cor[:,1,1]-Z_cor[:,0,1])-np.where(np.abs(denom_back)>1e-40,Zbc_z*Zex_arr/denom_back,np.nan+0j)
                Rc_arr=np.real(Zc_arr)
                Ze_arr=Z_cor[:,0,1]-Zbe_arr-np.where(np.abs(denom_back)>1e-40,Zbc_z*Rbi_arr/denom_back,np.nan+0j)
                Re_arr=np.real(Ze_arr)
            n0,n1=N_c//4,3*N_c//4
            def _med(arr): return _safe_median(arr[n0:n1])
            Cex_cold=_med(Cex_arr); Cbc_cold=_med(Cbc_arr); Rbi_cold=_med(Rbi_arr); Cbe_cold=_med(Cbe_arr)
            Rb_cold=_med(Rb_arr); Rc_cold=_med(Rc_arr); Re_cold=_med(Re_arr)
            st.dataframe(pd.DataFrame([("Cex",f"{Cex_cold*1e15:.4f}","fF"),("Cbc",f"{Cbc_cold*1e15:.4f}","fF"),
                ("Rbi",f"{Rbi_cold:.4f}","Ω"),("Cbe",f"{Cbe_cold*1e15:.4f}","fF"),
                ("Rb (Cold)",f"{Rb_cold:.4f}","Ω"),("Rc (Cold)",f"{Rc_cold:.4f}","Ω"),("Re (Cold)",f"{Re_cold:.4f}","Ω")],
                columns=["Parameter","Value","Unit"]),use_container_width=True,hide_index=True)
            cold_res=dict(Cex_cold=Cex_cold,Cbc_cold=Cbc_cold,Rbi_cold=Rbi_cold,Cbe_cold=Cbe_cold,
                          Rb_cold=Rb_cold,Rc_cold=Rc_cold,Re_cold=Re_cold)
        except Exception as e: st.error(f"Cold-HBT failed: {e}")

    st.markdown("---"); _render_sparams_comparison(S_raw,freq,z0,para_step1,fname)
    para_eff=_render_unified_pre_override(fname,para_step1,cold_res,rz12_Re)
    # carry open extra params into para_eff
    for cap in ["Cpbe","Cpce","Cpbc"]:
        para_eff[f"{cap}_mode"]=para_caps_ov.get(f"{cap}_mode","None")
        para_eff[f"{cap}_extra"]=para_caps_ov.get(f"{cap}_extra",0.0)
    for ck in ["Cpar_Lb","Cpar_Lc","Cpar_Le"]:
        para_eff[ck]=para_short_ov.get(ck,0.0)

    # ── Model selection ───────────────────────────────────────────────────────
    st.markdown("---"); st.markdown("### 🔘 Model Selection")
    col_mt,col_mpi,col_md=st.columns(3)
    do_T=col_mt.checkbox("T-model (Cheng 2022)",value=True,key=f"sel_T_{fname}")
    do_pi=col_mpi.checkbox("π-model (Cheng 2022)",value=True,key=f"sel_pi_{fname}")
    do_D=col_md.checkbox("Degachi (2008)",value=False,key=f"sel_D_{fname}")
    if not (do_T or do_pi or do_D): st.info("Select at least one model above."); return

    Y_ex1=ssm_peel_parasitics(S_raw,freq,z0,para_eff)
    res_ext_T=extra_T=res_ext_pi=extra_pi=None

    # ── Step 2 ────────────────────────────────────────────────────────────────
    if do_T or do_pi:
        st.markdown("---"); st.markdown("### 📌 Step 2 — Extrinsic Distributed Capacitances (Cheng [1])")
        if do_T and do_pi: col_T_f,col_pi_f=st.columns(2)
        else: col_T_f=col_pi_f=st
        if do_T:
            with col_T_f:
                st.markdown("**T-topology [Eqs. 13, 22]:**")
                st.latex(r"C_{bex}^T=\mathrm{Im}(Y_{11}+Y_{12})/\omega|_{\omega\to0}")
                st.latex(r"C_{bcx}=-(\mathrm{Im}(Y_{ms})\mathrm{Re}(Y_L)-\mathrm{Re}(Y_{ms})\mathrm{Im}(Y_L))/(\omega\cdot\mathrm{denom})")
        if do_pi:
            with col_pi_f:
                st.markdown("**π-topology [Eqs. 26–28]:**")
                st.latex(r"C_{bex}^\pi=(\mathrm{Re}(B)\mathrm{Re}(C)+\mathrm{Im}(B)\mathrm{Im}(C))/(\omega\,\mathrm{Im}(B))")
        with ThreadPoolExecutor(max_workers=2) as _ex:
            futs={}
            if do_T:  futs["T"] =_ex.submit(ssm_step2_extrinsic_T, Y_ex1,freq,n_low)
            if do_pi: futs["pi"]=_ex.submit(ssm_step2_extrinsic_pi,Y_ex1,freq,n_low)
            if "T"  in futs: res_ext_T, extra_T  =futs["T"].result()
            if "pi" in futs: res_ext_pi,extra_pi =futs["pi"].result()
        def _ext_df(res):
            return pd.DataFrame([{"Param":"Cbex","Value":f"{res['Cbex']*1e15:.4f}","Unit":"fF"},{"Param":"Cbcx","Value":f"{res['Cbcx']*1e15:.4f}","Unit":"fF"}])
        if do_T and do_pi:
            c_T,c_pi=st.columns(2)
            if res_ext_T: c_T.dataframe(_ext_df(res_ext_T),use_container_width=True,hide_index=True)
            if res_ext_pi: c_pi.dataframe(_ext_df(res_ext_pi),use_container_width=True,hide_index=True)
        elif do_T and res_ext_T: st.dataframe(_ext_df(res_ext_T),use_container_width=True,hide_index=True)
        elif do_pi and res_ext_pi: st.dataframe(_ext_df(res_ext_pi),use_container_width=True,hide_index=True)

    # ── Step 3 ────────────────────────────────────────────────────────────────
    st.markdown("---"); st.markdown("### 📌 Step 3 — Intrinsic Model Elements")
    res_int_T=extra_iT=res_int_pi=extra_ipi=None; res_int_D=extra_iD=None

    if do_T or do_pi:
        st.markdown("#### Cheng (2022) — Input: Y_ex2")
        if do_T and do_pi: col_T_f3,col_pi_f3=st.columns(2)
        else: col_T_f3=col_pi_f3=st
        if do_T and res_ext_T:
            with col_T_f3:
                st.markdown("**T [Eqs. 16, 29–33]:**")
                st.latex(r"\alpha=(Z_{12}-Z_{21})/(Z_{22}-Z_{21}),\;\alpha_0=|\alpha||_{\omega\to0}")
                st.latex(r"\tau_B=\sqrt{U-1}/\omega,\;\tau_C=-\arctan[V(1-V^2)^{-1/2}]/(2\omega)")
        if do_pi and res_ext_pi:
            with col_pi_f3:
                st.markdown("**π — Zhang et al. [4]:**")
                st.latex(r"g_m=(Z_{12}-Z_{21})/(Z_{bc}Z_{12})\Rightarrow G_{m0}=|g_m|,\;\tau=-\angle g_m/\omega")
        with ThreadPoolExecutor(max_workers=2) as _ex:
            futs3={}
            if do_T  and res_ext_T:  futs3["T"] =_ex.submit(ssm_step3_intrinsic_T, extra_T["Y_ex2"], freq,res_ext_T["Cbcx"], n_low)
            if do_pi and res_ext_pi: futs3["pi"]=_ex.submit(ssm_step3_intrinsic_pi,extra_pi["Y_ex2"],freq,res_ext_pi["Cbcx"],n_low)
            if "T"  in futs3: res_int_T, extra_iT  =futs3["T"].result()
            if "pi" in futs3: res_int_pi,extra_ipi =futs3["pi"].result()
        if rz12_Rbe is not None:
            if res_int_T:  res_int_T["Rbe"]=rz12_Rbe
            if res_int_pi: res_int_pi["Rbe"]=rz12_Rbe
            st.info(f"Rbe overridden from Re(Z₁₂): **{rz12_Rbe:.4f} Ω**")
        def _int_df_T(ri):
            return pd.DataFrame([("Rbi",f"{ri['Rbi']:.4f}","Ω"),("Rbe",f"{ri['Rbe']:.4f}" if ri['Rbe']<1000 else f"{ri['Rbe']*1e-3:.4f}k","Ω"),
                ("Cbe",f"{ri['Cbe']*1e15:.4f}" if ri['Cbe']<1e-12 else f"{ri['Cbe']*1e12:.4f}","fF" if ri['Cbe']<1e-12 else "pF"),
                ("Rbc",f"{ri['Rbc']*1e-3:.4f}","kΩ"),("Cbc",f"{ri['Cbc']*1e15:.4f}","fF"),
                ("α₀",f"{ri['alpha0']:.5f}",""),("τB",f"{ri['tauB']*1e12:.4f}","ps"),("τC",f"{ri['tauC']*1e12:.4f}","ps")],columns=["Symbol","Value","Unit"])
        def _int_df_pi(ri):
            return pd.DataFrame([("Rbi",f"{ri['Rbi']:.4f}","Ω"),("Rbe",f"{ri['Rbe']:.4f}" if ri['Rbe']<1000 else f"{ri['Rbe']*1e-3:.4f}k","Ω"),
                ("Cbe",f"{ri['Cbe']*1e15:.4f}" if ri['Cbe']<1e-12 else f"{ri['Cbe']*1e12:.4f}","fF" if ri['Cbe']<1e-12 else "pF"),
                ("Rbc",f"{ri['Rbc']*1e-3:.4f}","kΩ"),("Cbc",f"{ri['Cbc']*1e15:.4f}","fF"),
                ("Gm0",f"{ri['Gm0']*1e3:.4f}","mS"),("τ",f"{ri['tau']*1e12:.4f}","ps")],columns=["Symbol","Value","Unit"])
        if do_T and do_pi:
            c_T3,c_pi3=st.columns(2)
            if res_int_T:
                with c_T3: st.dataframe(_int_df_T(res_int_T),use_container_width=True,hide_index=True); _render_formula_trace_T()
            if res_int_pi:
                with c_pi3: st.dataframe(_int_df_pi(res_int_pi),use_container_width=True,hide_index=True); _render_formula_trace_pi()
        else:
            if res_int_T: st.dataframe(_int_df_T(res_int_T),use_container_width=True,hide_index=True); _render_formula_trace_T()
            if res_int_pi: st.dataframe(_int_df_pi(res_int_pi),use_container_width=True,hide_index=True); _render_formula_trace_pi()

    if do_D:
        st.markdown("---"); st.markdown("#### Degachi & Ghannouchi (2008) [2] — Input: Y_ex1")
        with st.spinner("Extracting Degachi parameters…"):
            res_int_D,extra_iD=ssm_step3_intrinsic_degachi(Y_ex1,freq,n_low)
        ri=res_int_D
        with st.expander("📊 Degachi — Fbi and F1 diagnostic fits",expanded=True):
            c1d,c2d=st.columns(2); omega2=extra_iD["omega2"]; Fbi=extra_iD["Fbi"]; F1=extra_iD["F1"]
            n_fit=max(4,2*len(freq)//3)
            for col_w,Farr,A,B,title in [(c1d,Fbi,ri["A0"],ri["B0"],"Fbi [Eq.8]"),(c2d,F1,ri["A1"],ri["B1"],"F1 [Eq.19]")]:
                fig_fit=go.Figure(); mask=np.isfinite(Farr)&(np.abs(Farr)<1e15)&(Farr>0)
                fig_fit.add_trace(go.Scatter(x=omega2[mask],y=Farr[mask],mode="markers",name="Meas.",marker=dict(size=5,color="#1f77b4")))
                if A>0 and mask.any():
                    x_fit=np.linspace(0,np.max(omega2[mask])*1.05,200)
                    fig_fit.add_trace(go.Scatter(x=x_fit,y=A+B*x_fit,mode="lines",name=f"Fit {A:.3e}+{B:.3e}·ω²",line=dict(color="#d62728",dash="dash",width=2)))
                fig_fit.update_layout(title=dict(text=title,font=dict(size=12)),xaxis_title="ω²",
                    plot_bgcolor="white",paper_bgcolor="white",height=280,margin=dict(l=55,r=10,t=40,b=42))
                fig_fit.update_xaxes(showgrid=True,gridcolor="#ebebeb"); fig_fit.update_yaxes(showgrid=True,gridcolor="#ebebeb")
                col_w.plotly_chart(fig_fit,use_container_width=True,key=f"dfit_{title[:4]}_{fname}")
        st.dataframe(pd.DataFrame([("Rbi",f"{ri['Rbi']:.4f}","Ω","[Eq.25]"),("Cbi",f"{ri['Cbi']*1e15:.4f}","fF","=Tbi/Rbi"),
            ("Rbe",f"{ri['Rbe']:.4f}" if ri['Rbe']<1000 else f"{ri['Rbe']*1e-3:.4f}k","Ω","[Eq.25]"),
            ("Cbe",f"{ri['Cbe']*1e15:.4f}" if ri['Cbe']<1e-12 else f"{ri['Cbe']*1e12:.4f}","fF" if ri['Cbe']<1e-12 else "pF","=Tbe/Rbe"),
            ("Rbc",f"{ri['Rbc']*1e-3:.4f}","kΩ",""),("Cbc",f"{ri['Cbc']*1e15:.4f}","fF",""),
            ("Rcx",f"{ri['Rcx']*1e-3:.2f}","kΩ","[Eq.26]"),("Ccx",f"{ri['Ccx']*1e15:.4f}","fF",""),
            ("Gm0",f"{ri['Gm0']*1e3:.4f}","mS",""),("τ",f"{ri['tau']*1e12:.4f}","ps","")],
            columns=["Symbol","Value","Unit","Note"]),use_container_width=True,hide_index=True)
        _render_formula_trace_degachi()

    # ── Cold cross-check ──────────────────────────────────────────────────────
    if cold_res is not None and (res_int_T or res_int_pi or res_int_D):
        st.markdown("---"); st.markdown("**Cold-HBT cross-check:**")
        hot_rbi=res_int_T["Rbi"] if res_int_T else (res_int_pi["Rbi"] if res_int_pi else (res_int_D["Rbi"] if res_int_D else None))
        hot_cbe=res_int_T["Cbe"] if res_int_T else (res_int_pi["Cbe"] if res_int_pi else (res_int_D["Cbe"] if res_int_D else None))
        hot_cbc=res_int_T["Cbc"] if res_int_T else (res_int_pi["Cbc"] if res_int_pi else (res_int_D["Cbc"] if res_int_D else None))
        cmp2=[]
        for sym,hot,cv,unit,sc in [("Rbi",hot_rbi,cold_res["Rbi_cold"],"Ω",1.0),("Cbe",hot_cbe,cold_res["Cbe_cold"],"fF",1e15),("Cbc",hot_cbc,cold_res["Cbc_cold"],"fF",1e15)]:
            hs=f"{hot*sc:.4f}" if hot is not None else "—"; cs=f"{cv*sc:.4f}" if cv is not None else "—"
            try: delta=f"{(cv-hot)*sc:+.4f}" if (hot is not None and cv is not None) else "—"
            except: delta="—"
            cmp2.append({"Symbol":sym,"Hot (RF)":hs,"Cold-HBT":cs,"Δ":delta,"Unit":unit})
        st.dataframe(pd.DataFrame(cmp2),use_container_width=True,hide_index=True)

    # ── Schematics ────────────────────────────────────────────────────────────
    st.markdown("---"); st.markdown("### 🔌 Circuit Topology Diagrams")
    topos=[]
    if do_T  and res_int_T:  topos.append(("T", {**para_eff,**res_ext_T, **res_int_T}))
    if do_pi and res_int_pi: topos.append(("pi",{**para_eff,**res_ext_pi,**res_int_pi}))
    if do_D  and res_int_D:  topos.append(("D", {**para_eff,**res_int_D}))
    if not topos: topos.append(("T" if do_T else ("pi" if do_pi else "D"),para_eff))
    for topo,all_p in topos:
        lbl={"T":"T-topology (Cheng 2022)","pi":"π-topology (Cheng 2022)","D":"Degachi (2008) augmented π"}.get(topo,"")
        st.markdown(f"**{lbl}**")
        try:
            fig_s=make_topology_fig(all_p,topo); st.pyplot(fig_s,use_container_width=True); plt.close(fig_s)
        except Exception as e: st.error(f"Schematic error: {e}")

    # ── Smith charts ──────────────────────────────────────────────────────────
    st.markdown("---"); st.markdown("### 📡 Measured vs. Modeled S-Parameters")
    st.caption("Pad params auto-synced from pre-extraction override. Use expanders below to fine-tune intrinsic/extrinsic.")
    S_sim_T=S_sim_pi=S_sim_D=None
    if do_T and res_ext_T and res_int_T:
        st.markdown("#### T-topology (Cheng)")
        S_sim_T=_render_topo_override_and_smith(fname,"T",S_raw,freq,z0,para_eff,res_ext_T,res_int_T)
    if do_pi and res_ext_pi and res_int_pi:
        st.markdown("#### π-topology (Cheng)")
        S_sim_pi=_render_topo_override_and_smith(fname,"pi",S_raw,freq,z0,para_eff,res_ext_pi,res_int_pi)
    if do_D and res_int_D:
        st.markdown("#### Degachi (2008)")
        S_sim_D=_render_degachi_override_and_smith(fname,S_raw,freq,z0,para_eff,res_int_D)

    # ── fT / fmax ─────────────────────────────────────────────────────────────
    if S_sim_T is not None or S_sim_pi is not None or S_sim_D is not None:
        st.markdown("---"); st.markdown("### 📊 fT and fmax — Measured vs Modeled")
        _render_ft_fmax_overlay(S_raw,S_sim_T,S_sim_pi,S_sim_D,freq,fname)

    # ── S2P DOWNLOAD SECTION ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📥 Download Modeled S2P Files")
    st.caption(
        "Forward-simulate Open, Short, and final DUT model with current parameters, "
        "then download as Touchstone .s2p files (dB/angle, 50 Ω) for independent verification.  \n"
        "**Open & Short** use the pad model (caps + extra elements + leads) from Step 1.  \n"
        "**DUT** uses the selected model from the Smith chart override values."
    )

    # Build param description dict for file header comments
    def _para_desc(p):
        d={}
        for k,v in p.items():
            if "mode" in k or "extra" in k or "Cpar" in k:
                if isinstance(v,str): d[k]=v
                elif isinstance(v,float) and v!=0.0: d[k]=f"{v:.6g}"
            elif isinstance(v,(int,float)) and np.isfinite(float(v)):
                d[k]=f"{v:.6g}"
        return d

    col_d1,col_d2,col_d3=st.columns(3)

    # Open S2P
    with col_d1:
        st.markdown("**Modeled Open dummy**")
        S_open_sim=_simulate_open_Sparams(para_eff,freq,z0)
        open_params={}
        for k in ["Cpbe","Cpce","Cpbc"]:
            open_params[k]=f"{para_eff.get(k,0)*1e15:.4f} fF"
            mode=para_eff.get(f"{k}_mode","None")
            if mode!="None":
                extra=para_eff.get(f"{k}_extra",0.0)
                unit="pH" if "L" in mode else "Ω"
                sc=1e12 if "L" in mode else 1.0
                open_params[f"{k}_extra"]=f"{mode}: {extra*sc:.4f} {unit}"
        open_bytes=_write_s2p(freq,S_open_sim,
            title=f"Open dummy — {Path(fname).stem}",params=open_params)
        st.download_button("📥 Open.s2p",data=open_bytes,
            file_name=f"model_open_{Path(fname).stem}.s2p",mime="text/plain",
            key=f"dl_open_{fname}",use_container_width=True)
        st.caption("Y_pad only — no series leads.")

    # Short S2P
    with col_d2:
        st.markdown("**Modeled Short dummy**")
        S_short_sim=_simulate_short_Sparams(para_eff,freq,z0)
        short_params={}
        for k in ["Rpb","Rpc","Rpe"]: short_params[{"Rpb":"Rb","Rpc":"Rc","Rpe":"Re"}[k]]=f"{para_eff.get(k,0):.4f} Ω"
        for k in ["Lb","Lc","Le"]: short_params[k]=f"{para_eff.get(k,0)*1e12:.4f} pH"
        for k in ["Cpar_Lb","Cpar_Lc","Cpar_Le"]:
            v=para_eff.get(k,0.0)
            if v>0: short_params[k]=f"{v*1e15:.4f} fF"
        short_bytes=_write_s2p(freq,S_short_sim,
            title=f"Short dummy — {Path(fname).stem}",params=short_params)
        st.download_button("📥 Short.s2p",data=short_bytes,
            file_name=f"model_short_{Path(fname).stem}.s2p",mime="text/plain",
            key=f"dl_short_{fname}",use_container_width=True)
        st.caption("Y_pad + inv(Z_ser) — DUT terminals shorted.")

    # DUT final model S2P
    with col_d3:
        st.markdown("**Modeled DUT S-parameters**")
        avail={}
        if S_sim_T  is not None: avail["T-topology (Cheng)"]=S_sim_T
        if S_sim_pi is not None: avail["π-topology (Cheng)"]=S_sim_pi
        if S_sim_D  is not None: avail["Degachi (2008)"]=S_sim_D
        if avail:
            chosen_model=st.selectbox("Model to download:",list(avail.keys()),key=f"dl_model_sel_{fname}")
            S_dut_sim=avail[chosen_model]
            dut_params={}
            for k in ["Cpbe","Cpce","Cpbc"]: dut_params[k]=f"{para_eff.get(k,0)*1e15:.4f} fF"
            for k in ["Rpb","Rpc","Rpe"]: dut_params[{"Rpb":"Rb","Rpc":"Rc","Rpe":"Re"}[k]]=f"{para_eff.get(k,0):.4f} Ω"
            for k in ["Lb","Lc","Le"]: dut_params[k]=f"{para_eff.get(k,0)*1e12:.4f} pH"
            dut_bytes=_write_s2p(freq,S_dut_sim,
                title=f"DUT {chosen_model} — {Path(fname).stem}",params=dut_params)
            st.download_button("📥 DUT.s2p",data=dut_bytes,
                file_name=f"model_dut_{Path(fname).stem}.s2p",mime="text/plain",
                key=f"dl_dut_{fname}",use_container_width=True)
            st.caption("Uses Smith chart override values (fine-tuned intrinsic/extrinsic).")
        else:
            st.info("Run at least one model above to enable DUT download.")

    # ── Parameter summary ─────────────────────────────────────────────────────
    st.markdown("---"); st.markdown("### 📋 Complete Parameter Summary")
    sum_rows=[]
    for sym,key,sc,unit in [("Cpbe","Cpbe",1e15,"fF"),("Cpce","Cpce",1e15,"fF"),("Cpbc","Cpbc",1e15,"fF"),
        ("Lb","Lb",1e12,"pH"),("Lc","Lc",1e12,"pH"),("Le","Le",1e12,"pH"),
        ("Rb (=Rpb)","Rpb",1,"Ω"),("Rc (=Rpc)","Rpc",1,"Ω"),("Re (=Rpe)","Rpe",1,"Ω")]:
        sum_rows.append({"Layer":"Pad Parasitics","Symbol":sym,"Value":f"{para_eff[key]*sc:.4f}","Unit":unit})
    # open extra elements
    for cap in ["Cpbe","Cpce","Cpbc"]:
        mode=para_eff.get(f"{cap}_mode","None")
        if mode!="None":
            extra=para_eff.get(f"{cap}_extra",0.0); unit_e="pH" if "L" in mode else "Ω"; sc_e=1e12 if "L" in mode else 1.0
            sum_rows.append({"Layer":"Open Extra","Symbol":f"{cap} {mode}","Value":f"{extra*sc_e:.4f}","Unit":unit_e})
    for cap,ck in [("Lb","Cpar_Lb"),("Lc","Cpar_Lc"),("Le","Cpar_Le")]:
        v=para_eff.get(ck,0.0)
        if v>0: sum_rows.append({"Layer":"Short Extra","Symbol":f"Cpar_{cap}","Value":f"{v*1e15:.4f}","Unit":"fF"})
    if cold_res is not None:
        for sym,key,sc,unit in [("Rb (Cold)","Rb_cold",1,"Ω"),("Rc (Cold)","Rc_cold",1,"Ω"),("Re (Cold)","Re_cold",1,"Ω"),
            ("Rbi","Rbi_cold",1,"Ω"),("Cbe (cold)","Cbe_cold",1e15,"fF"),("Cbc (cold)","Cbc_cold",1e15,"fF"),("Cex","Cex_cold",1e15,"fF")]:
            sum_rows.append({"Layer":"Cold-HBT","Symbol":sym,"Value":f"{cold_res[key]*sc:.4f}","Unit":unit})
    for topo,re,ri in [("T",res_ext_T,res_int_T),("π",res_ext_pi,res_int_pi)]:
        if re:
            sum_rows+=[{"Layer":f"{topo}-Extrinsic","Symbol":"Cbex","Value":f"{re['Cbex']*1e15:.4f}","Unit":"fF"},
                       {"Layer":f"{topo}-Extrinsic","Symbol":"Cbcx","Value":f"{re['Cbcx']*1e15:.4f}","Unit":"fF"}]
        if ri:
            if topo=="T":
                for sym,val,unit in [("Rbi",ri["Rbi"],"Ω"),("Rbe",ri["Rbe"],"Ω"),("Cbe",ri["Cbe"]*1e15,"fF"),
                    ("Rbc",ri["Rbc"]*1e-3,"kΩ"),("Cbc",ri["Cbc"]*1e15,"fF"),("α₀",ri["alpha0"],""),
                    ("τB",ri["tauB"]*1e12,"ps"),("τC",ri["tauC"]*1e12,"ps")]:
                    sum_rows.append({"Layer":"T-Intrinsic","Symbol":sym,"Value":f"{val:.4f}","Unit":unit})
            else:
                for sym,val,unit in [("Rbi",ri["Rbi"],"Ω"),("Rbe",ri["Rbe"],"Ω"),("Cbe",ri["Cbe"]*1e15,"fF"),
                    ("Rbc",ri["Rbc"]*1e-3,"kΩ"),("Cbc",ri["Cbc"]*1e15,"fF"),("Gm0",ri["Gm0"]*1e3,"mS"),("τ",ri["tau"]*1e12,"ps")]:
                    sum_rows.append({"Layer":"π-Intrinsic","Symbol":sym,"Value":f"{val:.4f}","Unit":unit})
    if res_int_D:
        ri=res_int_D
        for sym,val,unit in [("Rbi",ri["Rbi"],"Ω"),("Cbi",ri["Cbi"]*1e15,"fF"),("Rbe",ri["Rbe"],"Ω"),
            ("Cbe",ri["Cbe"]*1e15,"fF"),("Rbc",ri["Rbc"]*1e-3,"kΩ"),("Cbc",ri["Cbc"]*1e15,"fF"),
            ("Rcx",ri["Rcx"]*1e-3,"kΩ"),("Ccx",ri["Ccx"]*1e15,"fF"),("Gm0",ri["Gm0"]*1e3,"mS"),("τ",ri["tau"]*1e12,"ps")]:
            sum_rows.append({"Layer":"Degachi-Intrinsic","Symbol":sym,"Value":f"{val:.4f}","Unit":unit})
    if sum_rows:
        df_sum=pd.DataFrame(sum_rows); st.dataframe(df_sum,use_container_width=True,hide_index=True)
        buf=io.BytesIO(); df_sum.to_csv(buf,index=False)
        st.download_button("📥 Download SSM parameters (CSV)",data=buf.getvalue(),
            file_name=f"SSM_{Path(fname).stem}.csv",mime="text/csv",key=f"dl_ssm_{fname}")
"""
IOED RF Analyzer  v4.2 (build 16)
─────────────────────────────────────────────────────────────────────────────
Changes vs v3.0 (build 13):
  • Z-matrix contact resistance de-embedding applied GLOBALLY at import time
    — All Bode plots, metrics, π-model, S2P exports use de-embedded data
    — User inputs Rb, Rc, Re (or uses auto-estimated values from high-freq Z)
    — Mason U now valid across full frequency range after de-embedding
  • fT/fmax extraction: replaced manual extrap window with
    single-pole transfer function fitting (-20 dB/dec, -10 dB/dec auto-detect)
  • S2P/CSV export outputs de-embedded data
"""
import io, re, zipfile
from pathlib import Path
from datetime import datetime
try:
    from scipy.optimize import least_squares as _scipy_lsq
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="IOED RF Analyzer", layout="wide")
st.title("📡 IOED RF Analyzer  v4.2")
st.caption(
    "Atlas TCAD CSV / RF CSV / Touchstone S2P · "
    "Global Z-matrix contact R de-embedding · "
    "Kumar (2014) IEEE TCAD π-model · Single-pole fT/fmax fitting · CSV↔S2P"
)

PALETTE = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
           "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]

# ══════════════════════════════════════════════════════════════
# BUILT-IN REFERENCE DATA: UIUC Measurement (Lucas Yang dissertation)
# ══════════════════════════════════════════════════════════════
_UIUC_H21 = np.array([
    [1.0813,28.0078],[1.3296,28.0466],[1.5976,27.8912],[1.8233,27.9689],
    [2.1663,28.0855],[2.3477,28.0855],[2.6484,28.0466],[2.8052,28.1243],
    [3.0053,28.1243],[3.2195,28.0855],[3.4694,28.1632],[3.9138,28.1243],
    [4.4406,28.0855],[4.7299,28.0466],[5.1842,27.8912],[5.6505,27.8912],
    [5.8493,28.0078],[6.1587,27.8912],[6.5600,27.8524],[7.0675,27.7358],
    [7.5276,27.6582],[8.0173,27.5416],[8.6873,27.4251],[8.9399,27.3862],
    [9.5211,27.2309],[10.2572,27.0755],[10.9245,26.9589],[11.9045,26.7647],
    [12.3917,26.6870],[13.4248,26.4151],[14.3792,26.2209],[15.2250,25.9878],
    [16.0275,25.7159],[16.9679,25.3663],[18.0694,25.1332],[19.1339,24.9778],
    [19.9116,24.6670],[21.0855,24.5505],[22.1940,24.1620],[23.3648,23.9290],
    [25.1633,23.5017],[26.7956,23.2297],[28.6969,22.9190],[30.9058,22.4917],
    [32.7163,22.0644],[34.6329,21.6371],[37.2955,21.1321],[40.1610,20.5882],
    [42.7547,20.0832],[45.7760,19.5394],[49.2930,18.9956],
])  # freq_GHz, |h21|²_dB

_UIUC_U = np.array([
    [1.0726,33.8282],[1.2335,33.9529],[1.3319,33.9379],[1.3262,33.2747],
    [1.4084,33.9311],[1.5845,33.2118],[1.7446,33.8152],[1.8643,33.6787],
    [2.0242,33.7641],[2.1327,33.6289],[2.3778,33.6584],[2.5666,33.8413],
    [2.7746,33.8040],[2.9983,33.7604],[3.2301,33.4738],[3.4373,33.2927],
    [3.6349,33.3629],[3.8812,33.3251],[4.2053,33.4659],[4.5448,33.5088],
    [4.8970,33.1744],[5.3288,33.1370],[5.6946,33.0412],[5.9792,32.8025],
    [6.4159,32.6282],[6.6575,32.4982],[7.0622,32.5625],[7.6189,32.4109],
    [8.3190,32.5319],[8.9593,32.4326],[9.2934,31.7437],[9.4791,32.5137],
    [9.9694,31.7566],[10.5382,31.2750],[10.4910,32.0942],[11.3465,31.0680],
    [12.0154,30.5759],[12.6856,30.1535],[13.4782,29.9536],[13.9469,29.4445],
    [14.7168,29.0147],[15.8632,28.7539],[16.3148,27.9847],[17.1524,28.6601],
    [17.4985,27.8220],[18.7294,28.1042],[19.2364,27.5213],[20.0507,27.1562],
    [21.4087,27.1835],[21.3981,26.7499],[22.9520,26.6121],[24.0670,26.2824],
    [25.4975,26.0830],[26.9678,25.7285],[27.7269,24.9281],[29.1074,24.6505],
    [29.8720,25.0153],[31.3630,24.5786],[32.0297,23.8514],[32.9274,23.5734],
    [35.1865,23.6803],[34.9822,23.2407],[36.7753,24.8317],[37.3423,24.3779],
    [38.3068,25.5633],[38.8642,23.7076],[39.1063,23.0387],[39.6284,22.4315],
    [39.1685,24.4212],[39.9888,21.8986],[41.8204,21.4048],[42.6895,20.7469],
    [43.3626,22.2137],[44.7572,21.3459],[45.3394,20.6397],[45.1594,22.2396],
    [47.1378,20.0711],[48.3881,18.2046],[49.0082,17.9931],[49.9556,17.4459],
])  # freq_GHz, Mason_U_dB

def _dk(c):
    try:
        h=c.lstrip("#"); r,g,b=int(h[0:2],16),int(h[2:4],16),int(h[4:6],16)
        return f"#{max(0,r-45):02x}{max(0,g-45):02x}{max(0,b-45):02x}"
    except:
        return c


def _batch_inv(M):
    out=np.zeros_like(M)
    for i in range(len(M)):
        try:
            out[i]=np.linalg.inv(M[i])
        except:
            out[i]=np.full((2,2),np.nan+0j)
    return out

def _y_to_z(Y):
    return _batch_inv(Y)

def _s_to_h(S):
    s11,s12,s21,s22=S[:,0,0],S[:,0,1],S[:,1,0],S[:,1,1]
    den=((1-s11)*(1+s22)+s12*s21)
    H=np.zeros_like(S)
    H[:,0,0]=((1+s11)*(1+s22)-s12*s21)/(den+1e-30)*50.0
    H[:,0,1]=(2*s12)/(den+1e-30)
    H[:,1,0]=(-2*s21)/(den+1e-30)
    H[:,1,1]=((1-s11)*(1-s22)-s12*s21)/(den+1e-30)/50.0
    return H

def _ri_to_complex(a,b,fmt):
    if fmt=='ri': return a+1j*b
    if fmt=='ma': return a*np.exp(1j*np.deg2rad(b))
    if fmt=='db': return 10**(a/20.0)*np.exp(1j*np.deg2rad(b))
    raise ValueError(f"Unsupported s2p format: {fmt}")

def parse_s2p(file_obj):
    lines=file_obj.getvalue().decode('utf-8', errors='ignore').splitlines()
    f_unit='hz'; s_fmt='ma'; z0=50.0; vals=[]
    for line in lines:
        s=line.strip()
        if not s or s.startswith('!'):
            continue
        if s.startswith('#'):
            parts=s[1:].lower().split()
            for i,p in enumerate(parts):
                if p in ('hz','khz','mhz','ghz'): f_unit=p
                elif p in ('ma','ri','db'): s_fmt=p
                elif p=='r' and i+1<len(parts):
                    try: z0=float(parts[i+1])
                    except: pass
            continue
        vals.extend(s.split())
    vals=[float(x) for x in vals]
    n=len(vals)//9
    if n<=0:
        raise ValueError('S2P 檔案沒有完整的 2-port 資料列。')
    vals=np.array(vals[:n*9],dtype=float).reshape(n,9)
    scale={'hz':1.0,'khz':1e3,'mhz':1e6,'ghz':1e9}[f_unit]
    freq_hz=vals[:,0]*scale
    S=np.zeros((n,2,2),dtype=complex)
    S[:,0,0]=_ri_to_complex(vals[:,1],vals[:,2],s_fmt)
    S[:,1,0]=_ri_to_complex(vals[:,3],vals[:,4],s_fmt)
    S[:,0,1]=_ri_to_complex(vals[:,5],vals[:,6],s_fmt)
    S[:,1,1]=_ri_to_complex(vals[:,7],vals[:,8],s_fmt)
    Y=_s_to_y(S,z0)
    H=_s_to_h(S)
    stem,vce_fn,ib_fn=parse_bias_from_filename(getattr(file_obj,'name','uploaded.s2p'))
    # Z for s2p: compute from Y (no direct Z columns)
    Z_s2p = _batch_inv(Y)
    _avail_s2p = {'S':True,'Y':True,'Z':True,'H':True,'Z_direct':False,'source':'s2p'}
    return None,freq_hz,S,Y,Z_s2p,{},vce_fn,ib_fn,_avail_s2p,z0,H

def parse_any_rf(file_obj):
    name=getattr(file_obj,'name','uploaded').lower()
    if name.endswith('.s2p'):
        return parse_s2p(file_obj)   # now returns 11-tuple including Z
    df,freq_hz,S,Y,Z,meta,vce,ib=parse_csv_rf(file_obj)
    H=_s_to_h(S)
    avail={
        'S': all(c in df.columns for c in ['Real(S11)','Imag(S11)','Real(S12)','Imag(S12)','Real(S21)','Imag(S21)','Real(S22)','Imag(S22)']),
        'Y': all(c in df.columns for c in ['Real(Y11)','Imag(Y11)','Real(Y12)','Imag(Y12)','Real(Y21)','Imag(Y21)','Real(Y22)','Imag(Y22)']),
        'Z': all(c in df.columns for c in ['Real(Z11)','Imag(Z11)','Real(Z12)','Imag(Z12)','Real(Z21)','Imag(Z21)','Real(Z22)','Imag(Z22)']),
        'H': all(c in df.columns for c in ['Real(H11)','Imag(H11)','Real(H12)','Imag(H12)','Real(H21)','Imag(H21)','Real(H22)','Imag(H22)']),
        'source': 'csv'
    }
    # Z_direct: True if Z columns came directly from Atlas (not computed from S)
    avail['Z_direct'] = avail['Z']
    return df,freq_hz,S,Y,Z,meta,vce,ib,avail,50.0,H

def export_s2p_bytes(freq_hz,S,z0=50.0):
    lines=['! Exported by IOED_RF_Analyzer', f'# Hz S RI R {float(z0)}']
    for f,sm in zip(freq_hz,S):
        vals=[f,sm[0,0].real,sm[0,0].imag,sm[1,0].real,sm[1,0].imag,sm[0,1].real,sm[0,1].imag,sm[1,1].real,sm[1,1].imag]
        lines.append(' '.join(f'{v:.12e}' for v in vals))
    return ('\n'.join(lines)+'\n').encode('utf-8')

def export_csv_bytes(freq_hz,S,Y,H,meta=None):
    df=pd.DataFrame({'Frequency':freq_hz})
    for prefix,M in [('S',S),('Y',Y),('H',H)]:
        if M is None: continue
        for (r,c),nm in zip([(0,0),(0,1),(1,0),(1,1)],[f'{prefix}11',f'{prefix}12',f'{prefix}21',f'{prefix}22']):
            df[f'Real({nm})']=M[:,r,c].real
            df[f'Imag({nm})']=M[:,r,c].imag
    if meta:
        for k,v in meta.items():
            df[k]=v
    return df.to_csv(index=False).encode('utf-8')

# ══════════════════════════════════════════════════════════════
# PARSING
# ══════════════════════════════════════════════════════════════
def parse_bias_from_filename(name):
    stem=Path(name).stem; vce=None; ib=None
    m=re.search(r'[Vv][Cc][Ee][_\-]?([\d]+(?:p\d+)?)\s*[Vv]',stem)
    if m: vce=float(m.group(1).replace("p","."))
    m=re.search(r'[Ii][Bb][_\-]?([\d]+(?:p\d+)?)\s*([pnuUmM]?)[Aa]?',stem)
    if m:
        sc={'p':1e-12,'n':1e-9,'u':1e-6,'U':1e-6,'m':1e-3,'M':1e-3,'':1.}
        ib=float(m.group(1).replace("p","."))*sc.get(m.group(2),1.)
    return stem,vce,ib

def _read_param_block(df, prefix):
    """Read 2×2 complex matrix columns 'Real(P11)' etc. from DataFrame."""
    keys = [f'{prefix}11',f'{prefix}12',f'{prefix}21',f'{prefix}22']
    cols_r = [f'Real({k})' for k in keys]
    cols_i = [f'Imag({k})' for k in keys]
    if not all(c in df.columns for c in cols_r+cols_i):
        return None
    M = np.zeros((len(df),2,2), dtype=complex)
    for (r,c),k in zip([(0,0),(0,1),(1,0),(1,1)], keys):
        M[:,r,c] = (pd.to_numeric(df[f'Real({k})'],errors='coerce').values
                    +1j*pd.to_numeric(df[f'Imag({k})'],errors='coerce').values)
    return M

def parse_csv_rf(file_obj):
    """
    Parse Atlas TCAD or VNA CSV files.
    Priority for source matrices:
      1. Z-params (Real/Imag Z11/Z12/Z21/Z22) — Atlas direct Z output
      2. S-params (Real/Imag S11..S22)         — VNA or Atlas S output
      3. Y-params (Real/Imag Y11..Y22)          — Atlas direct Y output
    At least one of S or Z must be present.
    Returns Z_direct (or None) in addition to existing return values.
    """
    df=pd.read_csv(file_obj)
    df.columns=[str(c).strip().strip('"') for c in df.columns]
    if "Frequency" not in df.columns:
        raise ValueError("CSV 缺少 Frequency 欄位。")
    freq_hz=pd.to_numeric(df["Frequency"],errors="coerce").to_numpy(float)

    S = _read_param_block(df, 'S')
    Y = _read_param_block(df, 'Y')
    Z = _read_param_block(df, 'Z')

    if S is None and Z is None:
        raise ValueError(
            "CSV 缺少完整 S 或 Z 欄位。\n"
            "需要 Real(S11)/Imag(S11)…Real(S22)/Imag(S22) "
            "或 Real(Z11)/Imag(Z11)…Real(Z22)/Imag(Z22)。"
        )

    # Derive missing matrices
    if S is None:     S = _y_to_s(_batch_inv(Z), 50.)   # Z → Y → S
    if Y is None:     Y = _s_to_y(S, 50.)               # S → Y
    if Z is None:     Z = _batch_inv(Y)                  # Y → Z (numeric)

    meta={}
    for col in ["Emitter Voltage","Emitter Current","Base Voltage",
                "Base Current","Collector Voltage","Collector Current"]:
        if col in df.columns:
            vals=pd.to_numeric(df[col],errors="coerce").dropna()
            meta[col]=float(vals.iloc[0]) if len(vals) else np.nan
    vc=meta.get("Collector Voltage",np.nan); ve=meta.get("Emitter Voltage",0.)
    vce_m=float(vc-ve) if np.isfinite(vc) else None
    ib_m =meta.get("Base Current",None)
    if ib_m is not None and not np.isfinite(ib_m): ib_m=None
    return df,freq_hz,S,Y,Z,meta,vce_m,ib_m

# ══════════════════════════════════════════════════════════════
# MATRIX MATH
# ══════════════════════════════════════════════════════════════
def _s_to_y(S,z0=50.):
    s11,s12,s21,s22=S[:,0,0],S[:,0,1],S[:,1,0],S[:,1,1]
    d=(1+s11)*(1+s22)-s12*s21; Y=np.zeros_like(S)
    Y[:,0,0]=((1-s11)*(1+s22)+s12*s21)/(d*z0)
    Y[:,0,1]=-2.*s12/(d*z0); Y[:,1,0]=-2.*s21/(d*z0)
    Y[:,1,1]=((1+s11)*(1-s22)+s12*s21)/(d*z0)
    return Y

def _y_to_s(Y,z0=50.):
    S=np.zeros_like(Y); I=np.eye(2)
    for i in range(len(Y)):
        yn=Y[i]*z0
        try:    S[i]=np.dot(I-yn,np.linalg.inv(I+yn))
        except: S[i]=np.full((2,2),np.nan+0j)
    return S

# ══════════════════════════════════════════════════════════════
# RF METRICS
# ══════════════════════════════════════════════════════════════
def compute_metrics(Y,freq_hz):
    f=freq_hz*1e-9
    y11,y12,y21,y22=Y[:,0,0],Y[:,0,1],Y[:,1,0],Y[:,1,1]
    with np.errstate(divide='ignore',invalid='ignore'):
        h21=-y21/y11
        num_u=np.abs(y21-y12)**2
        den_u=4.*(y11.real*y22.real-y12.real*y21.real)
        U=np.where(den_u>0,num_u/den_u,np.nan)
        num_k=2.*y11.real*y22.real-(y12*y21).real
        K=num_k/(np.abs(y12*y21)+1e-60)
        MSG=np.abs(y21)/(np.abs(y12)+1e-30)
        MAG=MSG*(K-np.sqrt(np.clip(K**2-1.,0,None)))
        MAG_MSG=np.where(K>1.,MAG,MSG)
    return pd.DataFrame({
        "Freq (GHz)":f,
        "|h21|² (dB)":           10*np.log10(np.abs(h21)**2  +1e-30),
        "Mason U (dB)":           10*np.log10(np.abs(U)        +1e-30),
        "MAG/MSG (dB)":           10*np.log10(np.abs(MAG_MSG) +1e-30),
        "K Factor":               K,
        "fT Plateau (GHz)":       f*np.abs(h21),
        "fmax U Plateau (GHz)":   f*np.sqrt(np.abs(U)),
        "fmax MAG Plateau (GHz)": f*np.sqrt(np.abs(MAG_MSG)),
    })

def _do_extrap(fv, gv, pv, extrap_f_start, extrap_f_end, extrap_n_pts):
    """Legacy extrap — kept for backward compat but no longer primary."""
    em=(fv>=extrap_f_start)&(fv<=extrap_f_end); fe,ge=fv[em],gv[em]
    if len(fe)>extrap_n_pts:
        idx_=np.round(np.linspace(0,len(fe)-1,extrap_n_pts)).astype(int)
        fe,ge=fe[idx_],ge[idx_]
    vp=np.nanmax(pv) if not np.isnan(pv).all() else np.nan
    vex=np.nan
    if len(fe)>=2:
        with np.errstate(all='ignore'):
            m,c=np.polyfit(np.log10(fe),ge,1)
            if m<0: vex=10**(-c/m)
    return vex,vp

def fit_single_pole(freq_ghz, gain_db, target_slope=-20.0, min_rolloff_db=3.0):
    """
    Single-pole transfer function fitting for fT/fmax extraction.

    Finds the frequency region where the gain is rolling off with approximately
    `target_slope` dB/decade, then fits a line in log-frequency vs dB space
    and extrapolates to 0 dB.

    target_slope: -20 for h21/U (single-pole), -10 for MSG
    min_rolloff_db: minimum dB of rolloff needed to attempt fitting

    Returns: (f_cross_ghz, slope_actual, f_line, g_line, label)
    """
    ok = np.isfinite(gain_db) & (gain_db > -60)
    if ok.sum() < 4:
        return np.nan, np.nan, None, None, "No Data"

    fv, gv = freq_ghz[ok], gain_db[ok]

    # Need positive gain region
    pos = gv > 0
    if pos.sum() < 3:
        return np.nan, np.nan, None, None, "No Gain"

    # Check if 0 dB crossing exists
    above = gv >= 0
    crossings = np.where(above[:-1] & ~above[1:])[0]
    genuine_cross = None
    for idx in crossings[::-1]:
        cnt = sum(1 for j in range(idx, -1, -1) if above[j])
        if cnt >= 3:
            genuine_cross = idx
            break

    if genuine_cross is not None:
        # Direct 0 dB crossing — interpolate
        i = genuine_cross
        try:
            vcr = fv[i] + (0 - gv[i]) * (fv[i+1] - fv[i]) / (gv[i+1] - gv[i])
        except:
            vcr = np.nan
        return vcr, np.nan, None, None, "0dB Cross"

    # No crossing — need to extrapolate using single-pole fitting
    # Find the rolloff region: look for segments with slope near target_slope
    total_rolloff = np.nanmax(gv) - np.nanmin(gv)
    if total_rolloff < min_rolloff_db:
        # Not enough rolloff — use plateau estimate
        plat = fv * 10**(gv/20.0)  # GBP
        plat_valid = plat[np.isfinite(plat) & (plat > 0)]
        if len(plat_valid) > 3:
            return np.nan, np.nan, None, None, f"Plateau only (≈{np.median(plat_valid[-5:]):.0f} GHz)"
        return np.nan, np.nan, None, None, "Insufficient rolloff"

    # Use the upper portion of the spectrum where gain is rolling off
    # Find where gain starts decreasing monotonically
    lf = np.log10(fv)
    # Compute local slope using sliding 5-point window
    n = len(lf)
    if n < 5:
        # Too few points — simple linear fit on all positive-gain data
        pos_mask = gv > 0
        if pos_mask.sum() >= 2:
            m, c = np.polyfit(lf[pos_mask], gv[pos_mask], 1)
            if m < 0:
                f_cross = 10**(-c/m)
                f_line = np.logspace(lf[pos_mask][0], np.log10(max(f_cross*1.05, fv[-1])), 200)
                g_line = m * np.log10(f_line) + c
                return f_cross, m, f_line, g_line, f"Fit (slope={m:.1f})"
        return np.nan, np.nan, None, None, "Insufficient data"

    # Compute rolling slope (dB/decade)
    win = min(5, n//2)
    slopes = np.full(n, np.nan)
    for i in range(win, n):
        if lf[i] - lf[i-win] > 0.01:  # at least 0.01 decades span
            slopes[i] = (gv[i] - gv[i-win]) / (lf[i] - lf[i-win])

    # Find region where slope is within ±30% of target_slope
    tol = 0.3
    near_target = np.isfinite(slopes) & (slopes < target_slope * (1 - tol)) & (slopes > target_slope * (1 + tol))

    if near_target.sum() >= 2:
        # Fit in the region with target slope
        fit_mask = near_target & (gv > 0)
    else:
        # Fall back: use the last 60% of positive-gain data (high-freq rolloff)
        pos_idx = np.where(gv > 0)[0]
        if len(pos_idx) < 3:
            return np.nan, np.nan, None, None, "No rolloff region"
        start = pos_idx[len(pos_idx)//3]  # upper 2/3
        fit_mask = np.zeros(n, dtype=bool)
        fit_mask[start:] = True
        fit_mask &= (gv > 0) & np.isfinite(gv)

    if fit_mask.sum() < 2:
        return np.nan, np.nan, None, None, "Cannot fit"

    m, c = np.polyfit(lf[fit_mask], gv[fit_mask], 1)
    if m >= 0:
        return np.nan, np.nan, None, None, "Positive slope"

    # Validate: if actual slope is much shallower than target, the rolloff
    # hasn't truly started yet — use GBP plateau estimate instead
    if abs(m) < abs(target_slope) * 0.3:  # slope < 30% of expected
        plat = fv * 10**(gv/20.0)
        plat_valid = plat[np.isfinite(plat) & (plat > 0)]
        plat_est = float(np.median(plat_valid[-max(3,len(plat_valid)//3):])) if len(plat_valid)>0 else np.nan
        return plat_est, m, None, None, f"GBP plateau ≈{plat_est:.0f} GHz (slope={m:.1f}, need {target_slope:.0f})"

    f_cross = 10**(-c/m)
    f0 = max(fv[fit_mask][0] * 0.9, fv[0])
    f1 = max(f_cross * 1.05, fv[-1])
    f_line = np.logspace(np.log10(f0), np.log10(f1), 300)
    g_line = m * np.log10(f_line) + c
    g_plat = f_line * 10**(g_line/20.0)

    return f_cross, m, f_line, g_line, f"Single-pole fit (slope={m:.1f} dB/dec)"


def fit_single_pole_window(freq_ghz, gain_db, f_max_ghz=50.0, target_slope=-20.0):
    """
    Single-pole fit using only data up to f_max_ghz (e.g. 50 GHz).
    This mimics how literature extracts fT/fmax from VNA data limited to 50 GHz.
    """
    mask = (freq_ghz <= f_max_ghz) & np.isfinite(gain_db) & (gain_db > -60)
    if mask.sum() < 4:
        return np.nan, np.nan, None, None, "No Data in window"
    return fit_single_pole(freq_ghz[mask], gain_db[mask], target_slope=target_slope)


def extract_limit_dual(freq_ghz, gain_db, plateau_arr, f_min, f_max, fit_window_ghz=50.0):
    """
    Dual-mode extraction:
      1. Full-range: use all data → 0dB crossing or full-range single-pole fit
      2. Window: use only data up to fit_window_ghz → single-pole fit (literature-comparable)
    Returns: (full_val, full_method, win_val, win_method, plateau)
    """
    # ── Full range extraction ──
    vm_full = (freq_ghz >= f_min) & (freq_ghz <= f_max) & ~np.isnan(gain_db)
    full_val, full_plat, full_method = np.nan, np.nan, "No Data"
    if np.any(vm_full):
        fv, gv, pv = freq_ghz[vm_full], gain_db[vm_full], plateau_arr[vm_full]
        if np.nanmax(gv) > 0:
            full_plat = np.nanmax(pv) if not np.isnan(pv).all() else np.nan
            fc, sl, _, _, lb = fit_single_pole(fv, gv, target_slope=-20.0)
            full_val = fc
            full_method = lb

    # ── Window extraction (literature-comparable) ──
    vm_win = (freq_ghz >= f_min) & (freq_ghz <= fit_window_ghz) & ~np.isnan(gain_db)
    win_val, win_method = np.nan, "No Data in window"
    if np.any(vm_win):
        fv_w, gv_w = freq_ghz[vm_win], gain_db[vm_win]
        if np.nanmax(gv_w) > 0:
            fc_w, sl_w, _, _, lb_w = fit_single_pole(fv_w, gv_w, target_slope=-20.0)
            win_val = fc_w
            win_method = f"Fit≤{fit_window_ghz:.0f}GHz ({lb_w})"

    return full_val, full_method, win_val, win_method, full_plat


def extract_limit(freq_ghz, gain_db, plateau_arr, f_min, f_max,
                  extrap_f_start=10., extrap_f_end=40., extrap_n_pts=4,
                  force_extrap=False):
    """
    Extract fT/fmax using single-pole fitting (primary) or manual extrap (legacy).
    """
    vm = (freq_ghz >= f_min) & (freq_ghz <= f_max) & ~np.isnan(gain_db)
    if not np.any(vm):
        return np.nan, np.nan, "No Data"
    fv, gv, pv = freq_ghz[vm], gain_db[vm], plateau_arr[vm]
    if np.nanmax(gv) <= 0:
        return np.nan, np.nan, "No Gain"

    # Use single-pole fitting (auto-detect rolloff)
    f_cross, slope, f_line, g_line, label = fit_single_pole(fv, gv, target_slope=-20.0)

    # Plateau estimate from GBP
    vp = np.nanmax(pv) if not np.isnan(pv).all() else np.nan

    return f_cross, vp, label

# ══════════════════════════════════════════════════════════════
# INTRINSIC π-MODEL  (7-element, CORRECTED)
# ══════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════
# Rbb ANALYTICAL EXTRACTION  (Lucas Yang §4.3, Eq. 4.34)
# ══════════════════════════════════════════════════════════════
def extract_rbb_analytical(Y):
    """
    Analytically extract Rbb per frequency point (Lucas Yang Eq. 4.34 /
    Yang et al. IEEE TMT 2007 Eq. 14).

    Derivation: Y11/Y12 is independent of Rbb (Eq. 4.33):
        Y11/Y12 = -(Yπ+Yμ)/Yμ  (purely intrinsic)
    And:  1/Y11 = Rbb + 1/(Yπ+Yμ)
    Therefore:
        Rbb = Re(1/Y11) - Im(1/Y11) · Im(Y11/Y12) / Re(Y11/Y12)

    Valid when Y is measured after de-embedding lead inductances + series
    resistances (RE, RC, RBext), or can be used as estimate on raw Y.
    Returns Rbb(f) array — should be approximately frequency-independent
    for a correct model topology.
    """
    y11, y12 = Y[:,0,0], Y[:,0,1]
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_y11  = np.where(np.abs(y11)>1e-30, 1./y11, np.nan+0j)
        ratio    = np.where(np.abs(y12)>1e-30, y11/y12, np.nan+0j)
        rbb = (inv_y11.real
               - inv_y11.imag * ratio.imag / np.where(np.abs(ratio.real)>1e-30,
                                                        ratio.real, np.nan))
    return rbb   # complex array; physically meaningful part is real


def deembed_rbb(Y, rbb_val):
    """
    De-embed Rbb from measured 2-port Y-matrix.
    Series element at B-terminal only:
        Z_int = Z_meas − [[Rbb, 0], [0, 0]]
        Y_int = inv(Z_int)
    """
    Z = _batch_inv(Y)          # (N,2,2) Z-matrix
    Z[:,0,0] -= rbb_val        # remove Rbb from Z11 only
    return _batch_inv(Z)       # Y_int = inv(Z_int)


def embed_rbb(Y_int, rbb_val):
    """
    Embed Rbb back into intrinsic 2-port Y-matrix (inverse of deembed_rbb).
        Z_full = Z_int + [[Rbb, 0], [0, 0]]
        Y_full = inv(Z_full)
    """
    Z = _batch_inv(Y_int)
    Z[:,0,0] += rbb_val
    return _batch_inv(Z)


# ══════════════════════════════════════════════════════════════
# TCAD / No-Pad Direct De-embedding  (Z-matrix subtraction)
# ══════════════════════════════════════════════════════════════
def extract_extrinsic_R_from_Z(Y, freq_hz, f_low_ghz=2.0, Z_direct=None):
    """
    Analytically estimate extrinsic resistances from low-frequency Z-parameters.

    If Z_direct is provided (Atlas直接輸出的 Z 矩陣), uses it directly.
    Otherwise computes Z = inv(Y) from S→Y conversion.

    At f → 0, reactive terms vanish:
      Re(Z12)       → RE + re  (RE_contact + intrinsic emitter resistance re≈1/gm)
      Re(Z22 − Z12) → RC       (collector contact resistance)
      Re(Z11 − Z12) → RBext + Rbb (total base, split later)

    ⚠️ Re(Z12)|low ≠ pure RE (includes re = 1/gm)
       Caller must subtract re estimate from RE result.
    """
    # Prefer direct Atlas Z to avoid S→Y→Z conversion errors
    Z = Z_direct if Z_direct is not None else _batch_inv(Y)

    f_ghz = freq_hz * 1e-9
    mask = (f_ghz > 0) & (f_ghz <= f_low_ghz)
    if mask.sum() < 1:
        mask = np.ones(len(freq_hz), dtype=bool)

    Z11 = Z[:, 0, 0]; Z12 = Z[:, 0, 1]; Z22 = Z[:, 1, 1]
    return {
        "RE_plus_re": float(np.nanmedian(Z12[mask].real)),        # RE + re
        "RC":         float(np.nanmedian((Z22 - Z12)[mask].real)),
        "RBtot":      float(np.nanmedian((Z11 - Z12)[mask].real)),# RBext + Rbb
        "Z_source":   "direct" if Z_direct is not None else "computed_from_S",
    }


def deembed_extrinsic_Z(Y, RE, RC, RBext, Z_direct=None):
    """
    Remove extrinsic series resistances via Z-matrix subtraction.

    If Z_direct is provided (Atlas直接輸出), uses it directly — cleaner,
    no S→Y→Z rounding. Otherwise computes Z = inv(Y).

    Subtraction matrix (common-emitter, 2-port):
      Z_int = Z_meas − [[RE+RBext,  RE   ],
                         [RE,        RE+RC]]

    Physical meaning:
      RE   = emitter contact resistance (series at both ports via emitter)
      RC   = collector contact resistance (series at port 2)
      RBext= extrinsic base resistance (series at port 1 only)

    Returns Y_int = inv(Z_int).
    """
    Z = (Z_direct.copy() if Z_direct is not None else _batch_inv(Y))
    Z[:, 0, 0] -= (RE + RBext)
    Z[:, 0, 1] -= RE
    Z[:, 1, 0] -= RE
    Z[:, 1, 1] -= (RE + RC)
    return _batch_inv(Z)


# ══════════════════════════════════════════════════════════════
# INTRINSIC π-MODEL  (8-element, Rbb-aware)
# ══════════════════════════════════════════════════════════════
def compute_intrinsic_pi(Y, freq_hz, rbb_deembed=None):
    """
    8-element intrinsic π-model (Rbb-aware version).

    If rbb_deembed is None:
      → Extract directly from raw Y (fast, approximate when Rbb≠0)
      → Also computes Rbb(f) analytically for display/use

    If rbb_deembed is a scalar (Ω):
      → De-embed Rbb first: Z_int = Z_meas − [[Rbb,0],[0,0]]
      → Extract 8 elements from Y_int = inv(Z_int)
      → Extraction formulas now exact (topology matches)

    8-element intrinsic π formulas (applied to Y_int):
      Y12_int = −(Gμ + jωCμ)   →  Cμ=−Im/ω,  Gμ=−Re
      Y11_int+Y12_int = Gπ+jωCπ →  rπ=1/Gπ,   Cπ=Im/ω
      Y22_int+Y12_int = Go+jωCce →  ro=1/Go,   Cce=Im/ω
      Y21_int−Y12_int = gm·e^-jωτ → gm=|·|, τ=−∠/ω

    Also returns Rbb(f) column (analytical, regardless of rbb_deembed).
    """
    w  = 2*np.pi*freq_hz
    ws = np.where(w==0, 1e-30, w)

    # Rbb(f) analytical — always computed from raw Y
    rbb_f = extract_rbb_analytical(Y)

    # Use de-embedded Y if requested
    if rbb_deembed is not None and np.isfinite(float(rbb_deembed)):
        Y_use = deembed_rbb(Y, float(rbb_deembed))
    else:
        Y_use = Y

    y11, y12, y21, y22 = Y_use[:,0,0], Y_use[:,0,1], Y_use[:,1,0], Y_use[:,1,1]
    with np.errstate(divide='ignore', invalid='ignore'):
        Cmu = -np.imag(y12)/ws
        Gmu = -np.real(y12)
        Cpi =  np.imag(y11+y12)/ws
        Gpi =  np.real(y11+y12)
        Go  =  np.real(y22+y12)
        Cce =  np.imag(y22+y12)/ws
        rpi = np.where(np.abs(Gpi)>1e-30, 1./Gpi, np.nan)
        ro  = np.where(np.abs(Go) >1e-30, 1./Go,  np.nan)
        Ygm = y21 - y12
        gm  = np.abs(Ygm)
        tau = -np.angle(Ygm)/ws

    # ── Kumar (2014) Eq.8: R_B = Re(Z11-Z12), Eq.5: R_BC = -1/Re(Y12) ────
    with np.errstate(divide='ignore', invalid='ignore'):
        Rbc = np.where(np.abs(Gmu)>1e-30, 1./Gmu, np.nan)  # R_BC = -1/Re(Y12) (Kumar Eq.5)

    # ── Kumar (2014) Eq.13: fT = gm/(2π·(Cbe+Cbc)) ──────────────────────
    with np.errstate(divide='ignore', invalid='ignore'):
        Ci       = Cpi + Cmu                          # Ci = C_BE + C_BC
        fT_model = np.where(Ci > 1e-30,
                            gm / (2*np.pi*Ci),        # fT = gm/(2π·Ci)
                            np.nan) * 1e-9             # → GHz
        # ── Kumar (2014) Eq.14: fmax = sqrt(fT/(8π·C_BC·R_B)) ────────────
        Rbb_f_use = np.where(rbb_f.real > 0, rbb_f.real, np.nan)
        fmax_model = np.where(
            (Cmu > 1e-30) & np.isfinite(Rbb_f_use) & (fT_model > 0),
            np.sqrt(fT_model * 1e9 / (8*np.pi * Cmu * Rbb_f_use)),
            np.nan) * 1e-9                            # → GHz

    return pd.DataFrame({
        "Freq (GHz)":   freq_hz*1e-9,
        "Rbb (Ohm)":   rbb_f.real,       # R_B  — Kumar Eq.8 (analytical)
        # Kumar (2014) capacitance naming
        "C_BC (fF)":   Cmu*1e15,         # = Cmu  (B-C junction cap, Kumar Eq.3)
        "C_BE (fF)":   Cpi*1e15,         # = Cpi  (B-E junction cap, Kumar Eq.2)
        "C_CE (fF)":   Cce*1e15,         # = Cce  (C-E junction cap, Kumar Eq.4)
        # Kumar (2014) resistance naming
        "R_BC (Ohm)":  Rbc,              # = 1/Gmu (B-C junction res, Kumar Eq.5)
        "R_BE (Ohm)":  rpi,              # = rpi   (B-E junction res, Kumar Eq.7)
        "R_CE (Ohm)":  ro,               # = ro    (C-E junction res, Kumar Eq.6)
        # Transconductance
        "gm (mS)":     gm*1e3,          # Kumar Eq.11
        "Gmu (mS)":    Gmu*1e3,         # = 1/R_BC (conductance)
        "tau (ps)":    tau*1e12,
        # Analytical fT and fmax (Kumar Eq.13, 14)
        "fT_model (GHz)":   fT_model,
        "fmax_model (GHz)": fmax_model,
        # Legacy names (backward compat)
        "Cmu (fF)":    Cmu*1e15,
        "Cpi (fF)":    Cpi*1e15,
        "Cce (fF)":    Cce*1e15,
        "rpi (Ohm)":   rpi,
        "ro (Ohm)":    ro,
    })


def compute_avg(df_pi,f_min,f_max):
    mask=(df_pi["Freq (GHz)"]>=f_min)&(df_pi["Freq (GHz)"]<=f_max)
    sub=df_pi.loc[mask].copy()
    if sub.empty: sub=df_pi.copy()
    avg={}
    # Columns that represent physical quantities with sign constraints
    # Capacitances should be > 0, resistances should be > 0, gm should be > 0
    positive_cols = {"Cmu (fF)", "Cpi (fF)", "Cce (fF)",
                      "C_BC (fF)", "C_BE (fF)", "C_CE (fF)",
                      "rpi (Ohm)", "ro (Ohm)", "R_BE (Ohm)", "R_CE (Ohm)",
                      "gm (mS)", "fT_model (GHz)", "fmax_model (GHz)"}  # Rbb/Gmu/RBC NOT here
    # Gmu, Rbb may have either sign — do NOT force positive
    # Use median for resistance/conductance (prone to extreme outliers)
    median_cols = {"rpi (Ohm)", "ro (Ohm)", "R_BE (Ohm)", "R_CE (Ohm)", "R_BC (Ohm)", "Rbb (Ohm)"}
    for col in sub.columns:
        if col=="Freq (GHz)": continue
        v=pd.to_numeric(sub[col],errors="coerce").replace([np.inf,-np.inf],np.nan).dropna()
        # Filter: keep only physically reasonable values for constrained cols
        if col in positive_cols and len(v)>0:
            v = v[v > 0]
            # Remove outliers beyond 3× IQR
            if len(v) >= 4:
                q1, q3 = v.quantile(0.25), v.quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    v = v[(v >= q1 - 3*iqr) & (v <= q3 + 3*iqr)]
        if col in median_cols:
            avg[col]=float(v.median()) if len(v) else np.nan
        else:
            avg[col]=float(v.mean()) if len(v) else np.nan
    avg["Freq Range (GHz)"]=f"{float(sub['Freq (GHz)'].min()):.3f}–{float(sub['Freq (GHz)'].max()):.3f}"
    avg["Points"]=int(len(sub))
    return avg,sub

def reconstruct_Y(avg, freq_hz, rbb_val=None):
    """
    Rebuild intrinsic π-model Y matrix from averaged parameters.
    Supports both Kumar (2014) naming (C_BC/C_BE/R_BC/R_BE) and
    legacy naming (Cmu/Cpi/rpi/ro) — tries Kumar names first.
    If rbb_val is given (Ω), embed Rbb back via Z-matrix.
    """
    # Accept Kumar naming first, fall back to legacy
    Cmu = avg.get("C_BC (fF)", avg.get("Cmu (fF)", 0)) * 1e-15
    Cpi = avg.get("C_BE (fF)", avg.get("Cpi (fF)", 0)) * 1e-15
    Cce = avg.get("C_CE (fF)", avg.get("Cce (fF)", 0)) * 1e-15
    Gmu = avg.get("Gmu (mS)", 0) * 1e-3
    rpi_raw = avg.get("R_BE (Ohm)", avg.get("rpi (Ohm)", 0))
    ro_raw  = avg.get("R_CE (Ohm)", avg.get("ro (Ohm)", 0))
    rpi_ = rpi_raw; ro_ = ro_raw
    Gpi = 1./float(rpi_) if rpi_ and np.isfinite(float(rpi_)) and float(rpi_)!=0 else 0.
    Go  = 1./float(ro_)  if ro_  and np.isfinite(float(ro_))  and float(ro_) !=0 else 0.
    gm=avg.get("gm (mS)",0)*1e-3; tau=avg.get("tau (ps)",0)*1e-12
    w=2*np.pi*freq_hz; gmc=gm*np.exp(-1j*w*tau)
    Y=np.zeros((len(w),2,2),dtype=complex)
    Y[:,0,0]=Gpi+Gmu+1j*w*(Cpi+Cmu)
    Y[:,0,1]=   -Gmu-1j*w*Cmu
    Y[:,1,0]=gmc-Gmu-1j*w*Cmu
    Y[:,1,1]=Go +Gmu+1j*w*(Cce+Cmu)
    # Embed Rbb back if provided
    if rbb_val is not None and np.isfinite(float(rbb_val)) and float(rbb_val)!=0:
        Y = embed_rbb(Y, float(rbb_val))
    return Y

def reconstruct_Y_perfreq(df_pi, freq_hz, rbb_val=None):
    """
    Per-frequency reconstruction using per-freq extracted params.
    If rbb_val given, embeds Rbb back after building intrinsic Y:
        Z_full = Z_int + [[Rbb,0],[0,0]]  → exact round-trip vs measured Y.
    """
    w  = 2*np.pi*freq_hz

    def _col(name, scale=1.):
        v = pd.to_numeric(df_pi[name], errors="coerce").values * scale
        return np.where(np.isfinite(v), v, 0.)

    # Accept both Kumar naming and legacy naming
    def _col2(k1, k2, scale=1.):
        """Try column k1 first (Kumar), then k2 (legacy)."""
        if k1 in df_pi.columns:
            v = pd.to_numeric(df_pi[k1], errors="coerce").values * scale
        else:
            v = pd.to_numeric(df_pi[k2], errors="coerce").values * scale
        return np.where(np.isfinite(v), v, 0.)

    Cmu = _col2("C_BC (fF)", "Cmu (fF)", 1e-15)
    Cpi = _col2("C_BE (fF)", "Cpi (fF)", 1e-15)
    Cce = _col2("C_CE (fF)", "Cce (fF)", 1e-15)
    Gmu = _col("Gmu (mS)", 1e-3)
    rpi_col = "R_BE (Ohm)" if "R_BE (Ohm)" in df_pi.columns else "rpi (Ohm)"
    ro_col  = "R_CE (Ohm)" if "R_CE (Ohm)" in df_pi.columns else "ro (Ohm)"
    rpi = pd.to_numeric(df_pi[rpi_col], errors="coerce").values
    ro  = pd.to_numeric(df_pi[ro_col],  errors="coerce").values
    Gpi = np.where(np.isfinite(rpi) & (rpi!=0), 1./rpi, 0.)
    Go  = np.where(np.isfinite(ro)  & (ro !=0), 1./ro,  0.)
    gm  = _col("gm (mS)", 1e-3)
    tau = _col("tau (ps)", 1e-12)

    gmc = gm * np.exp(-1j*w*tau)
    Y   = np.zeros((len(w),2,2), dtype=complex)
    Y[:,0,0] = Gpi + Gmu + 1j*w*(Cpi+Cmu)
    Y[:,0,1] =     - Gmu - 1j*w*Cmu
    Y[:,1,0] = gmc - Gmu - 1j*w*Cmu
    Y[:,1,1] = Go  + Gmu + 1j*w*(Cce+Cmu)
    # Embed Rbb back if provided
    if rbb_val is not None and np.isfinite(float(rbb_val)) and float(rbb_val)!=0:
        Y = embed_rbb(Y, float(rbb_val))
    return Y

# ══════════════════════════════════════════════════════════════
# FULL HBT MODEL — Xu et al. (UIUC) 15-element Topology
# ══════════════════════════════════════════════════════════════
#
#  Port1(B_ext) ─[Lb+Rb]─ B' ─[Cbcx‖Rbcx]─────────────────── C' ─[Rc+Lc]─ Port2(C_ext)
#                          │                                    │
#                          ├─[Cbci+Rbci]───────────────────────┤  (intrinsic B-C series RC)
#                          │                                    │
#                          ├─[Rbe+Cje]─ E_int ─────────────────┤  (B-E series RC)
#                          │             │                      │
#                          │           [Le+Ree]          VCCS gm·e^-jωτ · V(B'−E)
#                          │             │               [ro between C' and E]
#                          │            GND                     │
#
#  Node numbering: 0=B_ext, 1=B', 2=C', 3=C_ext, 4=E_int   GND=reference
#
_FM_META = [
    # key,    label,       disp_unit, default_SI,  min_SI,     max_SI,    disp_scale
    ("Lb",   "Lb",        "pH",      1.00e-12,   0.001e-12,  30e-12,    1e12),
    ("Lc",   "Lc",        "pH",      1.00e-12,   0.001e-12,  30e-12,    1e12),
    ("Le",   "Le",        "pH",      1.00e-12,   0.001e-12,  30e-12,    1e12),
    ("Rb",   "Rb_total",  "Ω",       49.1,       0.01,       2000.,     1.),
    ("Rc",   "Rc",        "Ω",       6.4,        0.01,       500.,      1.),
    ("Ree",  "Ree",       "Ω",       16.3,       0.01,       500.,      1.),
    ("Cbcx", "Cbcx",      "fF",      2.00e-15,   0.01e-15,   500e-15,   1e15),
    ("Rbcx", "Rbcx",      "kΩ",      92e3,       100.,       10e6,      1e-3),
    ("Cbci", "Cbci",      "fF",      1.10e-15,   0.01e-15,   100e-15,   1e15),
    ("Rbci", "Rbci",      "kΩ",      146e3,      100.,       10e6,      1e-3),
    ("Cje",  "Cje",       "fF",      11.0e-15,   0.1e-15,    1000e-15,  1e15),
    ("Rbe",  "Rbe",       "Ω",       4.6,        0.001,      2000.,     1.),
    ("gm",   "gm",        "mS",      100e-3,     0.01e-3,    5000e-3,   1e3),
    ("tau",  "τ",         "ps",      0.5e-12,    0.,         20e-12,    1e12),
    ("ro",   "ro",        "Ω",       1000.,      1.,         1e7,       1.),
]
_FM_KEYS = [m[0] for m in _FM_META]

def build_Y2port_full_model(p, freq_hz):
    """
    Build exact 2-port Y matrix for the full Xu-topology HBT model.
    Uses 5-node nodal admittance matrix + Schur complement (Kron reduction).

    Element map (all series RC written as admittance = jωC/(1+jωRC)):
      Yb   = 1/(Rb + jωLb)      nodes 0↔1
      Yc   = 1/(Rc + jωLc)      nodes 2↔3
      Ye   = 1/(Ree + jωLe)     node 4 → GND
      Ybcx = jωCbcx/(1+jωRbcxCbcx)  nodes 1↔2  (parallel RC feedback)
      Ybci = jωCbci/(1+jωRbciCbci)  nodes 1↔2  (series RC B-C intrinsic)
      Ybe  = jωCje /(1+jωRbe Cje )  nodes 1↔4  (series RC B-E junction)
      Yro  = 1/ro                    nodes 2↔4
      VCCS = gm·exp(−jωτ)·V(1−4) → current into node2, out of node4
    """
    w   = 2*np.pi * np.asarray(freq_hz, dtype=float)
    N   = len(w)
    eps = 1e-30

    Lb   = float(p.get('Lb',   1e-12))
    Lc   = float(p.get('Lc',   1e-12))
    Le   = float(p.get('Le',   1e-12))
    Rb   = float(p.get('Rb',   49.1))
    Rc   = float(p.get('Rc',   6.4))
    Ree  = float(p.get('Ree',  16.3))
    Cbcx = float(p.get('Cbcx', 2e-15))
    Rbcx = float(p.get('Rbcx', 92e3))
    Cbci = float(p.get('Cbci', 1.1e-15))
    Rbci = float(p.get('Rbci', 146e3))
    Cje  = float(p.get('Cje',  11e-15))
    Rbe  = float(p.get('Rbe',  4.6))
    gm   = float(p.get('gm',   100e-3))
    tau  = float(p.get('tau',  0.5e-12))
    ro   = max(float(p.get('ro', 1000.)), 1e-3)

    # Admittances (N-length complex arrays)
    Yb   = 1./(Rb   + 1j*w*Lb  + eps)
    Yc   = 1./(Rc   + 1j*w*Lc  + eps)
    Ye   = 1./(Ree  + 1j*w*Le  + eps)
    Ybcx = 1j*w*Cbcx / (1. + 1j*w*Cbcx*Rbcx + eps)  # parallel RC  ‖
    Ybci = 1j*w*Cbci / (1. + 1j*w*Cbci*Rbci + eps)  # series  RC  +
    Ybe  = 1j*w*Cje  / (1. + 1j*w*Cje *Rbe  + eps)  # series  RC  +
    Yro  = (1./ro) * np.ones(N, dtype=complex)
    gmc  = gm * np.exp(-1j*w*tau)

    # 5×5 nodal admittance matrix (batch)
    Y = np.zeros((N, 5, 5), dtype=complex)

    def stmp(i, j, Ye_):   # passive between nodes i and j
        Y[:, i, i] += Ye_
        Y[:, j, j] += Ye_
        Y[:, i, j] -= Ye_
        Y[:, j, i] -= Ye_

    stmp(0, 1, Yb)          # Lb+Rb:  B_ext ↔ B'
    stmp(2, 3, Yc)          # Rc+Lc:  C'   ↔ C_ext
    Y[:, 4, 4] += Ye        # Le+Ree: E_int → GND (single-ended)
    stmp(1, 2, Ybcx)        # extrinsic B-C feedback (Cbcx ‖ Rbcx)
    stmp(1, 2, Ybci)        # intrinsic B-C (Cbci + Rbci series)
    stmp(1, 4, Ybe)         # B-E junction (Cje + Rbe series)
    stmp(2, 4, Yro)         # ro: C' ↔ E

    # VCCS: I = gmc·(V[1]−V[4]) enters node2, leaves node4
    Y[:, 2, 1] += gmc;  Y[:, 2, 4] -= gmc
    Y[:, 4, 1] -= gmc;  Y[:, 4, 4] += gmc

    # Kron reduction — keep ports [0,3], eliminate internal nodes [1,2,4]
    p_idx = np.array([0, 3])
    i_idx = np.array([1, 2, 4])
    Y_PP = Y[:, p_idx[:, None], p_idx[None, :]]   # (N,2,2)
    Y_PI = Y[:, p_idx[:, None], i_idx[None, :]]   # (N,2,3)
    Y_IP = Y[:, i_idx[:, None], p_idx[None, :]]   # (N,3,2)
    Y_II = Y[:, i_idx[:, None], i_idx[None, :]]   # (N,3,3)

    # Small diagonal regularisation for numerical stability
    Y_II = Y_II + np.eye(3)[None, :, :] * 1e-18
    try:
        Y_II_inv = np.linalg.inv(Y_II)
    except np.linalg.LinAlgError:
        return np.full((N, 2, 2), np.nan + 0j)

    return Y_PP - Y_PI @ Y_II_inv @ Y_IP   # Schur complement → (N,2,2)


def fit_full_model(freq_hz, S_meas, p0, lb, ub,
                   fix_extrinsic=False, max_iter=3000, ftol=1e-10, xtol=1e-10):
    """
    Fit full Xu-topology HBT model to measured S-params.

    Parameters
    ----------
    p0, lb, ub : dicts keyed by _FM_KEYS with SI values
    fix_extrinsic : if True, only optimise intrinsic params (Cbcx,Cbci,Rbci,Cje,Rbe,gm,tau,ro)
    Returns (fitted_dict, rms_pct, success_bool)
    """
    if not _SCIPY_OK:
        return None, np.nan, False

    extrinsic_keys = {"Lb", "Lc", "Le", "Rb", "Rc", "Ree", "Rbcx"}
    free_keys = [k for k in _FM_KEYS
                 if not (fix_extrinsic and k in extrinsic_keys)]

    x0  = np.array([p0[k]           for k in free_keys], dtype=float)
    lbv = np.array([max(lb[k], 1e-30 if k!='tau' else 0.) for k in free_keys], dtype=float)
    ubv = np.array([ub[k]           for k in free_keys], dtype=float)

    # Normalize to x0 scale for better convergence (avoid x0==0 for tau)
    x0_safe = np.where(np.abs(x0) > 1e-30, np.abs(x0), 1e-15)

    # Weight residuals by 1/|S| so all four S-params contribute equally
    S_abs   = np.abs(S_meas)
    S_w     = np.where(S_abs < 1e-6, 1.0, 1./S_abs)   # (N,2,2)

    def residual(x_norm):
        x = x_norm * x0_safe
        params = dict(zip(free_keys, x))
        # fixed params keep their p0 values
        for k in _FM_KEYS:
            if k not in params:
                params[k] = p0[k]
        try:
            Y_m = build_Y2port_full_model(params, freq_hz)
            S_m = _y_to_s(Y_m, 50.)
            if np.any(np.isnan(S_m)):
                return np.ones(8 * len(freq_hz)) * 1e4
        except Exception:
            return np.ones(8 * len(freq_hz)) * 1e4
        res = []
        for r, c in [(0,0),(0,1),(1,0),(1,1)]:
            d = (S_m[:,r,c] - S_meas[:,r,c]) * S_w[:,r,c]
            res.append(d.real); res.append(d.imag)
        return np.concatenate(res)

    lb_norm = lbv / x0_safe
    ub_norm = ubv / x0_safe
    x0_norm = np.ones_like(x0)

    result = _scipy_lsq(residual, x0_norm, bounds=(lb_norm, ub_norm),
                        method='trf', max_nfev=max_iter,
                        ftol=ftol, xtol=xtol, gtol=1e-12,
                        x_scale='jac', verbose=0)

    x_fit = result.x * x0_safe
    fitted = dict(zip(_FM_KEYS, [p0[k] for k in _FM_KEYS]))   # start with p0 (covers fixed)
    for k, v in zip(free_keys, x_fit):
        fitted[k] = float(v)

    # Compute RMS error on S-params
    try:
        Y_f = build_Y2port_full_model(fitted, freq_hz)
        S_f = _y_to_s(Y_f, 50.)
        rms = float(np.sqrt(np.mean(np.abs(S_f - S_meas)**2)) /
                    (np.sqrt(np.mean(np.abs(S_meas)**2)) + 1e-30) * 100.)
    except Exception:
        rms = np.nan

    ok = result.success or result.status in [1, 2, 3, 4]
    return fitted, rms, ok
def draw_pi_svg(avg):
    """
    Full HBT small-signal circuit showing ALL elements including
    Rbb, RE, RC in the extrinsic shell around the intrinsic π-model.
    """
    def _fv(k, pr=2):
        x = avg.get(k)
        return f"{float(x):.{pr}f}" if x is not None and np.isfinite(float(x)) else "N/A"

    # Colors
    CW="#2c3e50"     # wire / default
    CB="#2471a3"     # capacitor (blue)
    CG="#1e8449"     # conductance (green)
    CP="#6c3483"     # ro (purple)
    CO="#b9770e"     # VCCS gm (orange)
    CR="#c0392b"     # extrinsic resistors (red)
    CI="#7f7f7f"     # intrinsic boundary (gray)

    W, H = 960, 480
    # Key Y positions
    MY  = 190          # main horizontal rail
    TY  = 88           # top feedback
    FY  = 235          # fork point (below main rail)
    RY  = 355          # bottom rail
    EY  = H - 18       # emitter terminal
    # Key X positions
    Bx  = 14           # B external terminal
    LBx = 55           # Lb inductor region start
    RBx = 145          # Rbb resistor start
    Bpx = 230          # B' internal node
    Cpx = 730          # C' internal node
    RCx = 815          # RC resistor
    LCx = 875          # Lc inductor
    Cx  = 944          # C external terminal
    GX  = 480          # gm source / emitter column
    CPI_X = 120; RPI_X = 330
    RO_X  = 630; CCE_X  = 840

    def L(x1,y1,x2,y2,col=CW,sw=2):
        return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col}" stroke-width="{sw}"/>'
    def D(x,y,r=4.5,col=CW):
        return f'<circle cx="{x}" cy="{y}" r="{r}" fill="{col}"/>'
    def T(x,y,s,col="#222",sz=11,anch="middle",bold=False):
        fw="bold" if bold else "normal"
        return (f'<text x="{x}" y="{y}" font-size="{sz}" font-weight="{fw}" '
                f'text-anchor="{anch}" fill="{col}" font-family="Arial,sans-serif">{s}</text>')
    def hzig(x1,x2,y,col=CR,n=7,a=7):
        pts=[]
        for i in range(n+1):
            t=i/n; o=a*(-1)**i if 0<i<n else 0
            pts.append(f"{x1+t*(x2-x1):.1f},{y+o:.1f}")
        return f'<polyline points="{" ".join(pts)}" fill="none" stroke="{col}" stroke-width="2.5" stroke-linejoin="round"/>'
    def vzig(x,y1,y2,col=CG,n=9,a=8):
        pts=[]
        for i in range(n+1):
            t=i/n; o=a*(-1)**i if 0<i<n else 0
            pts.append(f"{x+o:.1f},{y1+t*(y2-y1):.1f}")
        return f'<polyline points="{" ".join(pts)}" fill="none" stroke="{col}" stroke-width="2.5" stroke-linejoin="round"/>'
    def vcap(x,y1,y2,col=CB,pw=18):
        m=(y1+y2)//2; g=9
        return (L(x,y1,x,m-g)+
                f'<line x1="{x-pw}" y1="{m-g}" x2="{x+pw}" y2="{m-g}" stroke="{col}" stroke-width="3.5"/>'+
                f'<line x1="{x-pw}" y1="{m+g}" x2="{x+pw}" y2="{m+g}" stroke="{col}" stroke-width="3.5"/>'+
                L(x,m+g,x,y2))
    def hcap(y,xl,xr,col=CB,ph=16):
        m=(xl+xr)//2; g=9
        return (L(xl,y,m-g,y)+
                f'<line x1="{m-g}" y1="{y-ph}" x2="{m-g}" y2="{y+ph}" stroke="{col}" stroke-width="3.5"/>'+
                f'<line x1="{m+g}" y1="{y-ph}" x2="{m+g}" y2="{y+ph}" stroke="{col}" stroke-width="3.5"/>'+
                L(m+g,y,xr,y))
    def csrc(xc,y1,y2,col=CO,r=28):
        cy=(y1+y2)//2; a1=cy+12; a2=cy-8
        return (L(xc,y1,xc,cy-r,col,2)+
                f'<circle cx="{xc}" cy="{cy}" r="{r}" fill="white" stroke="{col}" stroke-width="2.5"/>'+
                L(xc,a1,xc,a2+4,col,2)+
                f'<polygon points="{xc},{a2} {xc-6},{a2+12} {xc+6},{a2+12}" fill="{col}"/>'+
                L(xc,cy+r,xc,y2,col,2))
    def hinductor(x1,x2,y,col=CW,n=4):
        """Simple horizontal inductor symbol (humps)"""
        seg=(x2-x1)/n; r=seg/2.2; cx0=x1+seg/2
        s=f'<path d="M {x1},{y}'
        for i in range(n):
            cx=x1+i*seg+seg/2
            s+=f' A {r:.1f},{r:.1f} 0 0 1 {x1+(i+1)*seg:.1f},{y}'
        s+=f'" fill="none" stroke="{col}" stroke-width="2.2"/>'
        return s

    p=[]
    p.append(f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
             f'style="background:#fafafa;border:1px solid #ccc;border-radius:8px;width:100%;'
             f'font-family:Arial,sans-serif;">')

    # ── Title ─────────────────────────────────────────────────────────────
    _rbb_svg = avg.get("Rbb (Ohm)")
    _rbb_str = f"R_B={float(_rbb_svg):.1f}Ω" if _rbb_svg is not None and np.isfinite(float(_rbb_svg)) else ""
    p.append(T(W//2, 22, "HBT π Small-Signal Model  (Kumar 2014 IEEE TCAD)", "#222", 14, bold=True))
    p.append(T(W//2, 38,
               f"Avg: {avg.get('Freq Range (GHz)','N/A')} GHz · {avg.get('Points',0)} pts"
               + (f" · {_rbb_str}" if _rbb_str else ""),
               "#888", 10))

    # ── Intrinsic box ─────────────────────────────────────────────────────
    p.append(f'<rect x="{Bpx-10}" y="60" width="{Cpx-Bpx+20}" height="{RY-50}" '
             f'rx="8" fill="none" stroke="#bbb" stroke-width="1.5" stroke-dasharray="7,4"/>')
    p.append(T(Cpx+4, 74, "Intrinsic", "#bbb", 10, "start"))

    # ── External terminals ────────────────────────────────────────────────
    p.append(f'<circle cx="{Bx}" cy="{MY}" r="6" fill="#1f77b4"/>')
    p.append(T(Bx, MY-12, "B", "#1f77b4", 16, "middle", True))
    p.append(f'<circle cx="{Cx}" cy="{MY}" r="6" fill="#ff7f0e"/>')
    p.append(T(Cx, MY-12, "C", "#ff7f0e", 16, "middle", True))
    p.append(f'<circle cx="{GX}" cy="{EY}" r="6" fill="#2ca02c"/>')
    p.append(T(GX, EY+14, "E", "#2ca02c", 14, "middle", True))

    # ── B branch: B → [Rbb] → B' ─────────────────────────────────────────
    mid_rbb = (Bx + Bpx)//2
    p.append(L(Bx, MY, mid_rbb-28, MY))            # wire before Rbb
    p.append(hzig(mid_rbb-28, mid_rbb+28, MY, CR))  # Rbb zigzag
    p.append(L(mid_rbb+28, MY, Bpx, MY))            # wire after Rbb
    p.append(T(mid_rbb, MY-14, f"Rbb={_fv('Rbb (Ohm)')}Ω", CR, 10, "middle", True))
    p.append(T(mid_rbb, MY+18, "R_B (Kumar Eq.8: Z11-Z12)", CR, 9))

    # ── C branch: C' → [RC] → C ──────────────────────────────────────────
    mid_rc = (Cpx + Cx)//2
    p.append(L(Cpx, MY, mid_rc-26, MY))
    p.append(hzig(mid_rc-26, mid_rc+26, MY, CR))
    p.append(L(mid_rc+26, MY, Cx, MY))
    p.append(T(mid_rc, MY-14, "RC (集極接觸電阻)", CR, 10))
    # Use a placeholder (RC is not stored in avg_ dict - just label it)
    p.append(T(mid_rc, MY-14, "RC (集極接觸電阻)", CR, 10))

    # ── E branch: emitter → [RE] → E terminal ────────────────────────────
    mid_re_y = (RY + EY)//2
    p.append(L(GX, RY, GX, mid_re_y-18))
    p.append(vzig(GX, mid_re_y-18, mid_re_y+18, CR))
    p.append(L(GX, mid_re_y+18, GX, EY))
    p.append(T(GX+38, mid_re_y, "RE (射極接觸電阻)", CR, 9, "start"))

    # ── B' and C' node dots ───────────────────────────────────────────────
    p.append(D(Bpx, MY, 5)); p.append(T(Bpx, MY+22, "B'", "#555", 10))
    p.append(D(Cpx, MY, 5)); p.append(T(Cpx, MY+22, "C'", "#555", 10))

    # ── Cμ feedback (B'→top→C') ───────────────────────────────────────────
    p.append(L(Bpx, MY, Bpx, TY))
    p.append(hcap(TY, Bpx, Cpx))
    p.append(L(Cpx, TY, Cpx, MY))
    _gmu_ms = avg.get("Gmu (mS)")
    _rbc_v  = avg.get("R_BC (Ohm)")
    if _rbc_v is not None and np.isfinite(float(_rbc_v)):
        _rmu_str = f"‖ R_BC={float(_rbc_v):.0f}Ω"
    elif _gmu_ms and np.isfinite(float(_gmu_ms)) and abs(float(_gmu_ms)) > 1e-6:
        _rmu_val = 1./(float(_gmu_ms)*1e-3)
        _rmu_str = f"‖ R_BC={_rmu_val:.0f}Ω" if abs(_rmu_val)<1e6 else ""
    else:
        _rmu_str = ""
    p.append(T((Bpx+Cpx)//2, TY-18, f"C_BC = {_fv('C_BC (fF)')} fF  {_rmu_str}", CB, 11, bold=True))

    # ── B' fork down ──────────────────────────────────────────────────────
    p.append(L(Bpx, MY, Bpx, FY))
    p.append(L(Bpx, FY, CPI_X, FY))
    p.append(L(Bpx, FY, RPI_X, FY))
    lm = (FY+RY)//2
    # Cπ branch
    p.append(vcap(CPI_X, FY, RY))
    p.append(T(CPI_X-22, lm, f"C_BE={_fv('C_BE (fF)')}fF", CB, 10, "end"))
    # rπ branch
    p.append(L(RPI_X, FY, RPI_X, FY+12))
    p.append(vzig(RPI_X, FY+12, RY-12, CG))
    p.append(L(RPI_X, RY-12, RPI_X, RY))
    p.append(T(RPI_X+22, lm, f"R_BE={_fv('R_BE (Ohm)')}Ω", CG, 10, "start"))

    # ── C' fork down ──────────────────────────────────────────────────────
    p.append(L(Cpx, MY, Cpx, FY))
    p.append(L(Cpx, FY, RO_X, FY))
    p.append(L(Cpx, FY, CCE_X, FY))
    rm = (FY+RY)//2
    # ro branch
    p.append(L(RO_X, FY, RO_X, FY+12))
    p.append(vzig(RO_X, FY+12, RY-12, CP))
    p.append(L(RO_X, RY-12, RO_X, RY))
    p.append(T(RO_X-22, rm, f"R_CE={_fv('R_CE (Ohm)')}Ω", CP, 10, "end"))
    # Cce branch
    p.append(vcap(CCE_X, FY, RY))
    p.append(T(CCE_X+22, rm, f"C_CE={_fv('C_CE (fF)')}fF", CB, 10, "start"))

    # ── gm current source ─────────────────────────────────────────────────
    p.append(csrc(GX, MY, RY))
    p.append(T(GX+36, (MY+RY)//2-8, f"gm={_fv('gm (mS)')}mS  (Kumar Eq.11)", CO, 10, "start", True))
    p.append(T(GX+36, (MY+RY)//2+6, f"τ={_fv('tau (ps)')}ps", CO, 9, "start"))

    # ── vbe label ─────────────────────────────────────────────────────────
    p.append(T(Bpx-14, FY+30, "+", "#888", 12, "end"))
    p.append(T(Bpx-14, FY+47, "vbe", "#888", 10, "end"))
    p.append(T(Bpx-14, FY+64, "−", "#888", 12, "end"))

    # ── Bottom rail ───────────────────────────────────────────────────────
    p.append(L(CPI_X, RY, CCE_X, RY))
    for x in [RPI_X, GX, RO_X]: p.append(D(x, RY, 3.5))

    # ── Legend ────────────────────────────────────────────────────────────
    p.append(T(18, H-8,
               "Red: R_B/Rbb/RE/RC = extrinsic  |  Blue: C_BC/C_BE/C_CE  |  Green: R_BE/gm  |  Purple: R_CE  |  fT=gm/2pi/(C_BE+C_BC)  fmax=sqrt(fT/8pi/C_BC/Rbb)",
               "#666", 9, "start"))

    p.append('</svg>')
    return '<div style="overflow-x:auto;">' + ''.join(p) + '</div>'


# ══════════════════════════════════════════════════════════════
# SMITH CHART
# ══════════════════════════════════════════════════════════════
def _sgrid(max_r=3.):
    traces=[]; t=np.linspace(0,2*np.pi,500)
    kw=dict(mode='lines',showlegend=False,hoverinfo='skip')
    for r in np.arange(1.,max_r+.5,1.):
        lw=1.6 if r==1. else .9
        col='rgba(60,60,60,0.85)' if r==1. else 'rgba(170,170,170,0.6)'
        traces.append(go.Scatter(x=np.cos(t)*r,y=np.sin(t)*r,line=dict(color=col,width=lw),**kw))
    traces.append(go.Scatter(x=[-max_r,max_r],y=[0.,0.],
                              line=dict(color='rgba(100,100,100,0.6)',width=.8),**kw))
    gray='rgba(155,155,155,0.5)'
    for r in [0.,.2,.5,1.,2.,5.]:
        cx_=r/(r+1); rad=1./(r+1)
        xc,yc=cx_+rad*np.cos(t),rad*np.sin(t)
        m=np.sqrt(xc**2+yc**2)>max_r; xc[m]=np.nan; yc[m]=np.nan
        traces.append(go.Scatter(x=xc,y=yc,line=dict(color=gray,width=.8),**kw))
    for x in [.2,.5,1.,2.,5.]:
        for sign in [1,-1]:
            xv=sign*x; rx_=abs(1./xv)
            xc,yc=1.+rx_*np.cos(t),(1./xv)+rx_*np.sin(t)
            m=np.sqrt(xc**2+yc**2)>max_r; xc[m]=np.nan; yc[m]=np.nan
            traces.append(go.Scatter(x=xc,y=yc,line=dict(color=gray,width=.8),**kw))
    for lr in [0.,.5,1.,2.,5.]:
        lx_=lr/(lr+1)+1./(lr+1)
        traces.append(go.Scatter(x=[lx_],y=[.01],mode='text',text=[f" {lr}"],
                                  textfont=dict(size=8,color='rgba(100,100,100,0.7)'),
                                  showlegend=False,hoverinfo='skip'))
    return traces

def _slay(title,f_min,f_max,max_r,square_px=980):
    lim=max_r*1.05
    return dict(
        title=dict(text=title,font=dict(size=13)),
        autosize=False, width=square_px, height=square_px,
        xaxis=dict(title="Re(Γ)",range=[-lim,lim],showgrid=False,zeroline=False,
                   scaleanchor="y",scaleratio=1,
                   tickvals=[-3,-2,-1,0,1,2,3],constrain="domain"),
        yaxis=dict(title="Im(Γ)",range=[-lim,lim],showgrid=False,zeroline=False,
                   scaleanchor="x",scaleratio=1,constrain="domain"),
        plot_bgcolor="white",paper_bgcolor="white",
        margin=dict(l=60,r=60,t=55,b=65),hovermode="closest",
        legend=dict(x=1.02,y=1.,xanchor="left",yanchor="top",
                    bgcolor="rgba(255,255,255,0.92)",bordercolor="#ccc",borderwidth=1),
        annotations=[dict(x=.5,y=-.1,xref="paper",yref="paper",showarrow=False,
                          text=f"freq ({f_min*1000:.0f}MHz – {f_max:.1f}GHz)  ·  ●=低頻起始點",
                          font=dict(size=10,color="gray"),align="center")])

def make_smith(S,f_arr,f_min,f_max,title,max_r=2.,
               s11=True,s22=True,s21=True,s12=True,square_px=980,
               s_scales=None):
    if s_scales is None:
        s_scales = {'S11':1.0,'S22':1.0,'S21':1.0,'S12':1.0}
    mask=(f_arr>=f_min)&(f_arr<=f_max); Sp,fp=S[mask].copy(),f_arr[mask]
    fig=go.Figure()
    for tr in _sgrid(max_r): fig.add_trace(tr)
    for key,(r,c),col,dash,show in [
            ('S11',(0,0),'#1f77b4','solid',s11),('S22',(1,1),'#ff7f0e','dash',s22),
            ('S21',(1,0),'#2ca02c','dot',s21),  ('S12',(0,1),'#d62728','dashdot',s12)]:
        if not show: continue
        sc = s_scales.get(key, 1.0)
        sv=Sp[:,r,c].copy() * sc; sv[np.abs(sv)>max_r]=np.nan+1j*np.nan
        hov=[f"f={fv:.3f}GHz<br>Re={rv:.4f}<br>Im={iv:.4f}"
             for fv,rv,iv in zip(fp,sv.real,sv.imag)]
        fig.add_trace(go.Scatter(x=sv.real,y=sv.imag,mode='lines',
                                  line=dict(color=col,width=2.8,dash=dash),
                                  name=key + (f" ×{sc}" if sc!=1.0 else ""),
                                  text=hov,hoverinfo='text'))
        if len(sv)>0 and not np.isnan(sv.real[0]):
            fig.add_trace(go.Scatter(x=[sv.real[0]],y=[sv.imag[0]],mode='markers',
                                      marker=dict(size=6,color=col,
                                                  line=dict(color='white',width=1)),
                                      showlegend=False,hoverinfo='skip'))
    fig.update_layout(**_slay(f"Smith Chart — {title}",f_min,f_max,max_r,square_px))
    return fig

def make_smith_ver(S_meas, S_sim_avg, f_arr, f_min, f_max, title, max_r=2.,
                   S_sim_exact=None, square_px=980,
                   s_scales=None):
    """s_scales: dict like {'S11':1.0, 'S22':1.0, 'S21':1.0, 'S12':1.0} for individual magnification."""
    if s_scales is None:
        s_scales = {'S11':1.0,'S22':1.0,'S21':1.0,'S12':1.0}
    mask=(f_arr>=f_min)&(f_arr<=f_max)
    Sm,fp=S_meas[mask],f_arr[mask]
    Sa=S_sim_avg[mask]
    Se=S_sim_exact[mask] if S_sim_exact is not None else None

    fig=go.Figure()
    for tr in _sgrid(max_r): fig.add_trace(tr)

    for key,(r,c),col in [('S11',(0,0),'#1f77b4'),('S22',(1,1),'#ff7f0e'),
                            ('S21',(1,0),'#2ca02c'),('S12',(0,1),'#d62728')]:
        sc = s_scales.get(key, 1.0)
        # Measured: solid
        sv=Sm[:,r,c].copy() * sc; sv[np.abs(sv)>max_r]=np.nan+1j*np.nan
        fig.add_trace(go.Scatter(x=sv.real,y=sv.imag,mode='lines',
                                  line=dict(color=col,width=3.0,dash='solid'),
                                  name=f"{key} Meas" + (f" ×{sc}" if sc!=1.0 else ""),
                                  text=[f"f={fv:.3f}GHz" for fv in fp],hoverinfo='text'))
        if len(sv)>0 and not np.isnan(sv.real[0]):
            fig.add_trace(go.Scatter(x=[sv.real[0]],y=[sv.imag[0]],mode='markers',
                                      marker=dict(size=7,color=col,
                                                  line=dict(color='white',width=1.5)),
                                      showlegend=False,hoverinfo='skip'))
        # π-exact: dotted (per-frequency reconstruction)
        if Se is not None:
            se=Se[:,r,c].copy() * sc; se[np.abs(se)>max_r]=np.nan+1j*np.nan
            fig.add_trace(go.Scatter(x=se.real,y=se.imag,mode='lines',
                                      line=dict(color=col,width=2.0,dash='dot'),
                                      name=f"{key} π-exact",
                                      text=[f"f={fv:.3f}GHz" for fv in fp],hoverinfo='text'))
        # π-avg: dashed
        sa=Sa[:,r,c].copy() * sc; sa[np.abs(sa)>max_r]=np.nan+1j*np.nan
        fig.add_trace(go.Scatter(x=sa.real,y=sa.imag,mode='lines',
                                  line=dict(color=col,width=3.4,dash='dash'),
                                  name=f"{key} π-avg",
                                  text=[f"f={fv:.3f}GHz" for fv in fp],hoverinfo='text'))

    lay=_slay(f"π-Model Verification — {title}",f_min,f_max,max_r,square_px)
    fig.update_layout(**lay)
    return fig

# ══════════════════════════════════════════════════════════════
# BODE / PLATEAU
# ══════════════════════════════════════════════════════════════
def _lay(title,ytitle,yr,xr):
    return dict(
        title=dict(text=title,font=dict(size=13)),
        xaxis=dict(title="Frequency (GHz)",type="log",
                   range=[np.log10(max(xr[0],1e-4)),np.log10(xr[1])],
                   showgrid=True,gridcolor="#ebebeb",minor_showgrid=True),
        yaxis=dict(title=ytitle,
                   range=list(yr) if yr[0] is not None else None,
                   showgrid=True,gridcolor="#ebebeb"),
        legend=dict(x=1.,y=1.,xanchor="left",yanchor="top",
                    bgcolor="rgba(255,255,255,0.88)",bordercolor="#ccc",borderwidth=1),
        plot_bgcolor="white",paper_bgcolor="white",height=500,
        margin=dict(l=55,r=25,t=45,b=50),
        hovermode="x unified",template="plotly_white")

def _hl(fig,fval,color,label):
    if fval and np.isfinite(fval) and fval>0:
        fig.add_hline(y=fval,line=dict(color=color,width=1.2,dash="dot"),
                      annotation_text=f"{label}={fval:.3f}GHz",
                      annotation_position="right",annotation_font_size=9)

def make_bode(df,title,xr,yr,sh21,su,smag,color,extrap_lines=None):
    """
    extrap_lines: dict with optional keys 'h21', 'U', 'MAG'
      each value: (f_line, g_dB, g_plat, f_cross)
    Measured: solid lines, each metric different color
    Extrapolation: always dashed
    """
    fig=go.Figure(); f=df["Freq (GHz)"]
    hov="Freq:%{x:.4f}GHz<br>Gain:%{y:.4f}dB<extra></extra>"
    el=extrap_lines or {}
    # Distinct colors for each metric
    col_h21 = color
    col_U   = _dk(color)
    col_MAG = "#2ca02c"
    if sh21:
        fig.add_trace(go.Scatter(x=f,y=df["|h21|² (dB)"],name="|h21|²",
                                  line=dict(color=col_h21,width=2.5,dash='solid'),hovertemplate=hov))
        if 'h21' in el and el['h21'][0] is not None:
            fl,gd,_,fc=el['h21']
            fig.add_trace(go.Scatter(x=fl,y=gd,name=f"|h21|² extrap→{fc:.2f}GHz",
                                      line=dict(color=col_h21,width=2.5,dash="dash"),
                                      opacity=0.65,hovertemplate=hov))
            if np.isfinite(fc):
                fig.add_trace(go.Scatter(x=[fc],y=[0],mode='markers',
                                          marker=dict(size=10,symbol='star',color=col_h21),
                                          name=f"fT≈{fc:.2f}GHz",hovertemplate=f"fT≈{fc:.3f}GHz<extra></extra>"))
    if su:
        fig.add_trace(go.Scatter(x=f,y=df["Mason U (dB)"],name="Mason U",
                                  line=dict(color=col_U,width=2.5,dash='solid'),hovertemplate=hov))
        if 'U' in el and el['U'][0] is not None:
            fl,gd,_,fc=el['U']
            fig.add_trace(go.Scatter(x=fl,y=gd,name=f"U extrap→{fc:.2f}GHz",
                                      line=dict(color=col_U,width=2.5,dash="dash"),
                                      opacity=0.65,hovertemplate=hov))
            if np.isfinite(fc):
                fig.add_trace(go.Scatter(x=[fc],y=[0],mode='markers',
                                          marker=dict(size=10,symbol='star',color=col_U),
                                          name=f"fmax(U)≈{fc:.2f}GHz",hovertemplate=f"fmax(U)≈{fc:.3f}GHz<extra></extra>"))
    if smag:
        fig.add_trace(go.Scatter(x=f,y=df["MAG/MSG (dB)"],name="MAG/MSG",
                                  line=dict(color=col_MAG,width=2.5,dash='solid'),hovertemplate=hov))
        if 'MAG' in el and el['MAG'][0] is not None:
            fl,gd,_,fc=el['MAG']
            fig.add_trace(go.Scatter(x=fl,y=gd,name=f"MAG extrap→{fc:.2f}GHz",
                                      line=dict(color=col_MAG,width=2.5,dash="dash"),
                                      opacity=0.65,hovertemplate=hov))
            if np.isfinite(fc):
                fig.add_trace(go.Scatter(x=[fc],y=[0],mode='markers',
                                          marker=dict(size=10,symbol='star',color=col_MAG),
                                          name=f"fmax(MAG)≈{fc:.2f}GHz",hovertemplate=f"fmax(MAG)≈{fc:.3f}GHz<extra></extra>"))
    fig.add_hline(y=0,line_dash="dash",line_color="black")
    fig.update_layout(**_lay(f"Bode — {title}","Gain (dB)",yr,xr))
    return fig

def make_plateau(df,title,xr,sh21,su,smag,color,res=None,extrap_lines=None):
    cols=([df["fT Plateau (GHz)"].tolist()] if sh21 else [])+\
         ([df["fmax U Plateau (GHz)"].tolist()] if su else [])+\
         ([df["fmax MAG Plateau (GHz)"].tolist()] if smag else [])
    arr=np.array([v for sub in cols for v in sub if np.isfinite(v) and v>0])
    ym=float(np.quantile(arr,.97))*1.3 if len(arr) else 100
    hov="Freq:%{x:.4f}GHz<br>GBP:%{y:.4f}GHz<extra></extra>"
    el=extrap_lines or {}
    col_h21 = color
    col_U   = _dk(color)
    col_MAG = "#2ca02c"
    fig=go.Figure()
    if sh21:
        fig.add_trace(go.Scatter(x=df["Freq (GHz)"],y=df["fT Plateau (GHz)"],
                                  name="fT",line=dict(color=col_h21,width=2.5,dash='solid'),hovertemplate=hov))
        if res:
            _hl(fig,res.get("fT Cross/Extrap (GHz)"),col_h21,"fT")
        if 'h21' in el and el['h21'][0] is not None:
            fl,_,gp,fc=el['h21']
            fig.add_trace(go.Scatter(x=fl,y=gp,name=f"fT extrap ({fc:.2f}GHz)",
                                      line=dict(color=col_h21,width=2.5,dash="dash"),
                                      opacity=0.65,hovertemplate=hov))
            if np.isfinite(fc):
                ym=max(ym,fc*1.3)
                fig.add_trace(go.Scatter(x=[fc],y=[fc],mode='markers',
                                          marker=dict(size=10,symbol='star',color=col_h21),
                                          name=f"fT≈{fc:.2f}GHz",hovertemplate=f"fT≈{fc:.3f}GHz<extra></extra>"))
    if su:
        fig.add_trace(go.Scatter(x=df["Freq (GHz)"],y=df["fmax U Plateau (GHz)"],
                                  name="fmax(U)",line=dict(color=col_U,width=2.5,dash='solid'),hovertemplate=hov))
        if res:
            _hl(fig,res.get("fmax U Cross/Extrap (GHz)"),col_U,"fmax(U)")
        if 'U' in el and el['U'][0] is not None:
            fl,_,gp,fc=el['U']
            fig.add_trace(go.Scatter(x=fl,y=gp,name=f"fmax(U) extrap ({fc:.2f}GHz)",
                                      line=dict(color=col_U,width=2.5,dash="dash"),
                                      opacity=0.65,hovertemplate=hov))
            if np.isfinite(fc):
                ym=max(ym,fc*1.3)
                fig.add_trace(go.Scatter(x=[fc],y=[fc],mode='markers',
                                          marker=dict(size=10,symbol='star',color=col_U),
                                          name=f"fmax(U)≈{fc:.2f}GHz",hovertemplate=f"fmax(U)≈{fc:.3f}GHz<extra></extra>"))
    if smag:
        fig.add_trace(go.Scatter(x=df["Freq (GHz)"],y=df["fmax MAG Plateau (GHz)"],
                                  name="fmax(MAG)",line=dict(color=col_MAG,width=2.5,dash='solid'),hovertemplate=hov))
        if 'MAG' in el and el['MAG'][0] is not None:
            fl,_,gp,fc=el['MAG']
            fig.add_trace(go.Scatter(x=fl,y=gp,name=f"fmax(MAG) extrap ({fc:.2f}GHz)",
                                      line=dict(color=col_MAG,width=2.5,dash="dash"),
                                      opacity=0.65,hovertemplate=hov))
            if np.isfinite(fc):
                ym=max(ym,fc*1.3)
    fig.update_layout(**_lay(f"Plateau — {title}","GBP (GHz)",[0,ym],xr))
    return fig

# ══════════════════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════════════════
def _card(col,title,val,sub,color="#4A90D9"):
    col.markdown(
        f'<div style="padding:10px 14px;border-radius:8px;border-left:4px solid {color};'
        f'background:#f7f9fc;min-height:70px;margin-bottom:10px;">'
        f'<div style="font-size:.74rem;color:#666;white-space:nowrap;">{title}</div>'
        f'<div style="font-size:1.15rem;font-weight:700;color:#1a2e4a;">{val}</div>'
        f'<div style="font-size:.70rem;color:#888;margin-top:1px;">{sub}</div></div>',
        unsafe_allow_html=True)

def _fmt_card(vcr,vpl,method):
    if method in ["No Gain","No Data"]: return method
    # For ALL extrap modes: vcr holds the extrapolated 0-dB crossing → show it
    # For 0dB Cross: vcr is the direct crossing
    if np.isfinite(float(vcr)) if vcr is not None and vcr==vcr else False:
        return f"{float(vcr):.3f} GHz"
    if vpl is not None and np.isfinite(float(vpl)):
        return f"{float(vpl):.3f} GHz (plat.)"
    return "N/A"

def get_extrap_line_data(freq_ghz, gain_db, f_start=None, f_end=None, n_pts=None):
    """
    Auto-detect rolloff and return extrapolation line data using single-pole fitting.
    f_start/f_end/n_pts are kept for API compat but ignored — auto-detect is used.
    """
    ok = np.isfinite(gain_db) & (gain_db > -60)
    if ok.sum() < 3:
        return None, None, None, np.nan
    fv, gv = freq_ghz[ok], gain_db[ok]
    f_cross, slope, f_line, g_line, label = fit_single_pole(fv, gv, target_slope=-20.0)
    if f_line is None or not np.isfinite(f_cross):
        return None, None, None, np.nan
    g_plat = f_line * 10**(g_line/20.0)
    return f_line, g_line, g_plat, f_cross

def build_excel(sum_df,all_data):
    buf=io.BytesIO()
    with pd.ExcelWriter(buf,engine="openpyxl") as w:
        sum_df.to_excel(w,sheet_name="Summary",index=False)
        for k,d in all_data.items():
            base=re.sub(r'[:\\/*?\[\]]','_',Path(k).stem)[:28]
            d["df_metrics"].to_excel(w,sheet_name=base,index=False)
            d["df_pi"].to_excel(w,sheet_name=base[:24]+"_pi",index=False)
    return buf.getvalue()

# ══════════════════════════════════════════════════════════════
# FILE UPLOAD & PARSE
# ══════════════════════════════════════════════════════════════
uploaded_files=st.file_uploader(
    "📂 Upload RF CSV / S2P files",type=["csv","s2p"],accept_multiple_files=True,
    key=f"uploader_{st.session_state.get('upload_key',0)}")

if st.button("🗑️ 清除所有上傳檔案", key="clear_uploads"):
    st.session_state["upload_key"] = st.session_state.get("upload_key", 0) + 1
    st.rerun()

# ── Global Contact Resistance De-embedding ────────────────────
st.markdown("### 🔧 Contact Resistance De-embedding (Global)")
st.caption(
    "Atlas TCAD 模擬輸出的 S/Y/Z 包含 contact resistance 成分。"
    "在此輸入 Rb/Rc/Re 後，**所有後續分析（Bode、π-model、S2P 輸出）都使用去嵌後的數據**。\n\n"
    "Z_int = Z_meas − [[Rb+Re, Re], [Re, Rc+Re]]  →  Y_int = inv(Z_int)  →  S_int, H_int"
)
deembed_on = st.checkbox("啟用 Contact R De-embedding", value=False, key="global_deembed",
                          help="關閉後直接使用原始 CSV 數據（不做去嵌）")
gc1, gc2, gc3 = st.columns(3)
g_Rb = gc1.number_input("Rb — Base (Ω)", value=0.0, min_value=0.0, format="%.4f",
                          key="g_Rb", help="Atlas contact statement 中 Base 的 resistance / width")
g_Rc = gc2.number_input("Rc — Collector (Ω)", value=0.0, min_value=0.0, format="%.4f",
                          key="g_Rc", help="Atlas contact statement 中 Collector 的 resistance / width")
g_Re = gc3.number_input("Re — Emitter (Ω)", value=0.0, min_value=0.0, format="%.4f",
                          key="g_Re", help="Atlas contact statement 中 Emitter 的 resistance / width")

if deembed_on and uploaded_files:
    # Auto-estimate from first file if user left all zeros
    if g_Rb == 0 and g_Rc == 0 and g_Re == 0:
        st.info("💡 Rb/Rc/Re 皆為 0 — 將自動從高頻 Z 矩陣估算。如知道 Atlas 設定值，請手動輸入。")

st.divider()

all_data={}; errors={}
if uploaded_files:
    for f in uploaded_files:
        try:
            _,freq_hz,S,Y,Z,meta,vce_m,ib_m,avail,z0,H=parse_any_rf(f)
            stem,vce_fn,ib_fn=parse_bias_from_filename(f.name)

            # ── Apply global Z-matrix de-embedding ─────────────
            Rb_use, Rc_use, Re_use = g_Rb, g_Rc, g_Re
            deembed_applied = False

            if deembed_on:
                # Auto-estimate if user left zeros
                if Rb_use == 0 and Rc_use == 0 and Re_use == 0:
                    _ext = extract_extrinsic_R_from_Z(
                        Y, freq_hz, f_low_ghz=2.0,
                        Z_direct=Z if avail.get("Z_direct") else None)
                    # Use high-freq Z for auto-estimate
                    Rb_use = max(0., _ext.get("RBtot", 0.))
                    Rc_use = max(0., _ext.get("RC", 0.))
                    Re_use = max(0., _ext.get("RE_plus_re", 0.))

                if Rb_use > 0 or Rc_use > 0 or Re_use > 0:
                    Z_direct = Z if avail.get("Z_direct") else None
                    Y_de = deembed_extrinsic_Z(Y, Re_use, Rc_use, Rb_use, Z_direct=Z_direct)
                    S_de = _y_to_s(Y_de, z0)
                    H_de = _s_to_h(S_de)
                    Z_de = _batch_inv(Y_de)
                    # Replace ALL matrices with de-embedded versions
                    S, Y, Z, H = S_de, Y_de, Z_de, H_de
                    deembed_applied = True

            all_data[f.name]={
                "Label":      stem,
                "freq_hz":    freq_hz,
                "S":          S,
                "Y":          Y,
                "Z":          Z,
                "H":          H,
                "meta":       meta,
                "z0":         z0,
                "available":  avail,
                "Z_direct":   avail.get("Z_direct", False),
                "Vce (V)":    vce_m  if vce_m  is not None else vce_fn,
                "Ib (A)":     ib_m   if ib_m   is not None else ib_fn,
                "df_metrics": compute_metrics(Y,freq_hz),
                "df_pi":      compute_intrinsic_pi(Y,freq_hz),
                "deembed_applied": deembed_applied,
                "deembed_R":  {"Rb": Rb_use, "Rc": Rc_use, "Re": Re_use} if deembed_applied else None,
            }
        except Exception as e:
            errors[f.name]=str(e)

for fname,err in errors.items():
    st.error(f"**{fname}**: {err}")

if not all_data:
    st.info("請上傳 CSV 或 S2P 檔案。CSV 可包含 S/Y/Z/H；S2P 會自動展開後再分析。")
    st.stop()

# Show de-embedding status
for k, d in all_data.items():
    if d.get("deembed_applied"):
        r = d["deembed_R"]
        st.success(f"✅ **{Path(k).stem}**: De-embedded with Rb={r['Rb']:.4f}Ω  Rc={r['Rc']:.4f}Ω  Re={r['Re']:.4f}Ω")

selected_files=st.multiselect(
    "選擇要分析的檔案：",list(all_data.keys()),
    default=list(all_data.keys()),
    format_func=lambda x: Path(x).stem)

# ══════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════
tab_ov,tab_ind,tab_pi,tab_sum=st.tabs(
    ["📊 Overlay","📁 Individual","🔬 π-Model & Verification","📋 Summary"])

# ──────────────────────────────────────────────────────────────
# TAB 1 — OVERLAY (uploaded files only, no UIUC reference)
# ──────────────────────────────────────────────────────────────
with tab_ov:
    with st.expander("⚙️ Plot Settings",expanded=True):
        c1,c2,c3,c4=st.columns(4)
        fmin_ov=c1.number_input("Freq Min (GHz)",value=0.4,min_value=0.0001,format="%.4f",key="ov_f1")
        fmax_ov=c2.number_input("Freq Max (GHz)",value=300.,min_value=0.1,key="ov_f2")
        dmin_ov=c3.number_input("dB Min",value=0.,key="ov_d1")
        dmax_ov=c4.number_input("dB Max",value=35., key="ov_d2")
        t1,t2,t3=st.columns(3)
        h21_ov=t1.checkbox("|h21|² → fT",value=True,key="ov_h21")
        su_ov =t2.checkbox("Mason U → fmax(U)",value=True,key="ov_u")
        sm_ov =t3.checkbox("MAG/MSG → fmax(MAG)",value=True,key="ov_m")
        st.caption("fT/fmax 萃取：自動偵測 rolloff 區域，single-pole fitting 外插至 0 dB")

    xr_ov=(fmin_ov,fmax_ov); yr_ov=(dmin_ov,dmax_ov)

    st.markdown("### Bode Plot Overlay")
    fb=go.Figure()
    for i,n in enumerate(selected_files):
        d,c=all_data[n],PALETTE[i%len(PALETTE)]; df_=d["df_metrics"]; lbl=d["Label"]
        hov="Freq:%{x:.4f}GHz<br>Gain:%{y:.4f}dB<extra></extra>"
        fa_ov=df_["Freq (GHz)"].values
        col_U = _dk(c)
        col_MAG = "#2ca02c"
        if h21_ov:
            fb.add_trace(go.Scatter(x=df_["Freq (GHz)"],y=df_["|h21|² (dB)"],
                                     name=f"|h21|²–{lbl}",line=dict(color=c,width=2.5,dash='solid'),hovertemplate=hov))
            fl,gd,_,fc=get_extrap_line_data(fa_ov,df_["|h21|² (dB)"].values)
            if fl is not None:
                fb.add_trace(go.Scatter(x=fl,y=gd,name=f"|h21|² extrap–{lbl} ({fc:.1f}GHz)",
                                         line=dict(color=c,width=2.5,dash="dash"),opacity=0.6,hovertemplate=hov))
                if np.isfinite(fc):
                    fb.add_trace(go.Scatter(x=[fc],y=[0],mode='markers',
                                             marker=dict(size=9,symbol='star',color=c),
                                             name=f"fT≈{fc:.2f}GHz–{lbl}",showlegend=False,hoverinfo='skip'))
        if su_ov:
            fb.add_trace(go.Scatter(x=df_["Freq (GHz)"],y=df_["Mason U (dB)"],
                                     name=f"U–{lbl}",line=dict(color=col_U,width=2.5,dash='solid'),hovertemplate=hov))
            fl,gd,_,fc=get_extrap_line_data(fa_ov,df_["Mason U (dB)"].values)
            if fl is not None:
                fb.add_trace(go.Scatter(x=fl,y=gd,name=f"U extrap–{lbl} ({fc:.1f}GHz)",
                                         line=dict(color=col_U,width=2.5,dash="dash"),opacity=0.6,hovertemplate=hov))
                if np.isfinite(fc):
                    fb.add_trace(go.Scatter(x=[fc],y=[0],mode='markers',
                                             marker=dict(size=9,symbol='star',color=col_U),
                                             name=f"fmax(U)≈{fc:.2f}GHz–{lbl}",showlegend=False,hoverinfo='skip'))
        if sm_ov:
            fb.add_trace(go.Scatter(x=df_["Freq (GHz)"],y=df_["MAG/MSG (dB)"],
                                     name=f"MAG–{lbl}",line=dict(color=col_MAG,width=2.5,dash='solid'),hovertemplate=hov))

    fb.add_hline(y=0,line_dash="dash",line_color="black")

    fb.update_layout(**_lay("Overlay — Bode","Gain (dB)",yr_ov,xr_ov))
    st.plotly_chart(fb,use_container_width=True)

    st.markdown("### Plateau Plot Overlay")
    fp2=go.Figure(); av=[]
    for i,n in enumerate(selected_files):
        d,c=all_data[n],PALETTE[i%len(PALETTE)]; df_=d["df_metrics"]; lbl=d["Label"]
        hov="Freq:%{x:.4f}GHz<br>GBP:%{y:.4f}GHz<extra></extra>"
        fa_ov=df_["Freq (GHz)"].values
        col_U = _dk(c)
        col_MAG = "#2ca02c"
        if h21_ov:
            fp2.add_trace(go.Scatter(x=fa_ov,y=df_["fT Plateau (GHz)"],
                                      name=f"fT–{lbl}",line=dict(color=c,width=2.5,dash='solid'),hovertemplate=hov))
            av.extend(df_["fT Plateau (GHz)"].dropna().tolist())
            fl,_,gp,fc=get_extrap_line_data(fa_ov,df_["|h21|² (dB)"].values)
            if fl is not None:
                fp2.add_trace(go.Scatter(x=fl,y=gp,name=f"fT extrap–{lbl} ({fc:.1f}GHz)",
                                          line=dict(color=c,width=2.5,dash="dash"),opacity=0.6,hovertemplate=hov))
                if np.isfinite(fc): av.append(fc)
        if su_ov:
            fp2.add_trace(go.Scatter(x=fa_ov,y=df_["fmax U Plateau (GHz)"],
                                      name=f"fmax–{lbl}",line=dict(color=col_U,width=2.5,dash='solid'),hovertemplate=hov))
            av.extend(df_["fmax U Plateau (GHz)"].dropna().tolist())
            fl,_,gp,fc=get_extrap_line_data(fa_ov,df_["Mason U (dB)"].values)
            if fl is not None:
                fp2.add_trace(go.Scatter(x=fl,y=gp,name=f"fmax(U) extrap–{lbl} ({fc:.1f}GHz)",
                                          line=dict(color=col_U,width=2.5,dash="dash"),opacity=0.6,hovertemplate=hov))
                if np.isfinite(fc): av.append(fc)
        if sm_ov:
            fp2.add_trace(go.Scatter(x=fa_ov,y=df_["fmax MAG Plateau (GHz)"],
                                      name=f"fmax(MAG)–{lbl}",line=dict(color=col_MAG,width=2.5,dash='solid'),hovertemplate=hov))
            av.extend(df_["fmax MAG Plateau (GHz)"].dropna().tolist())
    arr_=np.array([v for v in av if np.isfinite(v) and v>0])
    ym_=float(np.quantile(arr_,.97))*1.3 if len(arr_) else 100
    fp2.update_layout(**_lay("Overlay — Plateau","GBP (GHz)",[0,ym_],xr_ov))
    st.plotly_chart(fp2,use_container_width=True)

# ──────────────────────────────────────────────────────────────
# TAB 2 — INDIVIDUAL (with UIUC reference overlay)
# ──────────────────────────────────────────────────────────────
with tab_ind:
    if not selected_files:
        st.info("請先選擇檔案。")
    else:
        with st.expander("⚙️ Chart & Extraction Settings",expanded=True):
            r1,r2,r3,r4=st.columns(4)
            fmin_id=r1.number_input("Freq Min (GHz)",value=0.4,min_value=0.0001,format="%.4f",key="id_f1")
            fmax_id=r2.number_input("Freq Max (GHz)",value=300.,min_value=0.1,key="id_f2")
            dmin_id=r3.number_input("dB Min",value=0.,key="id_d1")
            dmax_id=r4.number_input("dB Max",value=35.,key="id_d2")
            t1,t2,t3=st.columns(3)
            h21_id=t1.checkbox("|h21|² → fT",value=True,key="id_h21")
            su_id =t2.checkbox("Mason U → fmax(U)",value=True,key="id_u")
            sm_id =t3.checkbox("MAG/MSG → fmax(MAG)",value=True,key="id_m")

            st.caption("fT/fmax: 自動偵測 rolloff → single-pole fitting 外插至 0 dB")

            # ── Reference data overlay ─────────────────────────────
            st.markdown("**📎 Reference Data Overlay**")
            show_uiuc_id = st.checkbox("顯示 UIUC 量測數據", value=True, key="show_uiuc_id")
            with st.expander("自定義 Reference Data（可選）", expanded=False):
                ref_col1, ref_col2 = st.columns(2)
                ref_h21_file = ref_col1.file_uploader(
                    "|h21|² ref (CSV: freq_Hz, dB)", type=["csv"], key="ref_h21",
                    help="兩欄 CSV：頻率(Hz), |h21|²(dB)。無標題列。")
                ref_U_file = ref_col2.file_uploader(
                    "Mason U ref (CSV: freq_Hz, dB)", type=["csv"], key="ref_U")

            def _parse_ref_csv(file_obj):
                if file_obj is None: return None, None
                try:
                    raw = file_obj.getvalue().decode('utf-8', errors='ignore')
                    lines = [l.strip() for l in raw.splitlines() if l.strip() and not l.strip().startswith('#')]
                    data = np.array([[float(x) for x in l.split(',')] for l in lines if ',' in l])
                    if data.shape[1] >= 2 and len(data) >= 2:
                        return data[:,0] * 1e-9, data[:,1]
                except: pass
                return None, None
            ref_h21_f, ref_h21_g = _parse_ref_csv(ref_h21_file)
            ref_U_f, ref_U_g = _parse_ref_csv(ref_U_file)

            st.markdown("**文獻比較窗口**")
            fit_win_ghz = st.number_input("Single-pole fit 窗口上限 (GHz)", value=50.0,
                                            min_value=5.0, max_value=500.0, step=5.0,
                                            key="fit_win_ghz",
                                            help="模擬「文獻只量到 50 GHz」的條件。"
                                                 "用此窗口內的數據做 single-pole fit 外插。")

            st.markdown("**Smith Chart**")
            s1,s2,s3=st.columns(3)
            sfmin_id=s1.number_input("Smith Min (GHz)",value=0.4,min_value=0.0001,format="%.4f",key="id_sf1")
            sfmax_id=s2.number_input("Smith Max (GHz)",value=300.,min_value=0.1,key="id_sf2")
            sr_id   =s3.slider("|Γ| Max",0.5,15.,1.0,step=0.5,key="id_sr")
            b1,b2,b3,b4=st.columns(4)
            ss11=b1.checkbox("S11",value=True,key="id_s11"); ss22=b2.checkbox("S22",value=True,key="id_s22")
            ss21=b3.checkbox("S21",value=False,key="id_s21"); ss12=b4.checkbox("S12",value=True,key="id_s12")
            st.markdown("**S-Parameter Magnification**")
            sm1,sm2,sm3,sm4=st.columns(4)
            id_sc11=sm1.number_input("S11 ×",value=1.0,min_value=0.1,step=1.0,format="%.1f",key="id_sc11")
            id_sc22=sm2.number_input("S22 ×",value=1.0,min_value=0.1,step=1.0,format="%.1f",key="id_sc22")
            id_sc21=sm3.number_input("S21 ×",value=0.1,min_value=0.01,step=0.1,format="%.2f",key="id_sc21",
                                      help="|S21| 在低頻可達 10+，建議縮小 0.1× 才能在 Smith 圖中看到")
            id_sc12=sm4.number_input("S12 ×",value=1.0,min_value=0.1,step=1.0,format="%.1f",key="id_sc12")
            id_s_scales = {'S11':id_sc11,'S22':id_sc22,'S21':id_sc21,'S12':id_sc12}

        xr_id=(fmin_id,fmax_id); yr_id=(dmin_id,dmax_id)
        stabs=st.tabs([all_data[n]["Label"] for n in selected_files])
        fn_list=list(all_data.keys())

        for stab,n in zip(stabs,selected_files):
            c_=PALETTE[fn_list.index(n)%len(PALETTE)]
            d=all_data[n]; df_=d["df_metrics"]; fa=df_["Freq (GHz)"].values

            # ── Extrapolation line data (always compute; used in both Bode and Plateau) ──
            el_h21=get_extrap_line_data(fa,df_["|h21|² (dB)"].values)
            el_U  =get_extrap_line_data(fa,df_["Mason U (dB)"].values)
            el_MAG=get_extrap_line_data(fa,df_["MAG/MSG (dB)"].values)
            extrap_lines_id={'h21':el_h21,'U':el_U,'MAG':el_MAG}

            # ── Extract fT / fmax (dual mode) ────────────────────────────────
            fTc,fTp,ftm=extract_limit(fa,df_["|h21|² (dB)"].values,df_["fT Plateau (GHz)"].values,
                                       fmin_id,fmax_id)
            fUc,fUp,fUm=extract_limit(fa,df_["Mason U (dB)"].values,df_["fmax U Plateau (GHz)"].values,
                                       fmin_id,fmax_id)
            fMc,fMp,fMm=extract_limit(fa,df_["MAG/MSG (dB)"].values,df_["fmax MAG Plateau (GHz)"].values,
                                       fmin_id,fmax_id)

            # Dual extraction
            fT_full,fT_full_m, fT_win,fT_win_m, fT_plat = extract_limit_dual(
                fa, df_["|h21|² (dB)"].values, df_["fT Plateau (GHz)"].values,
                fmin_id, fmax_id, fit_win_ghz)
            fU_full,fU_full_m, fU_win,fU_win_m, _ = extract_limit_dual(
                fa, df_["Mason U (dB)"].values, df_["fmax U Plateau (GHz)"].values,
                fmin_id, fmax_id, fit_win_ghz)

            res_={"fT Cross/Extrap (GHz)":fTc,"fT Plateau (GHz)":fTp,"fT Method":ftm,
                  "fmax U Cross/Extrap (GHz)":fUc,"fmax U Plateau (GHz)":fUp,"fmax U Method":fUm,
                  "fmax MAG Cross/Extrap (GHz)":fMc,"fmax MAG Plateau (GHz)":fMp,"fmax MAG Method":fMm}

            with stab:
                # ── Dual extraction comparison table ──
                st.markdown("##### 📊 fT / fmax Extraction (Dual Mode)")
                dual_rows = []
                for _name, _full, _fm, _win, _wm in [
                    ("fT (|h21|²)",   fT_full, fT_full_m, fT_win, fT_win_m),
                    ("fmax (Mason U)", fU_full, fU_full_m, fU_win, fU_win_m),
                ]:
                    dual_rows.append({
                        "Metric": _name,
                        f"Full Range (GHz)": f"{_full:.1f}" if np.isfinite(_full) else "—",
                        "Full Method": _fm,
                        f"≤{fit_win_ghz:.0f} GHz Fit (GHz)": f"{_win:.1f}" if np.isfinite(_win) else "—",
                        "Window Method": _wm,
                        "Δ (%)": f"{(_win-_full)/_full*100:.1f}%" if np.isfinite(_full) and np.isfinite(_win) and _full>0 else "—",
                    })
                st.dataframe(pd.DataFrame(dual_rows), use_container_width=True, hide_index=True)

                # ── Info cards ──
                c1,c2,c3,c4,c5=st.columns(5)
                _card(c1,"Source",d.get("available",{}).get("source","csv").upper(),"","#888")
                _fT_disp = fT_full if np.isfinite(fT_full) else fT_win
                _fU_disp = fU_full if np.isfinite(fU_full) else fU_win
                _card(c2,"fT", f"{_fT_disp:.1f} GHz" if np.isfinite(_fT_disp) else "—", fT_full_m)
                _card(c3,"fmax(U)", f"{_fU_disp:.1f} GHz" if np.isfinite(_fU_disp) else "—", fU_full_m,"#d62728")
                _card(c4,"fmax MAG",_fmt_card(fMc,fMp,fMm),fMm,"#2ca02c")
                try:
                    Ic_val = abs(d.get("meta",{}).get("Collector Current",0))*1e3
                    beta_val = abs(d.get("meta",{}).get("Collector Current",0) / 
                                   d.get("meta",{}).get("Base Current",1))
                    _card(c5,"Bias",f"Ic={Ic_val:.2f}mA",f"β={beta_val:.1f}","#9467bd")
                except:
                    _card(c5,"Bias","—","","#9467bd")

                st.caption(f"Available: S={d.get('available',{}).get('S',True)} · Y={d.get('available',{}).get('Y',True)} · Z={d.get('available',{}).get('Z',False)} · H={d.get('available',{}).get('H',False)}")
                ta,tb,tc=st.tabs(["Bode Plot","Plateau Plot","Smith Chart"])
                with ta:
                    fig_bode_id=make_bode(df_,d["Label"],xr_id,yr_id,h21_id,su_id,sm_id,c_,
                                           extrap_lines=extrap_lines_id)
                    # Add UIUC reference
                    if show_uiuc_id:
                        if h21_id:
                            fig_bode_id.add_trace(go.Scatter(
                                x=_UIUC_H21[:,0], y=_UIUC_H21[:,1], mode='markers',
                                name="|h21|²–UIUC",
                                marker=dict(size=5, color='black', symbol='circle-open', line=dict(width=1.5)),
                                hovertemplate="f=%{x:.2f}GHz<br>%{y:.2f}dB<extra>UIUC</extra>"))
                        if su_id:
                            fig_bode_id.add_trace(go.Scatter(
                                x=_UIUC_U[:,0], y=_UIUC_U[:,1], mode='markers',
                                name="U–UIUC",
                                marker=dict(size=5, color='black', symbol='diamond-open', line=dict(width=1.5)),
                                hovertemplate="f=%{x:.2f}GHz<br>%{y:.2f}dB<extra>UIUC</extra>"))
                    # Custom reference
                    if ref_h21_f is not None and h21_id:
                        fig_bode_id.add_trace(go.Scatter(x=ref_h21_f, y=ref_h21_g, mode='markers',
                            name="|h21|²–Custom Ref",
                            marker=dict(size=5, color='gray', symbol='circle-open', line=dict(width=1.5))))
                    if ref_U_f is not None and su_id:
                        fig_bode_id.add_trace(go.Scatter(x=ref_U_f, y=ref_U_g, mode='markers',
                            name="U–Custom Ref",
                            marker=dict(size=5, color='gray', symbol='diamond-open', line=dict(width=1.5))))

                    st.plotly_chart(fig_bode_id,use_container_width=True)
                with tb:
                    st.plotly_chart(make_plateau(df_,d["Label"],xr_id,h21_id,su_id,sm_id,c_,
                                                  res_,extrap_lines=extrap_lines_id),
                                     use_container_width=True)
                with tc:
                    fig_smith=make_smith(d["S"],fa,sfmin_id,sfmax_id,d["Label"],
                                          sr_id,ss11,ss22,ss21,ss12,square_px=980,
                                          s_scales=id_s_scales)
                    st.plotly_chart(fig_smith,width="content")

                with st.expander("📋 Data Table"):
                    t1_,t2_=st.tabs(["Metrics","π-Model"])
                    with t1_: st.dataframe(df_.round(4),use_container_width=True,hide_index=True)
                    with t2_: st.dataframe(d["df_pi"].round(4),use_container_width=True,hide_index=True)

# ──────────────────────────────────────────────────────────────
# TAB 3 — π-MODEL & CLOSED-LOOP VERIFICATION
# ──────────────────────────────────────────────────────────────
with tab_pi:
    if not selected_files:
        st.info("請先選擇檔案。")
    else:
        with st.expander("📚 模型比較說明 — 本工具 vs IOED_HBT_RF_extract (Cheng 2022)", expanded=False):
            st.markdown("""
| 項目 | **本工具 (v3.0)** | **IOED_HBT_RF_extract (Cheng 2022)** |
|------|-----------------|--------------------------------------|
| **模型拓撲** | **Kumar (2014) 8-element π**：C_BC/C_BE/C_CE/R_BC/R_BE/R_CE/gm/τ + fT/fmax analytical + optional 15-element Xu | T & π dual topology (analytical peeling) |
| **De-embedding** | ① TCAD No-Pad (Z矩陣 RE/RC/RBext) + ② Rbb Y₁₁/Y₁₂ 解析法 | Open-Short-Thru (Gao 2015 §4.2)，需要 dummy pad |
| **Rbb 萃取** | 解析式 Eq. 4.34 (Yang IEEE TMT 2007): `Rbb = Re(1/Y₁₁) − Im(1/Y₁₁)·Im(Y₁₁/Y₁₂)/Re(Y₁₁/Y₁₂)` | Z 矩陣法 (T model): `Rbb ≈ Re(Z₁₁−Z₁₂)` |
| **是否需要 Open/Short pad** | **不需要（TCAD No-Pad 模式）** | 需要 Device Open + Device Short dummy |
| **本質參數萃取** | 直接從 Y_int 閉合解析：Cmu=-Im(Y₁₂)/ω, Gpi=Re(Y₁₁+Y₁₂)... | 解析 peeling 逐層剝離（T model先，再轉π）|
| **優點** | Atlas TCAD 直接可用；Rbb 解析式快速；Gμ = −Re(Y₁₂) 正確含電導 | 雙拓撲交叉驗證；量測資料 de-embedding 完整 |
| **適用場景** | TCAD 模擬（無 pad）、已做 Open/Short 的量測資料 | 量測資料（有 probe pad 與 device dummy） |
| **驗證方法** | Smith Chart π-exact/π-avg 重建誤差 | Measured vs Modeled S-param overlay |

**Atlas TCAD 直接輸出（無 Open/Short）→ 選 TCAD No-Pad 模式**：
1. Z₁₂ 低頻實部 → RE（射極接觸電阻）
2. (Z₂₂−Z₁₂) 低頻實部 → RC（集極接觸電阻）  
3. Z 矩陣相減解嵌入後，再用 Eq. 4.34 萃取 Rbb
4. 從 Y_int 萃取 8 個本質元素：Cmu, Gmu, Cpi, Cce, rpi, ro, gm, τ

**為什麼量測結果需要 Open/Short 但 TCAD 不需要**：
Atlas 模擬的是純半導體結構（沒有探針墊 pad），輸出的 S 參數只有接觸電阻和本質元件，不含 CPW pad 電容/電感。
因此跳過 Open/Short，直接用 Z 矩陣低頻法萃取外部 RE/RC 即可。
            """)
        pi_stabs=st.tabs([all_data[n]["Label"] for n in selected_files])
        fn_pi=list(all_data.keys())

        for pi_stab,n in zip(pi_stabs,selected_files):
            d=all_data[n]; df_pi=d["df_pi"]; fa_=d["df_metrics"]["Freq (GHz)"].values

            with pi_stab:
                st.markdown("#### ⚙️ Settings")
                pc1,pc2,pc3,pc4=st.columns(4)
                # Sensible defaults: middle third for avg, full range for Smith
                def_af1=float(fa_[max(0,len(fa_)//3)])
                def_af2=float(fa_[min(len(fa_)-1,2*len(fa_)//3)])
                af1=pc1.number_input("Avg Freq Min (GHz)",value=def_af1,min_value=0.0001,
                                      format="%.4f",key=f"pi_af1_{n}")
                af2=pc2.number_input("Avg Freq Max (GHz)",value=def_af2,min_value=0.0001,
                                      format="%.4f",key=f"pi_af2_{n}")
                vf1=pc3.number_input("Smith Min (GHz)",value=float(fa_[0]),min_value=0.0001,
                                      format="%.4f",key=f"pi_vf1_{n}")
                vf2=pc4.number_input("Smith Max (GHz)",value=float(fa_[-1]),min_value=0.0001,
                                      key=f"pi_vf2_{n}")
                vr=st.slider("|Γ| Max (verification)",0.5,15.,1.0,step=.5,key=f"pi_vr_{n}")

                # Define averaging band mask early (used by both TCAD and Rbb sections)
                _rbb_mask = (fa_ >= af1) & (fa_ <= af2)

                # ── De-embedding mode selector (simplified) ───────────
                st.markdown("---")
                st.markdown("**🔧 De-Embedding 設定**")
                _deembed_cols = st.columns([3, 2])
                _deembed_mode = _deembed_cols[0].radio(
                    "模式選擇",
                    ["不解嵌入 (直接萃取)",
                     "Rbb 解嵌入",
                     "Rbb + RE/RC 解嵌入 (進階)"],
                    key=f"pi_demode_{n}",
                    help=(
                        "**不解嵌入**：直接從 Y 萃取，適合 Atlas 已含所有電阻、或只需要參考值\n"
                        "**Rbb 解嵌入**：解除基極串聯電阻 Rbb，改善本質參數精度\n"
                        "**Rbb + RE/RC**：進階，需手動輸入 RE/RC 值（建議從 Atlas 設定取得）"
                    )
                )
                _deembed_cols[1].markdown(
                    '<div style="margin-top:4px;padding:8px 12px;border-radius:8px;'
                    'background:#f0f7ff;font-size:.80rem;border-left:3px solid #2471a3;">'
                    '<b>Atlas TCAD 建議</b><br>'
                    '✅ 先用「不解嵌入」確認資料正常<br>'
                    '→ 若 Smith avg 誤差大，試「Rbb 解嵌入」<br>'
                    '→ 若已知接觸電阻，才用第三選項<br>'
                    '<span style="color:#888;font-size:.73rem;">Rbb ≈ 本質基極電阻（電路圖中 B→B\' 的電阻)</span>'
                    '</div>',
                    unsafe_allow_html=True
                )
                st.markdown("")

                # ── RE/RC advanced de-embedding (mode 3 only) ────────────
                _RE_use = _RC_use = _RBext_use = 0.0
                _z_direct = d.get("Z_direct", False)
                _Z_mat    = d.get("Z", None)

                if _deembed_mode.startswith("Rbb + RE"):
                    st.markdown("**⚙️ 接觸電阻輸入（RE / RC）**")
                    st.caption(
                        "RE = 射極接觸電阻，RC = 集極接觸電阻。"
                        "請直接從 Atlas 模擬設定中取得（不可用自動估算，原因是低頻 Re(Z₁₂) 包含本質電阻 re = 1/gm）。"
                    )
                    if _z_direct and _Z_mat is not None:
                        st.success("✅ Atlas Z-param 直接輸出 — 使用原始 Z 矩陣做減法")
                    _tcde1, _tcde2, _tcde3 = st.columns(3)
                    _RE_use   = _tcde1.number_input("RE (Ω)  — 射極接觸電阻",
                                                     value=0.0, min_value=0., step=0.1,
                                                     format="%.3f", key=f"pi_RE_{n}")
                    _RC_use   = _tcde2.number_input("RC (Ω)  — 集極接觸電阻",
                                                     value=0.0, min_value=0., step=0.1,
                                                     format="%.3f", key=f"pi_RC_{n}")
                    _RBext_use= _tcde3.number_input("RBext (Ω) — 外部基極電阻（通常 0）",
                                                     value=0.0, min_value=0., step=0.1,
                                                     format="%.3f", key=f"pi_RBext_{n}")
                    _Y_deembed_tcad = deembed_extrinsic_Z(
                        d["Y"], _RE_use, _RC_use, _RBext_use,
                        Z_direct=_Z_mat if _z_direct else None
                    )
                    st.caption(f"已從 Z 矩陣減去 RE={_RE_use:.2f}Ω、RC={_RC_use:.2f}Ω、RBext={_RBext_use:.2f}Ω。")
                else:
                    _Y_deembed_tcad = d["Y"]   # no Z subtraction, use raw Y

                # ── Rbb de-embedding control (Lucas Yang §4.3 / Eq. 4.34) ──
                _show_rbb = not _deembed_mode.startswith("不解嵌入")
                if _show_rbb:
                    st.markdown("**Rbb (本質基極串聯電阻) De-Embedding**")
                rbb_cols = st.columns([2,2,1]) if _show_rbb else [None, None, None]

                # Rbb(f) from TCAD-deembedded Y (or raw Y if Rbb-only mode)
                _rbb_from_Y = extract_rbb_analytical(_Y_deembed_tcad)
                # _rbb_mask already defined above (after vr slider)
                _rbb_finite_vals = _rbb_from_Y.real[_rbb_mask]
                _rbb_finite_vals = _rbb_finite_vals[np.isfinite(_rbb_finite_vals)]
                _rbb_analytic = float(np.median(_rbb_finite_vals)) if len(_rbb_finite_vals)>0 else 0.0
                _rbb_analytic = max(0., _rbb_analytic)  # physically Rbb >= 0

                if _show_rbb:
                    _use_rbb = rbb_cols[2].checkbox("啟用 Rbb", value=True, key=f"pi_userbb_{n}")
                else:
                    _use_rbb = False
                if _show_rbb:
                    _rbb_man = rbb_cols[0].number_input(
                        "Rbb (Ω) — 手動設定 (0 = 不解嵌入)",
                        value=round(_rbb_analytic, 3),
                        min_value=0.0, max_value=5000.0, step=0.1, format="%.3f",
                        key=f"pi_rbb_{n}")
                    rbb_cols[1].markdown(
                        f'<div style="margin-top:28px;padding:6px 12px;border-radius:6px;'
                        f'background:#eef4fb;font-size:.85rem;">'
                        f'📐 Analytical median (avg band) = <b>{_rbb_analytic:.3f} Ω</b><br>'
                        f'<span style="font-size:.75rem;color:#777;">Lucas Yang Eq. 4.34</span></div>',
                        unsafe_allow_html=True)
                    _rbb_use = float(_rbb_man) if _use_rbb and float(_rbb_man)>0 else None
                else:
                    _rbb_use = None

                # Recompute df_pi with TCAD + Rbb de-embedding
                if _rbb_use is not None:
                    df_pi = compute_intrinsic_pi(_Y_deembed_tcad, d["freq_hz"], rbb_deembed=_rbb_use)
                else:
                    # Still use TCAD-deembedded Y if in TCAD mode (no Rbb step)
                    df_pi = compute_intrinsic_pi(_Y_deembed_tcad, d["freq_hz"], rbb_deembed=None)

                st.markdown("**S-Parameter Magnification** (for Smith Chart visualization)")
                sm1,sm2,sm3,sm4=st.columns(4)
                sc_s11=sm1.number_input("S11 ×",value=1.0,min_value=0.1,step=1.0,format="%.1f",key=f"pi_sc11_{n}")
                sc_s22=sm2.number_input("S22 ×",value=1.0,min_value=0.1,step=1.0,format="%.1f",key=f"pi_sc22_{n}")
                sc_s21=sm3.number_input("S21 ×",value=0.1,min_value=0.01,step=0.1,format="%.2f",key=f"pi_sc21_{n}",
                                        help="|S21| 在低頻可達 10+，建議縮小 0.1×")
                sc_s12=sm4.number_input("S12 ×",value=1.0,min_value=0.1,step=1.0,format="%.1f",key=f"pi_sc12_{n}")
                s_scales_pi = {'S11':sc_s11,'S22':sc_s22,'S21':sc_s21,'S12':sc_s12}

                avg_,df_band=compute_avg(df_pi,af1,af2)

                # ── Schematic ─────────────────────────────────────────────
                st.markdown("#### 🔌 π-Model Schematic")
                st.markdown(draw_pi_svg(avg_),unsafe_allow_html=True)

                # ── Averaged parameters table ──────────────────────────────
                st.markdown("#### 📊 Averaged Parameters")
                # ── Kumar (2014) naming table ──────────────────────────
                # Helper to safely get value
                def _ag(k, scale=1.):
                    v = avg_.get(k)
                    return round(float(v)*scale, 5) if v is not None and np.isfinite(float(v)) else None

                st.caption(
                    "**参數對應**（Kumar 2014 IEEE TCAD Eq.2-8）: "
                    "C_BC=Cmu=-Im(Y12)/w | C_BE=Cpi=Im(Y11+Y12)/w | C_CE=Im(Y22+Y12)/w | "
                    "R_BC=-1/Re(Y12) | R_BE=1/Re(Y11+Y12) | R_CE=1/Re(Y22+Y12) | "
                    "R_B(Rbb)=Re(Z11-Z12) | gm=|Y21-Y12|"
                )
                tbl=pd.DataFrame([{
                    "Avg band":        avg_.get("Freq Range (GHz)"),
                    "Pts":             avg_.get("Points"),
                    # Capacitances (Kumar Eq.2-4)
                    "C_BC/Cμ (fF)":   _ag("C_BC (fF)"),
                    "C_BE/Cπ (fF)":   _ag("C_BE (fF)"),
                    "C_CE (fF)":       _ag("C_CE (fF)"),
                    # Resistances (Kumar Eq.5-8)
                    "R_BC (Ω)":        _ag("R_BC (Ohm)"),
                    "R_BE/rπ (Ω)":    _ag("R_BE (Ohm)"),
                    "R_CE/ro (Ω)":    _ag("R_CE (Ohm)"),
                    "R_B/Rbb (Ω)":    _ag("Rbb (Ohm)"),
                    # Transconductance
                    "gm (mS)":         _ag("gm (mS)"),
                    "τ (ps)":          _ag("tau (ps)"),
                    # Analytical fT, fmax (Kumar Eq.13,14)
                    "fT_model (GHz)":  _ag("fT_model (GHz)"),
                    "fmax_model (GHz)":_ag("fmax_model (GHz)"),
                }])
                st.dataframe(tbl.round(4),use_container_width=True,hide_index=True)

                # ── Analytical fT / fmax cards (Kumar Eq.13, 14) ────────────
                st.markdown("**📐 Analytical fT / fmax from π-model (Kumar Eq.13 & 14)**")
                _ft_mod  = avg_.get("fT_model (GHz)")
                _fmax_mod= avg_.get("fmax_model (GHz)")
                _gm_avg  = avg_.get("gm (mS)", 0)
                _cbc_avg = avg_.get("C_BC (fF)", 0)
                _cbe_avg = avg_.get("C_BE (fF)", 0)
                _rbb_avg = avg_.get("Rbb (Ohm)", 0)
                _card_cols = st.columns(2)
                for _cc, _val, _lbl, _formula, _color in [
                    (_card_cols[0], _ft_mod,  "fT (analytical)",
                     f"gm/{2*3.14159:.2f}π·(C_BE+C_BC) = {_gm_avg:.2f}mS / 2π·({_cbe_avg:.2f}+{_cbc_avg:.2f})fF",
                     "#2471a3"),
                    (_card_cols[1], _fmax_mod,"fmax (analytical)",
                     f"√(fT/8π·C_BC·Rbb) = √(fT/8π·{_cbc_avg:.2f}fF·{_rbb_avg:.2f}Ω)",
                     "#d62728"),
                ]:
                    if _val is not None and np.isfinite(float(_val)):
                        _vstr = f"{float(_val):.2f} GHz"
                    else:
                        _vstr = "N/A (需啟用 Rbb 解嵌入)"
                    _cc.markdown(
                        f'<div style="padding:10px 14px;border-radius:8px;'
                        f'border-left:5px solid {_color};background:#f7f9fc;margin-bottom:8px;">'
                        f'<div style="font-size:.75rem;color:#666;">{_lbl}</div>'
                        f'<div style="font-size:1.4rem;font-weight:700;color:{_color};">{_vstr}</div>'
                        f'<div style="font-size:.70rem;color:#888;margin-top:3px;">{_formula}</div>'
                        f'</div>',
                        unsafe_allow_html=True)

                # ── Rbb(f) trend — key diagnostic ───────────────────────
                _rbb_all = _rbb_from_Y.real   # from TCAD-deembedded Y (or raw)
                _rbb_ok  = np.isfinite(_rbb_all)
                if _rbb_ok.any():
                    fig_rbb = go.Figure()
                    fig_rbb.add_trace(go.Scatter(
                        x=fa_[_rbb_ok], y=_rbb_all[_rbb_ok],
                        mode="lines", name="Rbb(f) analytical",
                        line=dict(color="#1f77b4", width=2)))
                    fig_rbb.add_hrect(
                        y0=float(np.nanpercentile(_rbb_all,10)),
                        y1=float(np.nanpercentile(_rbb_all,90)),
                        fillcolor="rgba(31,119,180,0.08)", line_width=0)
                    if _rbb_use is not None:
                        fig_rbb.add_hline(y=_rbb_use,
                            line=dict(color="#d62728", width=1.8, dash="dash"),
                            annotation_text=f"使用中 Rbb = {_rbb_use:.2f} Ω",
                            annotation_position="right", annotation_font_size=10)
                    fig_rbb.add_vrect(x0=af1,x1=af2,
                        fillcolor="rgba(255,200,0,0.18)",line_width=0,
                        annotation_text="Avg band",annotation_position="top left")
                    fig_rbb.update_layout(**_lay(
                        f"Rbb(f) — {d['Label']}  (應接近常數 → 拓撲正確)",
                        "Rbb (Ω)", [None,None], (float(fa_[0]),float(fa_[-1]))))
                    st.plotly_chart(fig_rbb, use_container_width=True)
                    st.caption(
                        "💡 **Rbb 診斷**：若曲線平坦（低頻到高頻幾乎不變）→ 拓撲正確，Rbb 即為有效值。"
                        "若 Rbb(f) 隨頻率快速變化 → 表示還有未解嵌入的寄生元件（Le/Lb/Lc 等電感）。"
                    )

                st.divider()

                # ── Parameter vs frequency trend ───────────────────────────
                st.markdown("#### 📈 Parameter vs Frequency")
                # Preferred display order for parameter selector
                _preferred = ["C_BC (fF)","C_BE (fF)","C_CE (fF)",
                               "R_BC (Ohm)","R_BE (Ohm)","R_CE (Ohm)",
                               "gm (mS)","tau (ps)","Rbb (Ohm)",
                               "fT_model (GHz)","fmax_model (GHz)"]
                param_choices = _preferred + [c for c in df_pi.columns
                                              if c not in _preferred and c!="Freq (GHz)"]
                sel_col=st.selectbox("選擇參數（Kumar 2014 命名）",param_choices,
                                      key=f"pi_sel_{n}")
                fig_tr=go.Figure()
                fig_tr.add_trace(go.Scatter(
                    x=df_pi["Freq (GHz)"],y=df_pi[sel_col],mode="lines",
                    name=sel_col,
                    line=dict(color=PALETTE[fn_pi.index(n)%len(PALETTE)],width=2.5)))
                fig_tr.add_vrect(x0=af1,x1=af2,
                                  fillcolor="rgba(255,200,0,0.18)",line_width=0,
                                  annotation_text="Avg band",annotation_position="top left")
                fig_tr.update_layout(**_lay(
                    f"{sel_col} — {d['Label']}",sel_col,[None,None],
                    (float(fa_[0]),float(fa_[-1]))))
                st.plotly_chart(fig_tr,use_container_width=True)

                st.divider()

                # ── Closed-loop verification ───────────────────────────────
                st.markdown("#### 🔁 Closed-Loop S-Param Reconstruction Verification")
                _rbb_note = (f"已啟用 Rbb = {_rbb_use:.2f} Ω 解嵌入 (Eq. 4.34 / Kumar R_B=Z11-Z12)"
                             if _rbb_use else "Rbb 解嵌入未啟用 → π-exact 可能仍有偏差")
                st.caption(
                    "**三條曲線說明：**  \n"
                    "· **Meas** (實線)：量測/模擬的原始 S 參數  \n"
                    "· **π-exact** (點線)：Kumar (2014) 8-element π-model per-freq 重建  "
                    "（C_BC/C_BE/C_CE/R_BC/R_BE/R_CE/gm/τ + Rbb 解嵌入）  \n"
                    "· **π-avg** (虛線)：頻段平均參數重建  \n\n"
                    f"🔧 **{_rbb_note}**  \n"
                    "⚠️ 若 π-exact 仍偏差 → 可能還有 RE/RC 未解嵌入 → 啟用「Rbb + RE/RC 解嵌入」模式，"
                    "或使用下方 Full Model 擬合。"
                )

                # Build both reconstructions
                Y_sim_avg   = reconstruct_Y(avg_, d["freq_hz"], rbb_val=_rbb_use)
                S_sim_avg   = _y_to_s(Y_sim_avg, 50.)
                Y_sim_exact = reconstruct_Y_perfreq(df_pi, d["freq_hz"], rbb_val=_rbb_use)
                S_sim_exact = _y_to_s(Y_sim_exact, 50.)

                # Error metrics — both inside the averaging band
                mask_e = (fa_>=af1) & (fa_<=af2)
                Sme = d["S"][mask_e]
                ok  = ~np.isnan(Sme).any(axis=(1,2))

                def _err_pct(S_sim_):
                    Sse = S_sim_[mask_e]
                    ok2 = ok & ~np.isnan(Sse).any(axis=(1,2))
                    if not ok2.any(): return np.nan
                    return float(np.max(np.abs(Sse[ok2]-Sme[ok2]))/np.max(np.abs(Sme[ok2]))*100)

                err_avg   = _err_pct(S_sim_avg)
                err_exact = _err_pct(S_sim_exact)

                met1,met2=st.columns(2)
                _rbb_lbl = (f"Rbb={_rbb_use:.1f}Ω 解嵌入後應趨近 0%"
                            if _rbb_use else "啟用 Rbb 解嵌入可大幅改善")
                for col_m, err, label, note in [
                        (met1, err_exact, "π-exact 重建誤差（Rbb+拓撲驗證）",
                         _rbb_lbl),
                        (met2, err_avg,   "π-avg 重建誤差（集總近似）",
                         f"平均頻段 {avg_.get('Freq Range (GHz)','?')} GHz"),
                ]:
                    if np.isfinite(err):
                        clr=("#1e8449" if err<2 else "#b9770e" if err<10 else "#922b21")
                        col_m.markdown(
                            f'<div style="padding:10px 16px;border-radius:8px;'
                            f'border-left:5px solid {clr};background:#f7f9fc;margin-bottom:8px;">'
                            f'<span style="font-size:.78rem;color:#666;">{label}</span><br>'
                            f'<span style="font-size:1.5rem;font-weight:700;color:{clr};">{err:.2f}%</span>'
                            f'<br><span style="font-size:.72rem;color:#888;">{note}</span></div>',
                            unsafe_allow_html=True)

                st.plotly_chart(
                    make_smith_ver(d["S"], S_sim_avg, fa_, vf1, vf2,
                                   d["Label"], vr, S_sim_exact=S_sim_exact,
                                   square_px=980, s_scales=s_scales_pi),
                    use_container_width=False)

                # ── fT_model vs frequency trend ─────────────────────
                _ft_arr = df_pi["fT_model (GHz)"].values
                _fm_arr = df_pi["fmax_model (GHz)"].values
                _ok_ft  = np.isfinite(_ft_arr)
                _ok_fm  = np.isfinite(_fm_arr)
                if _ok_ft.any() or _ok_fm.any():
                    fig_ft = go.Figure()
                    if _ok_ft.any():
                        fig_ft.add_trace(go.Scatter(
                            x=fa_[_ok_ft], y=_ft_arr[_ok_ft], mode="lines",
                            name="fT_model (Kumar Eq.13)",
                            line=dict(color="#2471a3", width=2.5)))
                    if _ok_fm.any():
                        fig_ft.add_trace(go.Scatter(
                            x=fa_[_ok_fm], y=_fm_arr[_ok_fm], mode="lines",
                            name="fmax_model (Kumar Eq.14)",
                            line=dict(color="#d62728", width=2.5, dash="dash")))
                    fig_ft.add_vrect(x0=af1, x1=af2,
                        fillcolor="rgba(255,200,0,0.18)", line_width=0,
                        annotation_text="Avg band", annotation_position="top left")
                    fig_ft.update_layout(**_lay(
                        f"fT / fmax from π-model — {d['Label']}",
                        "Frequency (GHz)", [None,None],
                        (float(fa_[0]), float(fa_[-1]))))
                    st.plotly_chart(fig_ft, use_container_width=True)
                    st.caption(
                        "fT_model = gm/(2pi*(C_BE+C_BC))  [Kumar 2014 Eq.13]  "
                        "| fmax_model = sqrt(fT/(8pi*C_BC*Rbb))  [Kumar 2014 Eq.14]  "
                        "| 曲線平坦 = 參數正確；隨頻率漂移 = 仍有外部寄生"
                    )

                with st.expander("📋 Freq-resolved π-model (averaging band)"):
                    st.dataframe(df_band.round(4),use_container_width=True,hide_index=True)

                # ─────────────────────────────────────────────────────────
                # FULL MODEL EXTRACTION (Xu / UIUC Topology)
                # ─────────────────────────────────────────────────────────
                st.divider()
                st.markdown("#### 🏗 Full Extrinsic+Intrinsic Model Extraction (Xu Topology)")
                st.caption(
                    "根據 UIUC Xu 論文電路圖實作：5節點導納矩陣 + Schur reduction → 2-port Y。  \n"
                    "**15 個參數**：Lb/Lc/Le、Rb/Rc/Ree（外部寄生）、Cbcx/Rbcx（外部 B-C）、"
                    "Cbci/Rbci（本質 B-C）、Cje/Rbe（B-E）、gm/τ（VCCS 轉導）、ro。  \n"
                    "使用 **scipy.least_squares TRF** 有界最佳化擬合量測 S 參數。"
                )

                if not _SCIPY_OK:
                    st.warning("⚠️ scipy 未找到，無法進行全模型擬合。請安裝 scipy (`pip install scipy`)。")
                else:
                    # ── Session-state key for this file's fit result ────
                    _sk = f"fm_result_{n}"
                    if _sk not in st.session_state:
                        st.session_state[_sk] = None

                    # ── Initial parameter inputs ─────────────────────────
                    with st.expander("📐 Initial Parameters & Bounds", expanded=False):
                        st.caption(
                            "預設值來自 Xu 論文電路圖。若上次已完成擬合，可點擊『載入上次結果作為初始值』。"
                        )
                        _prev = st.session_state[_sk]
                        if _prev and st.button("🔄 載入上次擬合結果作為初始值",
                                               key=f"fm_load_{n}"):
                            st.session_state[f"fm_loaded_{n}"] = _prev[0]

                        _loaded = st.session_state.get(f"fm_loaded_{n}", {})

                        st.markdown("**Extrinsic wire elements**")
                        _rc1 = st.columns(6)
                        _p0: dict = {}
                        _plb: dict = {}
                        _pub: dict = {}

                        # Render all 15 params in groups
                        _groups = [
                            ("Extrinsic wire",  ["Lb","Lc","Le","Rb","Rc","Ree"]),
                            ("Extrinsic B-C",   ["Cbcx","Rbcx"]),
                            ("Intrinsic B-C",   ["Cbci","Rbci"]),
                            ("Intrinsic B-E",   ["Cje","Rbe"]),
                            ("Transconductance",["gm","tau","ro"]),
                        ]
                        for _grp_lbl, _grp_keys in _groups:
                            st.markdown(f"**{_grp_lbl}**")
                            _gcols = st.columns(max(len(_grp_keys), 3))
                            for _ci, _k in enumerate(_grp_keys):
                                _meta = next(m for m in _FM_META if m[0]==_k)
                                _, _lbl, _dunit, _def_si, _min_si, _max_si, _dsc = _meta
                                _def_disp = float((_loaded.get(_k, _def_si)) * _dsc)
                                _v = _gcols[_ci].number_input(
                                    f"{_lbl} ({_dunit})",
                                    value=float(f"{_def_disp:.6g}"),
                                    min_value=float(_min_si * _dsc),
                                    max_value=float(_max_si * _dsc * 10),
                                    format="%.5g",
                                    key=f"fm_{_k}_{n}"
                                )
                                _p0[_k]  = float(_v / _dsc)
                                _plb[_k] = float(_min_si)
                                _pub[_k] = float(_max_si)

                    # ── Fit settings end of expander ─────────────────
                        _ffit_cols = st.columns(3)
                        _fm_fmin = _ffit_cols[0].number_input(
                            "Fit freq min (GHz)", value=1.0, min_value=0.01,
                            format="%.3f", key=f"fmf1_{n}")
                        _fm_fmax = _ffit_cols[1].number_input(
                            "Fit freq max (GHz)", value=min(float(fa_[-1]), 150.),
                            min_value=1.0, key=f"fmf2_{n}")
                        _fm_iter = int(_ffit_cols[2].number_input(
                            "Max func-evals", value=3000, min_value=200,
                            step=500, key=f"fmit_{n}"))

                    # ── Run controls (outside expander, always visible) ───
                    _fix_ext = st.checkbox(
                        "⚡ Stage-1: 先固定外部元件（Lb/Lc/Le/Rb/Rc/Ree/Rbcx），只擬合本質元件 → 收斂更快，建議初始值不佳時勾選",
                        value=False, key=f"fmfix_{n}")
                    # ── Run button ────────────────────────────────────────
                    _run_btn = st.button(
                        "▶ Run Full Model Extraction",
                        key=f"fmbtn_{n}",
                        type="primary",
                        help="執行 scipy least_squares (TRF) 擬合。約需 10-60 秒。"
                    )

                    if _run_btn:
                        _mask_fit = (fa_ >= _fm_fmin) & (fa_ <= _fm_fmax)
                        _fhz_fit  = d["freq_hz"][_mask_fit]
                        _S_fit    = d["S"][_mask_fit]
                        if len(_fhz_fit) < 10:
                            st.warning("擬合頻率點不足（< 10 點）。請擴大頻率範圍。")
                        else:
                            # Subsample to max 300 points for speed
                            if len(_fhz_fit) > 300:
                                _idx_ss = np.round(np.linspace(0, len(_fhz_fit)-1, 300)).astype(int)
                                _fhz_fit = _fhz_fit[_idx_ss]
                                _S_fit   = _S_fit[_idx_ss]

                            _prog_msg = ("Stage-1 擬合中（固定外部元件）…"
                                         if _fix_ext else "全模型擬合中，最多 " +
                                         f"{_fm_iter} 次函數評估…")
                            with st.spinner(_prog_msg):
                                if _fix_ext:
                                    # Stage 1: intrinsic-only
                                    _f1, _r1, _ok1 = fit_full_model(
                                        _fhz_fit, _S_fit, _p0, _plb, _pub,
                                        fix_extrinsic=True,
                                        max_iter=_fm_iter//2)
                                    if _f1 is not None:
                                        _p0_s2 = _f1
                                    else:
                                        _p0_s2 = _p0
                                    # Stage 2: all free
                                    _fit, _rms, _ok = fit_full_model(
                                        _fhz_fit, _S_fit, _p0_s2, _plb, _pub,
                                        fix_extrinsic=False,
                                        max_iter=_fm_iter)
                                else:
                                    _fit, _rms, _ok = fit_full_model(
                                        _fhz_fit, _S_fit, _p0, _plb, _pub,
                                        fix_extrinsic=False,
                                        max_iter=_fm_iter)

                            if _fit is not None:
                                st.session_state[_sk] = (_fit, _rms, _ok)
                                if _ok:
                                    st.success(f"✅ 擬合收斂！RMS 誤差 = {_rms:.3f}%")
                                else:
                                    st.warning(f"⚠️ 未完全收斂（已達函數評估上限），RMS = {_rms:.3f}%。"
                                               "可嘗試：增加 Max func-evals 或調整初始值。")
                            else:
                                st.error("❌ 擬合失敗。請檢查 scipy 安裝或參數範圍。")

                    # ── Show results if available ─────────────────────────
                    _res = st.session_state.get(_sk)
                    if _res is not None:
                        _fit, _rms, _ok = _res

                        st.markdown("##### 📊 Extracted Parameters")
                        _tbl_rows = []
                        for _k in _FM_KEYS:
                            _meta = next(m for m in _FM_META if m[0]==_k)
                            _, _lbl, _dunit, _def_si, *_ , _dsc = _meta
                            _init_si = st.session_state.get(f"fm_{_k}_{n}", _def_si / _dsc * _dsc)
                            # Re-read from widget if possible
                            _widget_val = st.session_state.get(f"fm_{_k}_{n}")
                            if _widget_val is not None:
                                _init_disp = float(_widget_val)
                            else:
                                _init_disp = float(_def_si * _dsc)
                            _fit_disp = float(_fit.get(_k, _def_si)) * _dsc
                            _tbl_rows.append({
                                "Parameter": _lbl,
                                "Unit": _dunit,
                                "Initial": round(_init_disp, 5),
                                "Fitted":  round(_fit_disp, 5),
                                "Δ (%)":   round((_fit_disp - _init_disp) /
                                                  (abs(_init_disp)+1e-30)*100., 2)
                            })
                        _tbl_df = pd.DataFrame(_tbl_rows)
                        st.dataframe(
                            _tbl_df.style.background_gradient(
                                subset=["Δ (%)"], cmap="RdYlGn_r", vmin=-50, vmax=50),
                            use_container_width=True, hide_index=True)

                        # ── Smith chart: meas vs full-model ──────────────
                        st.markdown("##### 🔁 Full-Model S-Param Verification (Smith Chart)")
                        _Y_full = build_Y2port_full_model(_fit, d["freq_hz"])
                        _S_full = _y_to_s(_Y_full, 50.)

                        # Reuse make_smith_ver: S_sim_avg = full-model, S_sim_exact = pi-exact
                        st.plotly_chart(
                            make_smith_ver(d["S"], _S_full, fa_, vf1, vf2,
                                           d["Label"] + " · Full Model",
                                           vr, S_sim_exact=S_sim_exact,
                                           square_px=980, s_scales=s_scales_pi),
                            use_container_width=False)

                        # Override legend labels via layout annotation
                        st.caption(
                            "**曲線說明（本圖）：**  \n"
                            "· **Meas** (實線)：量測 S 參數  \n"
                            "· **π-avg** → 重標為 **Full-Model fitted** (虛線)：Xu 15元素模型擬合結果  \n"
                            "· **π-exact** (點線)：8元素本質模型 per-freq 重建（供對比）"
                        )

                        # Per-S-param RMS error
                        _err_cols = st.columns(4)
                        for _ci, (_r, _c, _nm) in enumerate(
                                [(0,0,'S11'),(0,1,'S12'),(1,0,'S21'),(1,1,'S22')]):
                            _e = float(np.sqrt(np.mean(np.abs(
                                _S_full[:,_r,_c]-d["S"][:,_r,_c])**2)) /
                                (np.sqrt(np.mean(np.abs(d["S"][:,_r,_c])**2))+1e-30)*100)
                            _clr = "#1e8449" if _e < 3 else "#b9770e" if _e < 10 else "#922b21"
                            _err_cols[_ci].markdown(
                                f'<div style="padding:8px 12px;border-radius:6px;'
                                f'border-left:4px solid {_clr};background:#f7f9fc;">'
                                f'<span style="font-size:.75rem;color:#666;">{_nm} RMS 誤差</span><br>'
                                f'<span style="font-size:1.2rem;font-weight:700;color:{_clr};">'
                                f'{_e:.2f}%</span></div>',
                                unsafe_allow_html=True)

                        with st.expander("📋 Full fitted parameter dict (SI units)"):
                            st.json({k: float(v) for k, v in _fit.items()})

# ──────────────────────────────────────────────────────────────
# TAB 4 — SUMMARY
# ──────────────────────────────────────────────────────────────
with tab_sum:
    with st.expander("⚙️ Extraction Settings",expanded=False):
        s1,s2=st.columns(2)
        fmin_sm=s1.number_input("Freq Min (GHz)",value=0.4,min_value=0.0001,format="%.4f",key="sm_f1")
        fmax_sm=s2.number_input("Freq Max (GHz)",value=300.,min_value=0.1,key="sm_f2")
        st.caption("fT/fmax: single-pole fitting (auto-detect rolloff)")
        fit_win_sm = st.number_input("文獻比較窗口 (GHz)", value=50.0, min_value=5.0, step=5.0,
                                      key="sm_fit_win",
                                      help="同時用此窗口內的數據做 single-pole fit，產生文獻可比較值")

    rows=[]
    for k,d in all_data.items():
        fa_=d["df_metrics"]["Freq (GHz)"].values; dfm=d["df_metrics"]
        fTc,fTp,fTm=extract_limit(fa_,dfm["|h21|² (dB)"].values,   dfm["fT Plateau (GHz)"].values,    fmin_sm,fmax_sm)
        fUc,fUp,fUm=extract_limit(fa_,dfm["Mason U (dB)"].values,   dfm["fmax U Plateau (GHz)"].values, fmin_sm,fmax_sm)
        fMc,fMp,fMm=extract_limit(fa_,dfm["MAG/MSG (dB)"].values,   dfm["fmax MAG Plateau (GHz)"].values,fmin_sm,fmax_sm)

        # Dual extraction
        fT_full,fT_full_m, fT_win,fT_win_m, _ = extract_limit_dual(
            fa_, dfm["|h21|² (dB)"].values, dfm["fT Plateau (GHz)"].values,
            fmin_sm, fmax_sm, fit_win_sm)
        fU_full,fU_full_m, fU_win,fU_win_m, _ = extract_limit_dual(
            fa_, dfm["Mason U (dB)"].values, dfm["fmax U Plateau (GHz)"].values,
            fmin_sm, fmax_sm, fit_win_sm)

        # Extract DC bias info from CSV metadata
        meta = d.get("meta", {})
        Ic = meta.get("Collector Current", np.nan)
        Ib = meta.get("Base Current", np.nan)
        Vce_v = d.get("Vce (V)", np.nan)
        beta_dc = abs(Ic/Ib) if Ib is not None and Ic is not None and abs(Ib)>1e-15 else np.nan

        # gm from π-model (low-freq average)
        df_pi = d.get("df_pi")
        gm_avg = np.nan
        if df_pi is not None and "gm (mS)" in df_pi.columns:
            gm_vals = df_pi["gm (mS)"].iloc[:5]  # low-freq gm
            gm_avg = float(gm_vals.mean()) if len(gm_vals) > 0 else np.nan

        # Best fT/fmax: prefer cross, then plateau
        fT_best = fTc if np.isfinite(fTc) else fTp
        fmax_U_best = fUc if np.isfinite(fUc) else fUp
        fmax_MAG_best = fMc if np.isfinite(fMc) else fMp

        rows.append({
            "File": k,
            "Vce (V)": Vce_v,
            "Ic (mA)": round(abs(Ic)*1e3, 4) if np.isfinite(Ic) else None,
            "Ib (µA)": round(abs(Ib)*1e6, 3) if np.isfinite(Ib) else None,
            "β": round(beta_dc, 1) if np.isfinite(beta_dc) else None,
            "gm (mS)": round(gm_avg, 2) if np.isfinite(gm_avg) else None,
            "fT Full (GHz)": round(fT_full, 2) if np.isfinite(fT_full) else None,
            f"fT ≤{fit_win_sm:.0f}G (GHz)": round(fT_win, 2) if np.isfinite(fT_win) else None,
            "fmax Full (GHz)": round(fU_full, 2) if np.isfinite(fU_full) else None,
            f"fmax ≤{fit_win_sm:.0f}G (GHz)": round(fU_win, 2) if np.isfinite(fU_win) else None,
            "fT (GHz)": round(fT_best, 2) if np.isfinite(fT_best) else None,
            "fT Method": fTm,
            "fmax U (GHz)": round(fmax_U_best, 2) if np.isfinite(fmax_U_best) else None,
            "fmax U Method": fUm,
            "fmax MAG (GHz)": round(fmax_MAG_best, 2) if np.isfinite(fmax_MAG_best) else None,
        })
    sum_df=pd.DataFrame(rows)
    num_c=[c for c in sum_df.columns if any(k in c for k in ["GHz","Vce","mA","µA","mS","β"])]
    fmt={c:"{:.4f}" for c in num_c}
    for c in ["Ib (µA)"]: 
        if c in fmt: fmt[c]="{:.3f}"
    for c in ["β"]:
        if c in fmt: fmt[c]="{:.1f}"
    st.dataframe(sum_df.style.format(fmt,na_rep="—"),use_container_width=True,hide_index=True)

    # ══════════════════════════════════════════════════════════════
    # BIAS-DEPENDENT RF PLOTS
    # ══════════════════════════════════════════════════════════════
    st.divider()
    st.markdown("### 📈 Bias-Dependent RF Characteristics")

    # Build arrays for plotting
    Ic_arr = np.array([r.get("Ic (mA)") or np.nan for r in rows])
    fT_arr = np.array([r.get("fT Full (GHz)") or np.nan for r in rows])
    fmax_U_arr = np.array([r.get("fmax Full (GHz)") or np.nan for r in rows])
    fT_win_arr = np.array([r.get(f"fT ≤{fit_win_sm:.0f}G (GHz)") or np.nan for r in rows])
    fmax_win_arr = np.array([r.get(f"fmax ≤{fit_win_sm:.0f}G (GHz)") or np.nan for r in rows])
    fmax_MAG_arr = np.array([r.get("fmax MAG (GHz)") or np.nan for r in rows])
    beta_arr = np.array([r.get("β") or np.nan for r in rows])
    gm_arr = np.array([r.get("gm (mS)") or np.nan for r in rows])
    labels = [Path(r["File"]).stem for r in rows]

    # Need at least 2 valid Ic points for meaningful plots
    valid_ic = np.isfinite(Ic_arr) & (Ic_arr > 0)
    n_valid = valid_ic.sum()

    if n_valid < 2:
        st.warning(
            f"⚠️ 僅有 {n_valid} 個檔案含有效 Ic 數據。"
            "需要 ≥ 2 個不同 bias point 的 RF CSV 才能畫 bias-dependent 圖。\n\n"
            "**方法：** 在 Atlas 中對多個 bias point 各跑一次 AC 分析，產生多個 rf_XX.csv，全部上傳。"
        )
    else:
        # Sort by Ic for clean line plots
        sort_idx = np.argsort(Ic_arr)
        Ic_s = Ic_arr[sort_idx]
        fT_s = fT_arr[sort_idx]
        fmax_U_s = fmax_U_arr[sort_idx]
        fT_win_s = fT_win_arr[sort_idx]
        fmax_win_s = fmax_win_arr[sort_idx]
        fmax_MAG_s = fmax_MAG_arr[sort_idx]
        beta_s = beta_arr[sort_idx]
        gm_s = gm_arr[sort_idx]
        lbl_s = [labels[i] for i in sort_idx]

        hov_base = [f"{lbl_s[i]}<br>Ic={Ic_s[i]:.3f}mA" for i in range(len(Ic_s))]

        # ── Emitter area for Jc calculation (user input) ──
        st.markdown("**Emitter Area** (用於計算 Jc)")
        ea1, ea2 = st.columns(2)
        emitter_width = ea1.number_input("Emitter Width (µm)", value=0.3, min_value=0.01,
                                          format="%.2f", key="em_w",
                                          help="你的 HBT emitter stripe 寬度")
        emitter_length = ea2.number_input("Emitter Length (µm)", value=5.0, min_value=0.01,
                                           format="%.2f", key="em_l",
                                           help="你的 HBT emitter stripe 長度（= Atlas width）")
        A_em = emitter_width * emitter_length  # µm²
        Jc_s = Ic_s / A_em * 1e3  # mA/µm² → kA/cm²  (1 mA/µm² = 1e3 kA/cm²... no)
        # Ic in mA, A_em in µm² → Jc = Ic(mA) / A_em(µm²) * 1e-3(A) / 1e-8(cm²) = Ic/A_em * 1e5 A/cm² = Ic/A_em * 100 kA/cm²
        Jc_s = Ic_s * 1e-3 / (A_em * 1e-8)  * 1e-3  # → kA/cm²

        st.divider()

        # ── Plot: fT & fmax vs Ic ──
        st.markdown("#### fT & fmax vs Ic")
        fig_ft = go.Figure()
        mk = dict(size=8, line=dict(width=1.5, color='white'))
        if np.any(np.isfinite(fT_s)):
            fig_ft.add_trace(go.Scatter(
                x=Ic_s, y=fT_s, mode='lines+markers', name='fT',
                line=dict(color='#1f77b4', width=2.5),
                marker=dict(**mk, color='#1f77b4', symbol='circle'),
                text=[f"{h}<br>fT={fT_s[i]:.1f}GHz" for i,h in enumerate(hov_base)],
                hoverinfo='text'))
        if np.any(np.isfinite(fmax_U_s)):
            fig_ft.add_trace(go.Scatter(
                x=Ic_s, y=fmax_U_s, mode='lines+markers', name='fmax (Mason U)',
                line=dict(color='#d62728', width=2.5),
                marker=dict(**mk, color='#d62728', symbol='diamond'),
                text=[f"{h}<br>fmax(U)={fmax_U_s[i]:.1f}GHz" for i,h in enumerate(hov_base)],
                hoverinfo='text'))
        # Window fit values: dashed lines with open markers
        if np.any(np.isfinite(fT_win_s)):
            fig_ft.add_trace(go.Scatter(
                x=Ic_s, y=fT_win_s, mode='lines+markers',
                name=f'fT (≤{fit_win_sm:.0f}GHz fit)',
                line=dict(color='#1f77b4', width=2, dash='dot'),
                marker=dict(size=6, color='#1f77b4', symbol='circle-open', line=dict(width=1.5)),
                text=[f"{h}<br>fT_win={fT_win_s[i]:.1f}GHz" for i,h in enumerate(hov_base)],
                hoverinfo='text'))
        if np.any(np.isfinite(fmax_win_s)):
            fig_ft.add_trace(go.Scatter(
                x=Ic_s, y=fmax_win_s, mode='lines+markers',
                name=f'fmax (≤{fit_win_sm:.0f}GHz fit)',
                line=dict(color='#d62728', width=2, dash='dot'),
                marker=dict(size=6, color='#d62728', symbol='diamond-open', line=dict(width=1.5)),
                text=[f"{h}<br>fmax_win={fmax_win_s[i]:.1f}GHz" for i,h in enumerate(hov_base)],
                hoverinfo='text'))
        fig_ft.update_layout(
            title="fT & fmax vs Collector Current (solid=full range, dotted=≤" + f"{fit_win_sm:.0f}GHz fit)",
            xaxis_title="Ic (mA)", yaxis_title="Frequency (GHz)",
            xaxis=dict(type="log", showgrid=True, gridcolor="#ebebeb"),
            yaxis=dict(showgrid=True, gridcolor="#ebebeb", rangemode="tozero"),
            plot_bgcolor="white", paper_bgcolor="white", height=500,
            legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.9)"),
            hovermode="closest")
        st.plotly_chart(fig_ft, use_container_width=True)

        # ── Plot: fT & fmax vs Jc ──
        st.markdown("#### fT & fmax vs Jc")
        fig_jc = go.Figure()
        if np.any(np.isfinite(fT_s)):
            fig_jc.add_trace(go.Scatter(
                x=Jc_s, y=fT_s, mode='lines+markers', name='fT',
                line=dict(color='#1f77b4', width=2.5),
                marker=dict(**mk, color='#1f77b4', symbol='circle')))
        if np.any(np.isfinite(fmax_U_s)):
            fig_jc.add_trace(go.Scatter(
                x=Jc_s, y=fmax_U_s, mode='lines+markers', name='fmax (Mason U)',
                line=dict(color='#d62728', width=2.5),
                marker=dict(**mk, color='#d62728', symbol='diamond')))
        if np.any(np.isfinite(fmax_MAG_s)):
            fig_jc.add_trace(go.Scatter(
                x=Jc_s, y=fmax_MAG_s, mode='lines+markers', name='fmax (MAG)',
                line=dict(color='#2ca02c', width=2, dash='dash'),
                marker=dict(**mk, color='#2ca02c', symbol='triangle-up')))
        fig_jc.update_layout(
            title=f"fT & fmax vs Current Density (A_em={emitter_width}×{emitter_length} µm²)",
            xaxis_title="Jc (kA/cm²)", yaxis_title="Frequency (GHz)",
            xaxis=dict(type="log", showgrid=True, gridcolor="#ebebeb"),
            yaxis=dict(showgrid=True, gridcolor="#ebebeb", rangemode="tozero"),
            plot_bgcolor="white", paper_bgcolor="white", height=500,
            legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.9)"),
            hovermode="closest")
        st.plotly_chart(fig_jc, use_container_width=True)

        # ── Plot: β vs Ic ──
        st.markdown("#### β (DC Current Gain) vs Ic")
        fig_beta = go.Figure()
        if np.any(np.isfinite(beta_s)):
            fig_beta.add_trace(go.Scatter(
                x=Ic_s, y=beta_s, mode='lines+markers', name='β = Ic/Ib',
                line=dict(color='#9467bd', width=2.5),
                marker=dict(**mk, color='#9467bd', symbol='circle'),
                text=[f"{h}<br>β={beta_s[i]:.1f}" for i,h in enumerate(hov_base)],
                hoverinfo='text'))
        fig_beta.update_layout(
            title="DC Current Gain vs Collector Current",
            xaxis_title="Ic (mA)", yaxis_title="β",
            xaxis=dict(type="log", showgrid=True, gridcolor="#ebebeb"),
            yaxis=dict(showgrid=True, gridcolor="#ebebeb", rangemode="tozero"),
            plot_bgcolor="white", paper_bgcolor="white", height=450,
            hovermode="closest")
        st.plotly_chart(fig_beta, use_container_width=True)

        # ── Plot: gm vs Ic ──
        st.markdown("#### gm (Transconductance) vs Ic")
        fig_gm = go.Figure()
        if np.any(np.isfinite(gm_s)):
            fig_gm.add_trace(go.Scatter(
                x=Ic_s, y=gm_s, mode='lines+markers', name='gm (extracted)',
                line=dict(color='#ff7f0e', width=2.5),
                marker=dict(**mk, color='#ff7f0e', symbol='circle'),
                text=[f"{h}<br>gm={gm_s[i]:.2f}mS" for i,h in enumerate(hov_base)],
                hoverinfo='text'))
            # Ideal gm = Ic/Vt (Vt=26mV at 300K)
            Ic_ideal = np.logspace(np.log10(max(Ic_s[np.isfinite(Ic_s)].min(),0.001)),
                                    np.log10(Ic_s[np.isfinite(Ic_s)].max()), 100)
            gm_ideal = Ic_ideal / 0.026  # mA/26mV = mS
            fig_gm.add_trace(go.Scatter(
                x=Ic_ideal, y=gm_ideal, mode='lines', name='Ideal gm = Ic/Vt',
                line=dict(color='gray', width=1.5, dash='dash')))
        fig_gm.update_layout(
            title="Transconductance vs Collector Current",
            xaxis_title="Ic (mA)", yaxis_title="gm (mS)",
            xaxis=dict(type="log", showgrid=True, gridcolor="#ebebeb"),
            yaxis=dict(type="log", showgrid=True, gridcolor="#ebebeb"),
            plot_bgcolor="white", paper_bgcolor="white", height=450,
            legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.9)"),
            hovermode="closest")
        st.plotly_chart(fig_gm, use_container_width=True)

        # ── Plot: fT × fmax vs Ic ──
        st.markdown("#### fT × fmax vs Ic (Figure of Merit)")
        fig_fom = go.Figure()
        fom_U = fT_s * fmax_U_s
        fom_MAG = fT_s * fmax_MAG_s
        if np.any(np.isfinite(fom_U)):
            fig_fom.add_trace(go.Scatter(
                x=Ic_s, y=fom_U*1e-3, mode='lines+markers', name='fT × fmax(U)',
                line=dict(color='#e377c2', width=2.5),
                marker=dict(**mk, color='#e377c2', symbol='circle')))
        if np.any(np.isfinite(fom_MAG)):
            fig_fom.add_trace(go.Scatter(
                x=Ic_s, y=fom_MAG*1e-3, mode='lines+markers', name='fT × fmax(MAG)',
                line=dict(color='#8c564b', width=2, dash='dash'),
                marker=dict(**mk, color='#8c564b', symbol='diamond')))
        fig_fom.update_layout(
            title="Figure of Merit: fT × fmax vs Ic",
            xaxis_title="Ic (mA)", yaxis_title="fT × fmax (THz × GHz)",
            xaxis=dict(type="log", showgrid=True, gridcolor="#ebebeb"),
            yaxis=dict(showgrid=True, gridcolor="#ebebeb", rangemode="tozero"),
            plot_bgcolor="white", paper_bgcolor="white", height=450,
            legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.9)"),
            hovermode="closest")
        st.plotly_chart(fig_fom, use_container_width=True)


    st.divider()
    st.markdown("#### 📥 Downloads")

    # ── S2P Converter ──
    st.markdown("**S2P Converter**")
    if any(d.get("deembed_applied") for d in all_data.values()):
        st.caption("📌 輸出為 de-embedded 數據（已扣除 contact resistance）")
    conv_file=st.selectbox("Choose file for conversion", list(all_data.keys()), key="conv_file")
    cfile=all_data[conv_file]
    st.download_button("⬇️ Download as S2P",
                       data=export_s2p_bytes(cfile["freq_hz"], cfile["S"], cfile.get("z0",50.0)),
                       file_name=f"{Path(conv_file).stem}.s2p",
                       mime="text/plain",use_container_width=True)

    # ── Excel Download ──
    st.markdown("**Excel Export**")
    date=datetime.now().strftime("%Y%m%d")
    excel_name = st.text_input("Excel 檔名", value=f"IOED_RF_{date}",
                                key="excel_name", help="不需要加 .xlsx 副檔名")

    def build_excel_v2(sum_df, all_data, rows):
        buf=io.BytesIO()
        with pd.ExcelWriter(buf,engine="openpyxl") as w:
            sum_df.to_excel(w,sheet_name="Summary",index=False)
            bias_cols = [c for c in sum_df.columns]
            sum_df.to_excel(w,sheet_name="Bias_Dependent",index=False)
            for k,d in all_data.items():
                base=re.sub(r'[:\\/*?\[\]]','_',Path(k).stem)[:28]
                d["df_metrics"].to_excel(w,sheet_name=base,index=False)
                d["df_pi"].to_excel(w,sheet_name=base[:24]+"_pi",index=False)
        return buf.getvalue()

    st.download_button("📥 Download Excel",
                       data=build_excel_v2(sum_df,all_data,rows),
                       file_name=f"{excel_name}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       use_container_width=True)

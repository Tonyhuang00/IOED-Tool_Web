import io, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 初始化專屬 Key
if "gummel_uploader_key" not in st.session_state:
    st.session_state["gummel_uploader_key"] = 0

st.title("📈 IOED Sim Gummel Plot Analyzer")
st.caption("""
**Core Features:** Real-time Ideality Factor (n) Calculation & Target Calibration  

**📝 Changelog (版本紀錄):**
* **v2.0:** Added **Family Curve Bias** tab — interpolates Vbase from Gummel Ib data for voltage-driven family curve sweeps. Customizable step count & gap. Two-stage ramp (coarse 0.05V → fine 0.002V). Vbase rounded to 3 decimals. Auto-generates Atlas deck snippet. Included in Excel/CSV downloads.
* **v1.9:** Implemented Dynamic Data Slicing. Auto-scale Y-axis now strictly binds to the selected X-axis range, eliminating out-of-bound numerical artifacts (e.g., Beta exploding to 5000+).
""")

UIUC_CSV_STRING = """gummel_Ib,,gummel_Ic,,gummel_beta,
Vbase,Ib,Vbase,Ic,Vbase,Beta
0.205546687,4.12E-11,0.205208673,2.38E-10,0.200406342,7.052367946
0.212162102,6.73E-11,0.226503551,2.62E-10,0.201127677,6.652983857
0.219405258,9.35E-11,0.238672053,2.98E-10,0.202187534,6.259452708
0.224161074,1.26E-10,0.245770346,3.38E-10,0.202225499,5.907089877
0.232080781,1.66E-10,0.265037141,2.72E-10,0.202601985,5.560579986
0.244756304,2.39E-10,0.277205643,2.76E-10,0.205367104,5.078859262
0.255910764,3.29E-10,0.289374145,2.96E-10,0.206832554,4.068673385
0.267177896,4.28E-10,0.301542647,2.98E-10,0.207173154,4.50490953
0.276867629,6.48E-10,0.313711149,3.63E-10,0.210169137,3.679204267
0.284593662,9.54E-10,0.325879651,4.14E-10,0.211668791,3.250914787
0.293430312,1.37E-09,0.338048153,3.65E-10,0.214211266,2.737103077
0.305598814,1.83E-09,0.350216655,3.72E-10,0.216523336,2.263239002
0.317767316,2.50E-09,0.362385157,4.26E-10,0.224030233,1.840457982
0.329935818,3.43E-09,0.374553659,4.47E-10,0.230471202,1.417878933
0.34210432,4.86E-09,0.38621514,5.55E-10,0.238347791,1.161329046
0.354272822,6.89E-09,0.392806412,8.02E-10,0.248534849,1.045337016
0.366441324,9.39E-09,0.404974914,1.14E-09,0.257161092,0.726143483
0.378609826,1.25E-08,0.417143416,1.49E-09,0.26603195,0.534904116
0.389764286,1.69E-08,0.429746507,2.07E-09,0.270995334,0.335782677
0.400918746,2.28E-08,0.438776308,2.95E-09,0.28118021,0.240051509
0.413087248,3.05E-08,0.447564671,3.94E-09,0.291365465,0.140796713
0.42525575,4.04E-08,0.457705089,5.49E-09,0.301544646,0.097919969
0.437424252,5.39E-08,0.467845507,7.86E-09,0.311722308,0.069137739
0.449592754,7.28E-08,0.476971884,1.04E-08,0.321897692,0.061497279
0.461761256,9.58E-08,0.48609826,1.49E-08,0.332075355,0.032715049
0.473929758,1.27E-07,0.495224637,2.09E-08,0.34224998,0.032121846
0.48609826,1.69E-07,0.504351013,2.99E-08,0.352425364,0.024481386
0.498266762,2.25E-07,0.51347739,4.18E-08,0.362603026,-0.004300844
0.510435264,2.99E-07,0.522603766,6.09E-08,0.372777651,-0.004894047
0.522603766,3.98E-07,0.532406171,8.38E-08,0.382952276,-0.005487251
0.534772268,5.22E-07,0.541870561,1.27E-07,0.393125383,0.008014059
0.54694077,6.90E-07,0.552010979,1.79E-07,0.403298489,0.021515369
0.559109272,9.04E-07,0.562658419,2.61E-07,0.413476152,-0.007266861
0.571277774,1.20E-06,0.570263732,3.76E-07,0.423650777,-0.007860064
0.583446276,1.55E-06,0.580911172,5.33E-07,0.433825402,-0.008453268
0.595614778,2.01E-06,0.588854499,7.44E-07,0.44399699,0.019142555
0.60778328,2.64E-06,0.597497999,1.05E-06,0.454170855,0.025596608
0.619951782,3.47E-06,0.606769238,1.45E-06,0.46434548,0.025003405
0.632120284,4.55E-06,0.615895615,2.14E-06,0.474520105,0.024410202
0.644288786,5.68E-06,0.626036033,2.76E-06,0.48469473,0.023816998
0.656457288,7.41E-06,0.631106242,3.39E-06,0.494869356,0.023223795
0.66862579,9.71E-06,0.640268456,5.20E-06,0.505043981,0.022630591
0.680794292,1.26E-05,0.644295302,7.07E-06,0.515214809,0.057273671
0.692962794,1.67E-05,0.650373037,9.22E-06,0.525389434,0.056680468
0.705131296,2.17E-05,0.662976128,1.28E-05,0.535557226,0.119512574
0.717299798,2.82E-05,0.672247368,1.74E-05,0.545731091,0.125966627
0.7294683,3.68E-05,0.680649429,2.39E-05,0.55590192,0.160609707
0.741636802,4.80E-05,0.688230599,3.29E-05,0.56607123,0.2093473
0.753805304,6.13E-05,0.696765451,4.47E-05,0.57623978,0.265132149
0.765973806,7.92E-05,0.70672479,6.38E-05,0.586404535,0.356153282
0.778142308,1.01E-04,0.716060414,9.21E-05,0.596573085,0.411938132
0.79031081,1.29E-04,0.727295353,1.37E-04,0.606736321,0.517053778
0.802479312,1.63E-04,0.737146045,1.94E-04,0.616898797,0.62921668
0.814647814,2.04E-04,0.746707011,2.87E-04,0.627059755,0.755474096
0.826816316,2.54E-04,0.757607961,4.08E-04,0.637559613,0.884060824
0.838984818,3.09E-04,0.767857027,5.82E-04,0.647959498,1.041177812
0.85115332,3.68E-04,0.777128266,7.98E-04,0.657820098,1.257053011
0.863321822,4.38E-04,0.786254643,1.13E-03,0.667681566,1.464874202
0.875490324,5.20E-04,0.798085131,1.57E-03,0.677823161,1.770836662
0.887658826,6.22E-04,0.808563563,2.26E-03,0.68796134,2.108511777
0.898813286,7.31E-04,0.820732065,3.16E-03,0.697646849,2.450519913
,,0.832900567,4.25E-03,0.706196887,2.838596886
,,0.845069069,5.73E-03,0.714291029,3.260957721
,,0.857237571,7.26E-03,0.722094332,3.684593944
,,0.869406073,9.03E-03,0.728444705,4.102861675
,,0.881574575,1.09E-02,0.734504555,4.519468437
,,0.893743077,1.26E-02,0.741144636,4.94694777
,,0.901855411,1.31E-02,0.74692237,5.283870364
,,,,0.750999723,5.66367874
,,,,0.755029709,6.033422432
,,,,0.759060329,6.39729341
,,,,0.763087785,6.790527957
,,,,0.767113342,7.201380646
,,,,0.7711389,7.612233334
,,,,0.775506775,7.99370268
,,,,0.779143282,8.418001313
,,,,0.782873616,8.870818907
,,,,0.786896281,9.308518287
,,,,0.791364342,9.80939289
,,,,0.795431766,10.35629349
,,,,0.798943124,10.81793286
,,,,0.802957857,11.32925091
,,,,0.807418483,11.8991299
,,,,0.810980204,12.41795503
,,,,0.814997988,12.9009582
,,,,0.819008043,13.45569238
,,,,0.823019047,14.00161748
,,,,0.82703195,14.52992445
,,,,0.831041056,15.0934677
,,,,0.83505206,15.63939281
,,,,0.839361225,16.11608684
,,,,0.841738016,16.54045891
,,,,0.845594946,16.94251242
,,,,0.849100337,17.4595231
,,,,0.853989689,17.94750918
,,,,0.857485906,18.32483458
,,,,0.863498299,18.73221561
,,,,0.870974877,19.04051473
,,,,0.881150768,19.0281761
,,,,0.887637407,18.63139222
,,,,0.892089122,18.23472698
,,,,0.896200101,17.85276329
,,,,0.899277799,17.61767678"""


@st.cache_data
def load_uiuc_ref():
    df = pd.read_csv(io.StringIO(UIUC_CSV_STRING), skiprows=1)
    return {
        "V_Ib": df.iloc[:, 0].dropna(), "Ib": df.iloc[:, 1].dropna(),
        "V_Ic": df.iloc[:, 2].dropna(), "Ic": df.iloc[:, 3].dropna(),
        "V_Beta": df.iloc[:, 4].dropna(), "Beta": df.iloc[:, 5].dropna()
    }


uiuc_ref = load_uiuc_ref()


def normalize_cols(cols): return [str(c).strip().lower().replace(" ", "").replace("-", "") for c in cols]


def pick_column_by_keywords(df, include_kw, exclude_kw=None):
    if exclude_kw is None: exclude_kw = []
    candidates = [col for col in df.columns if
                  all(k in col.lower() for k in include_kw) and not any(k in col.lower() for k in exclude_kw)]
    return candidates[0] if candidates else None


def load_and_standardize(content, add_noise, ic_noise, ib_noise):
    df_raw = pd.read_csv(io.StringIO(content), comment="#").dropna(how="all")
    df_raw.columns = normalize_cols(df_raw.columns)
    vbase_col = pick_column_by_keywords(df_raw, ["base", "volt"]) or pick_column_by_keywords(df_raw, ["volt"],
                                                                                             ["collector"])
    ib_col, ic_col = pick_column_by_keywords(df_raw, ["base", "curr"]), pick_column_by_keywords(df_raw,
                                                                                                ["collector", "curr"])
    if not all([vbase_col, ib_col, ic_col]): raise ValueError("Missing Vbase/Ib/Ic columns.")
    out = pd.DataFrame({"Vbase": pd.to_numeric(df_raw[vbase_col], errors="coerce"),
                        "Ib_abs": pd.to_numeric(df_raw[ib_col], errors="coerce").abs(),
                        "Ic_abs": pd.to_numeric(df_raw[ic_col], errors="coerce").abs()}).dropna()

    if add_noise:
        out["Ic_abs"] = out["Ic_abs"] + ic_noise
        out["Ib_abs"] = out["Ib_abs"] + ib_noise

    out["Beta"] = (out["Ic_abs"] / out["Ib_abs"]).replace([np.inf, -np.inf], np.nan)
    out["Beta"] = out["Beta"].where((out["Vbase"] >= 0.0) & (out["Beta"] > 0), np.nan)
    return out.reset_index(drop=True)


def calc_ideality(v_array, i_array, v_min, v_max, vt):
    mask = (v_array >= v_min) & (v_array <= v_max)
    v_fit, i_fit = v_array[mask], i_array[mask]
    if len(v_fit) < 2: return np.nan
    # filter out zero/negative currents for log
    pos = i_fit > 0
    v_fit, i_fit = v_fit[pos], i_fit[pos]
    if len(v_fit) < 2: return np.nan
    slope, _ = np.polyfit(v_fit, np.log(i_fit), 1)
    return 1.0 / (vt * slope) if slope and not np.isnan(slope) else np.nan


def extract_metrics(df, n_vmin, n_vmax, Vt):
    n_ic = calc_ideality(df["Vbase"], df["Ic_abs"], n_vmin, n_vmax, Vt)
    n_ib = calc_ideality(df["Vbase"], df["Ib_abs"], n_vmin, n_vmax, Vt)
    max_ic = df["Ic_abs"].max()
    max_ib = df["Ib_abs"].max()
    df_beta = df[df["Vbase"] >= 0.45]
    max_beta = df_beta["Beta"].max() if not df_beta.empty else np.nan
    idx_beta = df_beta["Beta"].idxmax() if not df_beta.empty else None
    v_peak_beta = df.loc[idx_beta, "Vbase"] if pd.notna(idx_beta) else np.nan
    cond = df["Ic_abs"] >= 1e-9
    v_turn_on = df.loc[cond, "Vbase"].iloc[0] if cond.any() else np.nan
    return n_ic, n_ib, max_ic, max_ib, max_beta, v_peak_beta, v_turn_on


UIUC_COLOR = "#000000"
SIM_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
              "#17becf"]
LINE_IC = "solid"
LINE_IB = "dash"
LINE_BETA = "dashdot"

with st.sidebar:
    st.markdown("## ⚙️ Settings / 參數設定")

    st.markdown("#### 🔊 Noise Floor / 量測底噪")
    add_noise = st.checkbox("Inject Noise / 啟用模擬底噪", value=True)
    ic_noise = st.number_input("Ic Noise Limit (A)", value=2.34e-10, format="%.2e")
    ib_noise = st.number_input("Ib Noise Limit (A)", value=2.34e-10, format="%.2e")

    st.markdown("#### 📏 Plot Scale & Axis / 圖表範圍控制")
    y_scale = st.radio("Current Y-Axis Scale / 電流軸刻度", ["Log", "Linear"], index=0)

    col_x1, col_x2 = st.columns(2)
    x_min = col_x1.number_input("X Min (V)", value=0.2, step=0.1)
    x_max = col_x2.number_input("X Max (V)", value=0.9, step=0.1)

    st.divider()
    auto_y = st.checkbox("Auto Scale Y-Axes / Y軸自動縮放", value=True)

    cur_ymin = st.number_input("Current Min (A)", value=1e-12, format="%.1e", disabled=auto_y)
    cur_ymax = st.number_input("Current Max (A)", value=1e-2, format="%.1e", disabled=auto_y)

    col_by1, col_by2 = st.columns(2)
    beta_ymin = col_by1.number_input("Beta Min", value=0.0, step=1.0, disabled=auto_y)
    beta_ymax = col_by2.number_input("Beta Max", value=25.0, step=1.0, disabled=auto_y)

    st.markdown("#### 🔬 Physics Params / 物理參數")
    col_nv1, col_nv2 = st.columns(2)
    n_vmin = col_nv1.number_input("n Calc V_min (V) / 理想因子電壓下限", value=0.35, step=0.05, format="%.2f")
    n_vmax = col_nv2.number_input("n Calc V_max (V) / 理想因子電壓上限", value=0.65, step=0.05, format="%.2f")
    Vt = st.number_input("Thermal Voltage Vt (V)", value=0.02585, format="%.5f")

    st.markdown("#### 📐 Error Plot / 誤差圖設定")
    auto_err_y = st.checkbox("Auto Scale Error Y / 誤差Y軸自動", value=True)
    col_ey1, col_ey2 = st.columns(2)
    err_ymin = col_ey1.number_input("Error Y Min (%)", value=-100.0, step=10.0, disabled=auto_err_y)
    err_ymax = col_ey2.number_input("Error Y Max (%)", value=100.0, step=10.0, disabled=auto_err_y)
    col_av1, col_av2 = st.columns(2)
    avg_vmin = col_av1.number_input("Avg Calc V_min (V)", value=0.45, step=0.05, format="%.2f")
    avg_vmax = col_av2.number_input("Avg Calc V_max (V)", value=0.90, step=0.05, format="%.2f")

    st.markdown("#### 💾 Export / 匯出設定")
    file_prefix = st.text_input("Export File Name / 輸出檔名", value="Gummel_Results")

col_up1, col_up2 = st.columns([4, 1])
with col_up1:
    dut_files = st.file_uploader("Upload TonyPlot Gummel CSV / 上傳模擬 CSV", type=["csv"], accept_multiple_files=True,
                                 key=st.session_state["gummel_uploader_key"])
with col_up2:
    st.write("")
    st.write("")
    if st.button("🗑️ 清除所有上傳檔案", use_container_width=True, key="gummel_clear_btn"):
        st.session_state["gummel_uploader_key"] += 1
        if "gummel_ms_files" in st.session_state:
            st.session_state["gummel_ms_files"] = []
        if "gummel_prev_uploaded" in st.session_state:
            st.session_state["gummel_prev_uploaded"] = set()
        st.rerun()

all_data, errors = {}, {}
if dut_files:
    for f in dut_files:
        try:
            df = load_and_standardize(f.getvalue().decode("utf-8", errors="ignore"), add_noise, ic_noise, ib_noise)
            all_data[f.name] = df
        except Exception as e:
            errors[f.name] = str(e)
    for fname, err in errors.items(): st.error(f"**{fname}**: {err}")

uiuc_n_ic = calc_ideality(uiuc_ref["V_Ic"], uiuc_ref["Ic"], n_vmin, n_vmax, Vt)
uiuc_n_ib = calc_ideality(uiuc_ref["V_Ib"], uiuc_ref["Ib"], n_vmin, n_vmax, Vt)
uiuc_max_ic = uiuc_ref["Ic"].max()
uiuc_max_ib = uiuc_ref["Ib"].max()
_uiuc_beta_mask = uiuc_ref["V_Beta"] >= 0.45
uiuc_max_beta = uiuc_ref["Beta"][_uiuc_beta_mask].max() if _uiuc_beta_mask.any() else np.nan
_uiuc_beta_idx = uiuc_ref["Beta"][_uiuc_beta_mask].idxmax() if _uiuc_beta_mask.any() else None
uiuc_v_peak_beta = uiuc_ref["V_Beta"].iloc[_uiuc_beta_idx] if pd.notna(_uiuc_beta_idx) else np.nan
uiuc_v_turn_on = uiuc_ref["V_Ic"].iloc[(uiuc_ref["Ic"] >= 1e-9).idxmax()] if (uiuc_ref["Ic"] >= 1e-9).any() else np.nan

GLOBAL_LEGEND = dict(
    orientation="h",
    yanchor="bottom",
    y=1.05,
    xanchor="right",
    x=1,
    bgcolor="rgba(255,255,255,0.8)",
    bordercolor="#ccc",
    borderwidth=1
)


def update_axes(fig, y_title, log_y=False):
    yaxis_config = dict(
        title=y_title, type="log" if log_y else "linear",
        tickformat=".1e" if log_y else None, exponentformat="e" if log_y else "none",
        showgrid=True, gridcolor="#ebebeb"
    )

    if not auto_y:
        if log_y:
            yaxis_config["range"] = [np.log10(cur_ymin), np.log10(cur_ymax)]
        else:
            if "Beta" in y_title:
                yaxis_config["range"] = [beta_ymin, beta_ymax]
            else:
                yaxis_config["range"] = [cur_ymin, cur_ymax]

    fig.update_layout(
        xaxis=dict(title="Vbase (V)", range=[x_min, x_max], showgrid=True, gridcolor="#ebebeb"),
        yaxis=yaxis_config,
        plot_bgcolor="white", paper_bgcolor="white", height=550, margin=dict(l=55, r=25, t=45, b=50),
        hovermode="x unified",
        legend=GLOBAL_LEGEND
    )


tab1, tab2, tab3, tab4 = st.tabs(["📊 Overlay with UIUC", "🔍 Single Check", "📋 Summary", "🎛️ Family Curve Bias"])

with tab1:
    st.markdown("### 📊 Overlay simulated plots with UIUC Target")

    if all_data:
        file_options = list(all_data.keys())
        if "gummel_prev_uploaded" not in st.session_state:
            st.session_state["gummel_prev_uploaded"] = set()

        current_uploaded = set(file_options)
        new_files = current_uploaded - st.session_state["gummel_prev_uploaded"]

        if "gummel_ms_files" not in st.session_state:
            st.session_state["gummel_ms_files"] = file_options
        else:
            valid_selected = [f for f in st.session_state["gummel_ms_files"] if f in current_uploaded]
            for nf in new_files:
                if nf not in valid_selected:
                    valid_selected.append(nf)
            st.session_state["gummel_ms_files"] = valid_selected

        st.session_state["gummel_prev_uploaded"] = current_uploaded

        c_btn1, c_btn2, _ = st.columns([1.5, 1.5, 7])
        with c_btn1:
            if st.button("✅ 全部選取", use_container_width=True, key="gummel_sel_all"):
                st.session_state["gummel_ms_files"] = file_options
        with c_btn2:
            if st.button("❌ 全部清除選取", use_container_width=True, key="gummel_clr_all"):
                st.session_state["gummel_ms_files"] = []

        selected_files = st.multiselect(
            "📂 選擇要疊加顯示的模擬檔案 (支援輸入關鍵字搜尋)：",
            options=file_options,
            key="gummel_ms_files",
            format_func=lambda x: Path(x).stem
        )
    else:
        selected_files = []
        st.info("💡 等待上傳模擬資料中...下方為內建的 UIUC 基準曲線。")

    f_cur = go.Figure()

    m_u_ic = (uiuc_ref["V_Ic"] >= x_min) & (uiuc_ref["V_Ic"] <= x_max)
    m_u_ib = (uiuc_ref["V_Ib"] >= x_min) & (uiuc_ref["V_Ib"] <= x_max)
    f_cur.add_trace(go.Scatter(x=uiuc_ref["V_Ic"][m_u_ic], y=uiuc_ref["Ic"][m_u_ic], name="UIUC Ic",
                               line=dict(color=UIUC_COLOR, width=3, dash=LINE_IC)))
    f_cur.add_trace(go.Scatter(x=uiuc_ref["V_Ib"][m_u_ib], y=uiuc_ref["Ib"][m_u_ib], name="UIUC Ib",
                               line=dict(color=UIUC_COLOR, width=3, dash=LINE_IB)))

    if all_data:
        for i, k in enumerate(selected_files):
            d = all_data[k]
            c = SIM_COLORS[i % len(SIM_COLORS)]
            m_sim = (d["Vbase"] >= x_min) & (d["Vbase"] <= x_max)
            f_cur.add_trace(go.Scatter(x=d["Vbase"][m_sim], y=d["Ic_abs"][m_sim], name=f"Ic - {Path(k).stem}",
                                       line=dict(color=c, width=2, dash=LINE_IC)))
            f_cur.add_trace(go.Scatter(x=d["Vbase"][m_sim], y=d["Ib_abs"][m_sim], name=f"Ib - {Path(k).stem}",
                                       line=dict(color=c, width=2, dash=LINE_IB)))

    update_axes(f_cur, "Current (A)", y_scale == "Log")
    f_cur.update_layout(title="Current (Ic & Ib) Overlay")
    st.plotly_chart(f_cur, use_container_width=True)

    st.divider()

    f_beta = go.Figure()
    m_u_beta = (uiuc_ref["V_Beta"] >= x_min) & (uiuc_ref["V_Beta"] <= x_max)
    f_beta.add_trace(go.Scatter(x=uiuc_ref["V_Beta"][m_u_beta], y=uiuc_ref["Beta"][m_u_beta], name="UIUC Beta",
                                line=dict(color=UIUC_COLOR, width=3, dash=LINE_IC)))

    if all_data:
        for i, k in enumerate(selected_files):
            d = all_data[k]
            c = SIM_COLORS[i % len(SIM_COLORS)]
            m_sim = (d["Vbase"] >= x_min) & (d["Vbase"] <= x_max)
            f_beta.add_trace(go.Scatter(x=d["Vbase"][m_sim], y=d["Beta"][m_sim], name=f"Beta - {Path(k).stem}",
                                        line=dict(color=c, width=2, dash=LINE_IC)))

    update_axes(f_beta, "Beta (Linear)", False)
    f_beta.update_layout(title="Current Gain (Beta) Overlay")
    st.plotly_chart(f_beta, use_container_width=True)

with tab2:
    st.markdown("### 🔍 Single Check Calibration (Dual-Axis)")
    if not all_data:
        st.info("請先上傳模擬 CSV 檔案才能進行單一比對分析。")
    else:
        sc1, sc2 = st.columns([1, 4])
        with sc1:
            sel_sim = st.selectbox("Choose a single simulation to analyze:", list(all_data.keys()),
                                   key="gummel_sel_sim")

        with sc2:
            if sel_sim:
                d = all_data[sel_sim]

                f_dual = make_subplots(specs=[[{"secondary_y": True}]])
                m_u_ic = (uiuc_ref["V_Ic"] >= x_min) & (uiuc_ref["V_Ic"] <= x_max)
                m_u_ib = (uiuc_ref["V_Ib"] >= x_min) & (uiuc_ref["V_Ib"] <= x_max)
                m_u_beta = (uiuc_ref["V_Beta"] >= x_min) & (uiuc_ref["V_Beta"] <= x_max)

                f_dual.add_trace(go.Scatter(x=uiuc_ref["V_Ic"][m_u_ic], y=uiuc_ref["Ic"][m_u_ic], name="UIUC Ic",
                                            line=dict(color=UIUC_COLOR, width=3, dash=LINE_IC)), secondary_y=False)
                f_dual.add_trace(go.Scatter(x=uiuc_ref["V_Ib"][m_u_ib], y=uiuc_ref["Ib"][m_u_ib], name="UIUC Ib",
                                            line=dict(color=UIUC_COLOR, width=3, dash=LINE_IB)), secondary_y=False)
                f_dual.add_trace(
                    go.Scatter(x=uiuc_ref["V_Beta"][m_u_beta], y=uiuc_ref["Beta"][m_u_beta], name="UIUC Beta",
                               line=dict(color=UIUC_COLOR, width=3, dash=LINE_BETA)), secondary_y=True)

                sim_c = SIM_COLORS[list(all_data.keys()).index(sel_sim) % len(SIM_COLORS)]
                m_sim = (d["Vbase"] >= x_min) & (d["Vbase"] <= x_max)
                f_dual.add_trace(go.Scatter(x=d["Vbase"][m_sim], y=d["Ic_abs"][m_sim], name=f"Sim Ic",
                                            line=dict(color=sim_c, width=2.5, dash=LINE_IC)), secondary_y=False)
                f_dual.add_trace(go.Scatter(x=d["Vbase"][m_sim], y=d["Ib_abs"][m_sim], name=f"Sim Ib",
                                            line=dict(color=sim_c, width=2.5, dash=LINE_IB)), secondary_y=False)
                f_dual.add_trace(go.Scatter(x=d["Vbase"][m_sim], y=d["Beta"][m_sim], name=f"Sim Beta",
                                            line=dict(color=sim_c, width=2.5, dash=LINE_BETA)), secondary_y=True)

                y1_dict = dict(title="Current (A)", type="log" if y_scale == "Log" else "linear",
                               tickformat=".1e" if y_scale == "Log" else None,
                               exponentformat="e" if y_scale == "Log" else "none", showgrid=True, gridcolor="#ebebeb")
                y2_dict = dict(title="Beta (Linear)", type="linear", showgrid=False)

                if not auto_y:
                    if y_scale == "Log":
                        y1_dict["range"] = [np.log10(cur_ymin), np.log10(cur_ymax)]
                    else:
                        y1_dict["range"] = [cur_ymin, cur_ymax]
                    y2_dict["range"] = [beta_ymin, beta_ymax]

                f_dual.update_layout(
                    xaxis=dict(title="Vbase (V)", range=[x_min, x_max], showgrid=True, gridcolor="#ebebeb"),
                    yaxis=y1_dict,
                    yaxis2=y2_dict,
                    plot_bgcolor="white", paper_bgcolor="white", height=550, hovermode="x unified",
                    legend=GLOBAL_LEGEND
                )
                st.plotly_chart(f_dual, use_container_width=True)

                sim_metrics = extract_metrics(d, n_vmin, n_vmax, Vt)
                uiuc_metrics = (
                uiuc_n_ic, uiuc_n_ib, uiuc_max_ic, uiuc_max_ib, uiuc_max_beta, uiuc_v_peak_beta, uiuc_v_turn_on)
                labels = ["n (Ic Ideality)", "n (Ib Ideality)", "Max Ic (A)", "Max Ib (A)", "Max Beta",
                          "V_peak_beta (V)", "V_turn_on @1nA (V)"]


                def calc_err(sim, ref):
                    if pd.isna(sim) or pd.isna(ref) or ref == 0: return np.nan
                    return (sim - ref) / ref * 100


                comp_rows = []
                for lbl, s_val, u_val in zip(labels, sim_metrics, uiuc_metrics):
                    comp_rows.append({"Metric / 指標": lbl, "Simulated / 模擬值": s_val, "UIUC Target / 目標值": u_val,
                                      "Error (%) / 誤差": calc_err(s_val, u_val)})
                comp_df = pd.DataFrame(comp_rows)

                st.markdown("#### 🎯 Calibration Error Table / 對位誤差表")
                fmt_comp = {"Simulated / 模擬值": "{:.4e}", "UIUC Target / 目標值": "{:.4e}",
                            "Error (%) / 誤差": "{:+.2f}%"}
                st.dataframe(comp_df.style.format(fmt_comp, na_rep="—"), use_container_width=True, hide_index=True)

                # ── Per-Voltage Error Plot (0.2V ~ 0.9V) ──
                st.markdown("#### 📐 Current Error vs UIUC / 逐電壓誤差曲線")

                ERR_VMIN, ERR_VMAX = 0.20, 0.90
                v = d["Vbase"].values
                err_mask = (v >= ERR_VMIN) & (v <= ERR_VMAX)
                v_sel = v[err_mask]

                ic_ref = np.interp(v_sel, uiuc_ref["V_Ic"].values, uiuc_ref["Ic"].values)
                ib_ref = np.interp(v_sel, uiuc_ref["V_Ib"].values, uiuc_ref["Ib"].values)
                ic_err = np.where(ic_ref > 0, (d["Ic_abs"].values[err_mask] - ic_ref) / ic_ref * 100.0, np.nan)
                ib_err = np.where(ib_ref > 0, (d["Ib_abs"].values[err_mask] - ib_ref) / ib_ref * 100.0, np.nan)

                f_err = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                      subplot_titles=["Ic Relative Error (%)", "Ib Relative Error (%)"],
                                      vertical_spacing=0.12)
                f_err.add_trace(go.Scatter(x=v_sel, y=ic_err, name="Ic error",
                                           line=dict(color=sim_c, width=2),
                                           hovertemplate="V=%{x:.3f}V<br>Err=%{y:.1f}%"), row=1, col=1)
                f_err.add_trace(go.Scatter(x=v_sel, y=ib_err, name="Ib error",
                                           line=dict(color=sim_c, width=2, dash="dash"),
                                           hovertemplate="V=%{x:.3f}V<br>Err=%{y:.1f}%"), row=2, col=1)
                f_err.add_hline(y=0, line_dash="dot", line_color="gray", line_width=1, row=1, col=1)
                f_err.add_hline(y=0, line_dash="dot", line_color="gray", line_width=1, row=2, col=1)

                # Vertical lines showing average calculation range
                for row_i in [1, 2]:
                    f_err.add_vline(x=avg_vmin, line_dash="dash", line_color="blue",
                                    line_width=1.5, opacity=0.6, row=row_i, col=1,
                                    annotation_text=f"Avg start {avg_vmin:.2f}V" if row_i == 1 else None,
                                    annotation_position="top left" if row_i == 1 else None)
                    f_err.add_vline(x=avg_vmax, line_dash="dash", line_color="blue",
                                    line_width=1.5, opacity=0.6, row=row_i, col=1,
                                    annotation_text=f"Avg end {avg_vmax:.2f}V" if row_i == 1 else None,
                                    annotation_position="top right" if row_i == 1 else None)

                err_y_cfg = dict(title_text="Error (%)", showgrid=True, gridcolor="#ebebeb")
                if not auto_err_y:
                    err_y_cfg["range"] = [err_ymin, err_ymax]

                f_err.update_layout(height=550, plot_bgcolor="white", paper_bgcolor="white",
                                    hovermode="x unified", legend=GLOBAL_LEGEND,
                                    margin=dict(l=55, r=25, t=60, b=50))
                f_err.update_xaxes(title_text="Vbase (V)", range=[ERR_VMIN, ERR_VMAX],
                                   showgrid=True, gridcolor="#ebebeb", row=2, col=1)
                f_err.update_xaxes(showgrid=True, gridcolor="#ebebeb", row=1, col=1)
                f_err.update_yaxes(**err_y_cfg, row=1, col=1)
                f_err.update_yaxes(**err_y_cfg, row=2, col=1)
                st.plotly_chart(f_err, use_container_width=True)

                # Average error within avg_vmin ~ avg_vmax
                avg_mask = (v_sel >= avg_vmin) & (v_sel <= avg_vmax)
                avg_ic_err = np.nanmean(np.abs(ic_err[avg_mask])) if avg_mask.any() else np.nan
                avg_ib_err = np.nanmean(np.abs(ib_err[avg_mask])) if avg_mask.any() else np.nan
                ec1, ec2 = st.columns(2)
                ec1.metric(f"Ic Avg |Error| ({avg_vmin:.2f}–{avg_vmax:.2f}V)", f"{avg_ic_err:.2f} %")
                ec2.metric(f"Ib Avg |Error| ({avg_vmin:.2f}–{avg_vmax:.2f}V)", f"{avg_ib_err:.2f} %")

def compute_family_bias(df, num_steps, ib_step_uA):
    """Given a Gummel DataFrame, interpolate Vbase for each target Ib."""
    ib_vals = df["Ib_abs"].values
    vb_vals = df["Vbase"].values

    # Sort by Ib for monotonic interpolation
    sort_idx = np.argsort(ib_vals)
    ib_sorted = ib_vals[sort_idx]
    vb_sorted = vb_vals[sort_idx]

    # Remove duplicates (keep last for highest Vbe at given Ib)
    unique_mask = np.concatenate([np.diff(ib_sorted) > 0, [True]])
    ib_sorted = ib_sorted[unique_mask]
    vb_sorted = vb_sorted[unique_mask]

    rows = []
    for i in range(num_steps):
        target_ib = i * ib_step_uA * 1e-6  # convert µA to A
        if i == 0:
            rows.append({"Step": i, "Target Ib (µA)": i * ib_step_uA,
                         "Target Ib (A)": target_ib, "Vbase (V)": 0.0,
                         "Status": "Zero bias"})
            continue

        if target_ib < ib_sorted.min() or target_ib > ib_sorted.max():
            rows.append({"Step": i, "Target Ib (µA)": i * ib_step_uA,
                         "Target Ib (A)": target_ib, "Vbase (V)": np.nan,
                         "Status": "Out of range"})
        else:
            vb_interp = round(float(np.interp(target_ib, ib_sorted, vb_sorted)), 3)
            rows.append({"Step": i, "Target Ib (µA)": i * ib_step_uA,
                         "Target Ib (A)": target_ib, "Vbase (V)": vb_interp,
                         "Status": "OK"})
    return pd.DataFrame(rows)


with tab3:
    st.markdown("### 📋 Summary")
    if not all_data:
        st.info("請先上傳模擬 CSV 檔案產生摘要報表。")
    else:
        summary_rows = []
        for k, d in all_data.items():
            n_ic, n_ib, m_ic, m_ib, m_beta, v_pb, v_ton = extract_metrics(d, n_vmin, n_vmax, Vt)
            summary_rows.append(
                {"File": Path(k).stem, "n_Ic": n_ic, "n_Ib": n_ib, "Max Beta": m_beta, "V_peak_beta": v_pb,
                 "V_turn_on (@1nA)": v_ton})

        sum_df = pd.DataFrame(summary_rows)
        fmt = {"n_Ic": "{:.4f}", "n_Ib": "{:.4f}", "Max Beta": "{:.2f}", "V_peak_beta": "{:.3f}",
               "V_turn_on (@1nA)": "{:.3f}"}
        st.dataframe(sum_df.style.format(fmt, na_rep="—"), use_container_width=True, hide_index=True)

        # Compute family bias for all files (for download inclusion)
        _fc_num = st.session_state.get("fc_num_steps", 11)
        _fc_gap = st.session_state.get("fc_ib_step", 38.0)
        family_bias_all = {}
        for k, d in all_data.items():
            family_bias_all[k] = compute_family_bias(d, _fc_num, _fc_gap)

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            sum_df.to_excel(w, sheet_name="Summary", index=False)
            for k, v in all_data.items():
                v.to_excel(w, sheet_name=Path(k).stem[:31], index=False)
            for k, fb in family_bias_all.items():
                sheet_name = f"FC_{Path(k).stem}"[:31]
                fb.to_excel(w, sheet_name=sheet_name, index=False)

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.download_button("📥 Download Excel Report / 下載完整報告", data=buf.getvalue(),
                               file_name=f"{file_prefix}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               use_container_width=True)
        with col_d2:
            zbuf = io.BytesIO()
            with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("Summary.csv", sum_df.to_csv(index=False).encode())
                for k, d in all_data.items():
                    zf.writestr(f"{Path(k).stem}.csv", d.to_csv(index=False).encode())
                for k, fb in family_bias_all.items():
                    zf.writestr(f"FamilyBias_{Path(k).stem}.csv", fb.to_csv(index=False).encode())
            st.download_button("📦 Download ZIP (CSV) / 下載 CSV 壓縮包", data=zbuf.getvalue(),
                               file_name=f"{file_prefix}.zip", mime="application/zip", use_container_width=True)

with tab4:
    st.markdown("### 🎛️ Family Curve Bias — Ib → Vbase Lookup")
    st.caption(
        "從 Gummel 模擬資料中，以指定的 Ib 間距內插出對應的 Vbase 偏壓，"
        "直接用於 Atlas family curve voltage-driven sweep。\n\n"
        "**Ref (UIUC):** Offset voltage ≈ 0.05 V, Knee voltage ≈ 0.7 V @ Jc = 5 mA/µm²."
    )

    if not all_data:
        st.info("請先上傳模擬 Gummel CSV 檔案。")
    else:
        fc_col1, fc_col2, fc_col3 = st.columns(3)
        with fc_col1:
            fc_sel = st.selectbox("選擇 Gummel 模擬檔", list(all_data.keys()),
                                  key="fc_file_sel", format_func=lambda x: Path(x).stem)
        with fc_col2:
            fc_num_steps = st.number_input("Number of steps / 組數", min_value=2, max_value=50,
                                           value=11, step=1, key="fc_num_steps")
        with fc_col3:
            fc_ib_step = st.number_input("Ib step size (µA) / 間距", min_value=0.1, max_value=1000.0,
                                         value=38.0, step=1.0, format="%.1f", key="fc_ib_step")

        fc_max_ib = (fc_num_steps - 1) * fc_ib_step
        st.markdown(f"**Ib range:** 0 → {fc_max_ib:.1f} µA　({fc_num_steps} points, step = {fc_ib_step:.1f} µA)")

        d_fc = all_data[fc_sel]
        fc_df = compute_family_bias(d_fc, fc_num_steps, fc_ib_step)

        # --- Display table ---
        st.markdown("#### 📋 Bias Point Table")

        fc_display = fc_df.copy()
        fc_fmt = {
            "Target Ib (µA)": "{:.1f}",
            "Target Ib (A)": "{:.4e}",
            "Vbase (V)": "{:.3f}",
        }

        def highlight_status(val):
            if val == "Out of range":
                return "background-color: #ffcccc"
            elif val == "Zero bias":
                return "background-color: #e8e8e8"
            return ""

        styled = fc_display.style.format(fc_fmt, na_rep="—")
        try:
            styled = styled.applymap(highlight_status, subset=["Status"])
        except AttributeError:
            styled = styled.map(highlight_status, subset=["Status"])

        st.dataframe(styled, use_container_width=True, hide_index=True)

        # --- Quick copy block for Atlas deck ---
        st.markdown("#### 📝 Atlas Deck Snippet (Voltage-Driven Family Curve)")
        st.caption(
            "可直接複製貼入 Atlas input deck。Vbase 四捨五入至小數點下 3 位。\n"
            "Ramp 分兩段：大步進 (0.05V) 快速接近，再小步進 (0.002V) 精確到達目標 Vbe。"
        )

        valid_rows = fc_df[(fc_df["Status"] == "OK") & (fc_df["Vbase (V)"].notna())]
        deck_lines = ["# ── Family Curve (Voltage-Driven) ──",
                       "# Auto-generated from Gummel interpolation",
                       f"# Source: {Path(fc_sel).stem}",
                       f"# Ib step = {fc_ib_step} uA, {fc_num_steps} curves",
                       "# Vbase rounded to 3 decimal places",
                       "# Two-stage ramp: coarse (0.05V) → fine (0.002V)",
                       "",
                       "# Initial setup",
                       "solve init",
                       "solve prev",
                       ""]

        valid_list = list(valid_rows.iterrows())
        for i, (_, row) in enumerate(valid_list):
            idx = int(row["Step"])
            vb = row["Vbase (V)"]  # already rounded to 3 decimals
            # Stage 1 target: floor to 2 decimal places (coarse approach)
            vb_coarse = np.floor(vb * 100) / 100  # e.g. 0.739 → 0.73
            # Only use two-stage if there's meaningful distance
            if vb_coarse >= 0.05:
                deck_lines.append(f"# --- Curve {idx}: Ib ≈ {row['Target Ib (µA)']:.1f} uA, Vbe = {vb:.3f} V ---")
                deck_lines.append(f"# Stage 1: coarse ramp to {vb_coarse:.2f} V")
                deck_lines.append(f"solve vstep=0.05 vfinal={vb_coarse:.2f} name=base")
                if vb > vb_coarse:
                    deck_lines.append(f"# Stage 2: fine ramp to {vb:.3f} V")
                    deck_lines.append(f"solve vstep=0.002 vfinal={vb:.3f} name=base")
            else:
                deck_lines.append(f"# --- Curve {idx}: Ib ≈ {row['Target Ib (µA)']:.1f} uA, Vbe = {vb:.3f} V ---")
                deck_lines.append(f"solve vstep=0.002 vfinal={vb:.3f} name=base")

            deck_lines.append(f"log outf=family_{idx}.log")
            deck_lines.append("solve vcollector=0")
            deck_lines.append("solve vstep=0.025 vfinal=2.0 name=collector")
            deck_lines.append("log off")
            deck_lines.append("")
            # Reset for next curve (skip for the last one)
            if i < len(valid_list) - 1:
                deck_lines.append("solve init")
                deck_lines.append("solve prev")
                deck_lines.append("solve vstep=0.025 vfinal=0.65 name=collector")
            deck_lines.append("")

        deck_text = "\n".join(deck_lines)
        st.code(deck_text, language="bash")

        # --- Ib vs Vbase plot ---
        st.markdown("#### 📈 Ib vs Vbase Mapping")
        f_fc = go.Figure()

        # Full Gummel Ib curve
        m_fc = d_fc["Ib_abs"] > 0
        f_fc.add_trace(go.Scatter(
            x=d_fc["Vbase"][m_fc], y=d_fc["Ib_abs"][m_fc],
            name="Gummel Ib", mode="lines",
            line=dict(color="#1f77b4", width=2)
        ))

        # Target points
        ok_rows = fc_df[fc_df["Status"] == "OK"]
        f_fc.add_trace(go.Scatter(
            x=ok_rows["Vbase (V)"], y=ok_rows["Target Ib (A)"],
            name="Family targets", mode="markers+text",
            marker=dict(color="#d62728", size=10, symbol="diamond"),
            text=[f"{r['Target Ib (µA)']:.0f}µA" for _, r in ok_rows.iterrows()],
            textposition="top center", textfont=dict(size=9)
        ))

        f_fc.update_layout(
            xaxis=dict(title="Vbase (V)", showgrid=True, gridcolor="#ebebeb"),
            yaxis=dict(title="Ib (A)", type="log", showgrid=True, gridcolor="#ebebeb",
                       tickformat=".1e", exponentformat="e"),
            plot_bgcolor="white", paper_bgcolor="white",
            height=500, hovermode="x unified",
            legend=GLOBAL_LEGEND
        )
        st.plotly_chart(f_fc, use_container_width=True)

        # --- Download family bias CSV ---
        st.markdown("#### 💾 Download Family Bias Data")
        fc_csv = fc_df.to_csv(index=False).encode()
        st.download_button(
            "📥 Download Family Curve Bias CSV",
            data=fc_csv,
            file_name=f"FamilyBias_{Path(fc_sel).stem}.csv",
            mime="text/csv",
            use_container_width=True
        )
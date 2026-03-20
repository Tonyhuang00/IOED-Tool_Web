import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.title("🧪 SMU Interactive Plot Tool")

# =================================================
# Session state
# =================================================
if "hp_smu_map" not in st.session_state:
    st.session_state.hp_smu_map = {}

if "hp_df" not in st.session_state:
    st.session_state.hp_df = None

if "hp_file_name" not in st.session_state:
    st.session_state.hp_file_name = None


# =================================================
# Utilities
# =================================================
def read_table(raw):
    try:
        df = pd.read_csv(io.StringIO(raw), delim_whitespace=True, engine="python")
    except Exception:
        df = pd.read_csv(io.StringIO(raw), engine="python")
    df.columns = df.columns.str.strip()
    return df


def parse_smu_table(df, smu_map):
    parsed = {}
    for role, (vcol, icol) in smu_map.items():
        if vcol in df.columns and icol in df.columns:
            parsed[role.lower()] = (df[vcol].astype(float), df[icol].astype(float))
    return parsed


def group_by_ib(vc, y, ib):
    vc_axis = np.sort(vc.unique())
    groups = {}
    tmp_df = pd.DataFrame({"Vc": vc, "Y": y, "Ib": ib})
    for ib_val, group in tmp_df.groupby("Ib"):
        groups[ib_val] = group.set_index("Vc")["Y"].reindex(vc_axis).reset_index(drop=True)
    return vc_axis, groups


def format_ib_label(ib):
    a = abs(ib)
    if a >= 1e-3:
        return f"{ib * 1e3:.2g} mA"
    if a >= 1e-6:
        return f"{ib * 1e6:.2g} µA"
    return f"{ib:.2e} A"


def scale_current(y):
    ymax = np.nanmax(abs(y))
    if ymax < 1e-3:
        return y * 1e6, "µA"
    if ymax < 1:
        return y * 1e3, "mA"
    return y, "A"


def scale_power(y):
    ymax = np.nanmax(abs(y))
    if ymax < 1e-3:
        return y * 1e6, "µW"
    return y, "W"


def apply_axes(fig, xlim, ylim, grid, minor_grid, show_right_ticks):
    fig.update_xaxes(range=xlim if xlim else None, showgrid=grid, minor=dict(showgrid=minor_grid))
    fig.update_yaxes(range=ylim if ylim else None, showgrid=grid, minor=dict(showgrid=minor_grid),
                     ticks="outside", mirror=True if show_right_ticks else False)


# =================================================
# SMU assignment
# =================================================
st.subheader("① SMU Assignment (persistent)")

roles = ["Collector", "Base", "Emitter", "PD", "None"]
smu_cols = [("V1", "I1"), ("V2", "I2"), ("V3", "I3"), ("V4", "I4")]

for v, i in smu_cols:
    default = "None"
    for r, pair in st.session_state.hp_smu_map.items():
        if pair == (v, i):
            default = r.capitalize()
    try:
        default_index = roles.index(default.capitalize())
    except ValueError:
        default_index = 0
    choice = st.selectbox(f"{v}/{i}", roles, index=default_index, key=f"hp_smu_{v}_{i}")

    if choice != "None":
        st.session_state.hp_smu_map[choice.lower()] = (v, i)

st.markdown("---")

# =================================================
# File input
# =================================================
st.subheader("② Data File")
uploaded = st.file_uploader("Upload SMU data file", key="hp_file_uploader")
if uploaded:
    raw = uploaded.getvalue().decode("utf-8", errors="ignore")
    st.session_state.hp_df = read_table(raw)
    st.session_state.hp_file_name = uploaded.name or "data"

if st.session_state.hp_df is None:
    st.stop()

df = st.session_state.hp_df
parsed = parse_smu_table(df, st.session_state.hp_smu_map)

st.success(f"Loaded: {st.session_state.hp_file_name}")

st.markdown("---")

# =================================================
# Plot controls
# =================================================
st.subheader("③ Plot Settings")
dtype = st.selectbox("Data type", ["Family L-Ic-Vc", "Diode", "Gummel", "Family"], key="hp_dtype")

manual = st.checkbox("Manual axis limits", key="hp_manual_axis")
xlim = ylim = None
if manual:
    c1, c2 = st.columns(2)
    with c1:
        xlim = [st.number_input("X min", value=0.0, key="hp_xmin"), st.number_input("X max", value=1.0, key="hp_xmax")]
    with c2:
        ylim = [st.number_input("Y min", value=0.0, key="hp_ymin"), st.number_input("Y max", value=1.0, key="hp_ymax")]

grid = st.checkbox("Show grid", value=True, key="hp_grid")
minor_grid = st.checkbox("Show minor grid", value=False, key="hp_minor_grid")
responsivity = None
if dtype == "Family L-Ic-Vc":
    responsivity = st.number_input("Responsivity (A/W)", value=0.35, key="hp_responsivity")

# =================================================
# Plot
# =================================================
if st.button("📊 Show", key="hp_show_btn"):

    buf = io.BytesIO()
    writer = pd.ExcelWriter(buf, engine="openpyxl")

    try:
        if dtype == "Diode":
            if "base" not in parsed:
                st.error("Base SMU missing")
                st.stop()
            vb, _ = parsed["base"]

            if "emitter" in parsed:
                _, i = parsed["emitter"]
            elif "collector" in parsed:
                _, i = parsed["collector"]
            else:
                st.error("No current SMU found for diode")
                st.stop()

            y, unit = scale_current(abs(i))
            fig = go.Figure(go.Scatter(x=vb, y=y))
            fig.update_yaxes(type="log", title=f"I ({unit})")
            fig.update_xaxes(title="Vb")
            apply_axes(fig, xlim, ylim, grid, minor_grid, True)
            st.plotly_chart(fig, use_container_width=True)

            pd.DataFrame({"Vb": vb, "I": abs(i)}).to_excel(writer, sheet_name="Diode", index=False)

        elif dtype == "Gummel":
            for role in ["base", "collector", "emitter"]:
                if role not in parsed:
                    st.error(f"{role.capitalize()} SMU missing")
                    st.stop()
            vb, ib = parsed["base"]
            _, ic = parsed["collector"]
            _, ie = parsed["emitter"]

            ic_s, unit = scale_current(abs(ic))
            ie_s, _ = scale_current(abs(ie))
            beta = ic / ib.replace(0, np.nan)

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=vb, y=ic_s, name=f"Ic ({unit})"), secondary_y=False)
            fig.add_trace(go.Scatter(x=vb, y=ie_s, name=f"Ie ({unit})"), secondary_y=False)
            fig.add_trace(go.Scatter(x=vb, y=beta, name="β"), secondary_y=True)

            fig.update_yaxes(type="log", secondary_y=False)
            fig.update_xaxes(title="Vb")
            fig.update_xaxes(showgrid=grid)
            fig.update_yaxes(showgrid=grid, secondary_y=False)
            fig.update_yaxes(showgrid=grid, secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)

            pd.DataFrame({"Vb": vb, "Ib": ib, "Ic": ic, "Ie": ie, "Beta": beta}).to_excel(writer, sheet_name="Gummel",
                                                                                          index=False)

        else:
            if "collector" not in parsed or "base" not in parsed:
                st.error("Collector or Base SMU missing")
                st.stop()
            vc, ic = parsed["collector"]
            _, ib = parsed["base"]

            vc_axis, ic_groups = group_by_ib(vc, ic, ib)
            out_ic = pd.DataFrame({"Vc": vc_axis})
            fig_ic = go.Figure()
            for ib_val, series in ic_groups.items():
                y, unit = scale_current(series)
                fig_ic.add_trace(go.Scatter(x=vc_axis, y=y, name=f"Ib={format_ib_label(ib_val)}"))
                out_ic[f"Ic @ Ib={ib_val}"] = series
            fig_ic.update_xaxes(title="Vc")
            fig_ic.update_yaxes(title=f"Ic ({unit})")
            apply_axes(fig_ic, xlim, ylim, grid, minor_grid, True)

            if dtype == "Family":
                st.plotly_chart(fig_ic, use_container_width=True)
                out_ic.to_excel(writer, sheet_name="Electrical", index=False)
            else:
                if "pd" not in parsed:
                    st.error("PD SMU required for Family L-Ic-Vc")
                    st.stop()
                _, ipd = parsed["pd"]
                L = -ipd / responsivity

                vc_axis, L_groups = group_by_ib(vc, L, ib)
                fig_L = go.Figure()
                out_L = pd.DataFrame({"Vc": vc_axis})

                all_ic = pd.concat(ic_groups.values())
                ic_scaled, unit = scale_current(all_ic)

                start_idx = 0
                fig_ic = go.Figure()
                for ib_val, series in ic_groups.items():
                    end_idx = start_idx + len(series)
                    y = ic_scaled[start_idx:end_idx].values
                    fig_ic.add_trace(go.Scatter(x=vc_axis, y=y, name=f"Ib={format_ib_label(ib_val)}"))
                    out_ic[f"Ic @ Ib={ib_val}"] = series
                    start_idx = end_idx

                fig_ic.update_xaxes(title="Vc")
                fig_ic.update_yaxes(title=f"Ic ({unit})")
                apply_axes(fig_ic, xlim, ylim, grid, minor_grid, True)

                all_L = pd.concat(L_groups.values())
                L_scaled, unitL = scale_power(all_L)

                start_idx = 0
                fig_L = go.Figure()
                for ib_val, series in L_groups.items():
                    end_idx = start_idx + len(series)
                    y = L_scaled[start_idx:end_idx].values
                    fig_L.add_trace(go.Scatter(x=vc_axis, y=y, name=f"Ib={format_ib_label(ib_val)}"))
                    out_L[f"L @ Ib={ib_val}"] = series
                    start_idx = end_idx

                fig_L.update_xaxes(title="Vc")
                fig_L.update_yaxes(title=f"L ({unitL})")
                apply_axes(fig_L, xlim, ylim, grid, minor_grid, True)

                c1, c2 = st.columns(2)
                with c1:
                    st.plotly_chart(fig_ic, use_container_width=True)
                with c2:
                    st.plotly_chart(fig_L, use_container_width=True)

                out_ic.to_excel(writer, sheet_name="Electrical", index=False)
                out_L.to_excel(writer, sheet_name="Optical", index=False)

    except Exception as e:
        st.error(f"Error while plotting: {e}")
        st.stop()

    writer.close()
    st.download_button(
        "📥 Download Excel",
        data=buf.getvalue(),
        file_name=f"{st.session_state.hp_file_name}_processed.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
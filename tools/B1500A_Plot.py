import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# =================================================
# Utilities
# =================================================
def find_col_like(df, keywords):
    for c in df.columns:
        cl = c.lower()
        if all(k in cl for k in keywords):
            return c
    return None


def find_cols_starting(df, prefix):
    return [c for c in df.columns if c.lower().startswith(prefix.lower())]


def ideality_factor(v, i, vmin, vmax):
    mask = (v >= vmin) & (v <= vmax) & (i > 0)
    if mask.sum() < 2:
        return np.nan
    slope, _ = np.polyfit(v[mask], np.log(i[mask]), 1)
    q = 1.602e-19
    k = 1.381e-23
    T = 300
    return q / (slope * k * T)


# =================================================
# Sidebar
# =================================================
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Select Page",
    ["B1500A Viewer", "TLM Analysis"]
)

# =================================================
# B1500A VIEWER (EXCEL ONLY)
# =================================================
if page == "B1500A Viewer":

    st.title("🧪 B1500A Excel Viewer")

    uploaded = st.file_uploader("Upload Excel file (.xlsx)", type=["xlsx"])
    if not uploaded:
        st.stop()

    xls = pd.ExcelFile(uploaded)
    sheet = st.selectbox("Select sheet", xls.sheet_names)
    df = xls.parse(sheet)

    name = f"{uploaded.name} — {sheet}".lower()

    # Auto data type
    if "gummel" in name:
        default = "Gummel"
    elif "family" in name:
        default = "Family"
    else:
        default = "Diode"

    dtype = st.selectbox(
        "Data type",
        ["Diode", "Gummel", "Family"],
        index=["Diode", "Gummel", "Family"].index(default)
    )

    # =================================================
    # DIODE
    # =================================================
    if dtype == "Diode":
        vcol = df.columns[0]
        icol = df.columns[1]

        v = df[vcol].astype(float)
        i = abs(df[icol].astype(float))

        fig = go.Figure(go.Scatter(x=v, y=i))
        fig.update_yaxes(type="log", title="Current (A)")
        fig.update_xaxes(title="Voltage (V)")
        fig.update_layout(title="Diode I–V")
        st.plotly_chart(fig, use_container_width=True)

        vmin, vmax = st.slider(
            "Ideality factor voltage range (V)",
            0.2, 0.6, (0.35, 0.4)
        )

        n = ideality_factor(v, i, vmin, vmax)

        st.info(
            f"""
**Ideality factor (n) = {n:.2f}**

**Guide**
- n ≈ 1 → ideal diffusion current  
- n → 2 → recombination-dominated transport  
"""
        )

    # =================================================
    # GUMMEL
    # =================================================
    elif dtype == "Gummel":
        vb_col = find_col_like(df, ["vb"])
        ic_col = find_col_like(df, ["ic"])
        ib_col = find_col_like(df, ["ib"])

        if not all([vb_col, ic_col, ib_col]):
            st.error("Could not identify Vb / Ic / Ib columns")
            st.stop()

        vb = df[vb_col].astype(float)
        ic = df[ic_col].astype(float)
        ib = df[ib_col].astype(float)

        beta = ic / ib.replace(0, np.nan)
        beta_max = beta.max()
        beta_v = vb.loc[beta.idxmax()]

        fig = make_subplots(
            rows=1, cols=1,
            specs=[[{"secondary_y": True}]]
        )

        fig.add_trace(go.Scatter(x=vb, y=abs(ic), name="Ic"),
                      row=1, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=vb, y=abs(ib), name="Ib"),
                      row=1, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=vb, y=beta, name="β"),
                      row=1, col=1, secondary_y=True)

        fig.update_yaxes(type="log", secondary_y=False, title="Current (A)")
        fig.update_yaxes(
            type="linear",
            range=[0.0, 1.5 * beta_max],
            secondary_y=True,
            title="β"
        )
        fig.update_xaxes(title="Vb (V)")
        fig.update_layout(title="Gummel Plot")

        st.plotly_chart(fig, use_container_width=True)

        vmin, vmax = st.slider(
            "Ideality factor voltage range (V)",
            0.0, max(vb), (0.35, 0.4)
        )

        n_ic = ideality_factor(vb, abs(ic), vmin, vmax)
        n_ib = ideality_factor(vb, abs(ib), vmin, vmax)

        c1, c2 = st.columns(2)
        c1.metric("Collector Ideality Factor, n(Ic)", f"{n_ic:.2f}")
        c2.metric("Base Ideality Factor, n(Ib)", f"{n_ib:.2f}")
        c1.metric("Maximum Gain (beta)", f"{beta_max:.1f}")
        c2.metric("Maximum Gain Occurs at Voltage (V)", f"{beta_v:.3f}")

    # =================================================
    # FAMILY
    # =================================================
    else:
        vc_col = find_col_like(df, ["vc"])
        if not vc_col:
            st.error("Vc column not found")
            st.stop()

        vc = df[vc_col].astype(float)
        ic_cols = [c for c in df.columns if c.lower().startswith("ic")]

        if not ic_cols:
            st.error("No Ic curves found (expected 'Ic at Ib=...')")
            st.stop()

        fig = go.Figure()
        for c in ic_cols:
            fig.add_trace(go.Scatter(x=vc, y=df[c], name=c))

        fig.update_xaxes(title="Vc (V)")
        fig.update_yaxes(title="Ic (A)", range=[0, max(df[c]) * 1.2])
        fig.update_layout(title="Family I–V")
        st.plotly_chart(fig, use_container_width=True)

# =================================================
# TLM ANALYSIS
# =================================================
else:
    st.title("📐 TLM Analysis")

    Z = st.number_input("Pad width Z (µm)", value=80.0)

    R = {
        4: st.number_input("R @ 4 µm (Ω)", value=np.nan),
        8: st.number_input("R @ 8 µm (Ω)", value=np.nan),
        16: st.number_input("R @ 16 µm (Ω)", value=np.nan),
        32: st.number_input("R @ 32 µm (Ω)", value=np.nan),
    }

    x = np.array([k for k, v in R.items() if not np.isnan(v)])
    y = np.array([v for v in R.values() if not np.isnan(v)])

    if len(x) >= 2:
        slope, intercept = np.polyfit(x, y, 1)
        Rc = 0.5 * intercept
        Rsh = slope * Z
        LT = Rc / slope
        rho_c = Rc * LT * Z * 1e-8
        r2 = np.corrcoef(x, y)[0, 1] ** 2

        c1, c2, c3 = st.columns(3)
        c1.metric("Intercept (Ω)", f"{intercept:.2f}")
        c1.metric("Slope (Ω/µm)", f"{slope:.3f}")
        c2.metric("Contact Resistance, Rc (Ω)", f"{Rc:.2f}")
        c2.metric("Sheet Resistance, Rsh (Ω/□)", f"{Rsh:.1f}")
        c3.metric("Transfer Length, LT (µm)", f"{LT:.2f}")
        c3.metric("Specific Contact Resistivity, ρc (Ω·cm²)", f"{rho_c:.2e}")
        st.metric("Goodness, R²", f"{r2:.4f}")

        xs = np.linspace(0, 40, 200)
        ys = slope * xs + intercept

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Measured"))
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines",
            line=dict(dash="dash"),
            name="Linear fit"
        ))

        fig.update_xaxes(title="Spacing (µm)", range=[0, 40])
        fig.update_yaxes(title="Resistance (Ω)", range=[0, max(y) * 1.2])
        st.plotly_chart(fig, use_container_width=True)
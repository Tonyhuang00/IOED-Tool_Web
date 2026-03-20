import streamlit as st
import pandas as pd
import numpy as np
import io
import re
import os
import zipfile

st.title("🖥️ HBT Data Processing Tool")


# --- Utility functions ---
def detect_header_row(text):
    """Detect header row by searching for 'DataName' in the first column (case-insensitive)."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        first_cell = line.split(",")[0].strip()
        if re.fullmatch(r"dataname", first_cell, flags=re.IGNORECASE):
            return i
    return 0


def process_b1500a_file(file, template_cols, header_idx, preset):
    text = file.getvalue().decode("utf-8", errors="ignore")
    df = pd.read_csv(io.StringIO(text), header=header_idx, engine="python", on_bad_lines="skip")
    df.columns = df.columns.str.strip()

    if preset == "Ic-Vc Family preset":
        if not all(c in df.columns for c in ["Vc", "Ic", "Ib"]):
            raise ValueError("Missing Vc/Ic/Ib for Family curve")

        vc_axis = np.sort(df["Vc"].unique())
        out = pd.DataFrame({"Vc": vc_axis})

        for ib_val, grp in df.groupby("Ib"):
            series = (
                grp.set_index("Vc")["Ic"]
                .reindex(vc_axis)
                .reset_index(drop=True)
            )
            out[f"Ic at Ib={format_ib_label(ib_val)}"] = series

        return out

    return df[[c for c in template_cols if c in df.columns]]


def process_file(file, selected_cols, template_header_idx):
    text = file.getvalue().decode("utf-8", errors="ignore")
    header_idx = template_header_idx if template_header_idx is not None else detect_header_row(text)
    df = pd.read_csv(io.StringIO(text), header=header_idx, engine="python", on_bad_lines="skip")
    df.columns = df.columns.str.strip()
    df = df[[c for c in selected_cols if c in df.columns]]
    return df


def sanitize_sheet_name(name):
    name = name.split(";")[0].strip()
    name = re.sub(r'[:\\/*?\[\]]', '_', name)
    return name[:31]


def detect_preset_from_filename(filename):
    prefix = filename.strip().lower()[:2]
    if prefix == "tl":
        return "TLM preset"
    elif prefix == "bc":
        return "BC diode preset"
    elif prefix == "be":
        return "BE diode preset"
    elif prefix == "gu":
        return "Gummel preset"
    elif prefix == "fa":
        return "Ic-Vc Family preset"
    return "none"


def preset_to_type(preset):
    return {
        "Ic-Vc Family preset": "Family",
        "BE diode preset": "BE",
        "BC diode preset": "BC",
        "Gummel preset": "Gummel",
        "TLM preset": "TLM",
    }.get(preset, "Other")


def get_preset_columns(preset, available_cols):
    available_cols = [c.strip() for c in available_cols]
    presets = {
        "BC diode preset": ["Vb", "Ic_abs"],
        "BE diode preset": ["Vb", "Ie_abs"],
        "Gummel preset": ["Vb", "Ib", "Ic", "Beta"],
        "Ic-Vc Family preset": ["Vc", "Ib", "Ic"],
        "TLM preset": ["I1", "Rsa", "Vsa"],
    }
    wanted = presets.get(preset, [])
    return [c for c in available_cols if c in wanted]


def format_output_filename(preset):
    if preset == "BC diode preset":
        return "BC batch output.xlsx"
    elif preset == "BE diode preset":
        return "BE batch output.xlsx"
    elif preset == "Gummel preset":
        return "Gummel batch output.xlsx"
    elif preset == "Ic-Vc Family preset":
        return "Family batch output.xlsx"
    elif preset == "TLM preset":
        return "TLM batch output.xlsx"
    return "batch_output.xlsx"


def convert_units(val):
    units = [("A", 1), ("mA", 1e-3), ("uA", 1e-6), ("nA", 1e-9)]
    for unit, scale in units:
        if 1 <= val / scale < 1000:
            return val / scale, unit
    return val, "A"


def _lines(content):
    return [ln.rstrip() for ln in content.splitlines()]


def parse_var_headers(content):
    matches = re.findall(r'(?mi)^\s*VAR\s+(\S+)\s+MAG\s+(\d+)\s*$', content)
    return [(m[0], int(m[1])) for m in matches]


def parse_data_header(content):
    m = re.search(r'(?mi)^\s*DATA\s+(\S+)\s+MAG', content)
    return m.group(1) if m else None


def extract_var_list_blocks(lines):
    blocks = []
    i = 0
    while i < len(lines):
        tokens = lines[i].strip().split()
        if len(tokens) > 0 and tokens[0].upper() == "VAR_LIST_BEGIN":
            vals = []
            j = i + 1
            while j < len(lines):
                tks = lines[j].strip().split()
                if tks and tks[0].upper() == "VAR_LIST_END":
                    break
                for tok in tks:
                    try:
                        vals.append(float(tok.replace('E', 'e')))
                    except Exception:
                        continue
                j += 1
            blocks.append(vals)
            i = j + 1
        else:
            i += 1
    return blocks


def extract_data_blocks(lines):
    blocks = []
    i = 0
    while i < len(lines):
        tokens = lines[i].strip().split()
        if len(tokens) > 0 and tokens[0].upper() == "BEGIN":
            vals = []
            j = i + 1
            while j < len(lines):
                tks = lines[j].strip().split()
                if tks and tks[0].upper() == "END":
                    break
                for tok in tks:
                    try:
                        vals.append(float(tok.replace('E', 'e')))
                    except Exception:
                        continue
                j += 1
            blocks.append(vals)
            i = j + 1
        else:
            i += 1
    return blocks


def parse_gummel_from_text(content):
    lines = [ln.strip() for ln in content.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    rows = []
    for ln in lines:
        toks = ln.split()
        if len(toks) < 3: continue
        try:
            vce = float(toks[0].replace('E', 'e'))
            ib = float(toks[1].replace('E', 'e'))
            ic = float(toks[2].replace('E', 'e'))
            rows.append((vce, ib, ic))
        except Exception:
            continue
    if not rows: raise ValueError("No numeric rows found for Gummel format.")
    df = pd.DataFrame(rows, columns=["Vce", "Ib", "Ic"])
    df["Beta"] = df["Ic"] / df["Ib"].replace({0: np.nan})
    return df


def parse_diode_by_header(content):
    lines = _lines(content)
    var_hdrs = parse_var_headers(content)
    data_name = parse_data_header(content) or "DATA"
    var_blocks = extract_var_list_blocks(lines)
    data_blocks = extract_data_blocks(lines)
    if not var_hdrs or not var_blocks or not data_blocks:
        raise ValueError("Missing header or data for diode.")

    var_name, mag_count = var_hdrs[0]
    x = var_blocks[0][:mag_count]

    if any("#Base-Emitter Diode" in ln for ln in lines):
        diode_type = "BE"
    elif any("#Base-Collector Diode" in ln for ln in lines):
        diode_type = "BC"
    else:
        diode_type = "Diode"

    y_raw = data_blocks[0][:mag_count]
    y = []
    for val in y_raw:
        try:
            f = float(val)
            if diode_type in ["BE", "BC"]: f = abs(f)
            y.append(f)
        except:
            continue
    x = x[:len(y)]
    df = pd.DataFrame({var_name: x, data_name: y})
    return df, diode_type


def parse_family_by_header(content):
    lines = _lines(content)
    var_hdrs = parse_var_headers(content)
    data_names = []
    for ln in lines:
        t = ln.strip().split()
        if len(t) >= 3 and t[0].upper() == "DATA":
            data_names.append(t[1])
    var_blocks = extract_var_list_blocks(lines)
    data_blocks = extract_data_blocks(lines)
    if len(var_hdrs) < 2 or not data_blocks:
        raise ValueError("Missing header or data for Family format.")

    (var1_name, n1), (var2_name, n2) = var_hdrs[0], var_hdrs[1]
    var1_list = var_blocks[0][:n1]
    var2_list = var_blocks[1][:n2]
    df = pd.DataFrame({var1_name: var1_list})
    expected = n1 * n2

    ordered_pairs = []
    if "L" in data_names:
        idx = data_names.index("L")
        ordered_pairs.append(("L", data_blocks[idx]))
    if "Ic" in data_names:
        idx = data_names.index("Ic")
        ordered_pairs.append(("Ic", data_blocks[idx]))
    for name, block in zip(data_names, data_blocks):
        if name not in ["L", "Ic"]:
            ordered_pairs.append((name, block))

    for data_name, data in ordered_pairs:
        padded = np.full(expected, np.nan)
        padded[: min(len(data), expected)] = data[: min(len(data), expected)]
        arr = padded.reshape((n2, n1))
        for i in range(n2):
            ib_val = var2_list[i]
            col_name = f"{data_name} at {var2_name}={ib_val}"
            df[col_name] = arr[i, :]
    return df


def parse_gummel_vb_ib_ic(content):
    rows = []
    for ln in content.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"): continue
        toks = ln.split()
        if len(toks) < 3: continue
        try:
            vb = float(toks[0].replace("E", "e"))
            ib = float(toks[1].replace("E", "e"))
            ic = float(toks[2].replace("E", "e"))
            rows.append((vb, ib, ic))
        except Exception:
            continue
    if not rows: raise ValueError("No numeric rows found for Vb-Ib-Ic Gummel format.")
    df = pd.DataFrame(rows, columns=["Vb", "Ib", "Ic"])
    df["Beta"] = df["Ic"] / df["Ib"].replace({0: np.nan})
    return df


def parse_citi_file(content):
    var_hdrs = parse_var_headers(content)
    if not var_hdrs:
        try:
            return parse_gummel_vb_ib_ic(content), "Gummel"
        except Exception:
            return parse_gummel_from_text(content), "Gummel"
    elif len(var_hdrs) == 1:
        return parse_diode_by_header(content)
    else:
        return parse_family_by_header(content), "Family"


def parse_special_csv(content):
    lines = content.splitlines()
    numeric_rows = []
    for ln in lines:
        parts = ln.split(",")
        try:
            nums = [float(p) for p in parts]
            numeric_rows.append(nums)
        except Exception:
            continue
    if not numeric_rows: raise ValueError("No numeric data found in CSV.")
    header = ["Ib", "Vbe", "Ic", "L", "dL/dI", "dV/dI", "beta"]
    df = pd.DataFrame(numeric_rows, columns=header[: len(numeric_rows[0])])
    return df


def parse_smu_table(df, smu_map):
    out = {}
    for role, (vcol, icol) in smu_map.items():
        if vcol in df.columns and icol in df.columns:
            out[role] = (df[vcol].astype(float), df[icol].astype(float))
    return out


def group_family(vc, ic, ib):
    vc_axis = np.sort(vc.unique())
    grouped = {}
    for ib_val, grp in pd.DataFrame({"Vc": vc, "Ic": ic, "Ib": ib}).groupby("Ib"):
        series = grp.set_index("Vc")["Ic"].reindex(vc_axis).reset_index(drop=True)
        grouped[ib_val] = series
    return vc_axis, grouped


def format_ib_label(ib):
    val, unit = convert_units(abs(ib))
    sign = "-" if ib < 0 else ""
    return f"{sign}{val:.3g}{unit}"


page = st.sidebar.radio(
    "Choose page:",
    [
        "B1500A Smart Batch Tool",
        "TLM Resistance Avg",
        "E5270B citi File Tool",
        "HP4155A Data Processing Tool",
        "B1500A Column Selection & Batch"
    ],
)

if page == "B1500A Smart Batch Tool":
    st.header("🧠 B1500A Smart Batch Tool")
    st.caption("Upload multiple B1500A CSV files. The tool auto-detects measurement type and prepares batch downloads.")

    uploaded_files = st.file_uploader("Upload B1500A CSV files", type=["csv"], accept_multiple_files=True)
    if not uploaded_files: st.stop()

    processed = []
    for f in uploaded_files:
        try:
            raw = f.getvalue().decode("utf-8", errors="ignore")
            header_idx = detect_header_row(raw)
            preset = detect_preset_from_filename(f.name)
            group_type = preset_to_type(preset)

            df_tmp = pd.read_csv(io.StringIO(raw), header=header_idx, engine="python", on_bad_lines="skip")
            df_tmp.columns = df_tmp.columns.str.strip()
            cols = get_preset_columns(preset, df_tmp.columns.tolist())

            df_out = process_b1500a_file(f, cols, header_idx, preset)
            processed.append((f.name, preset, group_type, df_out))
        except Exception as e:
            st.error(f"{f.name}: {e}")

    if processed:
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname, _, _, df in processed:
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                    sheet = sanitize_sheet_name(fname.rsplit(".", 1)[0])
                    df.to_excel(writer, sheet_name=sheet, index=False)
                zf.writestr(f"{sheet}.xlsx", buf.getvalue())

        st.download_button("📦 Download each data as an Excel file (ZIP)", data=zip_buf.getvalue(),
                           file_name="B1500A_Individual_Files.zip", mime="application/zip")

        type_files = {}
        for fname, preset, gtype, df in processed:
            type_files.setdefault(gtype, []).append((fname, df))

        zip_buf2 = io.BytesIO()
        with zipfile.ZipFile(zip_buf2, "w", zipfile.ZIP_DEFLATED) as zf:
            for gtype, items in type_files.items():
                if not items: continue
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                    for fname, df in items:
                        sheet = sanitize_sheet_name(fname.rsplit(".", 1)[0])
                        df.to_excel(writer, sheet_name=sheet, index=False)
                out_name = {"Family": "IcVc_Family.xlsx", "BE": "BE_Diode.xlsx", "BC": "BC_Diode.xlsx",
                            "Gummel": "Gummel.xlsx", "TLM": "TLM.xlsx", "Other": "Other.xlsx"}.get(gtype,
                                                                                                   f"{gtype}.xlsx")
                zf.writestr(out_name, buf.getvalue())

        st.download_button("📊 Download grouped files by measurement type", data=zip_buf2.getvalue(),
                           file_name="B1500A_Grouped_By_Type.zip", mime="application/zip")

elif page == "B1500A Column Selection & Batch":
    st.header("R307B B1500A CSV Batch Processing Tool")
    st.header("Step 1: Upload a sample CSV")

    sample_file = st.file_uploader("Upload sample CSV", type=["csv"], key="sample")
    if sample_file:
        text = sample_file.getvalue().decode("utf-8", errors="ignore")
        header_idx = detect_header_row(text)
        df = pd.read_csv(io.StringIO(text), header=header_idx, engine="python", on_bad_lines="skip")
        df.columns = df.columns.str.strip()

        detected_preset = detect_preset_from_filename(sample_file.name)
        preset_choice = st.selectbox(
            "Choose a preset:",
            ["none", "BC diode preset", "BE diode preset", "Gummel preset", "Ic-Vc Family preset", "TLM preset"],
            index=["none", "BC diode preset", "BE diode preset", "Gummel preset", "Ic-Vc Family preset",
                   "TLM preset"].index(detected_preset) if detected_preset else 0
        )

        if preset_choice != "none":
            preset_cols = get_preset_columns(preset_choice, df.columns.tolist())
            selected_cols = preset_cols
        else:
            preset_cols = []
            selected_cols = df.columns.tolist()

        st.write("Detected header columns:")
        selected_cols = st.multiselect("Select columns to keep", df.columns.tolist(), default=selected_cols,
                                       key="col_select_widget")

        if preset_choice != "none" and set(selected_cols) != set(preset_cols):
            preset_choice = "none"

        if st.button("💾 Save template"):
            st.session_state.batch_template_cols = selected_cols
            st.session_state.batch_template_header_idx = header_idx
            st.session_state.batch_preset_choice = preset_choice
            st.success(f"Template saved with {len(selected_cols)} columns.")

    if "batch_template_cols" in st.session_state:
        st.subheader("Step 2: Upload batch CSV files")
        batch_files = st.file_uploader("Upload multiple CSV files", type=["csv"], accept_multiple_files=True,
                                       key="batch")

        if batch_files and st.button("⚡ Process Batch"):
            all_sheets = {}
            for f in batch_files:
                df_proc = process_file(f, st.session_state.batch_template_cols,
                                       st.session_state.batch_template_header_idx)
                raw_name = os.path.splitext(f.name)[0]
                sheet_name = sanitize_sheet_name(raw_name)
                all_sheets[sheet_name] = df_proc

            out_buffer = io.BytesIO()
            out_filename = format_output_filename(st.session_state.get("batch_preset_choice", "none"))
            with pd.ExcelWriter(out_buffer, engine="openpyxl") as writer:
                for sheet, df in all_sheets.items():
                    df.to_excel(writer, sheet_name=sheet, index=False)

            st.session_state.batch_ready = {"data": out_buffer.getvalue(), "filename": out_filename}

        if "batch_ready" in st.session_state:
            st.download_button("📥 Download processed Excel", data=st.session_state.batch_ready["data"],
                               file_name=st.session_state.batch_ready["filename"],
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            if st.button("🔄 Refresh"):
                for key in ["batch_template_cols", "batch_template_header_idx", "batch_preset_choice", "batch_ready"]:
                    if key in st.session_state: del st.session_state[key]
                st.rerun()

elif page == "TLM Resistance Avg":
    st.header("TLM Resistance Average Calculator")
    excel_file = st.file_uploader("Upload TLM Excel file. The file name should be 'TLM batch output.xlsx'.",
                                  type=["xlsx"], key="tlm_batch")
    if excel_file:
        xls = pd.ExcelFile(excel_file)
        sheet_names = xls.sheet_names
        st.write(f"Found sheets: {sheet_names}")
        resistance_avgs = []
        chosen_col = st.text_input("Enter column name for resistance (case-sensitive):", value="Rsa")

        if st.button("⚡ Process Averages"):
            for sheet in sheet_names:
                df = pd.read_excel(excel_file, sheet_name=sheet)
                avg_val = df[chosen_col].mean() if chosen_col in df.columns else None
                resistance_avgs.append((sheet, avg_val))
            result_df = pd.DataFrame(resistance_avgs, columns=["Sheet", "AvgResistance"])
            out_buffer = io.BytesIO()
            result_df.to_excel(out_buffer, index=False)
            st.dataframe(result_df)
            st.download_button("📥 Download averages Excel", data=out_buffer.getvalue(), file_name="TLM_avg_output.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

elif page == "E5270B citi File Tool":
    st.header("CITI File → Excel Converter")
    uploaded_files = st.file_uploader("Upload .citi or .txt files", accept_multiple_files=True,
                                      type=["citi", "txt", "csv"])
    if "citi_preview_visible" not in st.session_state:
        st.session_state.citi_preview_visible = False

    if uploaded_files:
        parsed_sheets = []
        for f in uploaded_files:
            try:
                raw = f.read().decode("utf-8", errors="ignore")
                name = f.name.rsplit(".", 1)[0]
                if f.name.lower().endswith(".csv"):
                    df = parse_special_csv(raw)
                    ftype = "CSV"
                else:
                    df, ftype = parse_citi_file(raw)
                parsed_sheets.append((name, df, ftype))

                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                    df.to_excel(writer, sheet_name=name[:31], index=False)
                st.download_button(f"📥 Download {name}_excel.xlsx", data=buf.getvalue(), file_name=f"{name}_excel.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                   key=f"dl_{name}")
            except Exception as e:
                st.error(f"❌ {f.name}: {e}")

        if parsed_sheets:
            if st.button("Show/Hide Preview"):
                st.session_state.citi_preview_visible = not st.session_state.citi_preview_visible

            if st.session_state.citi_preview_visible:
                st.subheader("File Preview")
                for fname, df, ftype in parsed_sheets:
                    st.write(f"{fname} ({ftype})")
                    st.dataframe(df.head(), height=200)

            st.subheader("Grouped Downloads")
            groups = ["BE", "BC", "Gummel", "Family"]
            for g in groups:
                group_sheets = [(n, df) for n, df, ftype in parsed_sheets if ftype == g]
                if group_sheets:
                    buf = io.BytesIO()
                    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                        for n, df in group_sheets:
                            df.to_excel(writer, sheet_name=n.rsplit(".", 1)[0][:31], index=False)
                    buf.seek(0)
                    st.download_button(f"📥 Download {g} Excel ({len(group_sheets)} files)", data=buf,
                                       file_name=f"{g.upper()} Batch Output.xlsx",
                                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            st.subheader("Download All")
            all_buf = io.BytesIO()
            with pd.ExcelWriter(all_buf, engine="openpyxl") as writer:
                for n, df, _ in parsed_sheets:
                    df.to_excel(writer, sheet_name=n.rsplit(".", 1)[0][:31], index=False)
            all_buf.seek(0)
            st.download_button("📦 Download ALL files", data=all_buf, file_name="All_CITI_Converted.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.info("Upload one or more .citi files (BE/BC/Family/Gummel).")

elif page == "HP4155A Data Processing Tool":
    st.header("HP4155A SMU Table Processor")
    uploaded_files = st.file_uploader("Upload raw data file(s)", accept_multiple_files=True)
    if not uploaded_files: st.stop()

    results = []
    for uploaded in uploaded_files:
        raw = uploaded.getvalue().decode("utf-8", errors="ignore")
        try:
            df = pd.read_csv(io.StringIO(raw), delim_whitespace=True, engine="python")
        except Exception:
            df = pd.read_csv(io.StringIO(raw), engine="python")
        df.columns = df.columns.str.strip()
        results.append((uploaded.name or "data", df))

    st.subheader("SMU role assignment")
    roles = ["Collector", "Base", "Emitter", "PD", "None"]
    smu_cols = [("V1", "I1"), ("V2", "I2"), ("V3", "I3"), ("V4", "I4")]
    smu_role_map = {}
    for vcol, icol in smu_cols:
        choice = st.selectbox(f"{vcol}/{icol}", roles, key=f"hp_batch_{vcol}_{icol}")
        if choice != "None": smu_role_map[choice.lower()] = (vcol, icol)

    st.subheader("Measurement type")
    meas = st.selectbox("Select measurement", ["family Ic-Vc", "family L-Ic-Vc", "gummel", "BE diode", "BC diode"])
    responsivity = st.number_input("Responsivity (A/W)", value=0.35,
                                   format="%.3f") if meas == "family L-Ic-Vc" else None

    if st.button("⚡ Process"):
        per_file_outputs = []
        for uploaded in uploaded_files:
            raw = uploaded.getvalue().decode("utf-8", errors="ignore")
            try:
                df = pd.read_csv(io.StringIO(raw), delim_whitespace=True, engine="python")
            except:
                df = pd.read_csv(io.StringIO(raw), engine="python")
            df.columns = df.columns.str.strip()
            fname = sanitize_sheet_name(uploaded.name or "data")
            parsed = parse_smu_table(df, smu_role_map)

            if meas in ["family Ic-Vc", "family L-Ic-Vc", "gummel"] and any(
                k not in parsed for k in ["collector", "base", "emitter"]): continue
            if meas in ["BE diode", "BC diode"] and any(k not in parsed for k in (
            ["base", "emitter"] if meas == "BE diode" else ["base", "collector"])): continue

            buf = io.BytesIO()
            if meas.startswith("family"):
                vc, ic = parsed["collector"]
                _, ib = parsed["base"]
                vc_axis, ic_groups = group_family(vc, ic, ib)
                out_elec = pd.DataFrame({"Vc": vc_axis})
                for ib_val, series in ic_groups.items():
                    out_elec[f"Ic at Ib={format_ib_label(ib_val)}"] = series

                if meas == "family Ic-Vc":
                    out_elec.to_excel(buf, index=False)
                else:
                    if "pd" not in parsed:
                        st.error(f"{uploaded.name}: PD SMU required for L-Ic-Vc")
                        continue
                    _, ipd = parsed["pd"]
                    L = -ipd / responsivity
                    vc_axis, L_groups = group_family(vc, L, ib)
                    out_opt = pd.DataFrame({"Vc": vc_axis})
                    for ib_val, series in L_groups.items():
                        out_opt[f"L at Ib={format_ib_label(ib_val)}"] = series
                    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                        out_elec.to_excel(writer, sheet_name="Electrical", index=False)
                        out_opt.to_excel(writer, sheet_name=f"Optical_responsivity_{responsivity}", index=False)

            elif meas == "gummel":
                vb, ib = parsed["base"]
                _, ic = parsed["collector"]
                out = pd.DataFrame({"Vb": vb, "Ib": ib, "Ic": ic, "Beta": ic / ib.replace({0: np.nan})})
                out.to_excel(buf, index=False)

            elif meas in ["BE diode", "BC diode"]:
                v, i = parsed["base"] if meas == "BE diode" else parsed["collector"]
                out = pd.DataFrame({"V": v, "I": abs(i)})
                out.to_excel(buf, index=False)

            per_file_outputs.append((fname, buf.getvalue()))

        st.subheader("Per-file downloads")
        for fname, data in per_file_outputs:
            st.download_button(f"📥 {fname}.xlsx", data=data, file_name=f"{fname}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               key=f"dl_hp_{fname}")

        if per_file_outputs:
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for fname, data in per_file_outputs: zf.writestr(f"{fname}.xlsx", data)
            st.download_button("📦 Download ALL (ZIP)", data=zip_buf.getvalue(), file_name="SMU_Table_ALL.zip",
                               mime="application/zip")
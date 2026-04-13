import streamlit as st

# ⚠️ 整個專案的網頁設定，統一在這裡宣告一次
st.set_page_config(page_title="IOED Lab Portal", layout="wide", page_icon="🔬")

def check_password():
    def password_entered():
        try:
            correct_pwd = st.secrets["APP_PASSWORD"]
        except Exception:
            correct_pwd = "IOED" # 本機測試預設密碼
        if st.session_state["pwd_input"] == correct_pwd:
            st.session_state["authenticated"] = True
            del st.session_state["pwd_input"]
        else:
            st.session_state["authenticated"] = False

    if st.session_state.get("authenticated", False):
        return True

    st.title("🔬 IOED Lab Unified Portal")
    st.divider()
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("Please enter IOED Lab Password. / 請輸入 IOED 實驗室專屬密碼。")
        st.text_input("Access Password / 存取密碼", type="password", on_change=password_entered, key="pwd_input")
        if "authenticated" in st.session_state and not st.session_state["authenticated"]:
            st.error("❌ Password Incorrect / 密碼錯誤")
    return False

# 1. 攔截未登入的使用者
if not check_password():
    st.stop()

# 2. 定義功能頁面 (指向 tools 資料夾內的 Python 檔)
# [現有功能]
gummel_page = st.Page("tools/IOED_Gummel_Analyzer_3.py", title="Sim Gummel Plot Analyzer", icon="📈")
sim_rf_page = st.Page("tools/IOED_RF_Analyzer_16.py", title="Sim RF Plot Analyzer", icon="📈")
rf_page = st.Page("tools/IOED_HBT_RF_extract.py", title="RF S-Parameter Extraction", icon="📡")

# [新增功能]
b1500a_page = st.Page("tools/B1500A_Plot.py", title="B1500A plot & TLM", icon="📊")
hp4155a_page = st.Page("tools/HP4155A_plot.py", title="HP4155A Quick Plot", icon="📉")
csv_process_page = st.Page("tools/csv_process.py", title="Measurement Data Muti-Process", icon="🗂️")

# 3. 建立側邊欄群組導航選單
pg = st.navigation({
    "高頻量測 (RF)": [rf_page],
    "元件模擬 (TCAD)": [gummel_page, sim_rf_page],
    "直流量測 (DC)": [b1500a_page, hp4155a_page],
    "資料處理 (Data)": [csv_process_page]
})

# 4. 側邊欄標題裝飾
st.sidebar.title("🔬 IOED Lab Portal")
st.sidebar.caption("整合式元件分析與萃取平台")
st.sidebar.divider()

# 5. 執行導航
pg.run()

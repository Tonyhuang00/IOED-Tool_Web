import importlib.util, subprocess, sys, os
from pathlib import Path

def ensure(pkg, pip_name=None):
    if importlib.util.find_spec(pkg) is None:
        print(f"Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name or pkg])

# 確保所有繪圖與資料處理套件都已安裝
for pkg, pip in [("openpyxl", None), ("pandas", None), ("streamlit", None),
                  ("numpy", None), ("plotly", None)]:
    ensure(pkg, pip)

# 指向你的整合主程式檔案 main.py
file = Path(__file__).parent / "IOED_Tool_Web.py"

if not file.exists():
    print(f"❌ 找不到主程式：{file.name}！")
    print("請確認 main.py 與 LAUNCH_Tool.py 在同一個資料夾底下。")
    input("按 Enter 鍵結束...")
    sys.exit(1)

print("==================================================")
print("🔬 正在啟動 IOED 整合式元件分析與萃取平台...")
print("==================================================")

try:
    # 自動使用 streamlit run 執行 main.py
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(file.resolve())])
except KeyboardInterrupt:
    print("\n伺服器已安全關閉。")

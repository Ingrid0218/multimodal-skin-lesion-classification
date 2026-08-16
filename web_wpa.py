import streamlit as tf
import sqlite3

DB_NAME = "mlb_wpa.db"

def query_state(inning, half, score_diff, outs, base_state, balls, strikes):
    """ 從 SQLite 查詢 RE 與 WP """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    sql = """
        SELECT re, wp FROM wpa_table 
        WHERE inning = ? 
          AND half = ? 
          AND score_diff = ? 
          AND outs = ? 
          AND base_state = ? 
          AND balls = ? 
          AND strikes = ?
    """
    # 進行分差的範圍夾擠 (±5分)
    score_diff_clamped = max(-5, min(5, score_diff))
    
    params = (int(inning), str(half), int(score_diff_clamped), int(outs), str(base_state), int(balls), int(strikes))
    
    try:
        cursor.execute(sql, params)
        result = cursor.fetchone()
        conn.close()
        return result, score_diff_clamped
    except Exception as e:
        conn.close()
        return None, score_diff_clamped

# === Streamlit 網頁介面設計 ===
tf.set_page_config(page_title="MLB RE288/WPA 計算機", page_icon="⚾", layout="centered")

tf.title("⚾ MLB RE288 / WPA 即時查詢網頁")
tf.divider()

# 使用 Streamlit 的 Layout 欄位功能排版
col1, col2, col3 = tf.columns(3)

with col1:
    inning = tf.number_input("局數 (Inning)", min_value=1, max_value=15, value=1, step=1)
    half = tf.selectbox("半局 (Half)", options=["Top (上半局)", "Bottom (下半局)"])
    # 轉換為資料庫對齊的字串
    db_half = "Top" if "Top" in half else "Bottom"

with col2:
    score_diff = tf.number_input("分差 (主隊 - 客隊)", min_value=-20, max_value=20, value=0, step=1)
    outs = tf.selectbox("出局數 (Outs)", options=[0, 1, 2])

with col3:
    # 這裡的壘包代碼預設使用你資料庫診斷出的 '000' 格式
    base_state = tf.selectbox("壘包狀態 (Bases)", options=["000", "100", "010", "001", "110", "101", "011", "111"], 
                             format_func=lambda x: {
                                 "000": "無人在壘 (000)", "100": "一壘有人 (100)", 
                                 "010": "二壘有人 (010)", "001": "三壘有人 (001)",
                                 "110": "一二壘有人 (110)", "101": "一三壘有人 (101)", 
                                 "011": "二三壘有人 (011)", "111": "滿壘 (111)"
                             }.get(x, x))
    
    # 球數選單
    balls = tf.selectbox("壞球 (Balls)", options=[0, 1, 2, 3])
    strikes = tf.selectbox("好球 (Strikes)", options=[0, 1, 2])

tf.divider()

# 執行查詢
res, clamped_diff = query_state(inning, db_half, score_diff, outs, base_state, balls, strikes)

# 顯示提示訊息（若分差有被收斂）
if score_diff != clamped_diff:
    tf.warning(f"💡 提示：輸入分差 {score_diff} 已自動校正為資料庫邊界分差 {clamped_diff} 分。")

if res:
    re_val, wp_val = res
    
    # 使用 Streamlit 美觀的 Metric 元件顯示數據
    m_col1, m_col2 = tf.columns(2)
    with m_col1:
        tf.metric(label="📊 預期得分 (RE Value)", value=f"{re_val:.3f}")
    with m_col2:
        tf.metric(label="📈 當前勝率 (Win Expectancy)", value=f"{float(wp_val)*100:.2f}%")
        
    # 進度條視覺化勝率
    tf.progress(float(wp_val))
else:
    tf.error("❌ 找不到對應的比賽狀態，請檢查此組合在現實棒球規則中是否存在。")
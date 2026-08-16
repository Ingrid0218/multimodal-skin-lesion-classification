import sqlite3
import sys

DB_NAME = "mlb_wpa.db"

def diagnose_db():
    """ 診斷工具：啟動時先撈一筆資料，確認資料庫裡的實際格式 """
    print("\n🔍 [系統診斷] 正在檢查資料庫實際欄位格式...")
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT inning, half, outs, base_state, balls, strikes, score_diff, re, wp FROM wpa_table LIMIT 1;")
        row = cursor.fetchone()
        conn.close()
        
        if row:
            print("-" * 55)
            print(" 資料庫真實數據範例：")
            print(f" 局數(inning): {row[0]} (型態: {type(row[0]).__name__})")
            print(f" 半局(half): '{row[1]}'")
            print(f" 出局(outs): {row[2]}")
            print(f" 壘包(base_state): '{row[3]}' <-- 💡請檢查你的輸入是否跟這外觀一樣！")
            print(f" 壞球(balls): {row[4]}")
            print(f" 好球(strikes): {row[5]}")
            print(f" 分差(score_diff): {row[6]}")
            print(f" RE: {row[7]}, WP: {row[8]}")
            print("-" * 55)
        else:
            print("❌ 警告：wpa_table 裡面似乎沒有任何資料！")
    except Exception as e:
        print(f"❌ 診斷失敗，錯誤訊息: {e}")

def banner():
    print("=" * 55)
    print(" ⚾ MLB RE288 / WPA 高速查詢工具 (自動容錯診斷版) ⚾")
    print("=" * 55)
    print(" 輸入格式: [局數] [半局] [分差] [出局] [壘包] [壞球] [好球]")
    print(" 範例說明: 9 b 0 2 000 3 2")
    print(" 容錯機制: 半局支援 t/b/top/down，分差超出 ±5 自動校正。")
    print(" 輸入 'q' 或 'exit' 可結束程式。")
    print("=" * 55)

def parse_half(half_input):
    """ 統一處理半局英文字母 """
    h = half_input.strip().lower()
    half_map = {
        't': 'Top', 'top': 'Top', 'u': 'Top', 'up': 'Top',
        'b': 'Bottom', 'Bottom': 'Down', 'd': 'Bottom', 'down': 'Bottom'
    }
    return half_map.get(h, None)

def query_state(inning, half, score_diff, outs, base_state, balls, strikes):
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
    # 這裡確保所有傳入參數型態正確，base_state 轉成字串
    params = (int(inning), str(half), int(score_diff), int(outs), str(base_state), int(balls), int(strikes))
    
    try:
        cursor.execute(sql, params)
        result = cursor.fetchone()
        conn.close()
        return result
    except Exception as e:
        print(f"❌ 查詢出錯: {e}")
        conn.close()
        return None

def main():
    diagnose_db()
    banner()
    
    while True:
        try:
            user_input = input("\n👉 請輸入比賽狀態: ").strip()
            if user_input.lower() in ['q', 'exit']:
                print("掰掰！祝你預測準確 ⚾")
                break
                
            args = user_input.split()
            if len(args) != 7:
                print("⚠️  格式錯誤！必須剛好輸入 7 個參數（用空格隔開）。")
                continue
                
            # 1. 處理半局容錯
            half = parse_half(args[1])
            if not half:
                print("⚠️  半局輸入錯誤！請用 T、B、Top 或 Down。")
                continue
                
            # 2. 處理分差 ±5 自動收斂
            raw_score_diff = int(args[2])
            score_diff = max(-5, min(5, raw_score_diff))
            if score_diff != raw_score_diff:
                print(f"💡 提示：分差 {raw_score_diff} 已自動收斂至邊界分差 {score_diff}")

            inning = int(args[0])
            outs = int(args[3])
            base_state = args[4]
            balls = int(args[5])
            strikes = int(args[6])
            
            # 3. 查表
            res = query_state(inning, half, score_diff, outs, base_state, balls, strikes)
            
            if res:
                re_val, wp_val = res
                print("-" * 40)
                print(f"📊 查詢結果 ({inning}局{half} / 分差:{score_diff} / {outs}死 / 球數:{balls}-{strikes})：")
                print(f"   預期得分 (RE): {re_val:.3f}")
                print(f"   當前勝率 (WP): {float(wp_val)*100:.2f}%")
                print("-" * 40)
            else:
                print("❌ 找不到對應的比賽狀態，請對照上方[系統診斷]的資料格式是否有輸入落差。")
                
        except ValueError:
            print("⚠️  型態錯誤！局數、分差、出局、球數必須是數字。")
        except KeyboardInterrupt:
            print("\n掰掰！")
            break

if __name__ == "__main__":
    main()
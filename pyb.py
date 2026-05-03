from pybaseball import statcast
import pandas as pd
import time

years = [2026, 2024, 2020, 2018, 2016]
all_summary = []

for year in years:
    print(f"正在處理 {year} 年數據...")
    
    # 為了效能與示範，我們抓取每年開季第一個月的數據作為代表
    # 如果要全賽季，請將日期改為 {year}-03-28 到 {year}-10-01，但請注意記憶體量
    start_date = f"{year}-03-28"
    end_date = f"{year}-10-31"
    
    try:
        # 抓取數據
        data = statcast(start_dt=start_date, end_dt=end_date)
        
        # 排除沒有擊球數據的資料 (必須有初速、仰角、距離其中之一)
        clean_data = data.dropna(subset=['launch_speed', 'launch_angle', 'hit_distance_sc'])
        
        # 按左右打 (stand) 分組並計算平均值
        summary = clean_data.groupby('stand').agg({
            'launch_speed': 'mean',
            'launch_angle': 'mean',
            'hit_distance_sc': 'mean'
        }).reset_index()
        
        # 加入年份標記
        summary['year'] = year
        all_summary.append(summary)
        
        # 稍微停頓，避免請求過快
        time.sleep(1)
        
    except Exception as e:
        print(f"{year} 年數據抓取失敗: {e}")

# 合併所有年份結果
final_report = pd.concat(all_summary, ignore_index=True)

# 重新整理欄位順序，方便閱讀
final_report = final_report[['year', 'stand', 'launch_speed', 'launch_angle', 'hit_distance_sc']]
final_report.columns = ['年份', '打席', '平均初速(mph)', '平均仰角(deg)', '平均飛行距離(ft)']

print("\n--- 多年份左右打擊球數據匯總 ---")
print(final_report.sort_values(['年份', '打席'], ascending=[False, True]))
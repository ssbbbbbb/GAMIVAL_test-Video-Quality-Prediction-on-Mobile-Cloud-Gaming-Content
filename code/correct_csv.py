import pandas as pd
import os

# 設定檔案路徑
input_csv = r"E:\FOLDER\computer_vision\data\LIVE-Meta-Gaming_metadata.csv"   # 原始 CSV 檔案
output_csv = r"E:\FOLDER\computer_vision\data\Metadata\LIVE-Meta-Gaming_metadata.csv" # 修改後儲存的 CSV 檔案

# 讀取 CSV
df = pd.read_csv(input_csv)

# 修改第一欄：只保留檔名（basename）
df.iloc[:, 0] = df.iloc[:, 0].apply(lambda x: os.path.basename(str(x)))

# 儲存結果
df.to_csv(output_csv, index=False)

print(f'✅ 已將第一欄轉為檔名並儲存至：{output_csv}')

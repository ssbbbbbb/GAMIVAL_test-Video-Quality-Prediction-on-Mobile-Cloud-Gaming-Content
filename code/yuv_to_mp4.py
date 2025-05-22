import pandas as pd

# 讀取原始 CSV
df = pd.read_csv(r"E:\FOLDER\computer_vision\data\Metadata\LIVE-Meta-MobileCloudGaming_metadata.csv")  # ← 請把檔名改成你的

# 假設第一欄是 File 名稱
col_name = df.columns[0]

# 只從第二列開始改 .yuv → .mp4
df.loc[1:, col_name] = df.loc[1:, col_name].apply(lambda x: x.replace('.yuv', '.mp4'))

# 輸出成新檔案
df.to_csv(r"E:\FOLDER\computer_vision\data\Metadata\LIVE-Meta-MobileCloudGaming_metadata1.csv", index=False)

print("✅ 完成：已另存為 updated_output.csv")

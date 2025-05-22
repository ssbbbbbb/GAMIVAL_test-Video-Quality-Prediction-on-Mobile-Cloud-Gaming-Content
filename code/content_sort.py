import pandas as pd
import numpy as np
import os
from sklearn.model_selection import GroupShuffleSplit

def generate_group_splits(csv_path, num_splits=6, output_file='LIVE-Meta-Gaming_idx.npy', base_seed=42):
    df = pd.read_csv(csv_path)

    # y, groups 是所有樣本對應的資訊
    groups = df['Content'].to_numpy()
    all_train_content = []
    all_test_content = []

    for i in range(num_splits):
        # 每次切分使用不同 seed（base_seed + i）
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=base_seed + i)
        train_idx, test_idx = next(gss.split(df, groups=groups))
        
        train_groups = np.unique(groups[train_idx])
        test_groups = np.unique(groups[test_idx])
        all_train_content.append(train_groups)
        all_test_content.append(test_groups)

    with open(output_file, 'wb') as f:
        np.save(f, all_train_content)
        np.save(f, all_test_content)

    print(f"✅ 已使用不同種子產生 {num_splits} 組 8:2 的內容切分檔：{output_file}")

if __name__ == '__main__':
    # 修改路徑與檔案名稱以符合你的實際環境
    csv_path = 'E:/FOLDER/computer_vision/GAMIVAL-ex/GAMIVAL-main/mos_files/LIVE-Meta-Gaming_metadata.csv'
    output_file = 'E:/FOLDER/computer_vision/GAMIVAL-ex/GAMIVAL-main/LIVE-Meta-Gaming_idx.npy'
    generate_group_splits(csv_path, num_splits=6, output_file=output_file)

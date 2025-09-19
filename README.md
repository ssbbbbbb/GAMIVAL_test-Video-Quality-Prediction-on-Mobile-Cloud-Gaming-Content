# GAMIVAL_test-Video-Quality-Prediction-on-Mobile-Cloud-Gaming-Content

## 第一步
利用CNN對影片資料集進行影片特徵提取

```
code code/demo_compute_CNN_feats.py
input Metadata/LIVE-Meta-MobileCloudGaming_metadata.csv
output feat_files/LIVE-Meta-Mobile-Cloud-Gaming_CNN_bicubic_feats.mat
```


## 第二步
利用NSS對影片資料集進行影片特徵提取

```
code code/demo_compute_NSS_feats.mat
input Metadata/LIVE-Meta-MobileCloudGaming_metadata.csv
output feat_files/LIVE-Meta-Mobile-Cloud-Gaming_NSS_bicubic_feats.mat
```


## 第三步
把CNN提取的特徵向量與NSS提取的特徵向量合併

```
code code/combineFeature.mat
input feat_files/LIVE-Meta-Mobile-Cloud-Gaming_CNN_bicubic_feats.mat
input feat_files/LIVE-Meta-Mobile-Cloud-Gaming_NSS_bicubic_feats.mat
output feat_files/LIVE-Meta-Mobile-Cloud-Gaming_GAMIVAL_feats.mat
```


## 第四步
將資料集按照content分成訓練集:資料集=8:2
避免訓練與測試資料相似

```
code code/content_sort.py
input Metadata/LIVE-Meta-MobileCloudGaming_metadata
output LIVE-Meta-Gaming_idx.npy
```


## 第五步
對SVR進行主觀評分訓練

```
code code/evaluate_bvqa_features_regression.py
input Metadata/LIVE-Meta-MobileCloudGaming_metadata
output result/LIVE-Meta-Mobile-Cloud-Gaming_GAMIVAL_SVR_corr.mat
```

LIVE-Meta-Mobile-Cloud-Gaming_GAMIVAL_SVR_corr.mat裡表格的內容分別是:

第一次:SRCC_train|KRCC_train|PLCC_train|RMSE_train|SRCC_test|KRCC_test|PLCC_test|RMSE_test  
第二次:...  
...  
第六次:...  

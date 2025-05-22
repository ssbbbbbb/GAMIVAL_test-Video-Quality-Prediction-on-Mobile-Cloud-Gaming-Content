import ffmpeg
import numpy as np
import imageio
np.random.seed(7)
import tensorflow as tf
tf.compat.v1.Session()
tf.random.set_seed(9)
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import densenet
import os
import pandas as pd
import argparse
import scipy.io
import math
import time

# ✅ 啟用 GPU 記憶體增長，避免 CUDA 記憶體被一口氣吃滿
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU 設定錯誤：{e}")

def test_video(NDGmodel, videopath, videoname, framerate):    
    probe = ffmpeg.probe(videoname) 
    video_info = next(x for x in probe['streams'] if x['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])
    
    out, err = (
        ffmpeg.input(videoname)
              .output('pipe:', format='rawvideo', pix_fmt='rgb24')
              .run(capture_stdout=True)
    )
    video = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
    
    preds_patch = []
    for k in range(math.floor(video.shape[0] / framerate)):
        # Frame 1
        ims1 = video[k * math.ceil(framerate)]
        h1, w1, _ = ims1.shape
        patch_rows = h1 // 299
        patch_cols = w1 // 299
        if patch_rows == 0 or patch_cols == 0:
            continue  # 略過太小的 frame
        patches1 = np.zeros((patch_rows, patch_cols, 299, 299, 3))
        for i in range(patch_rows):
            for j in range(patch_cols):
                patches1[i, j] = ims1[i*299:(i+1)*299, j*299:(j+1)*299]
        patches1 = densenet.preprocess_input(patches1.reshape((-1, 299, 299, 3)))
        pred_patch1 = NDGmodel.predict(patches1, verbose=0)

        # Frame 2
        ims2 = video[k * math.ceil(framerate) + math.floor(framerate / 2)]
        h2, w2, _ = ims2.shape
        patch_rows2 = h2 // 299
        patch_cols2 = w2 // 299
        if patch_rows2 == 0 or patch_cols2 == 0:
            continue  # 略過太小的 frame
        patches2 = np.zeros((patch_rows2, patch_cols2, 299, 299, 3))
        for i in range(patch_rows2):
            for j in range(patch_cols2):
                patches2[i, j] = ims2[i*299:(i+1)*299, j*299:(j+1)*299]
        patches2 = densenet.preprocess_input(patches2.reshape((-1, 299, 299, 3)))
        pred_patch2 = NDGmodel.predict(patches2, verbose=0)

        mean_vector = (np.mean(pred_patch1, axis=0) + np.mean(pred_patch2, axis=0)) / 2
        preds_patch.append(mean_vector)

    if len(preds_patch) == 0:
        raise ValueError("無有效 patch，請確認影片尺寸是否小於 299x299")
        
    return np.mean(preds_patch, axis=0), preds_patch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='LIVE-Meta-MobileCloudGaming')
    parser.add_argument('-mp', '--model', dest='model', default=r'E:\FOLDER\computer_vision\GAMIVAL-main\models\subjectiveDemo2_DMOS_Final.model')
    args = parser.parse_args()

    dataset_name = args.dataset_name
    model_path = args.model
    videopath = r"E:\FOLDER\computer_vision\data\mp4"
    output_path = r"E:\FOLDER\computer_vision\GAMIVAL-main\feat_files\LIVE-Meta-Mobile-Cloud-Gaming_CNN_bicubic_feats.mat"

    csv_file = os.path.join(r'E:\FOLDER\computer_vision\data\Metadata', dataset_name + '_metadata.csv')
    df = pd.read_csv(csv_file)
    videoname = df['File'].to_numpy()
    framerate = df['framerate'].to_numpy()

    feature_patch_total = []
    feats_frame_patch_total = np.empty((len(videoname), 1), dtype=object)

    NDGmodel = load_model(model_path)
    NDGmodel = tf.keras.Model(inputs=NDGmodel.input, outputs=NDGmodel.layers[-2].output)

    for i in range(len(videoname)):
        t0 = time.time()
        full_path = os.path.join(videopath, videoname[i].replace('/', os.sep))

        if not os.path.exists(full_path):
            print(f"找不到影片：{full_path}，跳過")
            continue

        try:
            feature_patch, feats_frame_patch = test_video(NDGmodel, videopath, full_path, framerate[i])
            feature_patch_total.append(feature_patch)
            feats_frame_patch_total[i, 0] = feats_frame_patch
            print(f"[{i+1}/{len(videoname)}] 處理完成 ({time.time() - t0:.2f} 秒)")
        except Exception as e:
            print(f"錯誤：{videoname[i]} - {str(e)}")
            continue

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    scipy.io.savemat(output_path, mdict={'feats_mat': np.asarray(feature_patch_total, dtype=np.float64)})
    print(f"特徵已儲存：{output_path}")

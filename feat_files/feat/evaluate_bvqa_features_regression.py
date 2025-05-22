# -*- coding: utf-8 -*-
import os
import sys
import time
import numpy as np
import scipy.io
import scipy.stats
import pandas
import argparse
from sklearn.impute import SimpleImputer
from sklearn.svm import SVR, LinearSVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from scipy.optimize import curve_fit
import warnings

warnings.filterwarnings("ignore")

class Logger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "a")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='GAME')
    parser.add_argument('--dataset_name', type=str, default='LIVE-Meta-Gaming')
    parser.add_argument('--feature_file', type=str, default='E:/FOLDER/computer_vision/GAMIVAL-main/feat_files/LIVE-Meta-Mobile-Cloud-Gaming_GAMIVAL_feats.mat')
    parser.add_argument('--out_file', type=str, default='E:/FOLDER/computer_vision/GAMIVAL-main/result/LIVE-Meta-Mobile-Cloud-Gaming_GAMIVAL_SVR_corr')
    parser.add_argument('--predicted_score', type=str, default='E:/FOLDER/computer_vision/GAMIVAL-main/predicted_score/LIVE-Meta-Mobile-Cloud-Gaming_GAMIVAL_SVR_predicted_score.mat')
    parser.add_argument('--best_parameter', type=str, default='E:/FOLDER/computer_vision/GAMIVAL-main/best_pamtr/LIVE-Meta-Mobile-Cloud-Gaming_GAMIVAL_SVR_pamtr')
    parser.add_argument('--log_file', type=str, default='E:/FOLDER/computer_vision/GAMIVAL-main/logs/LIVE-Meta-Mobile-Cloud-Gaming_GAMIVAL_SVR.log')
    parser.add_argument('--log_short', action='store_true')
    parser.add_argument('--use_parallel', action='store_true')
    parser.add_argument('--num_iterations', type=int, default=6)
    parser.add_argument('--max_thread_count', type=int, default=4)
    return parser.parse_args()

def logistic_func(X, b1, b2, b3, b4):
    logisticPart = 1 + np.exp(-(X - b3) / np.abs(b4))
    return b2 + (b1 - b2) / logisticPart

def compute_metrics(y_pred, y):
    SRCC = scipy.stats.spearmanr(y, y_pred)[0]
    try:
        KRCC = scipy.stats.kendalltau(y, y_pred)[0]
    except:
        KRCC = scipy.stats.kendalltau(y, y_pred, method='asymptotic')[0]
    beta_init = [np.max(y), np.min(y), np.mean(y_pred), 0.5]
    popt, _ = curve_fit(logistic_func, y_pred, y, p0=beta_init, maxfev=int(1e8))
    y_pred_log = logistic_func(y_pred, *popt)
    PLCC = scipy.stats.pearsonr(y, y_pred_log)[0]
    RMSE = np.sqrt(mean_squared_error(y, y_pred_log))
    return [SRCC, KRCC, PLCC, RMSE], y_pred_log

def formatted_print(snapshot, params, duration):
    print('======================================================')
    print('params: ', params)
    print('SRCC_train:', snapshot[0])
    print('KRCC_train:', snapshot[1])
    print('PLCC_train:', snapshot[2])
    print('RMSE_train:', snapshot[3])
    print('======================================================')
    print('SRCC_test:', snapshot[4])
    print('KRCC_test:', snapshot[5])
    print('PLCC_test:', snapshot[6])
    print('RMSE_test:', snapshot[7])
    print('======================================================')
    print(f' -- {duration:.2f} seconds elapsed...\n\n')

def final_avg(snapshot):
    def show(label, pos):
        values = [x[pos] for x in snapshot]
        print(f'{label}: mean={np.mean(values):.4f}, median={np.median(values):.4f}, std={np.std(values):.4f}')
    print('======================================================')
    print('Average training results:')
    show("SRCC Train", 0)
    show("KRCC Train", 1)
    show("PLCC Train", 2)
    show("RMSE Train", 3)
    print('======================================================')
    print('Average testing results:')
    show("SRCC Test", 4)
    show("KRCC Test", 5)
    show("PLCC Test", 6)
    show("RMSE Test", 7)
    print('\n')

def evaluate_bvqa(X_train, X_test, y_train, y_test, log_short):
    t_start = time.time()
    if X_train.shape[1] <= 4000:
        param_grid = {'C': np.logspace(1, 10, 10, base=2),
                      'gamma': np.logspace(-10, -6, 5, base=2)}
        grid = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=8, n_jobs=4, verbose=2)
    else:
        param_grid = {'C': [0.001, 0.01, 0.1, 1., 2.5, 5., 10.],
                      'epsilon': [0.001, 0.01, 0.1, 1., 2.5, 5., 10.]}
        grid = GridSearchCV(LinearSVR(max_iter=100), param_grid, n_jobs=4, cv=8, verbose=2)

    scaler = preprocessing.MinMaxScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    grid.fit(X_train, y_train)
    best_params = grid.best_params_

    if X_train.shape[1] <= 4000:
        model = SVR(**best_params)
    else:
        model = LinearSVR(**best_params)
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_metrics, _ = compute_metrics(y_train_pred, y_train)
    test_metrics, y_test_pred_log = compute_metrics(y_test_pred, y_test)
    if not log_short:
        formatted_print(train_metrics + test_metrics, best_params, time.time() - t_start)
    return best_params, train_metrics, test_metrics, y_test_pred_log

def main(args):
    base_path = 'E:/FOLDER/computer_vision/GAMIVAL-main'
    df = pandas.read_csv(os.path.join(base_path, 'mos_files', args.dataset_name + '_metadata.csv'))
    y = df['MOS'].to_numpy(dtype=np.float32)
    content = df['Content'].to_numpy()
    X = scipy.io.loadmat(args.feature_file)['feats_mat'].astype(np.float32)
    X[np.isinf(X)] = np.nan
    X = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(X)

    all_results = []
    best_parameters = []

    for i in range(args.num_iterations):
        split_file = os.path.join(base_path, f'{args.dataset_name}_idx.npy')
        if os.path.exists(split_file):
            with open(split_file, 'rb') as f:
                train_content = np.load(f, allow_pickle=True)[i]
                test_content = np.load(f, allow_pickle=True)[i]
            X_train = np.array([X[j] for j in range(len(content)) if content[j] in train_content])
            y_train = np.array([y[j] for j in range(len(content)) if content[j] in train_content])
            X_test = np.array([X[j] for j in range(len(content)) if content[j] in test_content])
            y_test = np.array([y[j] for j in range(len(content)) if content[j] in test_content])
       # else:
        #    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42+i)

        best_params, train_metrics, test_metrics, y_pred = evaluate_bvqa(X_train, X_test, y_train, y_test, args.log_short)
        all_results.append(train_metrics + test_metrics)
        best_parameters.append(best_params)

    final_avg(all_results)
    print(f'Finished {args.num_iterations} iterations.')

    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    os.makedirs(os.path.dirname(args.best_parameter), exist_ok=True)

    np.save(args.out_file + ".npy", np.array(all_results))
    scipy.io.savemat(args.out_file + ".mat", {'all_iterations': np.array(all_results)})
    scipy.io.savemat(args.best_parameter + ".mat", {'best_parameters': best_parameters})

if __name__ == '__main__':
    args = arg_parser()
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    sys.stdout = Logger(args.log_file)
    print(args)
    main(args)

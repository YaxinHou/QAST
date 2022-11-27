import os
import time
import logging
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import GradientBoostingClassifier as dbdt


def getnewf1(y_test, pre_label, min_class):
    labels = np.unique(y_test)
    p_class_macro, r_class_macro, f_class_macro, support_macro = \
        precision_recall_fscore_support(y_test, pre_label, labels=labels, average='macro', )
    p_class_micro, r_class_micro, f_class_micro, support_micro = \
        precision_recall_fscore_support(y_test, pre_label, labels=labels, average='micro', )
    p_class_weighted, r_class_weighted, f_class_weighted, support_weighted = \
        precision_recall_fscore_support(y_test, pre_label, labels=labels, average='weighted', )
    p_class_none, r_class_none, f_class_none, support_micro_none = \
        precision_recall_fscore_support(y_test, pre_label, labels=labels, average=None)
    pAR = 0
    for i in min_class:
        pAR = (pAR + r_class_none[i]) / len(min_class)

    acc = accuracy_score(y_test, pre_label)
    g_mean = 1
    for i in r_class_none:
        g_mean = g_mean * i
    g_mean = g_mean ** (1 / len(r_class_none))
    g_mean = round(g_mean, 4)

    # print(round(acc, 4), end='\t')
    # print(round(f_class_macro, 4), end='\t')
    # print(round(f_class_micro, 4), end='\t')
    # print(round(f_class_weighted, 4), end='\t')
    # print(round(g_mean, 4), end='\t')
    # print(round(r_class_macro, 4), end='\t')
    # print(round(pAR, 4))

    return acc, f_class_macro, f_class_micro, f_class_weighted, g_mean, r_class_macro, pAR


def get_log(log_path, logging_name):
    logger = logging.getLogger(logging_name)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path, mode='a')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def ags_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='dataset setting',
                        default='CWRU_Bearing_Dataset_1_20')

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = ags_parse()

    if not os.path.exists('./GBDT_LOG/'):
        os.makedirs('./GBDT_LOG/')

    logger = get_log('./GBDT_LOG/log-GBDT.txt', 'QAST')

    # the path of train and test data
    dir_path = '../data/splited_data'

    all_data_filename = ['abalone', 'balance', 'car', 'clave', 'contraceptive',
                         'dermatology', 'ecoli', 'flare', 'glass', 'yeast',
                         'mfcc', 'new-thyroid', 'nursery', 'pageblocks', 'satimage',
                         'shuttle', 'thyroid', 'winequality-red', 'winequality-white', 'log2',
                         'CWRU_Bearing_Dataset_1_10', 'CWRU_Bearing_Dataset_1_2',
                         'CWRU_Bearing_Dataset_1_20', 'CWRU_Bearing_Dataset_1_5',
                         'Gearbox_Fault_Diagnosis_1_10', 'Gearbox_Fault_Diagnosis_1_2',
                         'Gearbox_Fault_Diagnosis_1_20', 'Gearbox_Fault_Diagnosis_1_5',
                         'CM1', 'JM1', 'KC1', 'KC3', 'MC1', 'MC2',
                         'MW1', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5']

    # minority classes
    all_min_class_id = [[0, 1, 13, 14, 15, 16, 17, 18, 19], [1], [2, 3], [0], [2],
                        [0, 2, 3, 4, 5], [1, 2, 3, 4], [4, 5], [0, 3, 4, 5], [4, 5, 6, 7, 8],
                        [3], [0, 2], [1], [0, 2, 3], [2, 4, 5],
                        [0, 2], [1, 2], [0, 1, 5], [0, 1, 5], [3],
                        [1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9],
                        [1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9],
                        [1], [1],
                        [1], [1],
                        [1], [1], [1], [1], [1], [1],
                        [1], [1], [1], [1], [1], [1]]

    f = all_data_filename.index(args.dataset)
    file = all_data_filename[f] + '.xlsx'
    min_class = all_min_class_id[f]
    data_name = file.split('.')[0]

    logger.info('dataset：' + data_name)

    start_time = time.time()

    org_file_path = os.path.join(dir_path, file)

    # the path of generated data
    temp_path = data_name + '.csv'
    gen_dir = '../data/generated_data/' + data_name
    gen_path = os.path.join(gen_dir, temp_path)

    # original training data
    tr_data = pd.read_excel(org_file_path, header=None, index_col=None, sheet_name='Sheet1')
    tr_data = np.array(tr_data)
    tr_x = tr_data[:, 0:-1]
    tr_y = tr_data[:, -1]

    all_dict_max = {}
    max_num = []
    all_dict_min = {}
    min_num = {}
    for i in range(len(np.unique(tr_y))):
        if i in min_class:
            all_dict_min[i] = np.where(tr_y == i)
        else:
            all_dict_max[i] = np.where(tr_y == i)

    for k, v in all_dict_max.items():
        max_num.append(len(tr_y[v]))
    for k, v in all_dict_min.items():
        min_num[k] = len(tr_y[v])
    beta = min(max_num)
    gama = max(max_num)
    med = int((beta + gama) / 2)

    if med / beta >= 3:
        med = beta * 3

    # original testing data
    te_data = pd.read_excel(org_file_path, header=None, index_col=None, sheet_name='Sheet2')
    te_data = np.array(te_data)
    te_x = te_data[:, 0:-1]
    te_y = te_data[:, -1]

    # generated data
    ge_data = pd.read_csv(gen_path, header=None, index_col=None)
    ge_data = np.array(ge_data)
    ge_x = ge_data[:, 0:-1]
    ge_y = ge_data[:, -1]

    # add generated data to original training data
    min_dict = {}
    for i in min_class:
        min_dict[i] = np.where(ge_y == i)
    tr_y = list(tr_y)
    for k, v in min_dict.items():
        insert_num = med - min_num[k]
        data_sel = ge_x[v][0:insert_num]
        label_sel = ge_y[v][0:insert_num]
        tr_x = np.row_stack((tr_x, data_sel))
        label_sel = list(label_sel)
        tr_y.extend(label_sel)

    # data preprocessing
    from sklearn.preprocessing import StandardScaler

    mm = StandardScaler()
    tr_x = mm.fit_transform(tr_x)

    clf = dbdt()
    clf.fit(tr_x, tr_y)

    te_x = mm.transform(te_x)
    pred = clf.predict(te_x)

    Acc, F_class_macro, F_class_micro, F_class_weighted, G_mean, R_class, PAR = \
        getnewf1(te_y, pred, min_class)

    end_time = time.time()

    logger.info(str(Acc) + '  ' + str(F_class_macro) + '  ' + str(F_class_micro) + '  ' + str(
        F_class_weighted) + '  ' + str(G_mean) + '  ' + str(R_class) + '  ' + str(PAR) + '  ' + str(
        end_time - start_time))
    logger.info('++++++++++++++++Dividing line++++++++++++++++')
    logger.info(' ')

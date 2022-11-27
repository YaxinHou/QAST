import os
import logging
import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold


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

    logger = get_log('split_data.txt', 'QAST')

    # the path of original data
    dir_path = '../data/original_data'
    # the path to save train and test data
    splited_data_path = '../data/splited_data'

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

    f = all_data_filename.index(args.dataset)
    file = all_data_filename[f] + '.xlsx'
    data_name = file.split('.')[0]

    logger.info('dataset：' + data_name)

    org_file_path = os.path.join(dir_path, file)
    re_file_path = os.path.join(splited_data_path, file)

    data = pd.read_excel(org_file_path, header=0, index_col=None, sheet_name='Sheet1')
    data = np.array(data)
    x = data[:, 0:-1]
    y = data[:, -1]

    # Stratified sampling, each type of test set and training set are balanced
    skf = StratifiedKFold(n_splits=2, random_state=100, shuffle=True)
    splited_data = skf.split(x, y)

    for tr, te in splited_data:
        train = data[tr, :]
        test = data[te, :]
        train = pd.DataFrame(train)
        test = pd.DataFrame(test)
        with pd.ExcelWriter(re_file_path) as f:
            train.to_excel(f, header=False, index=False, sheet_name='Sheet1')
            test.to_excel(f, header=False, index=False, sheet_name='Sheet2')
        break
    print(file)
    logger.info('++++++++++++++++Dividing line++++++++++++++++')
    logger.info(' ')

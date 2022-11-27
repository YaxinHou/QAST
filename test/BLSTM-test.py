import os
import time
import math
import logging
import argparse
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

import torch
import torch.nn as nn
import torch.utils.data as Data


class BLSTM(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, num_layers=2, biFlag=True):
        super(BLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        if biFlag:
            self.bi_num = 2
        else:
            self.bi_num = 1
        self.biFlag = biFlag

        self.layer1 = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                              batch_first=True, bidirectional=self.biFlag)
        self.layer2 = nn.Linear(hidden_dim * self.bi_num, self.output_dim)

    def forward(self, x):
        after_padding_zeros = self.input_dim * math.ceil(x.size(1) / self.input_dim)
        input_of_x = torch.zeros(x.size(0), after_padding_zeros)
        input_of_x[:, 0:x.size(1)] = x[:, :]
        new_input_of_x = input_of_x.view(len(x), -1, self.input_dim)
        if cuda:
            new_input_of_x = new_input_of_x.cuda()
        out, (h_n, c_n) = self.layer1(new_input_of_x)
        y = self.layer2(out[:, -1, :])
        return y


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
    parser.add_argument('--pretrained', dest='pretrained', help="switch for using pretrained model",
                        action='store_true', default=False)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    
    args = ags_parse()

    cuda = True if torch.cuda.is_available() else False
    batch_size = 20
    epochs = 300

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

    log_filename = './BLSTM_LOG/re_log-BLSTM-' + str(data_name) + '.txt'
    log_name = 'QAST-result-' + str(data_name)
    result_logger = get_log(log_filename, log_name)

    # the path to save model
    model_dir = '../model/stage-2' + os.sep + data_name
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # the path of our trained model
    trained_model_dir = '../trained_model/stage-2' + os.sep + data_name

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

    rows_org, cols_org = tr_x.shape
    train_loader = Data.DataLoader(
        dataset=Data.TensorDataset((torch.from_numpy(tr_x)).float(), torch.from_numpy(np.array(tr_y)).float()),
        batch_size=batch_size,
        shuffle=True)
    dim_of_input = math.ceil(math.sqrt(cols_org))
    model = BLSTM(dim_of_input, len(np.unique(tr_y)))
    if not args.pretrained:
        loss_function = nn.CrossEntropyLoss()
        if cuda:
            model.cuda()
            loss_function.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.5, 0.9))
        for i in range(epochs):
            for seq, labels in train_loader:
                optimizer.zero_grad()
                if cuda:
                    seq = seq.cuda()
                    labels = labels.cuda()
                y_pred = model(seq)
                single_loss = loss_function(y_pred, labels.long())
                single_loss.backward()
                optimizer.step()
                print("Train Step:", i, " loss: ", single_loss)
            if cuda:
                pred = model(((torch.from_numpy(te_x)).float()).cuda())
            else:
                pred = model(((torch.from_numpy(te_x)).float()))
            _, pred = torch.max(torch.softmax(pred, dim=1), 1)
            Acc, F_class_macro, F_class_micro, F_class_weighted, G_mean, R_class, PAR = \
                getnewf1(te_y, pred.cpu(), min_class)
            torch.save(model.state_dict(), '{}/BLSTM-_{}_{}.pth'.format(model_dir, data_name, i))
            end_time = time.time()

            # print(round(end_time - start_time, 4))
            result_logger.info(str(i) + '  ' + str(Acc) + '  ' + str(F_class_macro) + '  ' + str(F_class_micro) +
                               '  ' + str(F_class_weighted) + '  ' + str(G_mean) + '  ' + str(R_class) + '  ' +
                               str(PAR) + '  ' + str(end_time - start_time))
            # print(file)
        result_logger.info(' ')
    else:
        if cuda:
            model.cuda()
            model_filename = '{}/BLSTM-_{}.pth'.format(trained_model_dir, data_name)
            model.load_state_dict(torch.load(model_filename))
            pred = model(((torch.from_numpy(te_x)).float())).cuda()
        else:
            model_filename = '{}/BLSTM-_{}.pth'.format(trained_model_dir, data_name)
            model.load_state_dict(torch.load(model_filename, map_location='cpu'))
            pred = model(((torch.from_numpy(te_x)).float()))
        _, pred = torch.max(torch.softmax(pred, dim=1), 1)
        Acc, F_class_macro, F_class_micro, F_class_weighted, G_mean, R_class, PAR = \
            getnewf1(te_y, pred.cpu(), min_class)
        result_logger.info(str(Acc) + '  ' + str(F_class_macro) + '  ' + str(F_class_micro) +
                           '  ' + str(F_class_weighted) + '  ' + str(G_mean) + '  ' + str(R_class) + '  ' +
                           str(PAR))

import os
import time
import logging

import argparse
import numpy as np
import pandas as pd
from random import sample

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier as dbdt

from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.ensemble import EasyEnsembleClassifier

from utils.transformer import DataTransformer
from model.model import Generator, Discriminator, Classify, Truncated_normal, Label_sampel


def mixup_data(feature, label, per_class_ids, min_class_id, sim_min_class_id):
    label_nums = len(np.unique(label))  # number of categories
    mixed_feature = torch.FloatTensor()
    mixed_label = torch.FloatTensor()
    for min_class in min_class_id:
        sim_min_class = sim_min_class_id[min_class]
        min_class_ids = per_class_ids[min_class]
        min_class_fea = feature[min_class_ids]
        min_class_lab = label[min_class_ids]

        for item in sim_min_class:
            lam = np.random.beta(1.0, 1.0)
            if lam < 0.5:
                lam = 1 - lam
            sim_min_class_ids = per_class_ids[item]
            sim_min_class_fea = feature[sim_min_class_ids]
            sim_min_class_lab = label[sim_min_class_ids]
            len1 = len(min_class_lab)
            len2 = len(sim_min_class_lab)
            div = int(len2 / len1)
            mod = len2 % len1
            index = torch.randperm(len2)
            new_min_class_fea = (torch.cat([min_class_fea.repeat(div, 1), min_class_fea[:mod, :]], dim=0))[index, :]
            new_min_class_lab = torch.cat((min_class_lab.repeat(div), min_class_lab[:mod, ]), dim=0)
            new_min_class_lab_one_hot = F.one_hot(new_min_class_lab.type(torch.int64), num_classes=label_nums)
            sim_min_class_lab_one_hot = F.one_hot(sim_min_class_lab.type(torch.int64), num_classes=label_nums)
            per_mixed_feature = lam * new_min_class_fea + (1 - lam) * sim_min_class_fea[index, :]
            per_mixed_label = lam * new_min_class_lab_one_hot + (1 - lam) * sim_min_class_lab_one_hot[index, :]
            if min(mixed_feature.shape) == 0:
                mixed_feature = per_mixed_feature
                mixed_label = per_mixed_label
            else:
                mixed_feature = torch.cat((mixed_feature, per_mixed_feature), 0)
                mixed_label = torch.cat((mixed_label, per_mixed_label), 0)
    _, mixed_y_label = mixed_label.max(1)

    return mixed_feature, mixed_y_label, mixed_label


def smote(feature, label, min_class_id, label_nums):
    size, _ = feature.shape
    smo = SMOTE(k_neighbors=3)
    X_smo1, Y_smo1 = smo.fit_resample(feature, label)
    X_smo2, Y_smo2 = smo.fit_resample(feature, label)

    smoted_x1, smoted_y1 = X_smo1[size:, :], Y_smo1[size:, ]
    smoted_x2, smoted_y2 = X_smo2[size:, :], Y_smo2[size:, ]

    result_smo_sample = np.array((pd.DataFrame(np.concatenate((np.hstack((smoted_x1, smoted_y1.reshape(-1, 1))),
                                                               np.hstack((smoted_x2, smoted_y2.reshape(-1,
                                                                                                       1))))))).drop_duplicates())

    result_smoted_x, result_smoted_y = result_smo_sample[:, :-1], result_smo_sample[:, -1]
    all_index_ids = np.where([result_smoted_y == min_class_id[0]])[1]
    for i in range(1, len(min_class_id)):
        per_ids = np.where([result_smoted_y == min_class_id[i]])[1]
        all_index_ids = np.concatenate((all_index_ids, per_ids))
    min_result_smoted_x = torch.from_numpy(result_smoted_x[all_index_ids, :])
    min_result_smoted_y = torch.from_numpy(result_smoted_y[all_index_ids,])
    min_result_smoted_emb_y = F.one_hot(min_result_smoted_y.type(torch.int64), num_classes=label_nums)

    return min_result_smoted_x, min_result_smoted_y, min_result_smoted_emb_y


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
    parser.add_argument('--train_dir', type=str, dest='train_dir', help='the path of train data',
                        default='./data/splited_data')
    parser.add_argument('--save_dir', type=str, dest='save_dir', help='the path of generated data dir',
                        default='./data/generated_data')
    parser.add_argument('--epochs', default=301, type=int, help='Number of DGC_training epochs')
    parser.add_argument('--Epochs', default=1001, type=int, help='Number of S_training epochs')
    parser.add_argument('--model_dir', default='./model', help='Number of training epochs')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = ags_parse()
    logger = get_log('stage-1.txt', 'QAST')
    result_logger = get_log('re_stage-1.txt', 'QAST-result')

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

    max_batch = [6, 8, 6, 6, 2, 8, 10, 6, 10, 6, 6, 6, 6, 75, 6, 6, 6, 27, 58, 318, 19, 19, 38, 19,
                 40, 40, 40, 40, 13, 30, 17, 8, 106, 3, 18, 22, 44, 13, 12, 11]
    min_batch = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 9, 2, 4,
                 4, 20, 2, 8, 2, 8, 6, 2, 2, 2, 2, 2, 2, 2, 2, 4]

    f = all_data_filename.index(args.dataset)
    data_filename = all_data_filename[f] + '.xlsx'
    min_class_id = all_min_class_id[f]
    max_batch_num = max_batch[f]
    min_batch_num = min_batch[f]

    start_time = time.time()
    data_name = data_filename.split('.')[0]
    logger.info('dataset：' + data_name)

    result_logger.info('dataset：' + data_name)

    model_name = data_filename.split('.')[0]

    filepath = os.path.join(args.train_dir, data_filename)

    # the path to save generated data
    args.save_dir = './data/generated_data' + os.sep + model_name
    # the path to save model
    args.model_dir = './model/stage-1' + os.sep + model_name

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    df = pd.read_excel(filepath, index_col=None, header=None, sheet_name='Sheet1')
    data = np.array(df)
    train = data[:, 0:-1]
    label = data[:, -1]

    label_nums = len(np.unique(label))  # number of categories

    # statistics on the number and proportion of each category
    per_class_count = []
    per_class_weight = []
    for c in range(len(np.unique(label))):
        num = np.sum(np.array(label == c))
        per_class_count.append(num)
        per_class_weight.append(np.clip(len(train) / num, 1, 150))

    per_class_ids = dict()
    ids = np.array(range(len(train)))
    for c in range(len(np.unique(label))):
        per_class_ids[c] = np.where([label == c])[1]

    max_class_id = [i for i in range(label_nums) if i not in min_class_id]
    sim_min_class_id = dict()
    for item in min_class_id:
        sample_num = min(len(max_class_id), 3)
        sim_min_class_id[item] = sample(max_class_id, sample_num)

    # mixup--mixed_feature, mixed_y_label, mixed_label
    mixed_feature, mixed_label, one_hot_mixed_label = \
        mixup_data(torch.from_numpy(train), torch.from_numpy(label), per_class_ids, min_class_id, sim_min_class_id)

    # smote--min_result_smoted_x, min_result_smoted_y, min_result_smoted_emb_y
    smote_feature, smote_label, one_hot_smote_label = \
        smote(torch.from_numpy(train), torch.from_numpy(label), min_class_id, label_nums)

    trans = DataTransformer()

    # gaussian transform
    trans.fit(train)
    train = trans.transform(train)
    mixed_feature = trans.transform(mixed_feature)
    smote_feature = trans.transform(smote_feature)

    rows_org, cols_org = train.shape


    # weight initialization
    def weight_init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=0.02)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.normal_(m.weight, mean=1, std=0.02)
            nn.init.constant_(m.bias, 0)


    train = torch.FloatTensor(train)
    label = torch.FloatTensor(label)
    mixed_feature = torch.FloatTensor(mixed_feature)
    mixed_label = mixed_label.type(torch.float32)
    one_hot_mixed_label = one_hot_mixed_label.type(torch.float32)

    smote_feature = torch.FloatTensor(smote_feature)
    smote_label = smote_label.type(torch.float32)
    one_hot_smote_label = one_hot_smote_label.type(torch.float32)

    # loss function
    CEL = nn.CrossEntropyLoss()
    dis_real_fake_loss = nn.BCELoss()

    # G:generator-generate samples
    # D:discriminator-distinguish true and false output is 0-1
    # C:classfier
    # S:one of the composition of the label committee
    G = Generator(cols_org + 1, cols_org, label_nums)
    D = Discriminator(cols_org + 1)
    C = Classify(cols_org, label_nums)
    S = Classify(cols_org, label_nums)

    G.apply(weight_init)
    D.apply(weight_init)
    C.apply(weight_init)
    S.apply(weight_init)

    optimizer_G = optim.Adam(G.parameters(), lr=3e-3, betas=(0.5, 0.9),
                             weight_decay=1e-4)  # weight decay (L2 penalty)
    optimizer_D = optim.Adam(D.parameters(), lr=3e-4, betas=(0.5, 0.9))
    optimizer_C = optim.Adam(C.parameters(), lr=3e-4, betas=(0.5, 0.9))
    optimizer_S = optim.Adam(S.parameters(), lr=3e-4, betas=(0.5, 0.9))

    # 1.SVC
    model_svc = SVC(C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False,
                    tol=0.001, cache_size=200, class_weight='balanced', verbose=False, max_iter=- 1,
                    decision_function_shape='ovr', break_ties=False, random_state=None)
    clf_svc = model_svc.fit(train, label)

    # 2.LinearSVC
    model_linsvc1 = LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0,
                              multi_class='ovr', fit_intercept=True, intercept_scaling=1,
                              class_weight='balanced', verbose=0, random_state=None, max_iter=1000)
    clf_linsvc = model_linsvc1.fit(train, label)

    # 3.smote + LinearSVC
    t = train.clone()
    l = label.clone()
    model_linsvc2 = LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0,
                              multi_class='ovr', fit_intercept=True, intercept_scaling=1,
                              class_weight='balanced', verbose=0, random_state=None, max_iter=1000)
    X_resampled, y_resampled = SMOTE(k_neighbors=3).fit_resample(t, l)
    clf_smo = model_linsvc2.fit(X_resampled, y_resampled)

    # # 4.ADASYN + LinearSVC
    # t = train.clone()
    # l = label.clone()
    # model_linsvc3 = LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0,
    #                           multi_class='ovr', fit_intercept=True, intercept_scaling=1,
    #                           class_weight='balanced', verbose=0, random_state=None, max_iter=1000)
    # X_resampled, y_resampled = ADASYN(n_neighbors=3, sampling_strategy='minority').fit_resample(t, l)
    # clf_ada = model_linsvc3.fit(X_resampled, y_resampled)

    t = train.clone()
    l = label.clone()
    gbdt1 = dbdt()
    smote_enn = SMOTEENN(smote=SMOTE(k_neighbors=3), random_state=0)
    X_resampled, y_resampled = smote_enn.fit_resample(t, l)
    clf_ada = gbdt1.fit(X_resampled, y_resampled)

    # 5.SMOTEENN + LinearSVC
    t = train.clone()
    l = label.clone()
    model_linsvc4 = LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0,
                              multi_class='ovr', fit_intercept=True, intercept_scaling=1,
                              class_weight='balanced', verbose=0, random_state=None, max_iter=1000)
    smote_enn = SMOTEENN(smote=SMOTE(k_neighbors=3), random_state=0)
    X_resampled, y_resampled = smote_enn.fit_resample(t, l)
    clf_smoenn = model_linsvc4.fit(X_resampled, y_resampled)

    # 6.SMOTETomek + LinearSVC
    t = train.clone()
    l = label.clone()
    model_linsvc5 = LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0,
                              multi_class='ovr', fit_intercept=True, intercept_scaling=1,
                              class_weight='balanced', verbose=0, random_state=None, max_iter=1000)
    smote_tomek = SMOTETomek(smote=SMOTE(k_neighbors=3), random_state=0)
    X_resampled, y_resampled = smote_tomek.fit_resample(t, l)
    clf_smotom = model_linsvc5.fit(X_resampled, y_resampled)

    # 7.EasyEnsembleClassifier
    eec = EasyEnsembleClassifier(random_state=0)
    clf_eec = eec.fit(train, label)

    # 8.MLP
    for i in range(args.Epochs):
        optimizer_S.zero_grad()
        output_s_real = S(train)
        loss_s_real = 0
        for c in range(len(np.unique(label))):
            per_loss_s_real = []
            label_c = [c]
            label_c = torch.FloatTensor(np.array(label_c))
            for index_c in per_class_ids[c]:
                per_loss_s_real.append(CEL(output_s_real[index_c].reshape(-1, label_nums), label_c.long()))
            per_loss_s_real.sort(reverse=True)
            if c not in min_class_id:
                per_loss_s_real[int(len(per_loss_s_real) / 3):] = (0,)
                per_loss_s_real = per_loss_s_real[:len(per_loss_s_real) - 1]
            else:
                pass
            per_loss_s_real = sum(per_loss_s_real) / len(per_loss_s_real)
            loss_s_real = loss_s_real + per_loss_s_real * (per_class_weight[c] / sum(per_class_weight))
        loss_s = loss_s_real
        loss_s.backward()
        optimizer_S.step()
        print('stage-1→epoch:', i, 'S_loss:', loss_s.item())
        if i % 50 == 0:
            logger.info('stage-1→epoch:' + str(i) + ' S_loss:' + str(loss_s.item()))
        if (i + 1) % 1001 == 0:
            torch.save(S.state_dict(), '{}/S-gen_{}.pth'.format(args.model_dir, i))

    new_train1 = torch.FloatTensor()
    new_label1 = []
    new_train2 = torch.FloatTensor()
    new_label2 = []

    for it in range(len(label)):
        if label[it] in min_class_id:
            new_label1.append(label[it])
            if min(new_train1.shape) == 0:
                new_train1 = train[it].reshape(1, -1)
            else:
                new_train1 = torch.cat((new_train1, train[it].reshape(1, -1)), 0)
        else:
            new_label2.append(label[it])
            if min(new_train2.shape) == 0:
                new_train2 = train[it].reshape(1, -1)
            else:
                new_train2 = torch.cat((new_train2, train[it].reshape(1, -1)), 0)

    new_label1 = torch.from_numpy(np.array(new_label1))
    new_label2 = torch.from_numpy(np.array(new_label2))
    skf1 = StratifiedKFold(n_splits=min_batch_num, shuffle=True)
    skf2 = StratifiedKFold(n_splits=max_batch_num, shuffle=True)

    for i in range(args.epochs):
        # tl is the label discriminator threshold-the quality of the label
        # se is the Semantic discriminator threshold-the quality of the feature
        tl = int(i / 100)
        se = min(max(0, i), 100) * 0.1 / 100 + \
             min(max(0, i - 100), 50) * 0.1 / 50 + \
             min(max(0, i - 150), 50) * 0.05 / 50 + \
             min(max(0, i - 200), 100) * 0.05 / 100

        min_x = []
        min_y = []
        for train_index, test_index in skf1.split(new_train1, new_label1):
            min_x.append(new_train1[test_index])
            min_y.append(new_label1[test_index])

        max_x = []
        max_y = []
        batch_sample = []
        flag = 0
        for train_index, test_index in skf2.split(new_train2, new_label2):
            max_x.append(new_train2[test_index])
            max_y.append(new_label2[test_index])
            new_sample_train = np.array(torch.cat((new_train2[test_index], min_x[flag % min_batch_num]), 0))
            new_sample_label = np.concatenate(
                (new_label2[test_index].reshape(-1, 1), min_y[flag % min_batch_num].reshape(-1, 1)), 0)
            new_sample_train_label = np.concatenate((new_sample_train, new_sample_label), 1)
            np.random.shuffle(new_sample_train_label)
            new_sample_train_label = torch.from_numpy(new_sample_train_label)
            batch_sample.append(new_sample_train_label)
            flag += 1

        for batch in range(max_batch_num):
            batch_train = batch_sample[batch][:, :-1]
            batch_label = batch_sample[batch][:, -1]
            batch_size, _ = batch_train.shape
            batch_per_class_ids = dict()
            for c in range(len(np.unique(batch_label))):
                batch_per_class_ids[c] = np.where([np.array(batch_label) == c])[1]
            batch_per_class_weight = []
            for c in range(len(np.unique(batch_label))):
                num = np.sum(np.array(np.array(batch_label) == c))
                batch_per_class_weight.append(np.clip(len(batch_train) / num, 1, 150))
            if i < 100:
                smote_num = int(batch_size / 4)
                mixup_num = int(batch_size / 4)
                random_num = batch_size - int(batch_size / 4) - int(batch_size / 4)
            elif i < 200:
                smote_num = int(batch_size / 8)
                mixup_num = int(batch_size / 8)
                random_num = batch_size - int(batch_size / 8) - int(batch_size / 8)
            else:
                smote_num = 0
                mixup_num = 0
                random_num = batch_size
            if i < 200:
                # mixup
                mixup_per_class_ids = dict()
                for c in np.unique(mixed_label):
                    mixup_per_class_ids[c] = np.where([np.array(mixed_label) == c])[1]
                    np.random.shuffle(mixup_per_class_ids[c])
                mixup_per_class_num = int(mixup_num / len(np.unique(mixed_label)))
                other_mixup_per_class_num = int(mixup_num % len(np.unique(mixed_label)))
                if other_mixup_per_class_num > 0:
                    all_mixup_sample_index = mixup_per_class_ids[np.unique(mixed_label)[0]][
                                             :mixup_per_class_num + 1]
                else:
                    all_mixup_sample_index = mixup_per_class_ids[np.unique(mixed_label)[0]][
                                             :mixup_per_class_num]
                for j in range(1, len(np.unique(mixed_label))):
                    c = np.unique(mixed_label)[j]
                    if j < other_mixup_per_class_num:
                        all_mixup_sample_index = np.concatenate(
                            [all_mixup_sample_index, mixup_per_class_ids[c][:mixup_per_class_num + 1]], axis=0)
                    else:
                        all_mixup_sample_index = np.concatenate(
                            [all_mixup_sample_index, mixup_per_class_ids[c][:mixup_per_class_num]], axis=0)
                np.random.shuffle(all_mixup_sample_index)
                new_mixed_feature = mixed_feature[all_mixup_sample_index, :]
                new_mixed_label = mixed_label[all_mixup_sample_index]
                new_one_hot_mixed_label = one_hot_mixed_label[all_mixup_sample_index, :]

                temp_new_mixed_feature = new_mixed_feature
                temp_new_mixed_label = new_mixed_label
                temp_new_one_hot_mixed_label = new_one_hot_mixed_label

                if len(all_mixup_sample_index) < mixup_num:
                    copy_nums = mixup_num / len(all_mixup_sample_index)
                    copy_num = mixup_num % len(all_mixup_sample_index)
                    if copy_nums > 1:
                        for copy_of_times in range(int(copy_nums) - 1):
                            temp_new_mixed_feature = torch.cat((temp_new_mixed_feature, new_mixed_feature))
                            temp_new_mixed_label = torch.cat((temp_new_mixed_label, new_mixed_label))
                            temp_new_one_hot_mixed_label = torch.cat(
                                (temp_new_one_hot_mixed_label, new_one_hot_mixed_label))
                    temp_new_mixed_feature = torch.cat(
                        (temp_new_mixed_feature, new_mixed_feature[0:copy_num, ]))
                    temp_new_mixed_label = torch.cat((temp_new_mixed_label, new_mixed_label[0:copy_num, ]))
                    temp_new_one_hot_mixed_label = torch.cat(
                        (temp_new_one_hot_mixed_label, new_one_hot_mixed_label[0:copy_num, ]))

                new_mixed_feature = temp_new_mixed_feature
                new_mixed_label = temp_new_mixed_label
                new_one_hot_mixed_label = temp_new_one_hot_mixed_label

                # smote
                smote_per_class_ids = dict()
                for c in np.unique(smote_label):
                    smote_per_class_ids[c] = np.where([np.array(smote_label) == c])[1]
                    np.random.shuffle(smote_per_class_ids[c])
                smote_per_class_num = int(smote_num / len(np.unique(smote_label)))
                other_smote_per_class_num = int(smote_num % len(np.unique(smote_label)))
                if other_smote_per_class_num > 0:
                    all_smote_sample_index = smote_per_class_ids[np.unique(smote_label)[0]][
                                             :smote_per_class_num + 1]
                else:
                    all_smote_sample_index = smote_per_class_ids[np.unique(smote_label)[0]][
                                             :smote_per_class_num]
                for j in range(1, len(np.unique(smote_label))):
                    c = np.unique(smote_label)[j]
                    if j < other_smote_per_class_num:
                        all_smote_sample_index = np.concatenate(
                            [all_smote_sample_index, smote_per_class_ids[c][:smote_per_class_num + 1]], axis=0)
                    else:
                        all_smote_sample_index = np.concatenate(
                            [all_smote_sample_index, smote_per_class_ids[c][:smote_per_class_num]], axis=0)
                np.random.shuffle(all_smote_sample_index)
                new_smote_feature = smote_feature[all_smote_sample_index, :]
                new_smote_label = smote_label[all_smote_sample_index]
                new_one_hot_smote_label = one_hot_smote_label[all_smote_sample_index, :]

                temp_new_smote_feature = new_smote_feature
                temp_new_smote_label = new_smote_label
                temp_new_one_hot_smote_label = new_one_hot_smote_label

                if len(all_smote_sample_index) < smote_num:
                    copy_nums = smote_num / len(all_smote_sample_index)
                    copy_num = smote_num % len(all_smote_sample_index)
                    if copy_nums > 1:
                        for copy_of_times in range(int(copy_nums) - 1):
                            temp_new_smote_feature = torch.cat((temp_new_smote_feature, new_smote_feature))
                            temp_new_smote_label = torch.cat((temp_new_smote_label, new_smote_label))
                            temp_new_one_hot_smote_label = torch.cat(
                                (temp_new_one_hot_smote_label, new_one_hot_smote_label))
                    temp_new_smote_feature = torch.cat(
                        (temp_new_smote_feature, new_smote_feature[0:copy_num, ]))
                    temp_new_smote_label = torch.cat((temp_new_smote_label, new_smote_label[0:copy_num, ]))
                    temp_new_one_hot_smote_label = torch.cat(
                        (temp_new_one_hot_smote_label, new_one_hot_smote_label[0:copy_num, ]))

                new_smote_feature = temp_new_smote_feature
                new_smote_label = temp_new_smote_label
                new_one_hot_smote_label = temp_new_one_hot_smote_label
            else:
                new_mixed_feature = torch.FloatTensor()
                new_mixed_label = torch.FloatTensor()
                new_one_hot_mixed_label = torch.FloatTensor()

                new_smote_feature = torch.FloatTensor()
                new_smote_label = torch.FloatTensor()
                new_one_hot_smote_label = torch.FloatTensor()

            # random
            smo_x = Truncated_normal(random_num, cols_org)
            smo_y, em_smo_y = Label_sampel(random_num, label_nums)

            total_x = torch.cat((smo_x, new_mixed_feature, new_smote_feature))
            total_y = torch.cat((smo_y, new_mixed_label, new_smote_label))
            total_em_y = torch.cat((em_smo_y, new_one_hot_mixed_label, new_one_hot_smote_label))

            fake = G(total_x, total_y.reshape(-1, 1), cols_org + 1, total_em_y)
            gen_samples = trans._apply_activate(fake)

            smo_y_c = []
            gen_samples_c = torch.FloatTensor()

            # Semantic screening
            # tl is the label discriminator threshold-the quality of the label
            # se is the Semantic discriminator threshold-the quality of the feature
            svc_pre = torch.from_numpy(clf_svc.predict(gen_samples.detach()))
            svc_pre_score = torch.softmax(torch.from_numpy(clf_svc.decision_function(gen_samples.detach())),
                                          dim=1)
            svc_pre_one_hot = F.one_hot(svc_pre.type(torch.int64), num_classes=label_nums)

            linsvc_pre = torch.from_numpy(clf_linsvc.predict(gen_samples.detach()))
            linsvc_pre_score = torch.softmax(
                torch.from_numpy(clf_linsvc.decision_function(gen_samples.detach())), dim=1)
            linsvc_pre_one_hot = F.one_hot(linsvc_pre.type(torch.int64), num_classes=label_nums)

            smo_pre = torch.from_numpy(clf_smo.predict(gen_samples.detach()))
            smo_pre_score = torch.softmax(torch.from_numpy(clf_smo.decision_function(gen_samples.detach())),
                                          dim=1)
            smo_pre_one_hot = F.one_hot(smo_pre.type(torch.int64), num_classes=label_nums)

            ada_pre = torch.from_numpy(clf_ada.predict(gen_samples.detach()))
            ada_pre_score = torch.softmax(torch.from_numpy(clf_ada.decision_function(gen_samples.detach())), dim=1)
            ada_pre_one_hot = F.one_hot(ada_pre.type(torch.int64), num_classes=label_nums)

            smoenn_pre = torch.from_numpy(clf_smoenn.predict(gen_samples.detach()))
            smoenn_pre_score = torch.softmax(
                torch.from_numpy(clf_smoenn.decision_function(gen_samples.detach())), dim=1)
            smoenn_pre_one_hot = F.one_hot(smoenn_pre.type(torch.int64), num_classes=label_nums)

            smotom_pre = torch.from_numpy(clf_smotom.predict(gen_samples.detach()))
            smotom_pre_score = torch.softmax(
                torch.from_numpy(clf_smotom.decision_function(gen_samples.detach())), dim=1)
            smotom_pre_one_hot = F.one_hot(smotom_pre.type(torch.int64), num_classes=label_nums)

            eec_pre = torch.from_numpy(clf_eec.predict(gen_samples.detach()))
            eec_pre_score = torch.softmax(torch.from_numpy(clf_eec.predict_proba(gen_samples.detach())), dim=1)
            eec_pre_one_hot = F.one_hot(eec_pre.type(torch.int64), num_classes=label_nums)

            output_s_fake = S(gen_samples.detach())
            ALL_ReturnVlaue, ALL_ReturnIndices = output_s_fake.max(1)
            output_s_fake_score = torch.softmax(output_s_fake, dim=1)
            output_s_fake_one_hot = F.one_hot(ALL_ReturnIndices.type(torch.int64), num_classes=label_nums)

            output_d_fake = D(gen_samples.detach(), total_y.reshape(-1, 1).detach())

            for it in range(len(total_y)):
                all_pre_probability = torch.cat((svc_pre_score[it].reshape(1, -1),
                                                 linsvc_pre_score[it].reshape(1, -1),
                                                 smo_pre_score[it].reshape(1, -1),
                                                 smoenn_pre_score[it].reshape(1, -1),
                                                 smotom_pre_score[it].reshape(1, -1),
                                                 eec_pre_score[it].reshape(1, -1),
                                                 output_s_fake_score[it].reshape(1, -1)), dim=0)
                all_pre_probability = all_pre_probability / torch.sum(all_pre_probability, dim=0)

                all_pre_one_hot = torch.cat((svc_pre_one_hot[it].reshape(1, -1),
                                             linsvc_pre_one_hot[it].reshape(1, -1),
                                             smo_pre_one_hot[it].reshape(1, -1),
                                             smoenn_pre_one_hot[it].reshape(1, -1),
                                             smotom_pre_one_hot[it].reshape(1, -1),
                                             eec_pre_one_hot[it].reshape(1, -1),
                                             output_s_fake_one_hot[it].reshape(1, -1)), dim=0)
                all_probability = torch.sum(all_pre_probability * all_pre_one_hot, dim=0)
                temp_c = total_y[it].item()
                all_count_num = torch.sum(all_pre_one_hot.T, dim=1)
                correct_num = all_count_num[int(temp_c)]
                correct_probability = all_probability[int(temp_c)]
                predict_count_probability = dict()
                for j in range(label_nums):
                    predict_count_probability[j] = {'class': j, 'correct_num': all_count_num[j],
                                                    'correct_probability': all_probability[j]}
                result_of_sort = sorted(predict_count_probability.keys(),
                                        key=lambda x: (- predict_count_probability[x]['correct_num'],
                                                       - predict_count_probability[x]['correct_probability']))
                if output_d_fake[it].item() >= se:
                    if correct_num >= tl:
                        smo_y_c.append(total_y[it])
                        if min(gen_samples_c.shape) == 0:
                            gen_samples_c = gen_samples[it].reshape(1, -1)
                        else:
                            gen_samples_c = torch.cat((gen_samples_c, gen_samples[it].reshape(1, -1)), 0)
                    else:
                        smo_y_c.append(result_of_sort[0])
                        if min(gen_samples_c.shape) == 0:
                            gen_samples_c = gen_samples[it].reshape(1, -1)
                        else:
                            gen_samples_c = torch.cat((gen_samples_c, gen_samples[it].reshape(1, -1)), 0)
                else:
                    ra = np.random.uniform(0, 1)
                    if ra >= 0.5:
                        if correct_num >= tl:
                            smo_y_c.append(total_y[it])
                            if min(gen_samples_c.shape) == 0:
                                gen_samples_c = gen_samples[it].reshape(1, -1)
                            else:
                                gen_samples_c = torch.cat((gen_samples_c, gen_samples[it].reshape(1, -1)), 0)
                        else:
                            smo_y_c.append(result_of_sort[0])
                            if min(gen_samples_c.shape) == 0:
                                gen_samples_c = gen_samples[it].reshape(1, -1)
                            else:
                                gen_samples_c = torch.cat((gen_samples_c, gen_samples[it].reshape(1, -1)), 0)
                    else:
                        pass
            smo_y_c = np.array(smo_y_c)
            # statistics on the number and proportion of each category
            per_class_count_smo = []
            per_class_weight_smo = []
            for c in range(len(np.unique(label))):
                num = np.sum(np.array(smo_y_c == c))
                per_class_count_smo.append(num)
                per_class_weight_smo.append(np.clip(len(smo_y_c) / (num + 1e-5), 1, 150))
            per_class_ids_smo = dict()
            ids_smo = np.array(range(len(total_y)))
            for c in range(len(np.unique(label))):
                per_class_ids_smo[c] = np.where([smo_y_c == c])[1]
            smo_y_c = torch.FloatTensor(smo_y_c)

            # train the discriminator
            # Discriminant loss, the discriminator only needs to distinguish whether it is a real sample or a synthetic sample
            optimizer_D.zero_grad()
            # feed real samples into the discriminator
            output_real = D(batch_train, batch_label.reshape(-1, 1))
            # feed synthetic samples into the discriminator
            output_fake = D(gen_samples, total_y.reshape(-1, 1))

            real_ones = torch.ones(batch_label.shape).reshape(-1, 1)
            fake_zeros = torch.zeros(total_y.shape).reshape(-1, 1)
            loss_d_dis = dis_real_fake_loss(output_real, real_ones) + dis_real_fake_loss(output_fake,
                                                                                         fake_zeros)

            train_la = torch.cat((batch_train, batch_label.reshape(-1, 1)), -1)
            gen_samples_la = torch.cat((gen_samples, total_y.reshape(-1, 1)), -1)
            alpha = torch.rand(1, 1)
            alpha = alpha.expand(train_la.size())
            interpolates = alpha * train_la + ((1 - alpha) * gen_samples_la)
            interpolated = torch.autograd.Variable(interpolates, requires_grad=True)
            disc_interpolates = D(interpolated)

            # calculate the gradient of outputs to inputs
            gradients = torch.autograd.grad(
                outputs=disc_interpolates, inputs=interpolated,
                grad_outputs=torch.ones(disc_interpolates.size(), device='cpu'),
                create_graph=True, retain_graph=True, only_inputs=True
            )[0]
            gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

            loss_d = loss_d_dis + gradient_penalty
            loss_d.backward(retain_graph=True)
            optimizer_D.step()

            # training classifier-softmax classifier
            # Classification loss-real samples + synthetic samples
            optimizer_C.zero_grad()
            output_c_real = C(batch_train)
            loss_g_org = 0
            for c in range(len(np.unique(batch_label))):
                per_loss_g_c = []
                label_c = [c]
                label_c = torch.FloatTensor(np.array(label_c))
                for index_c in batch_per_class_ids[c]:
                    per_loss_g_c.append(CEL(output_c_real[index_c].reshape(-1, label_nums), label_c.long()))
                per_loss_g_c.sort(reverse=True)
                if c not in min_class_id:
                    per_loss_g_c[int(len(per_loss_g_c) / 3):] = (0,)
                    per_loss_g_c = per_loss_g_c[:len(per_loss_g_c) - 1]
                else:
                    pass
                per_loss_g_c = sum(per_loss_g_c) / len(per_loss_g_c)
                loss_g_org = loss_g_org + per_loss_g_c * (
                        batch_per_class_weight[c] / sum(batch_per_class_weight))

            output_c_fake = C(gen_samples_c)
            loss_g_fake = 0
            for c in range(len(np.unique(label))):
                per_loss_g_c_smo = []
                label_c_smo = [c]
                label_c_smo = torch.FloatTensor(np.array(label_c_smo))
                for index_c in per_class_ids_smo[c]:
                    per_loss_g_c_smo.append(
                        CEL(output_c_fake[index_c].reshape(-1, label_nums), label_c_smo.long()))
                per_loss_g_c_smo.sort(reverse=True)
                if c not in min_class_id:
                    per_loss_g_c_smo[int(len(per_loss_g_c_smo) / 3):] = (0,)
                    per_loss_g_c_smo = per_loss_g_c_smo[:len(per_loss_g_c_smo) - 1]
                else:
                    pass
                if len(per_loss_g_c_smo) == 0:
                    per_loss_g_c_smo = 0
                else:
                    per_loss_g_c_smo = sum(per_loss_g_c_smo) / len(per_loss_g_c_smo)
                loss_g_fake = loss_g_fake + per_loss_g_c_smo * (
                        per_class_weight_smo[c] / sum(per_class_weight_smo))
            loss_c = loss_g_org + 2 * loss_g_fake
            loss_c.backward(retain_graph=True)
            optimizer_C.step()

            # train generator
            # Generator loss-reconstruction loss + discriminator loss + classifier loss
            optimizer_G.zero_grad()
            loss_g_rec = torch.norm((gen_samples - batch_train), p=2, dim=1).mean()

            output_d = D(gen_samples, total_y.reshape(-1, 1))
            label_d = torch.ones(total_y.shape).reshape(-1, 1)
            loss_g_d = dis_real_fake_loss(output_d, label_d)

            output_c = C(gen_samples)
            loss_g_c = CEL(output_c, total_y.long())

            loss_g = loss_g_rec + loss_g_d + loss_g_c
            loss_g.backward()
            optimizer_G.step()
            print('stage-2→epoch:', i, 'batch:', batch, 'D_loss:', loss_d.item(), 'C_loss:', loss_c.item(),
                  'G_loss:', loss_g.item())
            if i % 20 == 0 and batch == max_batch_num - 1:
                logger.info('stage-2→epoch:' + str(i) + ' D_loss:' + str(loss_d.item()) + ' C_loss:' + str(
                    loss_c.item()) + ' G_loss:' + str(loss_g.item()))
            if (i + 1) % 301 == 0 and batch == max_batch_num - 1:
                torch.save(G.state_dict(), '{}/G-gen_{}.pth'.format(args.model_dir, i))
                torch.save(D.state_dict(), '{}/D-gen_{}.pth'.format(args.model_dir, i))
                torch.save(C.state_dict(), '{}/C-gen_{}.pth'.format(args.model_dir, i))

    # save the generated data
    re_y_1 = []
    re_x_1 = torch.FloatTensor()
    re_y_2 = []
    re_x_2 = torch.FloatTensor()
    re_y_3 = []
    re_x_3 = torch.FloatTensor()
    re_y_4 = []
    re_x_4 = torch.FloatTensor()
    re_tempfilename = model_name + '.csv'
    re_gen_path = os.path.join(args.save_dir, re_tempfilename)
    for t in range(1, 101):
        sheetname = 'Sheet' + str(t)
        smo_x = Truncated_normal(rows_org, cols_org)
        smo_y, em_smo_y = Label_sampel(rows_org, label_nums)
        data = []
        fake = G(smo_x.detach(), (smo_y.reshape(-1, 1)).detach(), cols_org + 1, em_smo_y)
        fakeact = trans._apply_activate(fake)
        data.append(fakeact.detach().cpu().numpy())
        data = np.concatenate(data, axis=0)
        # save the corresponding label
        gen_y = smo_y.numpy()
        output = trans.inverse_transform(data, None)
        gen_data = np.column_stack((output, gen_y))
        output = pd.DataFrame(gen_data)

        data_check = np.array(output)
        te_x = data_check[:, 0:-1]
        te_y = data_check[:, -1]
        smo_te_x = trans.transform(te_x)
        smo_te_x = torch.FloatTensor(smo_te_x)
        te_x = torch.FloatTensor(te_x)
        te_y = torch.FloatTensor(te_y)

        svc_pre = torch.from_numpy(clf_svc.predict(smo_te_x.detach()))
        svc_pre_score = torch.softmax(torch.from_numpy(clf_svc.decision_function(smo_te_x.detach())), dim=1)
        svc_pre_one_hot = F.one_hot(svc_pre.type(torch.int64), num_classes=label_nums)

        linsvc_pre = torch.from_numpy(clf_linsvc.predict(smo_te_x.detach()))
        linsvc_pre_score = torch.softmax(torch.from_numpy(clf_linsvc.decision_function(smo_te_x.detach())), dim=1)
        linsvc_pre_one_hot = F.one_hot(linsvc_pre.type(torch.int64), num_classes=label_nums)

        smo_pre = torch.from_numpy(clf_smo.predict(smo_te_x.detach()))
        smo_pre_score = torch.softmax(torch.from_numpy(clf_smo.decision_function(smo_te_x.detach())), dim=1)
        smo_pre_one_hot = F.one_hot(smo_pre.type(torch.int64), num_classes=label_nums)

        ada_pre = torch.from_numpy(clf_ada.predict(smo_te_x.detach()))
        ada_pre_score = torch.softmax(torch.from_numpy(clf_ada.decision_function(smo_te_x.detach())), dim=1)
        ada_pre_one_hot = F.one_hot(ada_pre.type(torch.int64), num_classes=label_nums)

        smoenn_pre = torch.from_numpy(clf_smoenn.predict(smo_te_x.detach()))
        smoenn_pre_score = torch.softmax(torch.from_numpy(clf_smoenn.decision_function(smo_te_x.detach())), dim=1)
        smoenn_pre_one_hot = F.one_hot(smoenn_pre.type(torch.int64), num_classes=label_nums)

        smotom_pre = torch.from_numpy(clf_smotom.predict(smo_te_x.detach()))
        smotom_pre_score = torch.softmax(torch.from_numpy(clf_smotom.decision_function(smo_te_x.detach())), dim=1)
        smotom_pre_one_hot = F.one_hot(smotom_pre.type(torch.int64), num_classes=label_nums)

        eec_pre = torch.from_numpy(clf_eec.predict(smo_te_x.detach()))
        eec_pre_score = torch.softmax(torch.from_numpy(clf_eec.predict_proba(smo_te_x.detach())), dim=1)
        eec_pre_one_hot = F.one_hot(eec_pre.type(torch.int64), num_classes=label_nums)

        output_s_fake = S(smo_te_x.detach())
        ALL_ReturnVlaue, ALL_ReturnIndices = output_s_fake.max(1)
        output_s_fake_score = torch.softmax(output_s_fake, dim=1)
        output_s_fake_one_hot = F.one_hot(ALL_ReturnIndices.type(torch.int64), num_classes=label_nums)

        output_d_fake = D(smo_te_x.detach(), te_y.reshape(-1, 1).detach())  # 是否detach()

        for it in range(len(te_y)):
            all_pre_probability = torch.cat((svc_pre_score[it].reshape(1, -1),
                                             linsvc_pre_score[it].reshape(1, -1),
                                             smo_pre_score[it].reshape(1, -1),
                                             smoenn_pre_score[it].reshape(1, -1),
                                             smotom_pre_score[it].reshape(1, -1),
                                             eec_pre_score[it].reshape(1, -1),
                                             output_s_fake_score[it].reshape(1, -1)), dim=0)
            all_pre_probability = all_pre_probability / torch.sum(all_pre_probability, dim=0)

            all_pre_one_hot = torch.cat((svc_pre_one_hot[it].reshape(1, -1),
                                         linsvc_pre_one_hot[it].reshape(1, -1),
                                         smo_pre_one_hot[it].reshape(1, -1),
                                         smoenn_pre_one_hot[it].reshape(1, -1),
                                         smotom_pre_one_hot[it].reshape(1, -1),
                                         eec_pre_one_hot[it].reshape(1, -1),
                                         output_s_fake_one_hot[it].reshape(1, -1)), dim=0)
            all_probability = torch.sum(all_pre_probability * all_pre_one_hot, dim=0)
            temp_c = te_y[it].item()
            all_count_num = torch.sum(all_pre_one_hot.T, dim=1)
            correct_num = all_count_num[int(temp_c)]
            correct_probability = all_probability[int(temp_c)]
            predict_count_probability = dict()
            for j in range(label_nums):
                predict_count_probability[j] = {'class': j, 'correct_num': all_count_num[j],
                                                'correct_probability': all_probability[j]}
            result_of_sort = sorted(predict_count_probability.keys(),
                                    key=lambda x: (- predict_count_probability[x]['correct_num'],
                                                   - predict_count_probability[x]['correct_probability']))
            if output_d_fake[it].item() >= 0.3:
                if correct_num >= 7:
                    re_y_1.append(te_y[it])
                    if min(re_x_1.shape) == 0:
                        re_x_1 = te_x[it].reshape(1, -1)
                    else:
                        re_x_1 = torch.cat((re_x_1, te_x[it].reshape(1, -1)), 0)
                elif correct_num >= 5:
                    re_y_2.append(te_y[it])
                    if min(re_x_2.shape) == 0:
                        re_x_2 = te_x[it].reshape(1, -1)
                    else:
                        re_x_2 = torch.cat((re_x_2, te_x[it].reshape(1, -1)), 0)
                elif correct_num >= 3:
                    re_y_3.append(te_y[it])
                    if min(re_x_3.shape) == 0:
                        re_x_3 = te_x[it].reshape(1, -1)
                    else:
                        re_x_3 = torch.cat((re_x_3, te_x[it].reshape(1, -1)), 0)
                else:
                    vote_num = predict_count_probability[result_of_sort[0]]['correct_num']
                    if vote_num >= 7:
                        re_y_1.append(result_of_sort[0])
                        if min(re_x_1.shape) == 0:
                            re_x_1 = te_x[it].reshape(1, -1)
                        else:
                            re_x_1 = torch.cat((re_x_1, te_x[it].reshape(1, -1)), 0)
                    elif vote_num >= 5:
                        re_y_2.append(result_of_sort[0])
                        if min(re_x_2.shape) == 0:
                            re_x_2 = te_x[it].reshape(1, -1)
                        else:
                            re_x_2 = torch.cat((re_x_2, te_x[it].reshape(1, -1)), 0)
                    elif vote_num >= 3:
                        re_y_3.append(result_of_sort[0])
                        if min(re_x_3.shape) == 0:
                            re_x_3 = te_x[it].reshape(1, -1)
                        else:
                            re_x_3 = torch.cat((re_x_3, te_x[it].reshape(1, -1)), 0)
                    else:
                        pass
            else:
                pass
        print('—————————————————————————————sheet' + str(t) + '—————————————————————————————')
    re_x = np.array(torch.cat((re_x_1, re_x_2, re_x_3, re_x_4), 0))
    re_y = np.array(re_y_1 + re_y_2 + re_y_3 + re_y_4)
    re_data = np.column_stack((re_x, re_y))
    re_output = pd.DataFrame(re_data)
    if os.path.exists(re_gen_path):
        existing = pd.read_csv(re_gen_path, header=None, index_col=None)
        endout = pd.concat([existing, re_output], axis=0, ignore_index=False)
    else:
        endout = re_output
    endout.to_csv(re_gen_path, header=False, index=False)

    end_time = time.time()
    totaltime = end_time - start_time
    print(totaltime, '(', totaltime / 60, ')')
    logger.info(str(totaltime) + ' (' + str(totaltime / 60) + ')')
    logger.info('—————————————————————————————Dividing line—————————————————————————————')
    logger.info(' ')
    result_logger.info(str(totaltime) + ' (' + str(totaltime / 60) + ')')
    result_logger.info('—————————————————————————————Dividing line—————————————————————————————')
    result_logger.info(' ')

import os
import time
import logging

import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from sklearn.ensemble import GradientBoostingClassifier as dbdt

from sklearn.svm import SVC, LinearSVC

from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.ensemble import EasyEnsembleClassifier

from utils.transformer import DataTransformer
from model.model import Generator, Discriminator, Classify, Truncated_normal, Label_sampel


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
    args.model_dir = './trained_model/stage-1' + os.sep + model_name

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    df = pd.read_excel(filepath, index_col=None, header=None, sheet_name='Sheet1')
    data = np.array(df)
    train = data[:, 0:-1]
    label = data[:, -1]

    label_nums = len(np.unique(label))  # number of categories

    trans = DataTransformer()

    # gaussian transform
    trans.fit(train)
    train = trans.transform(train)

    rows_org, cols_org = train.shape

    train = torch.FloatTensor(train)
    label = torch.FloatTensor(label)

    # G:generator-generate samples
    # D:discriminator-distinguish true and false output is 0-1
    # C:classfier
    # S:one of the composition of the label committee
    G = Generator(cols_org + 1, cols_org, label_nums)
    D = Discriminator(cols_org + 1)
    C = Classify(cols_org, label_nums)
    S = Classify(cols_org, label_nums)

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

    S_filename = '{}/S-gen_{}.pth'.format(args.model_dir, 1000)
    S.load_state_dict(torch.load(S_filename))
    G_filename = '{}/G-gen_{}.pth'.format(args.model_dir, 300)
    G.load_state_dict(torch.load(G_filename))
    D_filename = '{}/D-gen_{}.pth'.format(args.model_dir, 300)
    D.load_state_dict(torch.load(D_filename))
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

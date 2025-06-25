import sys
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import argparse
from easydict import EasyDict

from model.HetDDI import HetDDI
from utils.data_loader import load_data, get_train_test
from train_test import train_one_epoch, test
from utils.pytorchtools import EarlyStopping
from utils.logger import Logger
import os

def run(args):
    # 고정된 랜덤 시드 설정
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # 데이터 경로구성 및 로딩
    data_path = os.path.join(args.data_path, args.kg_name+'+'+args.ddi_name)
    kg_g, smiles = load_data(data_path, device=device)
    train_sample, test_sample = get_train_test(
        data_path, fold_num=args.fold_num, label_type=args.label_type, condition=args.condition
        )

    scores = []
    for i in range(0, args.fold_num):
        # 훈련 및 테스트 데이터 분할
        train_x_left = train_sample[i][:, 0]
        train_x_right = train_sample[i][:, 1]
        train_y = train_sample[i][:, 2:]

        test_x_left = test_sample[i][:, 0]
        test_x_right = test_sample[i][:, 1]
        test_y = test_sample[i][:, 2:]

        # 레이블 타입에 따라 float/long Tensor로 변환
        if args.label_type == 'multi_class':
            train_y = torch.from_numpy(train_y).long()
            test_y = torch.from_numpy(test_y).long()
        else:
            train_y = torch.from_numpy(train_y).float()
            test_y = torch.from_numpy(test_y).float()

        # load model
        # 모델과 loss function 설정 (label_type에 따라 클래스 수 달라짐)
        if args.label_type == 'multi_class':
            model = HetDDI(kg_g, smiles, args.hidden_dim, args.num_layer, args.mode, 86, args.condition).to(device)
            loss_func = nn.CrossEntropyLoss()
        elif args.label_type == 'binary_class':
            model = HetDDI(kg_g, smiles, args.hidden_dim, args.num_layer, args.mode, 1, args.condition).to(device)
            loss_func = nn.BCEWithLogitsLoss()
        elif args.label_type == 'multi_label':
            model = HetDDI(kg_g, smiles, args.hidden_dim, args.num_layer, args.mode, 200, args.condition).to(device)
            loss_func = nn.BCEWithLogitsLoss()
        
        if i == 0:
            print(model)

        # divide parameters into two parts, weight_p has l2_norm but bias_bn_emb_p not
        # weight_decay 적용할 파라미터 구분 (bias, bn, embedding 제외)
        weight_p, bias_bn_emb_p = [], []
        for name, p in model.named_parameters():
            if 'bias' in name or 'bn' in name or 'embedding' in name:
                bias_bn_emb_p += [p]
            else:
                weight_p += [p]
        model_parameters = [
            {'params': weight_p, 'weight_decay': args.weight_decay},
            {'params': bias_bn_emb_p, 'weight_decay': 0},
        ]

        # optimizer 및 early stopping 설정
        optimizer = optim.Adam(model_parameters, lr=args.lr)
        early_stopping = EarlyStopping(patience=args.patience, verbose=True)

        best_test_score = None
        for epoch in range(args.epoch):
            # 1 epoch 학습
            train_one_epoch(model, loss_func, optimizer,
                            train_x_left, train_x_right, train_y,
                            i, epoch, args.batch_size,
                            args.label_type, device)

            # 평가
            test_score = test(model, loss_func,
                              test_x_left, test_x_right, test_y,
                              i, epoch, args.batch_size,
                              args.label_type, device)

            test_acc = test_score[0] # acc or auc
            if epoch > 50:
                early_stopping(test_acc, model)
                if early_stopping.counter == 0:
                    best_test_score = test_score
                if early_stopping.early_stop or epoch == args.epoch - 1:
                    break

            # epoch 결과 출력
            print(best_test_score)
            print("=" * 100)

        # 현재 fold 성능 저장
        scores.append(best_test_score)
        print('Test set score:', scores)

    # 전체 fold 결과 평균 계산
    scores = np.array(scores)
    scores = scores.mean(axis=0)

    # 최종 성능 출력 (label_type에 따라 출력 항목 다름)
    if args.label_type == 'multi_class':
        mean_kappa = scores[:, 4].mean()
        print(
            "\033[1;31mFinal DDI result:\033[0m\n"
            f"acc: {scores[0]:.3f}, "
            f"f1: {scores[1]:.3f}, "
            f"precision: {scores[2]:.3f}, "
            f"recall: {scores[3]:.3f}, "
            f"kappa: {scores[4]:.3f}"
        )
    elif args.label_type == 'binary_class':
        mean_auc = scores[:, 4].mean()
        print(
            "\033[1;31mFinal DDI result:\033[0m\n"
            f"acc: {scores[0]:.3f}, "
            f"f1: {scores[1]:.3f}, "
            f"precision: {scores[2]:.3f}, "
            f"recall: {scores[3]:.3f}, "
            f"auc: {scores[4]:.3f}"
        )
    elif args.label_type == 'multi_label':
        print(scores)


if __name__ == '__main__':
    ''' 
    ap = argparse.ArgumentParser(description='')
    ap.add_argument('--batch_size', type=int, default=2 ** 15)
    ap.add_argument('--fold_num', type=int, default=5)
    ap.add_argument('--hidden_dim', type=int, default=300, help='Dimension of the node hidden state. Default is 300.')
    ap.add_argument('--num_layer', type=int, default=3)
    ap.add_argument('--epoch', type=int, default=1000, help='Number of epochs.')
    ap.add_argument('--patience', type=int, default=50)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=1e-5)

    ap.add_argument('--label_type', type=str, choices=['multi_class', 'binary_class', 'multi_label'],
                    default='binary_class')
    ap.add_argument('--condition', type=str, choices=['s1', 's2', 's3'], default='s1')
    ap.add_argument('--mode', type=str, choices=['only_kg', 'only_mol', 'concat'], default='concat')
    ap.add_argument('--data_path', type=str, default='./data')
    ap.add_argument('--kg_name', type=str, default='DRKG')
    ap.add_argument('--ddi_name', type=str, choices=['DrugBank', "TWOSIDES"], default='DrugBank')

    args = ap.parse_args(args=[])
    print(args)
    '''
    
    args = EasyDict({
        'batch_size': 2 ** 15,
        'fold_num': 5,
        'hidden_dim': 300,
        'num_layer': 3,
        'epoch': 1000,
        'patience': 50,
        'lr': 1e-3,
        'weight_decay': 1e-5,
        'label_type': 'binary_class',
        'condition': 's1',
        'mode': 'concat',
        
        'data_path': './data',
        'kg_name': 'FOODRKG',
        'ddi_name': 'DrugBank',
    })

    # print 출력을 터미널과 로그 파일에 동시에 기록하도록 설정 & warning 무시
    sys.stdout = Logger(
        f'./log/ddi-dataset_{args.hidden_dim} '
        f'label-type_{args.label_type} '
        f'mode_{args.mode} '
        f'condition_{args.condition}.txt',
        sys.stdout
    )
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'); print('running on', device)

    run(args)


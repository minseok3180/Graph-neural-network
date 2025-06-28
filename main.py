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

    # load data
    data_path = os.path.join(args.data_path, f"{args.kg_name}+{args.ddi_name}")
    kg_g, smiles, f2c_dict = load_data(data_path, device=device)
    train_sample, test_sample = get_train_test(data_path, args.fold_num, args.label_type, args.condition)

    label_dims = {'multi_class': 86, 'binary_class': 1, 'multi_label': 200}
    loss_funcs = {
        'multi_class': nn.CrossEntropyLoss(),
        'binary_class': nn.BCEWithLogitsLoss(),
        'multi_label': nn.BCEWithLogitsLoss()
    }

    scores = []
    for fold in range(args.fold_num):
        # 훈련 및 테스트 데이터 분할
        txl, txr, ty = train_sample[fold][:, 0], train_sample[fold][:, 1], train_sample[fold][:, 2:]  # train x left, ..., train y
        vxl, vxr, vy = test_sample[fold][:, 0], test_sample[fold][:, 1], test_sample[fold][:, 2:]  # validation x left, ..., validation y

        ty = torch.from_numpy(ty).long() if args.label_type == 'multi_class' else torch.from_numpy(ty).float()
        vy = torch.from_numpy(vy).long() if args.label_type == 'multi_class' else torch.from_numpy(vy).float()

        # model + loss
        model = HetDDI(kg_g, smiles, args.hidden_dim, args.num_layer, args.mode, label_dims[args.label_type], args.condition).to(device)
        loss_func = loss_funcs[args.label_type]
        if fold == 0: print(model)

        # divide parameters into two parts, weight_p has l2_norm but bias_bn_emb_p not
        # weight_decay 적용할 파라미터 구분 (bias, bn, embedding 제외)
        weight_p, bias_bn_emb_p = [], []
        for name, p in model.named_parameters():
            (bias_bn_emb_p if any(x in name for x in ['bias', 'bn', 'embedding']) else weight_p).append(p)

        # optimizer 및 early stopping 설정
        optimizer = optim.Adam([
            {'params': weight_p, 'weight_decay': args.weight_decay},
            {'params': bias_bn_emb_p, 'weight_decay': 0},
        ], lr=args.lr)
        early_stopping = EarlyStopping(patience=args.patience, verbose=True)
        best_val_score = None
        
        for epoch in range(args.epoch):
            # 1 epoch 학습
            train_one_epoch(f2c_dict, model, loss_func, optimizer, txl, txr, ty, fold, epoch, args.batch_size, args.label_type, device)

            # 평가
            val_score = test(f2c_dict, model, loss_func, vxl, vxr, vy, fold, epoch, args.batch_size, args.label_type, device)

            val_acc = val_score[0] # acc or auc
            if epoch > 30:
                early_stopping(val_acc, model)
                if early_stopping.counter == 0:
                    best_val_score = val_score
                if early_stopping.early_stop or epoch == args.epoch - 1:
                    break

            # epoch 결과 출력
            print(best_val_score)
            print("=" * 100)

        # 현재 fold 성능 저장
        scores.append(best_val_score)
        print(f"[Fold {fold}] Best Score:", best_val_score)

    # 전체 fold 결과 평균 계산
    scores = np.array(scores).mean(axis=0)

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
        'epoch': 100,
        'patience': 50,
        'lr': 1e-3,
        'weight_decay': 1e-5,
        'label_type': 'multi_class',  # binary_class
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


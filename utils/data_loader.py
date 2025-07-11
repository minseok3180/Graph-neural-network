import os
import numpy as np
import dgl
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils import data
import pickle
from utils.KFold import KFold
import pandas as pd


def load_data(data_path='./data', device=torch.device('cpu')):
    kg_file = os.path.join(data_path, 'kg_data.pkl')
    if os.path.exists(kg_file):
        with open(kg_file, 'rb') as f:
            kg_g, e_feat = pickle.load(f)
        kg_g.to(device)

    else:
        edges, e_feat = [], []
        with open(os.path.join(data_path, 'edges.tsv')) as f:
            for line in tqdm(f, desc='loading kg'):
                h, r, t = map(int, line.strip().split('\t'))
                if h == t: continue  # self loop 제외
                edges += [[h, t], [t, h]]
                e_feat += [r + 1, r + 1]  # 0은 self-loop에 사용하기 위해 예약됨

        kg_g = dgl.graph(edges)  # dgl graph 자료
        kg_g = dgl.add_self_loop(dgl.remove_self_loop(kg_g))
        pad = kg_g.num_edges() - len(e_feat)
        e_feat = np.concatenate([e_feat, np.zeros(pad, dtype=np.int64)])  # self loop 명시적 생성, e_feat(list)에도 추가

        with open(os.path.join(data_path, 'nodes.tsv')) as f:
            nodes = []
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    node_id = int(parts[0])
                    node_type = int(parts[2])
                    nodes.append([node_id, node_type])

        kg_g.ndata['nodes'] = torch.tensor(nodes, dtype=torch.long)
        kg_g.edata['edges'] = torch.from_numpy(e_feat)
        kg_g.to(device)

        with open(kg_file, 'wb') as f:
            pickle.dump([kg_g, e_feat], f)

    with open(os.path.join(data_path, 'smiles.tsv')) as f:
        smiles_list = [line.strip().split('\t')[1] for line in f]  # [num_smiles]: smiles 데이터

    with open(os.path.join(data_path, 'f2cs.pkl'), 'rb') as f:
        f2c_dict = pickle.load(f)
    return kg_g.to(device), smiles_list, f2c_dict

def get_train_test(data_path='./data', fold_num=5, label_type='multi_class', condition='s1'):
    sample = pd.read_csv(os.path.join(data_path, 'ddi.tsv'), sep='\t').values
    kfold = KFold(n_splits=fold_num, shuffle=True, random_state=42, up_sample=(label_type == 'multi_class'), condition=condition)
    train_sample, test_sample = [], []

    for train, test in kfold.split(sample):
        train_sample.append(train)
        test_sample.append(test)

    if label_type not in ['binary_class', 'multi_label']:  # label_type = 'multi_class'인 경우
        return train_sample, test_sample
    
    # DDI Set (positive sample 기준)
    ddi_data = pd.read_csv(os.path.join(data_path, 'ddi.tsv'), sep='\t').values
    ddi_set = set('\t'.join(map(str, row)) for row in (ddi_data if label_type == 'multi_label' else ddi_data[:, :2]))

    for fold in range(fold_num):
        train, test = train_sample[fold], test_sample[fold]

        # drug ID pool 정의
        if condition == 's1':
            all_drugs = np.arange(max(train[:, :2].max(), test[:, :2].max()) + 1)
            train_drugs = test_drugs = all_drugs
        else:
            train_drugs = np.unique(train[:, :2])
            test_drugs = np.unique(test[:, :2])

        # binary label 1 할당
        if label_type == 'binary_class':
            train[:, 2] = 1
            test[:, 2] = 1

        # negative sample 생성 함수
        def generate_neg(pos, drugs):
            neg = []
            for i in tqdm(range(len(pos))):
                while True:
                    d1, d2 = np.random.choice(drugs, 2, replace=False)
                    if label_type == 'binary_class':
                        key1, key2 = f"{d1}\t{d2}", f"{d2}\t{d1}"
                    else:
                        key1 = f"{d1}\t{d2}\t{pos[i][2]}"
                        key2 = f"{d2}\t{d1}\t{pos[i][2]}"
                    if key1 not in ddi_set and key2 not in ddi_set:
                        neg.append([d1, d2, 0])
                        break
            return np.array(neg, dtype=pos.dtype)

        train_neg = generate_neg(train, train_drugs)
        test_neg = generate_neg(test, test_drugs)

        # multi-label 확장
        if label_type == 'multi_label':
            for s in [train, test]:
                s[:] = np.concatenate([s, s[:, 2:]], axis=1)
                s[:, 2] = 1  # positive flag

            for s_neg, s_pos in [(train_neg, train), (test_neg, test)]:
                s_neg = np.concatenate([s_neg, np.zeros((len(s_neg), 1), dtype=s_neg.dtype)], axis=1)
                s_neg[:, 3] = s_pos[:, 3]  # 같은 label 붙이기

        # 통합
        train_sample[fold] = np.concatenate([train, train_neg])
        test_sample[fold] = np.concatenate([test, test_neg])

    return train_sample, test_sample


# inference용 모든 데이터 가져오는 dataloader
def get_all_data(data_path='./data', label_type='multi_class', condition='s1'):
    sample = pd.read_csv(os.path.join(data_path, 'ddi.tsv'), sep='\t').values

    if label_type not in ['binary_class', 'multi_label']:
        return sample  # multi_class는 pos만 사용

    ddi_data = pd.read_csv(os.path.join(data_path, 'ddi.tsv'), sep='\t').values
    ddi_set = set('\t'.join(map(str, row)) for row in (ddi_data if label_type == 'multi_label' else ddi_data[:, :2]))

    if condition == 's1':
        all_drugs = np.arange(sample[:, :2].max() + 1)
    else:
        all_drugs = np.unique(sample[:, :2])

    def generate_neg(pos, drugs):
        neg = []
        for i in tqdm(range(len(pos)), desc='Generating inference negative samples'):
            while True:
                d1, d2 = np.random.choice(drugs, 2, replace=False)
                if label_type == 'binary_class':
                    key1, key2 = f"{d1}\t{d2}", f"{d2}\t{d1}"
                else:
                    key1 = f"{d1}\t{d2}\t{pos[i][2]}"
                    key2 = f"{d2}\t{d1}\t{pos[i][2]}"
                if key1 not in ddi_set and key2 not in ddi_set:
                    neg.append([d1, d2, 0])
                    break
        return np.array(neg, dtype=pos.dtype)

    sample[:, 2] = 1  # positive
    neg_sample = generate_neg(sample, all_drugs)

    if label_type == 'multi_label':
        sample = np.concatenate([sample, sample[:, 2:]], axis=1)
        sample[:, 2] = 1
        neg_sample = np.concatenate([neg_sample, np.zeros((len(neg_sample), 1), dtype=neg_sample.dtype)], axis=1)
        neg_sample[:, 3] = sample[:, 3]  # label 맞춰줌

    all_data = np.concatenate([sample, neg_sample])
    return all_data


if __name__ == '__main__':
    os.chdir('../')
    # load_data()
    # get_train_test()

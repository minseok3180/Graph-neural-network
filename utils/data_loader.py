import os
import numpy as np
import dgl
import tqdm
import torch
from torch.utils import data
import pickle
from utils.KFold import KFold
import pandas as pd


def load_data(data_path='./data', device=torch.device('cpu')):
    kg_file = os.path.join(data_path, 'kg_data.pkl')
    if os.path.exists(kg_file):
        with open(kg_file, 'rb') as f:
            kg_g, e_feat = pickle.load(f)
    else:
        edges, e_feat = [], []
        with open(os.path.join(data_path, 'edges.tsv')) as f:
            for line in tqdm.tqdm(f, desc='loading kg'):
                h, r, t = map(int, line.strip().split('\t'))
                if h == t: continue  # self loop 제외
                edges += [[h, t], [t, h]]
                e_feat += [r + 1, r + 1]  # 0은 self-loop에 사용하기 위해 예약됨

        kg_g = dgl.graph(edges)  # dgl graph 자료
        kg_g = dgl.add_self_loop(dgl.remove_self_loop(kg_g))
        pad = kg_g.num_edges() - len(e_feat)
        e_feat = np.concatenate([e_feat, np.zeros(pad, dtype=np.int64)])  # self loop 명시적 생성, e_feat(list)에도 추가

        with open(os.path.join(data_path, 'nodes.tsv')) as f:
            nodes: list = [list(map(int, line.strip().split('\t')[::2])) for line in f]  # [객체id * 2] -> 한 행당 '객체, 범주' 형태로 이뤄짐 
        kg_g.ndata['nodes'] = torch.tensor(nodes)  # [num_nodes, 2]
        kg_g.edata['edges'] = torch.from_numpy(e_feat)  # [num_edges]: label class 정보 포함

        with open(kg_file, 'wb') as f:
            pickle.dump([kg_g, e_feat], f)

    with open(os.path.join(data_path, 'smiles.tsv')) as f:
        smiles_list = [line.strip().split('\t')[1] for line in f]  # [num_smiles]: smiles 데이터

    return kg_g.to(device), smiles_list

def update_food_data_with_indices(data_path='./data'):
    # 경로 설정
    smiles_file = os.path.join(data_path, 'smiles.tsv')
    food_node_file = os.path.join(data_path, 'food_node.tsv')
    food_edge_file = os.path.join(data_path, 'food_edge.tsv')
    food_node_indexed_file = os.path.join(data_path, 'food_node_indexed.tsv')
    food_edge_indexed_file = os.path.join(data_path, 'food_edge_indexed.tsv')

    # 1. smiles.tsv 로드 및 dict 생성
    smiles_df = pd.read_csv(smiles_file, sep='\t', header=None, names=['idx', 'smiles'])
    smiles_to_idx = {s: i for i, s in zip(smiles_df['idx'], smiles_df['smiles'])}
    next_smiles_idx = max(smiles_df['idx']) + 1 if not smiles_df.empty else 0
    print(next_smiles_idx)

    # 2. food_node.tsv 로드
    node_df = pd.read_csv(food_node_file, sep='\t')
    food_list = node_df[node_df['node_type'] == 'food']['id'].tolist()
    molecule_list = node_df[node_df['node_type'] == 'molecule']['id'].tolist()

    # 3. smiles 추가 indexing
    for smiles in molecule_list:
        if smiles not in smiles_to_idx:
            smiles_to_idx[smiles] = next_smiles_idx
            smiles_df.loc[len(smiles_df)] = [next_smiles_idx, smiles]
            next_smiles_idx += 1

    # 4. food indexing
    food_to_idx = {food: i for i, food in enumerate(sorted(set(food_list)))}

    # 5. food_node_indexed.tsv 생성
    indexed_nodes = []
    for _, row in node_df.iterrows():
        if row['node_type'] == 'food':
            idx = food_to_idx[row['id']]
        else:  # molecule
            idx = smiles_to_idx[row['id']]
        indexed_nodes.append([idx, row['id'], row['node_type']])
    pd.DataFrame(indexed_nodes, columns=['id', 'name', 'node_type']).to_csv(
        food_node_indexed_file, sep='\t', index=False
    )

    # 6. food_edge.tsv 처리 및 indexing
    edge_df = pd.read_csv(food_edge_file, sep='\t')
    edge_rows = []
    for _, row in edge_df.iterrows():
        source = row['source']
        target = row['target']
        weight = row['weight']
        if source not in food_to_idx:
            food_to_idx[source] = len(food_to_idx)
        if target not in smiles_to_idx:
            smiles_to_idx[target] = next_smiles_idx
            smiles_df.loc[len(smiles_df)] = [next_smiles_idx, target]
            next_smiles_idx += 1
        edge_rows.append([food_to_idx[source], smiles_to_idx[target], weight])

    pd.DataFrame(edge_rows, columns=['source', 'target', 'weight']).to_csv(
        food_edge_indexed_file, sep='\t', index=False
    )

    # 7. smiles.tsv 갱신 저장
    smiles_df = smiles_df.sort_values('idx').reset_index(drop=True)
    smiles_df.to_csv(smiles_file, sep='\t', index=False, header=False)

    return food_node_indexed_file, food_edge_indexed_file, smiles_file

def get_train_test(data_path='./data', fold_num=5, label_type='multi_class', condition='s1'):
    sample = pd.read_csv(os.path.join(data_path, 'ddi.tsv'), sep='\t').values
    kfold = KFold(n_splits=fold_num, shuffle=True, random_state=42, up_sample=(label_type == 'multi_class'), condition=condition)
    train_sample, test_sample = [], []

    for train, test in kfold.split(sample):
        train_sample.append(train)
        test_sample.append(test)

    # generate negative samples
    if label_type == 'binary_class' or label_type == 'multi_label':
        if label_type == 'multi_label':
            all_ddi = pd.read_csv(os.path.join(data_path, 'ddi.tsv'), sep='\t').values
        else:
            all_ddi = pd.read_csv(os.path.join(data_path, 'ddi.tsv'), sep='\t').values[:, :2]
        ddi = set()
        for item in all_ddi:
            ddi.add('\t'.join([str(_) for _ in item]))

        for fold in range(fold_num):
            if condition == 's1':
                train_drug = np.arange(max(train_sample[fold][:, :2].max(), test_sample[fold][:, :2].max())+1)
                test_drug = train_drug
            else:
                train_drug = np.unique(train_sample[fold][:, :2])
                test_drug = np.unique(test_sample[fold][:, :2])

            if label_type == 'binary_class':
                train_sample[fold][:, 2] = 1
                test_sample[fold][:, 2] = 1

            sample_neg = []
            sample_pos = np.concatenate([train_sample[fold], test_sample[fold]])
            for i in range(len(train_sample[fold])):
                while True:
                    d1 = train_drug[np.random.randint(len(train_drug))]
                    d2 = train_drug[np.random.randint(len(train_drug))]
                    if d1 == d2:
                        continue
                    if label_type == 'binary_class' and ('\t'.join([str(d1), str(d2)]) not in ddi and '\t'.join([str(d2), str(d1)]) not in ddi):
                        sample_neg.append([d1, d2, 0])
                        break
                    elif label_type == 'multi_label' and ('\t'.join([str(d1), str(d2), str(sample_pos[i][2])]) not in ddi and '\t'.join([str(d2), str(d1), str(sample_pos[i][2])]) not in ddi):
                        sample_neg.append([d1, d2, 0])
                        break
            for i in range(len(train_sample[fold]), len(train_sample[fold])+len(test_sample[fold])):
                while True:
                    d1 = test_drug[np.random.randint(len(test_drug))]
                    d2 = test_drug[np.random.randint(len(test_drug))]
                    if d1 == d2:
                        continue
                    if label_type == 'binary_class' and ('\t'.join([str(d1), str(d2)]) not in ddi and '\t'.join([str(d2), str(d1)]) not in ddi):
                        sample_neg.append([d1, d2, 0])
                        break
                    elif label_type == 'multi_label' and ('\t'.join([str(d1), str(d2), str(sample_pos[i][2])]) not in ddi and '\t'.join([str(d2), str(d1), str(sample_pos[i][2])]) not in ddi):
                        sample_neg.append([d1, d2, 0])
                        break
            train_sample_neg = np.array(sample_neg[:len(train_sample[fold])])
            test_sample_neg = np.array(sample_neg[len(train_sample[fold]):])
            if label_type == 'multi_label':
                train_sample[fold] = np.concatenate([train_sample[fold], train_sample[fold][:, 2:]], axis=-1)
                test_sample[fold] = np.concatenate([test_sample[fold], test_sample[fold][:, 2:]], axis=-1)
                train_sample[fold][:, 2] = 1
                test_sample[fold][:, 2] = 1

                train_sample_neg = np.concatenate([train_sample_neg, np.zeros((train_sample_neg.shape[0], 1), dtype=train_sample_neg.dtype)], axis=-1)
                test_sample_neg = np.concatenate([test_sample_neg, np.zeros((test_sample_neg.shape[0], 1), dtype=test_sample_neg.dtype)], axis=-1)
                train_sample_neg[:, 3] = train_sample[fold][:, 3]
                test_sample_neg[:, 3] = test_sample[fold][:, 3]

            train_sample[fold] = np.concatenate([train_sample[fold], train_sample_neg])
            test_sample[fold] = np.concatenate([test_sample[fold], test_sample_neg])

    return train_sample, test_sample

if __name__ == '__main__':
    os.chdir('../')
    # load_data()
    # get_train_test()
    update_food_data_with_indices(data_path='./data/FOODRKG+DrugBank')

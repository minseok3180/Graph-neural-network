import os
import pandas as pd
from rapidfuzz import process, fuzz
import pandas as pd
import os
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd

##########################################################################################
def build_fuzzy_food_mapper(kg_nodes_file, threshold=40):
    # 1. load kg_nodes.tsv
    kg_df = pd.read_csv(kg_nodes_file, sep='\t', header=None, names=['index', 'name', 'type'])

    # 2. extract food names (without "Food::")
    kg_food_df = kg_df[kg_df['name'].str.startswith('Food::')]
    foodname_to_kgid = {}
    food_names_cleaned = []

    for idx, name in zip(kg_food_df['index'], kg_food_df['name']):
        clean_name = name.replace('Food::', '', 1).strip()
        foodname_to_kgid[clean_name] = idx
        food_names_cleaned.append(clean_name)

    # 3. define fuzzy matching function
    def match_fuzzy_name(name):
        name = name.strip()
        if name in foodname_to_kgid:
            return foodname_to_kgid[name], name, 100
        match, score, _ = process.extractOne(name, food_names_cleaned, scorer=fuzz.ratio)
        if score >= threshold:
            return foodname_to_kgid[match], match, score
        else:
            return None, None, score

    return match_fuzzy_name
##########################################################################################
def update_food_data_with_indices(data_path='./data'):
    # 경로 설정
    smiles_file = os.path.join(data_path, 'smiles.tsv')
    new_smiles_file = os.path.join(data_path, 'new_smiles.tsv')
    food_node_file = os.path.join(data_path, 'food_nodes.tsv')
    food_edge_file = os.path.join(data_path, 'food_edges.tsv')
    food_node_indexed_file = os.path.join(data_path, 'new_food_nodes.tsv')
    food_edge_indexed_file = os.path.join(data_path, 'new_food_edges.tsv')
    kg_nodes_file = os.path.join(data_path, 'new_kg_nodes.tsv')

    # 1. smiles.tsv 로드 및 dict 생성
    smiles_df = pd.read_csv(smiles_file, sep='\t', header=None, names=['idx', 'smiles'])
    smiles_to_idx = {s: i for i, s in zip(smiles_df['smiles'], smiles_df['idx'])}
    next_smiles_idx = max(smiles_df['idx']) + 1 if not smiles_df.empty else 0

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

    # 4. kg_nodes.tsv 로드하여 food name → index 매핑 생성
    kg_df = pd.read_csv(kg_nodes_file, sep='\t', header=None, names=['index', 'name', 'type'])
    kg_food_df = kg_df[kg_df['name'].str.startswith('Food::')]
    foodname_to_kgid = {
        name.split('Food::')[1]: idx
        for idx, name in zip(kg_food_df['index'], kg_food_df['name'])
    }
    
    # 5. food_node_indexed.tsv 생성
    indexed_nodes = []
    for line_num, row in node_df.iterrows():
        node_type = row['node_type']
        raw_id = row['id']
        if node_type == 'food':
            if raw_id in foodname_to_kgid:
                idx = foodname_to_kgid[raw_id]
            else:
                print(f"[food_node.tsv] Line {line_num}: '{raw_id}' not found in kg_nodes.tsv → keeping original value")
                idx = raw_id  # 원래 문자열 그대로
        else:  # molecule
            idx = smiles_to_idx[raw_id]
        indexed_nodes.append([idx, raw_id, node_type])
    pd.DataFrame(indexed_nodes, columns=['id', 'name', 'node_type']).to_csv(
        food_node_indexed_file, sep='\t', index=False
    )

    # 6. food_edge.tsv 처리 및 indexing
    edge_df = pd.read_csv(food_edge_file, sep='\t')
    edge_rows = []
    for line_num, row in edge_df.iterrows():
        source = row['source']
        target = row['target']
        weight = row['weight']

        if source in foodname_to_kgid:
            source_idx = foodname_to_kgid[source]
        else:
            print(f"[food_edge.tsv] Line {line_num}: source '{source}' not found in kg_nodes.tsv → keeping original value")
            source_idx = source  # 원래 문자열 그대로

        if target not in smiles_to_idx:
            smiles_to_idx[target] = next_smiles_idx
            smiles_df.loc[len(smiles_df)] = [next_smiles_idx, target]
            next_smiles_idx += 1
        target_idx = smiles_to_idx[target]

        edge_rows.append([source_idx, target_idx, weight])

    pd.DataFrame(edge_rows, columns=['source', 'target', 'weight']).to_csv(
        food_edge_indexed_file, sep='\t', index=False
    )

    # 7. smiles.tsv 갱신 저장
    smiles_df = smiles_df.sort_values('idx').reset_index(drop=True)
    smiles_df.to_csv(new_smiles_file, sep='\t', index=False, header=False)

    return food_node_indexed_file, food_edge_indexed_file, smiles_file

##########################################################################################
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

def run_bert_node_edge_matching(data_path='./data'):
    # 경로 설정
    food_edge_indexed_file = os.path.join(data_path, 'new_food_edges.tsv')
    food_node_indexed_file = os.path.join(data_path, 'new_food_nodes.tsv')
    kg_nodes_file = os.path.join(data_path, 'new_kg_nodes.tsv')
    updated_edge_file = os.path.join(data_path, 'bert_food_edges.tsv')
    updated_node_file = os.path.join(data_path, 'bert_food_nodes.tsv')
    log_file = os.path.join(data_path, 'bert_matching_all.log')

    # 모델 로드
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # KG node 로드 및 음식 이름 인코딩
    kg_df = pd.read_csv(kg_nodes_file, sep='\t', header=None, names=['index', 'name', 'type'])
    kg_food_df = kg_df[kg_df['name'].str.startswith('Food::')]
    kg_food_names = kg_food_df['name'].str.replace('Food::', '', n=1).tolist()
    kg_food_embeddings = model.encode(kg_food_names, convert_to_tensor=True)
    kg_food_index_map = dict(zip(kg_food_names, kg_food_df['index']))

    # 캐시, 로그
    matched_cache = {}
    log_lines = []

    # === food_edges 처리 ===
    edge_df = pd.read_csv(food_edge_indexed_file, sep='\t')
    new_edges = []

    for _, row in tqdm(edge_df.iterrows()):
        source, target, weight = row['source'], row['target'], row['weight']
        if isinstance(source, str) and not source.isdigit():
            if source not in matched_cache:
                query_embedding = model.encode(source, convert_to_tensor=True)
                hits = util.semantic_search(query_embedding, kg_food_embeddings, top_k=1)[0]
                best = hits[0]
                matched_name = kg_food_names[best['corpus_id']]
                matched_index = kg_food_index_map[matched_name]
                matched_cache[source] = matched_index
                log_lines.append(f"[EDGE MATCH] '{source}' → '{matched_name}' (ID: {matched_index}, Score: {best['score']:.4f})")
            source = matched_cache[source]
        new_edges.append([source, target, weight])

    pd.DataFrame(new_edges, columns=['source', 'target', 'weight']).to_csv(updated_edge_file, sep='\t', index=False)

    # === food_nodes 처리 ===
    node_df = pd.read_csv(food_node_indexed_file, sep='\t')
    new_nodes = []

    for _, row in tqdm(node_df.iterrows()):
        nid, name, ntype = row['id'], row['name'], row['node_type']
        if row['node_type'] == 'food' and isinstance(nid, str) and not nid.isdigit():
            if nid not in matched_cache:
                query_embedding = model.encode(nid, convert_to_tensor=True)
                hits = util.semantic_search(query_embedding, kg_food_embeddings, top_k=1)[0]
                best = hits[0]
                matched_name = kg_food_names[best['corpus_id']]
                matched_index = kg_food_index_map[matched_name]
                matched_cache[nid] = matched_index
                log_lines.append(f"[NODE MATCH] '{nid}' → '{matched_name}' (ID: {matched_index}, Score: {best['score']:.4f})")
            nid = matched_cache[nid]
        new_nodes.append([nid, name, ntype])

    pd.DataFrame(new_nodes, columns=['id', 'name', 'node_type']).to_csv(updated_node_file, sep='\t', index=False)

    # 로그 저장
    with open(log_file, 'w', encoding='utf-8') as f:
        for line in log_lines:
            f.write(line + '\n')

    print(f"✅ BERT 매칭 완료:\n- {updated_edge_file}\n- {updated_node_file}\n📄 로그: {log_file}")


if __name__ == '__main__':
    # # os.chdir('../')
    # # update_food_data_with_indices(data_path='./data/FOODRKG+DrugBank')
    ##########################################################################################

    # matcher = build_fuzzy_food_mapper('./data/FOODRKG+DrugBank/new_kg_nodes.tsv')

    # test_names = ['Highbush blueberry', 'Chilli pepper [Green]', 'Black huckleberry']

    # for name in test_names:
    #     idx, matched, score = matcher(name)
    #     if idx is not None:
    #         print(f"[MATCHED] '{name}' → '{matched}' (ID: {idx}, Score: {score})")
    #     else:
    #         print(f"[FAILED] '{name}' → No match found (Score: {score})")
    ##########################################################################################
    
    # # SBERT 모델 로딩 (한글이면 klue/roberta-small 등, 영어면 all-MiniLM 등)
    # model = SentenceTransformer('all-MiniLM-L6-v2')

    # kg_df = pd.read_csv('./data/FOODRKG+DrugBank/new_kg_nodes.tsv', sep='\t', header=None, names=['index', 'name', 'type'])
    # kg_food_df = kg_df[kg_df['name'].str.startswith('Food::')]

    # # 이름 정제
    # food_names = kg_food_df['name'].str.replace('Food::', '', n=1).tolist()
    # food_embeddings = model.encode(food_names, convert_to_tensor=True)
    # index_mapping = dict(zip(food_names, kg_food_df['index']))

    # def bert_match_food_name(query_name, top_k=1, threshold=0.4):
    #     query_embedding = model.encode(query_name, convert_to_tensor=True)
    #     hits = util.semantic_search(query_embedding, food_embeddings, top_k=top_k)[0]
    #     score = hits[0]['score']
    #     if score >= threshold:
    #         matched_name = food_names[hits[0]['corpus_id']]
    #         matched_idx = index_mapping[matched_name]
    #         return matched_idx, matched_name, score
    #     else:
    #         return None, None, score

    # query_list = ['Highbush blueberry', 'Chilli pepper [Green]', 'Black huckleberry']

    # for name in query_list:
    #     idx, matched, score = bert_match_food_name(name)
    #     if idx:
    #         print(f"[BERT MATCHED] '{name}' → '{matched}' (ID: {idx}, Score: {score:.3f})")
    #     else:
    #         print(f"[FAILED] '{name}' → No match (Score: {score:.3f})")
    ##########################################################################################
    # run_bert_node_edge_matching('./data/FOODRKG+DrugBank')
    ##########################################################################################
    # import pandas as pd

    # # 경로 설정
    # file_path = './data/FOODRKG+DrugBank/new_ddi.tsv'
    # filtered_path = './data/FOODRKG+DrugBank/filtered_new_ddi.tsv'
    # removed_path = './data/FOODRKG+DrugBank/removed_ddi.tsv'

    # # 파일 로드
    # df = pd.read_csv(file_path, sep='\t', header=None, names=['drug1', 'drug2', 'label'])

    # # 필터 조건: drug1 또는 drug2 중 하나라도 200000 이상인 행
    # mask_removed = (df['drug1'] >= 200000) | (df['drug2'] >= 200000)

    # # 제거된 행
    # removed_df = df[mask_removed]

    # # 유지할 행 (반대 조건)
    # filtered_df = df[~mask_removed]

    # # 결과 저장
    # filtered_df.to_csv(filtered_path, sep='\t', index=False, header=False)
    # removed_df.to_csv(removed_path, sep='\t', index=False, header=False)

    # print(f"✅ 필터링 완료\n- 유지된 행: {filtered_path}\n- 제거된 행: {removed_path}")
    ##########################################################################################

    import pandas as pd
    from collections import defaultdict

    # 파일 경로 설정
    edge_file = './data/FOODRKG+DrugBank/food_edges.tsv'

    # 데이터 로드
    df = pd.read_csv(edge_file, sep='\t')  # header 포함되어 있다고 가정

    # defaultdict로 초기화
    food_to_compounds = defaultdict(list)

    # 데이터 순회하며 dict 구성
    for _, row in df.iterrows():
        food_id = int(row['source'])
        compound_id = int(row['target'])
        weight = float(row['weight'])
        if weight > 0:
            food_to_compounds[food_id].append((compound_id, weight))

    import pickle

    # 저장
    with open('./data/FOODRKG+DrugBank/F2Cs.pkl', 'wb') as f:
        pickle.dump(food_to_compounds, f)

    # 로드
    with open('./data/FOODRKG+DrugBank/F2Cs.pkl', 'rb') as f:
        food_to_compounds = pickle.load(f)

    # 예시 출력
    example_id = 98261
    print(f"Food {example_id} has compounds:", food_to_compounds[example_id][:5])

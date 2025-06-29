import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.hgnn import HGNN
from model.decoder import Mlp
from model.mol import Mol
from torch.utils.checkpoint import checkpoint


class HetDDI(nn.Module):
    def __init__(self,
                 kg_g,
                 smiles,
                 num_hidden,
                 num_layer,
                 mode,
                 class_num,
                 condition
                 ):
        super(HetDDI, self).__init__()

        self.smiles = smiles
        self.device = kg_g.device
        self.mode = mode
        self.drug_num = len(smiles)

        dropout = 0.1
        if self.mode == 'only_kg' or self.mode == 'concat':
            self.kg = HGNN(kg_g, kg_g.edata['edges'], kg_g.ndata['nodes'], num_hidden, num_layer=num_layer)
            # self.kg.load_state_dict(torch.load('./kg_weight.pth'))  #####
            self.kg_size = self.kg.get_output_size()
            self.kg_fc = nn.Sequential(nn.Linear(self.kg_size, self.kg_size),
                                       nn.BatchNorm1d(self.kg_size),
                                       nn.Dropout(dropout),
                                       nn.ReLU(),

                                       nn.Linear(self.kg_size, self.kg_size),
                                       nn.BatchNorm1d(self.kg_size),
                                       nn.Dropout(dropout),
                                       nn.ReLU(),

                                       nn.Linear(self.kg_size, self.kg_size),
                                       nn.BatchNorm1d(self.kg_size),
                                       nn.Dropout(dropout),
                                       nn.ReLU()
                                       )

        if self.mode == 'only_mol' or self.mode == 'concat':
            self.mol = Mol(smiles, num_hidden, num_layer, self.device, condition)
            self.mol_size = self.mol.gnn.get_output_size()
            self.mol_fc = nn.Sequential(nn.Linear(self.mol_size, self.mol_size),
                                        nn.BatchNorm1d(self.mol_size),
                                        nn.Dropout(dropout),
                                        nn.ReLU(),

                                        nn.Linear(self.mol_size, self.mol_size),
                                        nn.BatchNorm1d(self.mol_size),
                                        nn.Dropout(dropout),
                                        nn.ReLU(),

                                        nn.Linear(self.mol_size, self.mol_size),
                                        nn.BatchNorm1d(self.mol_size),
                                        nn.Dropout(dropout),
                                        nn.ReLU()
                                        )

        if self.mode == 'only_kg':
            self.decoder = Mlp(self.kg_size, 0, class_num=class_num)
        elif self.mode == 'only_mol':
            self.decoder = Mlp(0, self.mol_size, class_num=class_num)
        else:
            self.decoder = Mlp(self.kg_size, self.mol_size, class_num=class_num)

    def forward(self, left, right, f2c):
        
        def resolve_entity(idx):
            if idx < 90000:
                return mol_emb[idx]
            else:
                cids = f2c[idx]['compounds']  # [N]
                ws = torch.tensor(f2c[idx]['weights'], device=mol_emb.device, dtype=mol_emb.dtype)

                compound_vecs = mol_emb[cids]  # [N, D]
                return (ws.unsqueeze(1) * compound_vecs).sum(dim=0)  # [D]

        if self.mode == 'only_kg':
            # kg_emb = checkpoint(self.kg)[:self.drug_num]
            kg_emb = self.kg()[:self.drug_num]
            kg_emb = self.kg_fc(kg_emb)

            left_kg_emb = kg_emb[left]
            right_kg_emb = kg_emb[right]

            return self.decoder(left_kg_emb, right_kg_emb)

        elif self.mode == 'only_mol':
            mol_emb = self.mol()
            mol_emb = self.mol_fc(mol_emb)

            left_mol_emb = torch.stack([resolve_entity(i.item()) for i in left])
            right_mol_emb = torch.stack([resolve_entity(i.item()) for i in right])

            return self.decoder(left_mol_emb, right_mol_emb)

        else:
            # kg_emb = checkpoint(self.kg)[:self.drug_num]
            kg_emb = self.kg()
            print("kg_emb.shape:", kg_emb.shape)
            kg_emb = self.kg_fc(kg_emb)
            print("kg_emb.shape after fc:", kg_emb.shape)

            left_kg_emb = kg_emb[left]
            right_kg_emb = kg_emb[right]

            mol_emb = self.mol()
            print("mol_emb.shape before fc:", mol_emb.shape)
            mol_emb = self.mol_fc(mol_emb)
            print("mol_emb.shape:", mol_emb.shape)

            left_mol_emb = torch.stack([resolve_entity(i.item()) for i in left])
            right_mol_emb = torch.stack([resolve_entity(i.item()) for i in right])

            left_emb = torch.concat([left_kg_emb, left_mol_emb], dim=-1)
            right_emb = torch.concat([right_kg_emb, right_mol_emb], dim=-1)

            return self.decoder(left_emb, right_emb)


if __name__ == '__main__':
    import warnings
    from easydict import EasyDict
    from utils.data_loader import load_data
    from utils.logger import Logger

    args = EasyDict({
        'hidden_dim': 300,
        'num_layer': 4,
        'epoch': 100,
        'lr': 1e-3,
        'data_path': './data',
        'kg_name': 'FOODRKG',
        'ddi_name': 'DrugBank',
    })

    os.makedirs('./log', exist_ok=True)
    sys.stdout = Logger(
        f'./log/pretraining-hidden_dim_{args.hidden_dim} '
        f'num_layer_{args.num_layer} '
        f'epoch_{args.epoch}.txt',
        sys.stdout
    )
    warnings.filterwarnings("ignore", category=UserWarning)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('running on', device)

    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    data_path = os.path.join(args.data_path, f"{args.kg_name}+{args.ddi_name}")
    kg_g, smiles, f2c_dict = load_data(data_path, device=device)

    kg_model = HGNN(kg_g, kg_g.edata['edges'], kg_g.ndata['nodes'], args.hidden_dim, num_layer=args.num_layer).to(device)

    class KGPretrainLoss(nn.Module):
        def __init__(self, embedding_dim, num_rel):
            super().__init__()
            self.relation_classifier = nn.Linear(embedding_dim * 2, num_rel)

        def forward(self, h_emb, t_emb, edge_labels):
            x = torch.cat([h_emb, t_emb], dim=-1)
            logits = self.relation_classifier(x)
            return F.cross_entropy(logits, edge_labels)

    num_rel = int(kg_g.edata['edges'].max().item()) + 1
    epochs = args.epoch
    loss_fn = KGPretrainLoss(kg_model.get_output_size(), num_rel).to(device)
    optimizer = torch.optim.Adam(kg_model.parameters(), lr=args.lr)

    src, dst = kg_g.edges()
    edge_labels = kg_g.edata['edges']

    kg_model.train()
    for epoch in range(epochs):
        node_emb = kg_model()
        h = node_emb[src]
        t = node_emb[dst]
        loss = loss_fn(h, t, edge_labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"[Epoch {epoch}] Loss: {loss.item():.4f}")

    kg_model.eval()
    torch.save(kg_model.state_dict(), './kg_weight.pth')
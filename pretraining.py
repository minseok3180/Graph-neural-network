class KGPretrainLoss(nn.Module):
    def __init__(self, embedding_dim, num_rel):
        super().__init__()
        self.relation_classifier = nn.Linear(embedding_dim * 2, num_rel)

    def forward(self, h_emb, t_emb, edge_labels):
        x = torch.cat([h_emb, t_emb], dim=-1)
        logits = self.relation_classifier(x)
        return F.cross_entropy(logits, edge_labels)

model = HGNN(kg_g, kg_g.edata['edges'], kg_g.ndata['nodes'], hidden_dim, num_layer).to(device)


model.train()
loss_fn = KGPretrainLoss(model.get_output_size(), num_rel).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# edge triplet
src, dst = kg_g.edges()
edge_labels = kg_g.edata['edges']  # shape [num_edges]

for epoch in range(epochs):
    node_emb = model()  # 전체 노드 embedding
    h = node_emb[src]
    t = node_emb[dst]
    
    loss = loss_fn(h, t, edge_labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print(f"[Epoch {epoch}] Loss: {loss.item():.4f}")
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------- GNN MODELS ---------------- #
class GCN(nn.Module):
    def __init__(self, in_c, hid_c, out_c):
        super().__init__()
        self.conv1 = GCNConv(in_c, hid_c)
        self.conv2 = GCNConv(hid_c, out_c)
        self.hidden = None
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        self.hidden = x.detach().cpu()
        return self.conv2(x, edge_index)

class GAT(nn.Module):
    def __init__(self, in_c, hid_c, out_c):
        super().__init__()
        self.conv1 = GATConv(in_c, hid_c, heads=2, concat=True)
        self.conv2 = GATConv(hid_c * 2, out_c, heads=1, concat=False)
        self.hidden = None
    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        self.hidden = x.detach().cpu()
        return x

class GraphSAGE(nn.Module):
    def __init__(self, in_c, hid_c, out_c):
        super().__init__()
        self.conv1 = SAGEConv(in_c, hid_c)
        self.conv2 = SAGEConv(hid_c, out_c)
        self.hidden = None
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        self.hidden = x.detach().cpu()
        return self.conv2(x, edge_index)

class GINNet(nn.Module):
    def __init__(self, in_c, hid_c, out_c):
        super().__init__()
        nn1 = nn.Sequential(nn.Linear(in_c, hid_c), nn.ReLU(), nn.Linear(hid_c, hid_c))
        nn2 = nn.Sequential(nn.Linear(hid_c, hid_c), nn.ReLU(), nn.Linear(hid_c, out_c))
        self.conv1 = GINConv(nn1)
        self.conv2 = GINConv(nn2)
        self.hidden = None
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        self.hidden = x.detach().cpu()
        return self.conv2(x, edge_index)

class HybridGCNGAT(nn.Module):
    def __init__(self, in_c, hid_c, out_c):
        super().__init__()
        self.gcn = GCNConv(in_c, hid_c)
        self.gat = GATConv(hid_c, hid_c, heads=1)
        self.fc = nn.Linear(hid_c, out_c)
        self.hidden = None
    def forward(self, x, edge_index):
        x = F.relu(self.gcn(x, edge_index))
        x = F.relu(self.gat(x, edge_index))
        self.hidden = x.detach().cpu()
        return self.fc(x)

MODELS = {
    "GCN": GCN,
    "GAT": GAT,
    "GraphSAGE": GraphSAGE,
    "GIN": GINNet,
    "Hybrid": HybridGCNGAT
}

# ---------------- DATA LOADER ---------------- #
def load_data():
    features = pd.read_csv("elliptic_txs_features.csv", header=None)
    labels = pd.read_csv("elliptic_txs_classes.csv")
    edges = pd.read_csv("elliptic_txs_edgelist.csv", header=None, names=["source", "target"])

    features = features.rename(columns={0: "tx_id", 1: "time_step"})
    labels = labels.rename(columns={"txId": "tx_id", "class": "label"})
    labels["label"] = labels["label"].replace({"unknown": -1, "1": 0, "2": 1})

    for col in ["tx_id"]:
        features[col] = pd.to_numeric(features[col], errors="coerce")
        labels[col] = pd.to_numeric(labels[col], errors="coerce")
    for col in ["source", "target"]:
        edges[col] = pd.to_numeric(edges[col], errors="coerce")

    full_data = features.merge(labels, on="tx_id", how="left")
    all_ids = pd.concat([features["tx_id"], edges["source"], edges["target"]]).dropna().unique()
    tx_id_map = {id_: i for i, id_ in enumerate(sorted(all_ids.astype(int)))}

    full_data["mapped_id"] = full_data["tx_id"].map(tx_id_map)
    edges["source"] = edges["source"].map(tx_id_map)
    edges["target"] = edges["target"].map(tx_id_map)
    edge_index = torch.tensor(edges.dropna().astype(int).values.T, dtype=torch.long)

    num_nodes = len(tx_id_map)
    x = torch.zeros((num_nodes, 165))
    y = torch.full((num_nodes,), -1, dtype=torch.long)

    known = full_data[full_data["label"] != -1]
    ids = known["mapped_id"].values
    x[ids] = torch.tensor(known.iloc[:, 2:167].values, dtype=torch.float)
    y[ids] = torch.tensor(known["label"].values, dtype=torch.long)

    print(f"Loaded: {x.shape[0]} nodes, {edge_index.shape[1]} edges")
    return Data(x=x, edge_index=edge_index, y=y)

# ---------------- TRAIN + ENSEMBLE ---------------- #
def train_and_get_embeddings(data, model_cls, name):
    valid_mask = data.y >= 0
    idx = torch.where(valid_mask)[0]
    y_np = data.y[valid_mask].cpu().numpy()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_embs, all_labels = [], []
    gnn_metrics = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(idx, y_np)):
        print(f"Fold {fold+1}")
        model = model_cls(data.num_features, 64, 2).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()
        train_idx = idx[train_idx]
        test_idx = idx[test_idx]

        for _ in range(30):
            model.train()
            out = model(data.x.to(device), data.edge_index.to(device))
            loss = loss_fn(out[train_idx], data.y[train_idx].to(device))
            loss.backward()
            opt.step()
            opt.zero_grad()

        model.eval()
        out = model(data.x.to(device), data.edge_index.to(device)).cpu().detach()
        pred = out[test_idx].argmax(dim=1)
        true = data.y[test_idx].cpu()
        gnn_metrics.append({
            "Acc": accuracy_score(true, pred),
            "Prec": precision_score(true, pred),
            "Rec": recall_score(true, pred),
            "F1": f1_score(true, pred)
        })

        _ = model(data.x.to(device), data.edge_index.to(device))
        emb = model.hidden[test_idx].numpy()
        lbl = data.y[test_idx].cpu().numpy()
        all_embs.append(emb)
        all_labels.append(lbl)

    avg_gnn = pd.DataFrame(gnn_metrics).mean().to_dict()
    return np.vstack(all_embs), np.hstack(all_labels), avg_gnn

def run_ensemble(X, y):
    results = {}
    for clf_name, clf in {
        "RF": RandomForestClassifier(n_estimators=100, max_depth=10),
        "XGB": XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss'),
        "LGBM": LGBMClassifier(n_estimators=100)
    }.items():
        clf.fit(X, y)
        pred = clf.predict(X)
        results[clf_name] = {
            "Acc": accuracy_score(y, pred),
            "Prec": precision_score(y, pred),
            "Rec": recall_score(y, pred),
            "F1": f1_score(y, pred)
        }
    return results

# ---------------- MAIN ---------------- #
if __name__ == "__main__":
    data = load_data()
    summary_rows = []

    for name, Model in MODELS.items():
        print(f"\nRunning {name}...")
        X, y, gnn_scores = train_and_get_embeddings(data, Model, name)
        ens_scores = run_ensemble(X, y)

        summary_rows.append({
            "Model": name,
            **{f"GNN_{k}": v for k, v in gnn_scores.items()},
            **{f"{clf}_{k}": v for clf, res in ens_scores.items() for k, v in res.items()}
        })

    df = pd.DataFrame(summary_rows).round(4)
    print("\nPerformance Comparison Table:")
    print(df)

    os.makedirs("figures", exist_ok=True)
    df.to_csv("figures/final_performance_table.csv", index=False)

    plt.figure(figsize=(12, 6))
    df.set_index("Model")[["GNN_F1", "RF_F1", "XGB_F1", "LGBM_F1"]].plot(kind="bar")
    plt.title("F1 Score Comparison")
    plt.ylabel("F1 Score")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("figures/f1_barplot.png")
    plt.close()

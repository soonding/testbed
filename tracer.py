import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GraphConv, GINConv
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from captum.attr import IntegratedGradients
from umap import UMAP
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# ------------------ GNN Model Definitions ------------------ #
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
        self.att_weights = None
        self.hidden = None
    def forward(self, x, edge_index):
        x, _ = self.conv1(x, edge_index, return_attention_weights=True)
        x = F.elu(x)
        self.hidden = x.detach().cpu()
        x, att = self.conv2(x, edge_index, return_attention_weights=True)
        self.att_weights = att
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

class GraphConvNet(nn.Module):
    def __init__(self, in_c, hid_c, out_c):
        super().__init__()
        self.conv1 = GraphConv(in_c, hid_c)
        self.conv2 = GraphConv(hid_c, out_c)
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

MODELS = {
    "GCN": GCN,
    "GAT": GAT,
    "GraphSAGE": GraphSAGE,
    "GraphConv": GraphConvNet,
    "GIN": GINNet,
}
def load_data():
    features = pd.read_csv("elliptic_txs_features.csv", header=None)
    labels = pd.read_csv("elliptic_txs_classes.csv")
    edges = pd.read_csv("elliptic_txs_edgelist.csv", header=None, names=["source", "target"])

    features = features.rename(columns={0: "tx_id", 1: "time_step"})
    labels = labels.rename(columns={"txId": "tx_id", "class": "label"})
    labels["label"] = labels["label"].replace({"unknown": -1, "1": 0, "2": 1}).astype(int)

    # Clean and merge
    features["tx_id"] = pd.to_numeric(features["tx_id"], errors="coerce")
    labels["tx_id"] = pd.to_numeric(labels["tx_id"], errors="coerce")
    edges["source"] = pd.to_numeric(edges["source"], errors="coerce")
    edges["target"] = pd.to_numeric(edges["target"], errors="coerce")

    full_data = features.merge(labels, on="tx_id", how="left")
    all_ids = pd.concat([features["tx_id"], edges["source"], edges["target"]]).dropna().unique()
    tx_id_map = {id_: i for i, id_ in enumerate(sorted(all_ids))}

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

    print(f"✅ Loaded: {x.shape[0]} nodes, {edge_index.shape[1]} edges")

    # --------- 3D Class Distribution Visualization ---------
    try:
        sample = known.sample(n=min(3000, len(known)), random_state=42)
        reducer = UMAP(n_components=3, random_state=42)
        umap_proj = reducer.fit_transform(sample.iloc[:, 2:167].values)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(umap_proj[:, 0], umap_proj[:, 1], umap_proj[:, 2],
                             c=sample["label"], cmap="coolwarm", alpha=0.6)
        ax.set_title("3D Class Distribution")
        plt.savefig("class_dist_3d.png")
        plt.close()
    except Exception as e:
        print(f"⚠️ Failed 3D class distribution plot: {e}")

    # --------- 3D Illicit Ratio Visualization (per time step) ---------
    try:
        time_grouped = known.groupby("time_step")["label"].value_counts().unstack().fillna(0)
        illicit_ratio = (time_grouped[1] / (time_grouped[0] + time_grouped[1])).fillna(0)
        steps = time_grouped.index.values
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(steps, illicit_ratio, zs=0, zdir='y', label='Illicit Ratio')
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Illicit Ratio")
        ax.set_zlabel("Z")
        ax.set_title("3D Illicit Ratio over Time")
        plt.savefig("figure_illicit_ratio_3d.png")
        plt.close()
    except Exception as e:
        print(f"⚠️ Failed 3D illicit ratio plot: {e}")

    return Data(x=x, edge_index=edge_index, y=y)
from umap import UMAP  # 상단에 import 추가
from mpl_toolkits.mplot3d import Axes3D

def train_and_evaluate(data, ModelClass, name):
    valid_mask = data.y >= 0
    idx = torch.where(valid_mask)[0]
    y_np = data.y[valid_mask].cpu().numpy()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    preds, labels, times, probs = [], [], [], []
    final_model = None

    for fold, (train_i, test_i) in enumerate(skf.split(idx, y_np)):
        print(f"🔁 Fold {fold+1}")
        train_idx, test_idx = idx[train_i], idx[test_i]
        model = ModelClass(data.num_features, 64, 2).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()

        for _ in range(50):
            model.train()
            out = model(data.x.to(device), data.edge_index.to(device))
            loss = loss_fn(out[train_idx], data.y[train_idx].to(device))
            loss.backward()
            opt.step()
            opt.zero_grad()

        model.eval()
        out = model(data.x.to(device), data.edge_index.to(device))
        pred = out[test_idx].argmax(dim=1).cpu().detach()
        prob = F.softmax(out[test_idx], dim=1).cpu().detach().numpy()

        preds.extend(pred)
        labels.extend(data.y[test_idx].cpu())
        times.extend(data.x[test_idx, 0].cpu())
        probs.extend(prob)
        final_model = model

    return preds, labels, times, probs, final_model

def save_results(name, y_true, y_pred, probs):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"📊 {name} → Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")
    with open(f"report_{name}.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nF1 Score: {f1:.4f}\n")

    df = pd.DataFrame({"true": y_true, "pred": y_pred, "prob_1": [p[1] for p in probs]})
    df.to_csv(f"results_{name}.csv", index=False)

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix ({name})")
    plt.savefig(f"confusion_matrix_{name}.png")
    plt.close()

    fpr, tpr, _ = roc_curve(y_true, [p[1] for p in probs])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.legend()
    plt.title(f"ROC Curve ({name})")
    plt.savefig(f"roc_curve_{name}.png")
    plt.close()

def plot_f1_over_time(y, pred, times, name):
    df = pd.DataFrame({"time": times, "y": y, "pred": pred})
    def safe_f1(g):
        mask = (g["y"] >= 0) & (g["pred"] >= 0)
        if mask.sum() < 2 or len(g["y"][mask].unique()) < 2:
            return None
        return f1_score(g["y"][mask], g["pred"][mask])
    try:
        f1s = df.groupby("time", group_keys=False).apply(safe_f1).dropna()
        if f1s.empty:
            print(f"⚠️ No valid F1 scores to plot for {name}")
            return
        f1s.plot(marker="o")
        plt.title(f"F1 over Time ({name})")
        plt.xlabel("Time Step")
        plt.ylabel("F1 Score")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"f1_over_time_{name}.png")
        plt.close()
    except Exception as e:
        print(f"⚠️ F1-over-time plotting failed for {name}: {e}")

def run_xai(data, model, name, num_nodes=3):
    print(f"🔍 Running Integrated Gradients for {num_nodes} nodes...")
    model.eval()
    model.cpu()
    edge_index = data.edge_index.cpu()
    y = data.y.cpu()
    x = data.x.cpu()

    nodes = torch.where(y >= 0)[0][:num_nodes]
    explanations = []

    for node_idx in nodes:
        try:
            subset, sub_edge_index, mapping, _ = k_hop_subgraph(node_idx.item(), 2, edge_index, relabel_nodes=True)
            x_sub = x[subset].clone().detach().requires_grad_(True)
            ig = IntegratedGradients(lambda inputs: model(inputs, sub_edge_index))
            attr = ig.attribute(x_sub, target=int(y[node_idx]), n_steps=25)
            topk = torch.topk(attr[mapping].abs(), k=10)
            top_features = topk.indices.tolist()
            print(f"🔥 {name} Node {int(node_idx)} Top IG Features: {top_features}")
            explanations.append({
                "model": name,
                "node": int(node_idx),
                "top_features": top_features
            })
        except Exception as e:
            print(f"⚠️ IG failed for node {int(node_idx)}: {e}")

    pd.DataFrame(explanations).to_csv(f"xai_ig_top_features_{name}.csv", index=False)

    if hasattr(model, 'att_weights') and model.att_weights is not None:
        try:
            attn = model.att_weights[1].detach().cpu().numpy()
            plt.hist(attn, bins=100)
            plt.title(f"GAT Attention ({name})")
            plt.savefig(f"gat_attention_{name}.png")
            plt.close()
        except Exception as e:
            print(f"⚠️ Failed to visualize GAT attention: {e}")
def visualize_class_distribution_3d(full_data, max_points=3000):
    print("📊 Creating 3D class distribution and illicit ratio figures...")
    labeled = full_data[full_data["label"] >= 0]
    sampled = labeled.sample(n=min(max_points, len(labeled)), random_state=42)

    features = sampled.iloc[:, 2:167].values
    labels = sampled["label"].values

    try:
        reducer = UMAP(n_components=3, random_state=42)
        embedding = reducer.fit_transform(features)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c=labels, cmap="coolwarm", alpha=0.7)
        ax.set_title("3D UMAP Projection of Labeled Transactions")import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GraphConv, GINConv
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from captum.attr import IntegratedGradients
from umap import UMAP
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# ------------------ GNN Model Definitions ------------------ #
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
        self.att_weights = None
        self.hidden = None
    def forward(self, x, edge_index):
        x, _ = self.conv1(x, edge_index, return_attention_weights=True)
        x = F.elu(x)
        self.hidden = x.detach().cpu()
        x, att = self.conv2(x, edge_index, return_attention_weights=True)
        self.att_weights = att
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

class GraphConvNet(nn.Module):
    def __init__(self, in_c, hid_c, out_c):
        super().__init__()
        self.conv1 = GraphConv(in_c, hid_c)
        self.conv2 = GraphConv(hid_c, out_c)
        self.hidden = None
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        self.hidden = x.detach().cpu()
        return self.conv2(x, edge_index)

class GINNet(nn.Module):
    def __init__(self, in_c, hid_c, out_c):
        super().__init__()
        nn1 = nn.Sequential(nn.Linear(in_c, hid_c), nn.ReLU(), nn.Linear(hid_c, hid_c))import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GraphConv, GINConv
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from captum.attr import IntegratedGradients
from umap import UMAP
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# ------------------ GNN Model Definitions ------------------ #
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
        self.att_weights = None
        self.hidden = None
    def forward(self, x, edge_index):
        x, _ = self.conv1(x, edge_index, return_attention_weights=True)
        x = F.elu(x)
        self.hidden = x.detach().cpu()
        x, att = self.conv2(x, edge_index, return_attention_weights=True)
        self.att_weights = att
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

class GraphConvNet(nn.Module):
    def __init__(self, in_c, hid_c, out_c):
        super().__init__()
        self.conv1 = GraphConv(in_c, hid_c)
        self.conv2 = GraphConv(hid_c, out_c)
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

MODELS = {
    "GCN": GCN,
    "GAT": GAT,
    "GraphSAGE": GraphSAGE,
    "GraphConv": GraphConvNet,
    "GIN": GINNet,
}
def load_data():
    features = pd.read_csv("elliptic_txs_features.csv", header=None)
    labels = pd.read_csv("elliptic_txs_classes.csv")
    edges = pd.read_csv("elliptic_txs_edgelist.csv", header=None, names=["source", "target"])

    features = features.rename(columns={0: "tx_id", 1: "time_step"})
    labels = labels.rename(columns={"txId": "tx_id", "class": "label"})
    labels["label"] = labels["label"].replace({"unknown": -1, "1": 0, "2": 1}).astype(int)

    # Clean and merge
    features["tx_id"] = pd.to_numeric(features["tx_id"], errors="coerce")
    labels["tx_id"] = pd.to_numeric(labels["tx_id"], errors="coerce")
    edges["source"] = pd.to_numeric(edges["source"], errors="coerce")
    edges["target"] = pd.to_numeric(edges["target"], errors="coerce")

    full_data = features.merge(labels, on="tx_id", how="left")
    all_ids = pd.concat([features["tx_id"], edges["source"], edges["target"]]).dropna().unique()
    tx_id_map = {id_: i for i, id_ in enumerate(sorted(all_ids))}

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

    print(f"✅ Loaded: {x.shape[0]} nodes, {edge_index.shape[1]} edges")

    # --------- 3D Class Distribution Visualization ---------
    try:
        sample = known.sample(n=min(3000, len(known)), random_state=42)
        reducer = UMAP(n_components=3, random_state=42)
        umap_proj = reducer.fit_transform(sample.iloc[:, 2:167].values)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(umap_proj[:, 0], umap_proj[:, 1], umap_proj[:, 2],
                             c=sample["label"], cmap="coolwarm", alpha=0.6)
        ax.set_title("3D Class Distribution")
        plt.savefig("class_dist_3d.png")
        plt.close()
    except Exception as e:
        print(f"⚠️ Failed 3D class distribution plot: {e}")

    # --------- 3D Illicit Ratio Visualization (per time step) ---------
    try:
        time_grouped = known.groupby("time_step")["label"].value_counts().unstack().fillna(0)
        illicit_ratio = (time_grouped[1] / (time_grouped[0] + time_grouped[1])).fillna(0)
        steps = time_grouped.index.values
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(steps, illicit_ratio, zs=0, zdir='y', label='Illicit Ratio')
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Illicit Ratio")
        ax.set_zlabel("Z")
        ax.set_title("3D Illicit Ratio over Time")
        plt.savefig("figure_illicit_ratio_3d.png")
        plt.close()
    except Exception as e:
        print(f"⚠️ Failed 3D illicit ratio plot: {e}")

    return Data(x=x, edge_index=edge_index, y=y)
from umap import UMAP  # 상단에 import 추가
from mpl_toolkits.mplot3d import Axes3D

def train_and_evaluate(data, ModelClass, name):
    valid_mask = data.y >= 0
    idx = torch.where(valid_mask)[0]
    y_np = data.y[valid_mask].cpu().numpy()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    preds, labels, times, probs = [], [], [], []
    final_model = None

    for fold, (train_i, test_i) in enumerate(skf.split(idx, y_np)):
        print(f"🔁 Fold {fold+1}")
        train_idx, test_idx = idx[train_i], idx[test_i]
        model = ModelClass(data.num_features, 64, 2).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()

        for _ in range(50):
            model.train()
            out = model(data.x.to(device), data.edge_index.to(device))
            loss = loss_fn(out[train_idx], data.y[train_idx].to(device))
            loss.backward()
            opt.step()
            opt.zero_grad()

        model.eval()
        out = model(data.x.to(device), data.edge_index.to(device))
        pred = out[test_idx].argmax(dim=1).cpu().detach()
        prob = F.softmax(out[test_idx], dim=1).cpu().detach().numpy()

        preds.extend(pred)
        labels.extend(data.y[test_idx].cpu())
        times.extend(data.x[test_idx, 0].cpu())
        probs.extend(prob)
        final_model = model

    return preds, labels, times, probs, final_model

def save_results(name, y_true, y_pred, probs):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"📊 {name} → Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")
    with open(f"report_{name}.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nF1 Score: {f1:.4f}\n")

    df = pd.DataFrame({"true": y_true, "pred": y_pred, "prob_1": [p[1] for p in probs]})
    df.to_csv(f"results_{name}.csv", index=False)

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix ({name})")
    plt.savefig(f"confusion_matrix_{name}.png")
    plt.close()

    fpr, tpr, _ = roc_curve(y_true, [p[1] for p in probs])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.legend()
    plt.title(f"ROC Curve ({name})")
    plt.savefig(f"roc_curve_{name}.png")
    plt.close()

def plot_f1_over_time(y, pred, times, name):
    df = pd.DataFrame({"time": times, "y": y, "pred": pred})
    def safe_f1(g):
        mask = (g["y"] >= 0) & (g["pred"] >= 0)
        if mask.sum() < 2 or len(g["y"][mask].unique()) < 2:
            return None
        return f1_score(g["y"][mask], g["pred"][mask])
    try:
        f1s = df.groupby("time", group_keys=False).apply(safe_f1).dropna()
        if f1s.empty:
            print(f"⚠️ No valid F1 scores to plot for {name}")
            return
        f1s.plot(marker="o")
        plt.title(f"F1 over Time ({name})")
        plt.xlabel("Time Step")
        plt.ylabel("F1 Score")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"f1_over_time_{name}.png")
        plt.close()
    except Exception as e:
        print(f"⚠️ F1-over-time plotting failed for {name}: {e}")

def run_xai(data, model, name, num_nodes=3):
    print(f"🔍 Running Integrated Gradients for {num_nodes} nodes...")
    model.eval()
    model.cpu()
    edge_index = data.edge_index.cpu()
    y = data.y.cpu()
    x = data.x.cpu()

    nodes = torch.where(y >= 0)[0][:num_nodes]
    explanations = []

    for node_idx in nodes:
        try:
            subset, sub_edge_index, mapping, _ = k_hop_subgraph(node_idx.item(), 2, edge_index, relabel_nodes=True)
            x_sub = x[subset].clone().detach().requires_grad_(True)
            ig = IntegratedGradients(lambda inputs: model(inputs, sub_edge_index))
            attr = ig.attribute(x_sub, target=int(y[node_idx]), n_steps=25)
            topk = torch.topk(attr[mapping].abs(), k=10)
            top_features = topk.indices.tolist()
            print(f"🔥 {name} Node {int(node_idx)} Top IG Features: {top_features}")
            explanations.append({
                "model": name,
                "node": int(node_idx),
                "top_features": top_features
            })
        except Exception as e:
            print(f"⚠️ IG failed for node {int(node_idx)}: {e}")

    pd.DataFrame(explanations).to_csv(f"xai_ig_top_features_{name}.csv", index=False)

    if hasattr(model, 'att_weights') and model.att_weights is not None:
        try:
            attn = model.att_weights[1].detach().cpu().numpy()
            plt.hist(attn, bins=100)
            plt.title(f"GAT Attention ({name})")
            plt.savefig(f"gat_attention_{name}.png")
            plt.close()
        except Exception as e:
            print(f"⚠️ Failed to visualize GAT attention: {e}")
def visualize_class_distribution_3d(full_data, max_points=3000):
    print("📊 Creating 3D class distribution and illicit ratio figures...")
    labeled = full_data[full_data["label"] >= 0]
    sampled = labeled.sample(n=min(max_points, len(labeled)), random_state=42)

    features = sampled.iloc[:, 2:167].values
    labels = sampled["label"].values

    try:
        reducer = UMAP(n_components=3, random_state=42)
        embedding = reducer.fit_transform(features)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c=labels, cmap="coolwarm", alpha=0.7)
        ax.set_title("3D UMAP Projection of Labeled Transactions")
        fig.colorbar(scatter)
        plt.savefig("figure_3D_umap_projection.png")
        plt.close()
    except Exception as e:
        print(f"⚠️ UMAP projection failed: {e}")

    try:
        class_ratio = labeled.groupby("time_step")["label"].value_counts(normalize=True).unstack().fillna(0)
        class_ratio.columns = ["licit", "illicit"]
        fig = plt.figure(figsize=(12, 6))import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GraphConv, GINConv
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from captum.attr import IntegratedGradients
from umap import UMAP
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# ------------------ GNN Model Definitions ------------------ #
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
        self.att_weights = None
        self.hidden = None
    def forward(self, x, edge_index):
        x, _ = self.conv1(x, edge_index, return_attention_weights=True)
        x = F.elu(x)
        self.hidden = x.detach().cpu()
        x, att = self.conv2(x, edge_index, return_attention_weights=True)
        self.att_weights = att
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

class GraphConvNet(nn.Module):
    def __init__(self, in_c, hid_c, out_c):
        super().__init__()
        self.conv1 = GraphConv(in_c, hid_c)
        self.conv2 = GraphConv(hid_c, out_c)
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

MODELS = {
    "GCN": GCN,
    "GAT": GAT,
    "GraphSAGE": GraphSAGE,
    "GraphConv": GraphConvNet,
    "GIN": GINNet,
}
def load_data():
    features = pd.read_csv("elliptic_txs_features.csv", header=None)
    labels = pd.read_csv("elliptic_txs_classes.csv")
    edges = pd.read_csv("elliptic_txs_edgelist.csv", header=None, names=["source", "target"])

    features = features.rename(columns={0: "tx_id", 1: "time_step"})
    labels = labels.rename(columns={"txId": "tx_id", "class": "label"})
    labels["label"] = labels["label"].replace({"unknown": -1, "1": 0, "2": 1}).astype(int)

    # Clean and merge
    features["tx_id"] = pd.to_numeric(features["tx_id"], errors="coerce")
    labels["tx_id"] = pd.to_numeric(labels["tx_id"], errors="coerce")
    edges["source"] = pd.to_numeric(edges["source"], errors="coerce")
    edges["target"] = pd.to_numeric(edges["target"], errors="coerce")

    full_data = features.merge(labels, on="tx_id", how="left")
    all_ids = pd.concat([features["tx_id"], edges["source"], edges["target"]]).dropna().unique()
    tx_id_map = {id_: i for i, id_ in enumerate(sorted(all_ids))}

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

    print(f"✅ Loaded: {x.shape[0]} nodes, {edge_index.shape[1]} edges")

    # --------- 3D Class Distribution Visualization ---------
    try:
        sample = known.sample(n=min(3000, len(known)), random_state=42)
        reducer = UMAP(n_components=3, random_state=42)
        umap_proj = reducer.fit_transform(sample.iloc[:, 2:167].values)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(umap_proj[:, 0], umap_proj[:, 1], umap_proj[:, 2],
                             c=sample["label"], cmap="coolwarm", alpha=0.6)
        ax.set_title("3D Class Distribution")
        plt.savefig("class_dist_3d.png")
        plt.close()
    except Exception as e:
        print(f"⚠️ Failed 3D class distribution plot: {e}")

    # --------- 3D Illicit Ratio Visualization (per time step) ---------
    try:
        time_grouped = known.groupby("time_step")["label"].value_counts().unstack().fillna(0)
        illicit_ratio = (time_grouped[1] / (time_grouped[0] + time_grouped[1])).fillna(0)
        steps = time_grouped.index.values
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(steps, illicit_ratio, zs=0, zdir='y', label='Illicit Ratio')
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Illicit Ratio")
        ax.set_zlabel("Z")
        ax.set_title("3D Illicit Ratio over Time")
        plt.savefig("figure_illicit_ratio_3d.png")
        plt.close()
    except Exception as e:
        print(f"⚠️ Failed 3D illicit ratio plot: {e}")

    return Data(x=x, edge_index=edge_index, y=y)
from umap import UMAP  # 상단에 import 추가
from mpl_toolkits.mplot3d import Axes3D

def train_and_evaluate(data, ModelClass, name):
    valid_mask = data.y >= 0
    idx = torch.where(valid_mask)[0]
    y_np = data.y[valid_mask].cpu().numpy()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    preds, labels, times, probs = [], [], [], []
    final_model = None

    for fold, (train_i, test_i) in enumerate(skf.split(idx, y_np)):
        print(f"🔁 Fold {fold+1}")
        train_idx, test_idx = idx[train_i], idx[test_i]
        model = ModelClass(data.num_features, 64, 2).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()

        for _ in range(50):
            model.train()
            out = model(data.x.to(device), data.edge_index.to(device))
            loss = loss_fn(out[train_idx], data.y[train_idx].to(device))
            loss.backward()
            opt.step()
            opt.zero_grad()

        model.eval()
        out = model(data.x.to(device), data.edge_index.to(device))
        pred = out[test_idx].argmax(dim=1).cpu().detach()
        prob = F.softmax(out[test_idx], dim=1).cpu().detach().numpy()

        preds.extend(pred)
        labels.extend(data.y[test_idx].cpu())
        times.extend(data.x[test_idx, 0].cpu())
        probs.extend(prob)
        final_model = model

    return preds, labels, times, probs, final_model

def save_results(name, y_true, y_pred, probs):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"📊 {name} → Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")
    with open(f"report_{name}.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nF1 Score: {f1:.4f}\n")

    df = pd.DataFrame({"true": y_true, "pred": y_pred, "prob_1": [p[1] for p in probs]})
    df.to_csv(f"results_{name}.csv", index=False)

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix ({name})")
    plt.savefig(f"confusion_matrix_{name}.png")
    plt.close()

    fpr, tpr, _ = roc_curve(y_true, [p[1] for p in probs])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.legend()
    plt.title(f"ROC Curve ({name})")
    plt.savefig(f"roc_curve_{name}.png")
    plt.close()

def plot_f1_over_time(y, pred, times, name):
    df = pd.DataFrame({"time": times, "y": y, "pred": pred})
    def safe_f1(g):
        mask = (g["y"] >= 0) & (g["pred"] >= 0)
        if mask.sum() < 2 or len(g["y"][mask].unique()) < 2:
            return None
        return f1_score(g["y"][mask], g["pred"][mask])
    try:
        f1s = df.groupby("time", group_keys=False).apply(safe_f1).dropna()
        if f1s.empty:
            print(f"⚠️ No valid F1 scores to plot for {name}")
            return
        f1s.plot(marker="o")
        plt.title(f"F1 over Time ({name})")
        plt.xlabel("Time Step")
        plt.ylabel("F1 Score")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"f1_over_time_{name}.png")
        plt.close()
    except Exception as e:
        print(f"⚠️ F1-over-time plotting failed for {name}: {e}")

def run_xai(data, model, name, num_nodes=3):
    print(f"🔍 Running Integrated Gradients for {num_nodes} nodes...")
    model.eval()
    model.cpu()
    edge_index = data.edge_index.cpu()
    y = data.y.cpu()
    x = data.x.cpu()

    nodes = torch.where(y >= 0)[0][:num_nodes]
    explanations = []

    for node_idx in nodes:
        try:
            subset, sub_edge_index, mapping, _ = k_hop_subgraph(node_idx.item(), 2, edge_index, relabel_nodes=True)
            x_sub = x[subset].clone().detach().requires_grad_(True)
            ig = IntegratedGradients(lambda inputs: model(inputs, sub_edge_index))
            attr = ig.attribute(x_sub, target=int(y[node_idx]), n_steps=25)
            topk = torch.topk(attr[mapping].abs(), k=10)
            top_features = topk.indices.tolist()
            print(f"🔥 {name} Node {int(node_idx)} Top IG Features: {top_features}")
            explanations.append({
                "model": name,
                "node": int(node_idx),
                "top_features": top_features
            })
        except Exception as e:
            print(f"⚠️ IG failed for node {int(node_idx)}: {e}")

    pd.DataFrame(explanations).to_csv(f"xai_ig_top_features_{name}.csv", index=False)

    if hasattr(model, 'att_weights') and model.att_weights is not None:
        try:
            attn = model.att_weights[1].detach().cpu().numpy()
            plt.hist(attn, bins=100)
            plt.title(f"GAT Attention ({name})")
            plt.savefig(f"gat_attention_{name}.png")
            plt.close()
        except Exception as e:
            print(f"⚠️ Failed to visualize GAT attention: {e}")
def visualize_class_distribution_3d(full_data, max_points=3000):
    print("📊 Creating 3D class distribution and illicit ratio figures...")
    labeled = full_data[full_data["label"] >= 0]
    sampled = labeled.sample(n=min(max_points, len(labeled)), random_state=42)

    features = sampled.iloc[:, 2:167].values
    labels = sampled["label"].values

    try:
        reducer = UMAP(n_components=3, random_state=42)
        embedding = reducer.fit_transform(features)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c=labels, cmap="coolwarm", alpha=0.7)
        ax.set_title("3D UMAP Projection of Labeled Transactions")
        fig.colorbar(scatter)
        plt.savefig("figure_3D_umap_projection.png")
        plt.close()
    except Exception as e:
        print(f"⚠️ UMAP projection failed: {e}")

    try:
        class_ratio = labeled.groupby("time_step")["label"].value_counts(normalize=True).unstack().fillna(0)
        class_ratio.columns = ["licit", "illicit"]
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111, projection="3d")
        ts = class_ratio.index.values
        ax.bar(ts, class_ratio["licit"], zs=0, zdir='y', label="Licit")
        ax.bar(ts, class_ratio["illicit"], zs=1, zdir='y', label="Illicit")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Class")
        ax.set_zlabel("Ratio")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Licit", "Illicit"])
        plt.legend()
        plt.title("3D Class Distribution Over Time")
        plt.savefig("figure_illicit_ratio_3D.png")
        plt.close()
    except Exception as e:
        print(f"⚠️ 3D class dist plot failed: {e}")

# ------------------ Main ------------------ #

if __name__ == "__main__":
    data = load_data()

    features = pd.read_csv("elliptic_txs_features.csv", header=None)
    features = features.rename(columns={0: "tx_id", 1: "time_step"})
    labels = pd.read_csv("elliptic_txs_classes.csv")
    labels = labels.rename(columns={"txId": "tx_id", "class": "label"})
    labels["label"] = labels["label"].replace({"unknown": -1, "1": 0, "2": 1}).astype(int)
    full_data = features.merge(labels, on="tx_id", how="left")

    visualize_class_distribution_3d(full_data)

    for name, Model in MODELS.items():
        print(f"\n🚀 Running {name}...")
        pred, label, time, prob, model = train_and_evaluate(data, Model, name)
        save_results(name, label, pred, prob)
        plot_f1_over_time(label, pred, time, name)
        run_xai(data, model, name)

        ax = fig.add_subplot(111, projection="3d")
        ts = class_ratio.index.values
        ax.bar(ts, class_ratio["licit"], zs=0, zdir='y', label="Licit")
        ax.bar(ts, class_ratio["illicit"], zs=1, zdir='y', label="Illicit")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Class")
        ax.set_zlabel("Ratio")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Licit", "Illicit"])
        plt.legend()
        plt.title("3D Class Distribution Over Time")
        plt.savefig("figure_illicit_ratio_3D.png")
        plt.close()
    except Exception as e:
        print(f"⚠️ 3D class dist plot failed: {e}")

# ------------------ Main ------------------ #

if __name__ == "__main__":
    data = load_data()

    features = pd.read_csv("elliptic_txs_features.csv", header=None)
    features = features.rename(columns={0: "tx_id", 1: "time_step"})
    labels = pd.read_csv("elliptic_txs_classes.csv")
    labels = labels.rename(columns={"txId": "tx_id", "class": "label"})
    labels["label"] = labels["label"].replace({"unknown": -1, "1": 0, "2": 1}).astype(int)
    full_data = features.merge(labels, on="tx_id", how="left")

    visualize_class_distribution_3d(full_data)

    for name, Model in MODELS.items():
        print(f"\n🚀 Running {name}...")
        pred, label, time, prob, model = train_and_evaluate(data, Model, name)
        save_results(name, label, pred, prob)
        plot_f1_over_time(label, pred, time, name)
        run_xai(data, model, name)

        nn2 = nn.Sequential(nn.Linear(hid_c, hid_c), nn.ReLU(), nn.Linear(hid_c, out_c))
        self.conv1 = GINConv(nn1)
        self.conv2 = GINConv(nn2)
        self.hidden = None
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        self.hidden = x.detach().cpu()
        return self.conv2(x, edge_index)

MODELS = {
    "GCN": GCN,
    "GAT": GAT,
    "GraphSAGE": GraphSAGE,
    "GraphConv": GraphConvNet,
    "GIN": GINNet,
}
def load_data():
    features = pd.read_csv("elliptic_txs_features.csv", header=None)
    labels = pd.read_csv("elliptic_txs_classes.csv")
    edges = pd.read_csv("elliptic_txs_edgelist.csv", header=None, names=["source", "target"])

    features = features.rename(columns={0: "tx_id", 1: "time_step"})
    labels = labels.rename(columns={"txId": "tx_id", "class": "label"})
    labels["label"] = labels["label"].replace({"unknown": -1, "1": 0, "2": 1}).astype(int)

    # Clean and merge
    features["tx_id"] = pd.to_numeric(features["tx_id"], errors="coerce")
    labels["tx_id"] = pd.to_numeric(labels["tx_id"], errors="coerce")
    edges["source"] = pd.to_numeric(edges["source"], errors="coerce")
    edges["target"] = pd.to_numeric(edges["target"], errors="coerce")

    full_data = features.merge(labels, on="tx_id", how="left")
    all_ids = pd.concat([features["tx_id"], edges["source"], edges["target"]]).dropna().unique()
    tx_id_map = {id_: i for i, id_ in enumerate(sorted(all_ids))}

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

    print(f"✅ Loaded: {x.shape[0]} nodes, {edge_index.shape[1]} edges")

    # --------- 3D Class Distribution Visualization ---------
    try:
        sample = known.sample(n=min(3000, len(known)), random_state=42)
        reducer = UMAP(n_components=3, random_state=42)
        umap_proj = reducer.fit_transform(sample.iloc[:, 2:167].values)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(umap_proj[:, 0], umap_proj[:, 1], umap_proj[:, 2],
                             c=sample["label"], cmap="coolwarm", alpha=0.6)
        ax.set_title("3D Class Distribution")
        plt.savefig("class_dist_3d.png")
        plt.close()
    except Exception as e:
        print(f"⚠️ Failed 3D class distribution plot: {e}")

    # --------- 3D Illicit Ratio Visualization (per time step) ---------
    try:
        time_grouped = known.groupby("time_step")["label"].value_counts().unstack().fillna(0)
        illicit_ratio = (time_grouped[1] / (time_grouped[0] + time_grouped[1])).fillna(0)
        steps = time_grouped.index.values
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(steps, illicit_ratio, zs=0, zdir='y', label='Illicit Ratio')
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Illicit Ratio")
        ax.set_zlabel("Z")
        ax.set_title("3D Illicit Ratio over Time")
        plt.savefig("figure_illicit_ratio_3d.png")
        plt.close()
    except Exception as e:
        print(f"⚠️ Failed 3D illicit ratio plot: {e}")

    return Data(x=x, edge_index=edge_index, y=y)
from umap import UMAP  # 상단에 import 추가
from mpl_toolkits.mplot3d import Axes3D

def train_and_evaluate(data, ModelClass, name):
    valid_mask = data.y >= 0
    idx = torch.where(valid_mask)[0]
    y_np = data.y[valid_mask].cpu().numpy()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    preds, labels, times, probs = [], [], [], []
    final_model = None

    for fold, (train_i, test_i) in enumerate(skf.split(idx, y_np)):
        print(f"🔁 Fold {fold+1}")
        train_idx, test_idx = idx[train_i], idx[test_i]
        model = ModelClass(data.num_features, 64, 2).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()

        for _ in range(50):
            model.train()
            out = model(data.x.to(device), data.edge_index.to(device))
            loss = loss_fn(out[train_idx], data.y[train_idx].to(device))
            loss.backward()
            opt.step()
            opt.zero_grad()

        model.eval()
        out = model(data.x.to(device), data.edge_index.to(device))
        pred = out[test_idx].argmax(dim=1).cpu().detach()
        prob = F.softmax(out[test_idx], dim=1).cpu().detach().numpy()

        preds.extend(pred)
        labels.extend(data.y[test_idx].cpu())
        times.extend(data.x[test_idx, 0].cpu())
        probs.extend(prob)
        final_model = model

    return preds, labels, times, probs, final_model

def save_results(name, y_true, y_pred, probs):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"📊 {name} → Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")
    with open(f"report_{name}.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nF1 Score: {f1:.4f}\n")

    df = pd.DataFrame({"true": y_true, "pred": y_pred, "prob_1": [p[1] for p in probs]})
    df.to_csv(f"results_{name}.csv", index=False)

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix ({name})")
    plt.savefig(f"confusion_matrix_{name}.png")
    plt.close()

    fpr, tpr, _ = roc_curve(y_true, [p[1] for p in probs])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.legend()
    plt.title(f"ROC Curve ({name})")
    plt.savefig(f"roc_curve_{name}.png")
    plt.close()

def plot_f1_over_time(y, pred, times, name):
    df = pd.DataFrame({"time": times, "y": y, "pred": pred})
    def safe_f1(g):
        mask = (g["y"] >= 0) & (g["pred"] >= 0)
        if mask.sum() < 2 or len(g["y"][mask].unique()) < 2:
            return None
        return f1_score(g["y"][mask], g["pred"][mask])
    try:
        f1s = df.groupby("time", group_keys=False).apply(safe_f1).dropna()
        if f1s.empty:
            print(f"⚠️ No valid F1 scores to plot for {name}")
            return
        f1s.plot(marker="o")
        plt.title(f"F1 over Time ({name})")
        plt.xlabel("Time Step")
        plt.ylabel("F1 Score")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"f1_over_time_{name}.png")
        plt.close()
    except Exception as e:
        print(f"⚠️ F1-over-time plotting failed for {name}: {e}")

def run_xai(data, model, name, num_nodes=3):
    print(f"🔍 Running Integrated Gradients for {num_nodes} nodes...")
    model.eval()
    model.cpu()
    edge_index = data.edge_index.cpu()
    y = data.y.cpu()
    x = data.x.cpu()

    nodes = torch.where(y >= 0)[0][:num_nodes]
    explanations = []

    for node_idx in nodes:
        try:
            subset, sub_edge_index, mapping, _ = k_hop_subgraph(node_idx.item(), 2, edge_index, relabel_nodes=True)
            x_sub = x[subset].clone().detach().requires_grad_(True)
            ig = IntegratedGradients(lambda inputs: model(inputs, sub_edge_index))
            attr = ig.attribute(x_sub, target=int(y[node_idx]), n_steps=25)
            topk = torch.topk(attr[mapping].abs(), k=10)
            top_features = topk.indices.tolist()
            print(f"🔥 {name} Node {int(node_idx)} Top IG Features: {top_features}")
            explanations.append({
                "model": name,
                "node": int(node_idx),
                "top_features": top_features
            })
        except Exception as e:
            print(f"⚠️ IG failed for node {int(node_idx)}: {e}")

    pd.DataFrame(explanations).to_csv(f"xai_ig_top_features_{name}.csv", index=False)

    if hasattr(model, 'att_weights') and model.att_weights is not None:
        try:
            attn = model.att_weights[1].detach().cpu().numpy()
            plt.hist(attn, bins=100)
            plt.title(f"GAT Attention ({name})")
            plt.savefig(f"gat_attention_{name}.png")
            plt.close()
        except Exception as e:
            print(f"⚠️ Failed to visualize GAT attention: {e}")
def visualize_class_distribution_3d(full_data, max_points=3000):
    print("📊 Creating 3D class distribution and illicit ratio figures...")
    labeled = full_data[full_data["label"] >= 0]
    sampled = labeled.sample(n=min(max_points, len(labeled)), random_state=42)

    features = sampled.iloc[:, 2:167].values
    labels = sampled["label"].values

    try:
        reducer = UMAP(n_components=3, random_state=42)
        embedding = reducer.fit_transform(features)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c=labels, cmap="coolwarm", alpha=0.7)
        ax.set_title("3D UMAP Projection of Labeled Transactions")
        fig.colorbar(scatter)
        plt.savefig("figure_3D_umap_projection.png")
        plt.close()
    except Exception as e:
        print(f"⚠️ UMAP projection failed: {e}")

    try:
        class_ratio = labeled.groupby("time_step")["label"].value_counts(normalize=True).unstack().fillna(0)
        class_ratio.columns = ["licit", "illicit"]
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111, projection="3d")
        ts = class_ratio.index.values
        ax.bar(ts, class_ratio["licit"], zs=0, zdir='y', label="Licit")
        ax.bar(ts, class_ratio["illicit"], zs=1, zdir='y', label="Illicit")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Class")
        ax.set_zlabel("Ratio")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Licit", "Illicit"])
        plt.legend()
        plt.title("3D Class Distribution Over Time")
        plt.savefig("figure_illicit_ratio_3D.png")
        plt.close()
    except Exception as e:
        print(f"⚠️ 3D class dist plot failed: {e}")

# ------------------ Main ------------------ #

if __name__ == "__main__":
    data = load_data()

    features = pd.read_csv("elliptic_txs_features.csv", header=None)
    features = features.rename(columns={0: "tx_id", 1: "time_step"})
    labels = pd.read_csv("elliptic_txs_classes.csv")
    labels = labels.rename(columns={"txId": "tx_id", "class": "label"})
    labels["label"] = labels["label"].replace({"unknown": -1, "1": 0, "2": 1}).astype(int)
    full_data = features.merge(labels, on="tx_id", how="left")

    visualize_class_distribution_3d(full_data)

    for name, Model in MODELS.items():
        print(f"\n🚀 Running {name}...")
        pred, label, time, prob, model = train_and_evaluate(data, Model, name)
        save_results(name, label, pred, prob)
        plot_f1_over_time(label, pred, time, name)
        run_xai(data, model, name)

        fig.colorbar(scatter)
        plt.savefig("figure_3D_umap_projection.png")
        plt.close()
    except Exception as e:
        print(f"⚠️ UMAP projection failed: {e}")

    try:
        class_ratio = labeled.groupby("time_step")["label"].value_counts(normalize=True).unstack().fillna(0)
        class_ratio.columns = ["licit", "illicit"]
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111, projection="3d")
        ts = class_ratio.index.values
        ax.bar(ts, class_ratio["licit"], zs=0, zdir='y', label="Licit")
        ax.bar(ts, class_ratio["illicit"], zs=1, zdir='y', label="Illicit")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Class")
        ax.set_zlabel("Ratio")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Licit", "Illicit"])
        plt.legend()
        plt.title("3D Class Distribution Over Time")
        plt.savefig("figure_illicit_ratio_3D.png")
        plt.close()
    except Exception as e:
        print(f"⚠️ 3D class dist plot failed: {e}")

# ------------------ Main ------------------ #

if __name__ == "__main__":
    data = load_data()

    features = pd.read_csv("elliptic_txs_features.csv", header=None)
    features = features.rename(columns={0: "tx_id", 1: "time_step"})
    labels = pd.read_csv("elliptic_txs_classes.csv")
    labels = labels.rename(columns={"txId": "tx_id", "class": "label"})
    labels["label"] = labels["label"].replace({"unknown": -1, "1": 0, "2": 1}).astype(int)
    full_data = features.merge(labels, on="tx_id", how="left")

    visualize_class_distribution_3d(full_data)

    for name, Model in MODELS.items():
        print(f"\n🚀 Running {name}...")
        pred, label, time, prob, model = train_and_evaluate(data, Model, name)
        save_results(name, label, pred, prob)
        plot_f1_over_time(label, pred, time, name)
        run_xai(data, model, name)

(chatoauth) root@testbed:~/tracer/dataset# cat ensemble_hybrid.py 
# ensemble_final.py
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder, label_binarize
from captum.attr import IntegratedGradients
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap import UMAP

# Ensure figure output directory
os.makedirs("figures", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------- Data Loading ---------------------- #
def load_data():
    tx_features = pd.read_csv("txs_features.csv", header=None, low_memory=False)
    tx_classes = pd.read_csv("txs_classes.csv")
    edges = pd.read_csv("txs_edgelist.csv").map(str)

    tx_classes.columns = ["tx_id", "label"]
    tx_classes["time_step"] = -1
    tx_classes["tx_id"] = tx_classes["tx_id"].astype(str)
    tx_features.iloc[:, 0] = tx_features.iloc[:, 0].astype(str)

    full_data = pd.merge(tx_features, tx_classes, left_on=0, right_on="tx_id", how="left")
    full_data = full_data.drop(columns=["tx_id"])
    full_data = full_data.replace("?", np.nan).dropna()
    feature_values = full_data.iloc[:, 1:-2].astype(np.float32).values
    x = torch.tensor(feature_values, dtype=torch.float)

    y = full_data["label"].fillna(-1).astype(int).values
    time = torch.tensor(full_data["time_step"].fillna(-1).astype(int).values, dtype=torch.long)

    valid_indices = y >= 0
    le = LabelEncoder()
    y[valid_indices] = le.fit_transform(y[valid_indices])
    y = torch.tensor(y, dtype=torch.long)

    ids = full_data.iloc[:, 0].astype(str).values
    id_map = {id_: i for i, id_ in enumerate(ids)}
    edge_index = edges.values
    edge_index = np.array([[id_map.get(src, -1), id_map.get(dst, -1)] for src, dst in edge_index])
    edge_index = edge_index[(edge_index[:, 0] != -1) & (edge_index[:, 1] != -1)]
    edge_index = torch.tensor(edge_index, dtype=torch.long).T

    data = Data(x=x, y=y, edge_index=edge_index)
    data.time = time
    return data, full_data, le

# ---------------------- GNN Model ---------------------- #
class GNNModel(nn.Module):
    def __init__(self, in_channels, out_channels, model_type="GCN"):
        super().__init__()
        if model_type == "GCN":
            self.conv1 = GCNConv(in_channels, 64)
            self.conv2 = GCNConv(64, out_channels)
        elif model_type == "GAT":
            self.conv1 = GATConv(in_channels, 64, heads=2, concat=False)
            self.conv2 = GATConv(64, out_channels, heads=1, concat=False)
        elif model_type == "SAGE":
            self.conv1 = SAGEConv(in_channels, 64)
            self.conv2 = SAGEConv(64, out_channels)
        elif model_type == "GIN":
            self.conv1 = GINConv(nn.Sequential(nn.Linear(in_channels, 64), nn.ReLU(), nn.Linear(64, 64)))
            self.conv2 = GINConv(nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, out_channels)))
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

    def get_embeddings(self, x, edge_index):
        return F.relu(self.conv1(x, edge_index)).detach()

class HybridGCNGAT(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.gcn = GCNConv(in_channels, 64)
        self.gat = GATConv(64, 64, heads=1, concat=True)
        self.fc = nn.Linear(64, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.gcn(x, edge_index))
        self.embeddings = F.relu(self.gat(x, edge_index))
        return self.fc(self.embeddings)

    def get_embeddings(self, x, edge_index):
        x = F.relu(self.gcn(x, edge_index))
        return F.relu(self.gat(x, edge_index)).detach()


# ---------------------- Evaluation & Visualization ---------------------- #
def eval_model(pred, true):
    return (
        accuracy_score(true, pred),
        precision_score(true, pred, average="macro", zero_division=0),
        recall_score(true, pred, average="macro", zero_division=0),
        f1_score(true, pred, average="macro", zero_division=0),
    )

def eval_ensemble(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    return eval_model(pred, y_test)

def plot_confusion(y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    plt.title(f"Confusion Matrix - {name}")
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    plt.savefig(f"figures/confusion_{name}.png")
    plt.close()

def plot_roc(y_true, y_probs, name):
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    n_classes = y_probs.shape[1]
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    plt.figure()
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {i} (AUC={roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.title(f"Multiclass ROC Curve - {name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/roc_{name}.png")
    plt.close()

def plot_f1_time(times, y_true, y_pred, name):
    df = pd.DataFrame({"time": times, "y": y_true, "pred": y_pred})
    df = df[df["y"] >= 0]
    f1_by_time = df.groupby("time").apply(lambda g: f1_score(g["y"], g["pred"], average="macro", zero_division=0))
    f1_by_time.plot(marker="o")
    plt.title(f"F1 Score Over Time - {name}")
    plt.ylabel("F1 Score")
    plt.xlabel("Time Step")
    plt.grid()
    plt.savefig(f"figures/f1_time_{name}.png")
    plt.close()

def plot_embeddings(x, y, method, name):
    if method == "pca":
        reducer = PCA(n_components=2)
    elif method == "tsne":
        reducer = TSNE(n_components=2, random_state=42)
    else:
        reducer = UMAP(n_components=2, random_state=42)
    emb_2d = reducer.fit_transform(x)
    plt.figure()
    plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=y, cmap="coolwarm", s=2, alpha=0.7)
    plt.title(f"{method.upper()} Visualization - {name}")
    plt.savefig(f"figures/{method}_{name}.png")
    plt.close()

# ---------------------- Main Train Loop ---------------------- #
def train_and_evaluate(data, full_data, model_type):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    labeled_idx = (data.y >= 0).nonzero(as_tuple=True)[0]
    y_all = data.y[labeled_idx].cpu().numpy()
    gnn_scores, rf_scores, xgb_scores, lgbm_scores = [], [], [], []
    final_preds, final_labels, final_probs, final_times = [], [], [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(labeled_idx, y_all)):
        train_mask = labeled_idx[train_idx]
        test_mask = labeled_idx[test_idx]

        # ✅ Hybrid 모델 분기 처리
        if model_type == "Hybrid":
            model = HybridGCNGAT(data.num_features, data.y.max().item() + 1).to(device)
        else:
            model = GNNModel(data.num_features, data.y.max().item() + 1, model_type).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)

        for _ in range(60):
            model.train()
            out = model(data.x.to(device), data.edge_index.to(device))
            loss = F.cross_entropy(out[train_mask], data.y[train_mask].to(device))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        out = model(data.x.to(device), data.edge_index.to(device)).cpu()
        pred = out[test_mask].argmax(dim=1)
        gnn_scores.append(eval_model(pred, data.y[test_mask].cpu()))

        emb = model.get_embeddings(data.x.to(device), data.edge_index.to(device)).cpu().numpy()
        X_train, y_train = emb[train_mask], data.y[train_mask].cpu().numpy()
        X_test, y_test = emb[test_mask], data.y[test_mask].cpu().numpy()

        final_preds.extend(pred.numpy())
        final_labels.extend(y_test)
        final_probs.extend(F.softmax(out[test_mask], dim=1).detach().numpy())
        final_times.extend(data.time[test_mask].numpy())

        rf_scores.append(eval_ensemble(RandomForestClassifier(n_estimators=200, max_depth=10), X_train, y_train, X_test, y_test))
        xgb_scores.append(eval_ensemble(XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, use_label_encoder=False, eval_metric="mlogloss", verbosity=0), X_train, y_train, X_test, y_test))
        lgbm_scores.append(eval_ensemble(LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=31), X_train, y_train, X_test, y_test))

    name = model_type
    final_probs_arr = np.array(final_probs)
    plot_confusion(final_labels, final_preds, name)
    plot_roc(final_labels, final_probs_arr, name)
    plot_f1_time(final_times, final_labels, final_preds, name)
    plot_embeddings(data.x.numpy(), data.y.numpy(), "pca", name + "_raw")
    plot_embeddings(model.get_embeddings(data.x.to(device), data.edge_index.to(device)).cpu().numpy(), data.y.numpy(), "umap", name + "_hidden")

    return (
        np.mean(gnn_scores, axis=0),
        np.mean(rf_scores, axis=0),
        np.mean(xgb_scores, axis=0),
        np.mean(lgbm_scores, axis=0),
    )

# ---------------------- Entry ---------------------- #
if __name__ == "__main__":
    data, full_data, le = load_data()
    print(f"Loaded: {data.num_nodes} nodes | {data.num_edges} edges | {data.num_features} features")

    results = []
    for model_type in ["GCN", "GAT", "SAGE", "GIN", "Hybrid"]:
        print(f"\nRunning {model_type}...")
        gnn, rf, xgb, lgbm = train_and_evaluate(data, full_data, model_type)
        results.append({
            "Model": model_type,
            "GNN_Acc": gnn[0], "GNN_Prec": gnn[1], "GNN_Rec": gnn[2], "GNN_F1": gnn[3],
            "RF_Acc": rf[0], "RF_Prec": rf[1], "RF_Rec": rf[2], "RF_F1": rf[3],
            "XGB_Acc": xgb[0], "XGB_Prec": xgb[1], "XGB_Rec": xgb[2], "XGB_F1": xgb[3],
            "LGBM_Acc": lgbm[0], "LGBM_Prec": lgbm[1], "LGBM_Rec": lgbm[2], "LGBM_F1": lgbm[3],
        })

    df_results = pd.DataFrame(results)
    print("\nPerformance Comparison Table:")
    print(df_results.round(4))
    df_results.to_csv("figures/performance_summary.csv", index=False)

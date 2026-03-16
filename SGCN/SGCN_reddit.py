import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import warnings
import random
import os

warnings.filterwarnings('ignore')

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"All random seeds set to {seed}")
    print(f"CUDA available: {torch.cuda.is_available()}")

seed_everything(42)

def load_and_clean_data(file_path='soc-redditHyperlinks-body.tsv'):
    print(f"Loading Reddit Hyperlink data from {file_path}...")
    df = pd.read_csv(file_path, sep='\t')
    
    df = df.rename(columns={
        'SOURCE_SUBREDDIT': 'source',
        'TARGET_SUBREDDIT': 'target',
        'LINK_SENTIMENT': 'sign',
        'TIMESTAMP': 'timestamp'
    })
    
    df = df[['source', 'target', 'sign', 'timestamp']]
    
    df = df[df['sign'].isin([1, -1])]
    
    df['label'] = (df['sign'] + 1) // 2
    
    df.drop_duplicates(subset=['source', 'target'], keep='first', inplace=True)
    
    print("Data after cleaning:")
    print(df.head())
    print(f"Total edges: {len(df)}")
    print(f"Positive edges (+1): {len(df[df['sign'] == 1])}")
    print(f"Negative edges (-1): {len(df[df['sign'] == -1])}")
    print(f"Unique subreddits (nodes): {df['source'].nunique() + df['target'].nunique() - len(set(df['source']) & set(df['target']))}")
    
    return df

def perform_eda(df):
    print("\n" + "="*60)
    print("EXPLORATORY DATA ANALYSIS - Reddit Subreddit Hyperlinks")
    print("="*60)
    
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(x='sign', data=df, hue='sign', palette=['salmon', 'skyblue'], legend=False)
    plt.title('Distribution of Link Sentiment', fontsize=16, fontweight='bold')
    plt.xlabel('Sentiment', fontsize=12)
    plt.ylabel('Number of Hyperlinks', fontsize=12)
    plt.xticks(ticks=[0, 1], labels=['Negative (-1)', 'Positive (+1)'])
    
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                    (p.get_x() + p.get_width()/2., p.get_height()), 
                    ha='center', va='center', xytext=(0, 9), 
                    textcoords='offset points', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('reddit_sentiment_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    df['datetime'] = pd.to_datetime(df['timestamp'])
    plt.figure(figsize=(12, 5))
    df['datetime'].hist(bins=60, color='purple', alpha=0.8, edgecolor='white')
    plt.title('Temporal Distribution of Hyperlinks (2014–2017)', fontsize=16, fontweight='bold')
    plt.xlabel('Year')
    plt.ylabel('Number of Hyperlinks')
    plt.tight_layout()
    plt.savefig('reddit_time_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    out_deg = df['source'].value_counts()
    in_deg = df['target'].value_counts()
    total_deg = out_deg.add(in_deg, fill_value=0).sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    plt.loglog(range(1, len(total_deg)+1), total_deg.values, 
               marker='o', linestyle='none', markersize=4, alpha=0.7, color='teal')
    plt.title('Node Degree Distribution (Log-Log Scale)', fontsize=16, fontweight='bold')
    plt.xlabel('Rank')
    plt.ylabel('Degree')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('reddit_degree_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印统计信息
    pos_ratio = (df['sign'] == 1).mean()
    print(f"Positive links ratio: {pos_ratio:.4%}")
    print(f"Total unique subreddits: {len(set(df['source']) | set(df['target']))}")
    print(f"Time range: {df['datetime'].min().date()} → {df['datetime'].max().date()}")

def get_nodes_map(df):
    all_nodes = np.unique(np.concatenate([df['source'].values, df['target'].values]))
    node_map = {node: idx for idx, node in enumerate(all_nodes)}
    num_nodes = len(all_nodes)
    return node_map, num_nodes

def get_edge_indices(df, node_map):
    src = [node_map[s] for s in df['source']]
    tgt = [node_map[t] for t in df['target']]
    edge_index = torch.tensor([src, tgt], dtype=torch.long)
    edge_label = torch.tensor(df['label'].values, dtype=torch.float)
    return edge_index, edge_label

def sparse_propagate(x, edge_index, num_nodes):
    row, col = edge_index
    deg = torch.bincount(row, minlength=num_nodes).float()
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    msg = norm.view(-1, 1) * x[col]
    out = torch.zeros_like(x)
    out.scatter_add_(0, row.view(-1, 1).expand_as(msg), msg)
    return out

class SignedGraphConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear_pos = nn.Linear(3 * in_channels, out_channels)
        self.linear_neg = nn.Linear(3 * in_channels, out_channels)
        nn.init.xavier_uniform_(self.linear_pos.weight)
        nn.init.xavier_uniform_(self.linear_neg.weight)

    def forward(self, x_pos, x_neg, pos_edge_index, neg_edge_index, num_nodes):
        pos_agg_pos = sparse_propagate(x_pos, pos_edge_index, num_nodes)
        neg_agg_neg = sparse_propagate(x_neg, neg_edge_index, num_nodes)
        pos_agg_neg = sparse_propagate(x_pos, neg_edge_index, num_nodes)
        neg_agg_pos = sparse_propagate(x_neg, pos_edge_index, num_nodes)

        h_pos_concat = torch.cat([x_pos, pos_agg_pos, neg_agg_neg], dim=1)
        h_neg_concat = torch.cat([x_neg, neg_agg_pos, pos_agg_neg], dim=1)

        z_pos = self.linear_pos(h_pos_concat)
        z_neg = self.linear_neg(h_neg_concat)
        return F.relu(z_pos), F.relu(z_neg)

class SignedGCN(nn.Module):
    def __init__(self, num_nodes, in_channels=32, hidden_channels=32, out_channels=16, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, in_channels)
        self.conv1 = SignedGraphConvolution(in_channels, hidden_channels)
        self.conv2 = SignedGraphConvolution(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, pos_edge_index, neg_edge_index, num_nodes):
        x = self.embedding.weight
        x_pos = x_neg = x
        x_pos, x_neg = self.conv1(x_pos, x_neg, pos_edge_index, neg_edge_index, num_nodes)
        x_pos = F.dropout(x_pos, p=self.dropout, training=self.training)
        x_neg = F.dropout(x_neg, p=self.dropout, training=self.training)
        x_pos, x_neg = self.conv2(x_pos, x_neg, pos_edge_index, neg_edge_index, num_nodes)
        return torch.cat([x_pos, x_neg], dim=1)

class SignedLinkPredictor(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.lin = nn.Linear(embedding_dim * 2, 1)

    def forward(self, z, edge_index):
        src, tgt = edge_index
        h = torch.cat([z[src], z[tgt]], dim=1)
        return self.lin(h).squeeze()

def train_and_evaluate(df, epochs=500, lr=0.01, patience=40):
    node_map, num_nodes = get_nodes_map(df)
    full_edge_index, full_edge_label = get_edge_indices(df, node_map)
    num_edges = full_edge_index.size(1)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics = {'micro_f1': [], 'macro_f1': [], 'binary_f1': [], 'auc': []}

    for fold, (train_val_idx, test_idx) in enumerate(kf.split(np.arange(num_edges))):
        print(f"\n{'='*20} Fold {fold+1}/5 {'='*20}")

        test_edge_index = full_edge_index[:, test_idx]
        test_label = full_edge_label[test_idx]

        train_val_indices = np.random.permutation(train_val_idx)
        split = int(0.9 * len(train_val_indices))
        train_idx, val_idx = train_val_indices[:split], train_val_indices[split:]

        train_edge_index = full_edge_index[:, train_idx]
        train_label = full_edge_label[train_idx]
        val_edge_index = full_edge_index[:, val_idx]
        val_label = full_edge_label[val_idx]

        pos_edge_index = train_edge_index[:, train_label == 1]
        neg_edge_index = train_edge_index[:, train_label == 0]

        model = SignedGCN(num_nodes)
        predictor = SignedLinkPredictor(embedding_dim=32)
        optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=lr, weight_decay=5e-4)
        criterion = nn.BCEWithLogitsLoss()

        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0
        train_losses, val_losses = [], []

        for epoch in range(epochs):
            model.train(); predictor.train()
            optimizer.zero_grad()
            z = model(pos_edge_index, neg_edge_index, num_nodes)
            out = predictor(z, train_edge_index)
            loss = criterion(out, train_label)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            model.eval(); predictor.eval()
            with torch.no_grad():
                z_val = model(pos_edge_index, neg_edge_index, num_nodes)
                val_out = predictor(z_val, val_edge_index)
                val_loss = criterion(val_out, val_label).item()
                val_losses.append(val_loss)

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = {'model': copy.deepcopy(model.state_dict()), 'predictor': copy.deepcopy(predictor.state_dict())}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            if epoch % 100 == 0:
                print(f"Epoch {epoch:03d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f}")

        if best_state:
            model.load_state_dict(best_state['model'])
            predictor.load_state_dict(best_state['predictor'])

        plt.figure(figsize=(8,5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.title(f'Fold {fold+1} Training Curve')
        plt.legend()
        plt.savefig(f'reddit_sgcn_fold_{fold+1}_loss.png', dpi=300, bbox_inches='tight')
        plt.close()

        model.eval(); predictor.eval()
        with torch.no_grad():
            z_test = model(pos_edge_index, neg_edge_index, num_nodes)
            logits = predictor(z_test, test_edge_index)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            truth = test_label.cpu().numpy()

            micro = f1_score(truth, preds, average='micro')
            macro = f1_score(truth, preds, average='macro')
            binary = f1_score(truth, preds, average='binary')
            auc = roc_auc_score(truth, probs)

            fold_metrics['micro_f1'].append(micro)
            fold_metrics['macro_f1'].append(macro)
            fold_metrics['binary_f1'].append(binary)
            fold_metrics['auc'].append(auc)

            print(f"Fold {fold+1} → AUC: {auc:.4f} | Binary-F1: {binary:.4f}")

    print("\n" + "="*60)
    print("FINAL RESULTS (5-Fold CV) - Reddit Hyperlink Network + SGCN")
    print("="*60)
    for k in fold_metrics:
        print(f"{k.replace('_', ' ').title():<12}: {np.mean(fold_metrics[k]):.4f} ± {np.std(fold_metrics[k]):.4f}")

if __name__ == "__main__":
    df = load_and_clean_data('soc-redditHyperlinks-body.tsv')
    perform_eda(df)
    train_and_evaluate(df, epochs=300, lr=0.01, patience=30)
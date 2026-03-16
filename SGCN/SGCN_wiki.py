import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import copy
import warnings

warnings.filterwarnings('ignore')

import random
import os

def seed_everything(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"All random seeds set to {seed}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

seed_everything(42)

# Step 1: Data Loading and Cleaning
def load_and_clean_data(file_path='wiki_RfA.csv'):
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    # The dataset might have different column names, ensure we target source, target, sign
    # Based on previous snippet: source, target, sign
    df.dropna(subset=['source', 'target', 'sign'], inplace=True)
    
    # Filter for valid signs (1: Support, -1: Oppose)
    df = df[df['sign'].isin([1, -1])]
    
    # Create binary label: 1 -> 1 (Positive), -1 -> 0 (Negative)
    df['label'] = (df['sign'] + 1) // 2
    
    # Remove duplicates if any
    df.drop_duplicates(subset=['source', 'target'], inplace=True)
    
    print("Data after cleaning:")
    print(df.head())
    print(f"Total edges: {len(df)}")
    print(f"Positive edges: {len(df[df['label'] == 1])}")
    print(f"Negative edges: {len(df[df['label'] == 0])}")
    return df

# Step 2: Build Graph Helpers
def get_nodes_map(df):
    all_nodes = np.unique(np.concatenate([df['source'].values, df['target'].values]))
    node_map = {node: idx for idx, node in enumerate(all_nodes)}
    num_nodes = len(all_nodes)
    return node_map, num_nodes

def get_edge_indices(df, node_map):
    # Convert source/target to indices
    src = [node_map[s] for s in df['source']]
    tgt = [node_map[t] for t in df['target']]
    edge_index = torch.tensor([src, tgt], dtype=torch.long)
    edge_label = torch.tensor(df['label'].values, dtype=torch.float)
    return edge_index, edge_label

# Basic Sparse Aggregation Helper (similar to PyG's MessagePassing but manual)
def sparse_propagate(x, edge_index, num_nodes):
    """
    Performs the aggregation: D^-0.5 * A * D^-0.5 * X
    """
    row, col = edge_index
    
    # Calculate degree
    deg = torch.bincount(row, minlength=num_nodes).float()
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    
    # Normalized weights for edges
    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    
    # Message passing (Scatter Add)
    # msg = norm.view(-1, 1) * x[col]
    # To save memory, we can do sparse mm if we built a sparse matrix, 
    # but here we stick to the scatter_add implementation for flexibility
    msg = norm.view(-1, 1) * x[col]
    
    out = torch.zeros_like(x)
    # scatter_add_(dim, index, src)
    out.scatter_add_(0, row.view(-1, 1).expand_as(msg), msg)
    
    return out

# Step 3: Define SGCN Model Layers
class SignedGraphConvolution(nn.Module):
    """
    Signed Graph Convolutional Layer based on Derr et al. (ICDM 2018).
    Maintains two representations: Balanced (Pos) and Unbalanced (Neg).
    """
    def __init__(self, in_channels, out_channels):
        super(SignedGraphConvolution, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Based on SGCN paper, we concatenate [h_u, agg_pos, agg_neg]
        # So the linear layer input dimension is 3 * in_channels
        self.linear_pos = nn.Linear(3 * in_channels, out_channels)
        self.linear_neg = nn.Linear(3 * in_channels, out_channels)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.linear_pos.weight)
        nn.init.xavier_uniform_(self.linear_neg.weight)

    def forward(self, x_pos, x_neg, pos_edge_index, neg_edge_index, num_nodes):
        # 1. Aggregate information
        # Positive Aggregation (Friend of Friend)
        pos_agg_pos = sparse_propagate(x_pos, pos_edge_index, num_nodes)
        # Negative Aggregation (Enemy of Enemy -> Balance theory says this is 'balanced' usually, 
        # but in SGCN derivation:
        # h_pos update uses: x_pos (self), agg(x_pos, pos_edges), agg(x_neg, neg_edges)
        # h_neg update uses: x_neg (self), agg(x_neg, pos_edges), agg(x_pos, neg_edges)
        
        neg_agg_neg = sparse_propagate(x_neg, neg_edge_index, num_nodes)
        
        pos_agg_neg = sparse_propagate(x_pos, neg_edge_index, num_nodes)
        neg_agg_pos = sparse_propagate(x_neg, pos_edge_index, num_nodes)
        
        # 2. Concatenate features for Positive (Balanced) Embedding
        # [Self, Pos_Neighbors_Pos_State, Neg_Neighbors_Neg_State]
        # Rationale: Friend is friend, Enemy of Enemy is friend (structural balance)
        h_pos_concat = torch.cat([x_pos, pos_agg_pos, neg_agg_neg], dim=1)
        z_pos = self.linear_pos(h_pos_concat)
        
        # 3. Concatenate features for Negative (Unbalanced) Embedding
        # [Self, Pos_Neighbors_Neg_State, Neg_Neighbors_Pos_State]
        # Rationale: Friend of Enemy is Enemy, Enemy of Friend is Enemy
        h_neg_concat = torch.cat([x_neg, neg_agg_pos, pos_agg_neg], dim=1)
        z_neg = self.linear_neg(h_neg_concat)
        
        return F.relu(z_pos), F.relu(z_neg)

class SignedGCN(nn.Module):
    def __init__(self, num_nodes, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(SignedGCN, self).__init__()
        
        # Initial embedding can be learnable or just features. 
        # Here we use learnable embeddings as in many link prediction tasks without rich features.
        self.embedding = nn.Embedding(num_nodes, in_channels)
        
        self.conv1 = SignedGraphConvolution(in_channels, hidden_channels)
        self.conv2 = SignedGraphConvolution(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, pos_edge_index, neg_edge_index, num_nodes):
        # Initial features: unique embedding for each node
        x = self.embedding.weight
        
        # We start with same x for both pos/neg channels or split?
        # Standard SGCN often initializes both with the node features.
        x_pos = x
        x_neg = x
        
        # Layer 1
        x_pos, x_neg = self.conv1(x_pos, x_neg, pos_edge_index, neg_edge_index, num_nodes)
        x_pos = F.dropout(x_pos, p=self.dropout, training=self.training)
        x_neg = F.dropout(x_neg, p=self.dropout, training=self.training)
        
        # Layer 2
        x_pos, x_neg = self.conv2(x_pos, x_neg, pos_edge_index, neg_edge_index, num_nodes)
        
        # Final embedding is usually concatenation of both representations
        z = torch.cat([x_pos, x_neg], dim=1)
        return z

class SignedLinkPredictor(nn.Module):
    """
    Logistic Regression on concatenated node embeddings.
    Input: [z_u, z_v] -> Probability of Positive Link
    """
    def __init__(self, embedding_dim):
        super(SignedLinkPredictor, self).__init__()
        # Input is z_u (dim) + z_v (dim)
        self.lin = nn.Linear(embedding_dim * 2, 1)
        
    def forward(self, z, edge_index):
        src, tgt = edge_index
        z_src = z[src]
        z_tgt = z[tgt]
        # Concatenate features
        h = torch.cat([z_src, z_tgt], dim=1)
        return self.lin(h).squeeze()

# Step 4: Training and Evaluation with 5-fold CV
def train_and_evaluate(df, epochs=100, lr=0.01, patience=20):
    node_map, num_nodes = get_nodes_map(df)
    full_edge_index, full_edge_label = get_edge_indices(df, node_map)
    num_edges = full_edge_index.size(1)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_metrics = {'micro_f1': [], 'macro_f1': [], 'binary_f1': [], 'auc': []}
    
    for fold, (train_val_idx, test_idx) in enumerate(kf.split(np.arange(num_edges))):
        print(f"\nFold {fold+1}/5")
        
        # --- Data Splitting ---
        # Test Set
        test_edge_index = full_edge_index[:, test_idx]
        test_label = full_edge_label[test_idx]
        
        # Train/Val Split
        train_val_indices = np.random.permutation(train_val_idx)
        split_point = int(len(train_val_indices) * 0.9)
        train_idx = train_val_indices[:split_point]
        val_idx = train_val_indices[split_point:]
        
        train_edge_index = full_edge_index[:, train_idx]
        train_label = full_edge_label[train_idx]
        
        val_edge_index = full_edge_index[:, val_idx]
        val_label = full_edge_label[val_idx]
        
        # --- Construct Positive and Negative Graphs for Training ---
        # SGCN requires separate edge indices for + and - edges to perform aggregation
        # We only use TRAINING edges to build the graph structure
        
        train_pos_mask = (train_label == 1)
        train_neg_mask = (train_label == 0)
        
        pos_edge_index = train_edge_index[:, train_pos_mask]
        neg_edge_index = train_edge_index[:, train_neg_mask]
        
        # --- Model Initialization ---
        # Embedding dim: 64 total (32 per channel)
        in_dim = 32
        hidden_dim = 32
        out_dim = 16 # Final vector will be 16+16 = 32
        
        model = SignedGCN(num_nodes, in_dim, hidden_dim, out_dim)
        predictor = SignedLinkPredictor(embedding_dim=out_dim*2)
        
        optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), 
                                     lr=lr, weight_decay=5e-4)
        criterion = torch.nn.BCEWithLogitsLoss()
        
        # --- Training Loop ---
        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            model.train()
            predictor.train()
            optimizer.zero_grad()
            
            # Forward pass: Learn node embeddings using graph structure
            z = model(pos_edge_index, neg_edge_index, num_nodes)
            
            # Predict on training edges
            out = predictor(z, train_edge_index)
            loss = criterion(out, train_label)
            
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
            # Validation
            model.eval()
            predictor.eval()
            with torch.no_grad():
                # Note: In transductive learning, we usually use the same embeddings z learned from structure
                # to predict validation edges.
                # Re-computing z is technically redundant in eval mode if graph didn't change, 
                # but good practice to ensure consistency.
                z_val = model(pos_edge_index, neg_edge_index, num_nodes)
                val_out = predictor(z_val, val_edge_index)
                val_loss = criterion(val_out, val_label).item()
                val_losses.append(val_loss)
            
            if epoch % 50 == 0:
                print(f'Epoch {epoch:03d}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')
            
            # Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    'model': copy.deepcopy(model.state_dict()),
                    'predictor': copy.deepcopy(predictor.state_dict())
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        # --- Evaluation ---
        # Load best model
        if best_state:
            model.load_state_dict(best_state['model'])
            predictor.load_state_dict(best_state['predictor'])
        
        # Plot Loss
        plt.figure()
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.title(f'Fold {fold+1} Loss')
        plt.legend()
        plt.savefig(f'sgcn_fold_{fold+1}_loss.png')
        plt.close()
        
        model.eval()
        predictor.eval()
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
            
            print(f"Fold {fold+1} Result - AUC: {auc:.4f}, Binary F1: {binary:.4f}")

    # Average metrics
    print("\nAverage SGCN Metrics over 5 Folds:")
    for key in fold_metrics:
        avg = np.mean(fold_metrics[key])
        std = np.std(fold_metrics[key])
        print(f"{key}: {avg:.4f} (+/- {std:.4f})")

if __name__ == "__main__":
    try:
        df = load_and_clean_data('wiki_RfA.csv')
        train_and_evaluate(df, epochs=300)
    except FileNotFoundError:
        print("Error: 'wiki_RfA.csv' not found. Please upload the dataset.")
    except Exception as e:
        print(f"An error occurred: {e}")
import sys
import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, roc_auc_score
from types import SimpleNamespace

# Add SGCN src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../SGCN/src'))

from sgcn import SignedGCNTrainer
from utils import read_graph

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SGCN_DIR = os.path.join(BASE_DIR, '../SGCN')
INPUT_PATH = os.path.join(SGCN_DIR, 'input/bitcoin_alpha.csv')
TEMP_EDGE_PATH = os.path.join(SGCN_DIR, 'input/bitcoin_alpha_sgcn_edges.csv')
EMBEDDING_PATH = os.path.join(SGCN_DIR, 'output/embedding/bitcoin_alpha_sgcn.csv')
WEIGHTS_PATH = os.path.join(SGCN_DIR, 'output/weights/bitcoin_alpha_sgcn.csv')
LOG_PATH = os.path.join(SGCN_DIR, 'logs/bitcoin_alpha_logs.json')
PLOT_PATH = os.path.join(BASE_DIR, 'sgcn_bitcoin_tsne.png')

def preprocess():
    print("Preprocessing Bitcoin Alpha dataset...")
    # Read raw CSV (Source, Target, Rating, Time)
    # Assuming standard format: source, target, rating, time. No header.
    df = pd.read_csv(INPUT_PATH, header=None, names=['source', 'target', 'rating', 'time'])
    
    # Map nodes to contiguous IDs
    nodes = pd.concat([df['source'], df['target']]).unique()
    node_map = {node: i for i, node in enumerate(nodes)}
    
    df['source_id'] = df['source'].map(node_map)
    df['target_id'] = df['target'].map(node_map)
    
    # Sign: 1 if rating > 0, -1 if rating < 0
    # Filter out 0 ratings if any
    # Convert rating to numeric, coercing errors to NaN
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df = df.dropna(subset=['rating'])
    
    df = df[df['rating'] != 0].copy()
    df['sign'] = df['rating'].apply(lambda x: 1 if x > 0 else -1)
    
    # Re-map IDs after filtering!
    # If we filter rows, some nodes might disappear or IDs might become non-contiguous relative to the new set.
    # But more importantly, the 'nodes' set calculated earlier included everything.
    # Let's recalculate the node map based on the filtered dataframe to be safe and ensure contiguous 0..N-1 IDs.
    
    nodes = pd.concat([df['source'], df['target']]).unique()
    node_map = {node: i for i, node in enumerate(nodes)}
    
    df['source_id'] = df['source'].map(node_map)
    df['target_id'] = df['target'].map(node_map)
    
    # SGCN's read_graph expects a CSV where the first 3 columns are read
    df[['source_id', 'target_id', 'sign']].to_csv(TEMP_EDGE_PATH, index=False, header=True)
    print(f"Saved processed edges to {TEMP_EDGE_PATH}")
    return len(nodes)

def run_sgcn():
    # Define arguments
    args = SimpleNamespace(
        edge_path=TEMP_EDGE_PATH,
        features_path=TEMP_EDGE_PATH, # Using structural features (SVD on adjacency)
        embedding_path=EMBEDDING_PATH,
        regression_weights_path=WEIGHTS_PATH,
        log_path=LOG_PATH,
        epochs=100,
        reduction_iterations=30,
        reduction_dimensions=64,
        seed=42,
        lamb=1.0,
        test_size=0.2,
        learning_rate=0.01,
        weight_decay=1e-5,
        layers=[32, 32],
        spectral_features=True
    )
    
    print("Reading graph...")
    edges = read_graph(args)
    
    print("Initializing SGCN Trainer...")
    trainer = SignedGCNTrainer(args, edges)
    trainer.setup_dataset()
    
    print("Training model...")
    trainer.create_and_train_model()
    
    # Custom Evaluation
    # print("\nEvaluating with custom metrics...")
    # evaluate_metrics(trainer)
    
    # Save
    trainer.save_model()
    
    # Save logs to specific result file
    results = trainer.logs["performance"][-1] # Last epoch results
    print("\nFinal Results:")
    print(f"AUC: {results[1]}")
    print(f"Micro-F1: {results[2]}")
    print(f"Macro-F1: {results[3]}")
    print(f"Binary-F1: {results[4]}")
    
    with open(os.path.join(SGCN_DIR, 'bitcoin_results.txt'), 'w') as f:
        f.write(f"AUC: {results[1]}\n")
        f.write(f"Micro-F1: {results[2]}\n")
        f.write(f"Macro-F1: {results[3]}\n")
        f.write(f"Binary-F1: {results[4]}\n")
    
    return trainer

# def evaluate_metrics(trainer):
    # ... (Commenting out as we now use internal logging)


def visualize_tsne():
    print(f"Generating t-SNE plot from {EMBEDDING_PATH}...")
    if not os.path.exists(EMBEDDING_PATH):
        print("Embedding file not found.")
        return

    df = pd.read_csv(EMBEDDING_PATH)
    # Columns: id, x_0, x_1...
    vectors = df.iloc[:, 1:].values
    
    # Downsample if needed
    if len(vectors) > 5000:
        indices = np.random.choice(len(vectors), 5000, replace=False)
        vectors = vectors[indices]
        
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    vecs_2d = tsne.fit_transform(vectors)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(vecs_2d[:, 0], vecs_2d[:, 1], s=10, alpha=0.6)
    plt.title("t-SNE of SGCN Embeddings (Bitcoin Alpha)")
    plt.savefig(PLOT_PATH)
    print(f"Saved plot to {PLOT_PATH}")

if __name__ == "__main__":
    preprocess()
    run_sgcn()
    visualize_tsne()

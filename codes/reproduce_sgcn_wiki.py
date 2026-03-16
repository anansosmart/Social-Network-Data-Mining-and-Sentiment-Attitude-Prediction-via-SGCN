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
INPUT_PATH = os.path.join(SGCN_DIR, 'input/wiki_RfA.csv')
TEMP_EDGE_PATH = os.path.join(SGCN_DIR, 'input/wiki_sgcn_edges.csv')
EMBEDDING_PATH = os.path.join(SGCN_DIR, 'output/embedding/wiki_sgcn.csv')
WEIGHTS_PATH = os.path.join(SGCN_DIR, 'output/weights/wiki_sgcn.csv')
LOG_PATH = os.path.join(SGCN_DIR, 'logs/wiki_logs.json')
PLOT_PATH = os.path.join(BASE_DIR, 'sgcn_wiki_tsne.png')

def preprocess():
    print("Preprocessing Wiki RfA dataset...")
    df = pd.read_csv(INPUT_PATH)
    
    # Columns: source, target, sign, text
    sources = df['source']
    targets = df['target']
    signs = df['sign']
    
    nodes = pd.concat([sources, targets]).unique()
    node_map = {node: i for i, node in enumerate(nodes)}
    
    df_out = pd.DataFrame()
    df_out['source_id'] = sources.map(node_map)
    df_out['target_id'] = targets.map(node_map)
    df_out['sign'] = signs.astype(int)
    
    df_out[['source_id', 'target_id', 'sign']].to_csv(TEMP_EDGE_PATH, index=False, header=True)
    print(f"Saved processed edges to {TEMP_EDGE_PATH}")
    return len(nodes)

def run_sgcn():
    args = SimpleNamespace(
        edge_path=TEMP_EDGE_PATH,
        features_path=TEMP_EDGE_PATH,
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
    
    trainer.save_model()
    
    results = trainer.logs["performance"][-1]
    print("\nFinal Results:")
    print(f"AUC: {results[1]}")
    print(f"Micro-F1: {results[2]}")
    print(f"Macro-F1: {results[3]}")
    print(f"Binary-F1: {results[4]}")
    
    with open(os.path.join(SGCN_DIR, 'wiki_results.txt'), 'w') as f:
        f.write(f"AUC: {results[1]}\n")
        f.write(f"Micro-F1: {results[2]}\n")
        f.write(f"Macro-F1: {results[3]}\n")
        f.write(f"Binary-F1: {results[4]}\n")

# def evaluate_metrics(trainer):
    # ... (Commenting out)

def visualize_tsne():
    print(f"Generating t-SNE plot from {EMBEDDING_PATH}...")
    if not os.path.exists(EMBEDDING_PATH):
        print("Embedding file not found.")
        return

    df = pd.read_csv(EMBEDDING_PATH)
    vectors = df.iloc[:, 1:].values
    
    if len(vectors) > 5000:
        indices = np.random.choice(len(vectors), 5000, replace=False)
        vectors = vectors[indices]
        
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    vecs_2d = tsne.fit_transform(vectors)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(vecs_2d[:, 0], vecs_2d[:, 1], s=10, alpha=0.6)
    plt.title("t-SNE of SGCN Embeddings (Wiki RfA)")
    plt.savefig(PLOT_PATH)
    print(f"Saved plot to {PLOT_PATH}")

if __name__ == "__main__":
    preprocess()
    run_sgcn()
    visualize_tsne()

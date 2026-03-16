import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Define paths
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
REDDIT_EMB_PATH = os.path.join(DATA_DIR, 'reddit.embeddings')
BITCOIN_EMB_PATH = os.path.join(DATA_DIR, 'bitcoin.embeddings')
REDDIT_PLOT_PATH = os.path.join(DATA_DIR, 'reddit_tsne.png')
BITCOIN_PLOT_PATH = os.path.join(DATA_DIR, 'bitcoin_tsne.png')

def load_embeddings(path):
    print(f"Loading embeddings from {path}...")
    ids = []
    vectors = []
    with open(path, 'r') as f:
        header = f.readline().split()
        num_nodes, dim = int(header[0]), int(header[1])
        for line in f:
            parts = line.split()
            ids.append(int(parts[0]))
            vectors.append([float(x) for x in parts[1:]])
    return np.array(ids), np.array(vectors)

def visualize_tsne(emb_path, plot_path, title, sample_size=5000):
    if not os.path.exists(emb_path):
        print(f"Error: {emb_path} not found.")
        return

    ids, vectors = load_embeddings(emb_path)
    
    # Downsample if too large for t-SNE speed
    if len(ids) > sample_size:
        print(f"Downsampling from {len(ids)} to {sample_size} points for visualization...")
        indices = np.random.choice(len(ids), sample_size, replace=False)
        vectors = vectors[indices]
    
    print(f"Running t-SNE on {len(vectors)} points...")
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    vectors_2d = tsne.fit_transform(vectors)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], alpha=0.6, s=10)
    plt.title(title)
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.grid(True, alpha=0.3)
    plt.savefig(plot_path)
    print(f"Saved plot to {plot_path}")
    plt.close()

if __name__ == "__main__":
    visualize_tsne(REDDIT_EMB_PATH, REDDIT_PLOT_PATH, 't-SNE Visualization of Reddit Embeddings')
    visualize_tsne(BITCOIN_EMB_PATH, BITCOIN_PLOT_PATH, 't-SNE Visualization of Bitcoin Alpha Embeddings')

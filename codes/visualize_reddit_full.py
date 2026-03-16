import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Define paths
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
REDDIT_EMB_PATH = os.path.join(DATA_DIR, 'reddit.embeddings')
REDDIT_FULL_PLOT_PATH = os.path.join(DATA_DIR, 'reddit_tsne_full.png')

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

def visualize_tsne_full(emb_path, plot_path, title):
    if not os.path.exists(emb_path):
        print(f"Error: {emb_path} not found.")
        return

    ids, vectors = load_embeddings(emb_path)
    
    print(f"Running t-SNE on ALL {len(vectors)} points... (This may take a while)")
    
    # Using 'auto' learning rate and PCA initialization for better convergence/speed on larger datasets
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto', n_jobs=-1)
    vectors_2d = tsne.fit_transform(vectors)
    
    plt.figure(figsize=(12, 10))
    # Using smaller marker size and transparency for dense plot
    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], alpha=0.3, s=1, c='b')
    plt.title(title)
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.grid(True, alpha=0.2)
    plt.savefig(plot_path, dpi=300) # Higher DPI for detail
    print(f"Saved plot to {plot_path}")
    plt.close()

if __name__ == "__main__":
    visualize_tsne_full(REDDIT_EMB_PATH, REDDIT_FULL_PLOT_PATH, 't-SNE Visualization of All Reddit Embeddings')

import pandas as pd
import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from deepwalk import graph
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score

# Define paths
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(DATA_DIR, 'soc-sign-bitcoinalpha.csv')
EDGELIST_PATH = os.path.join(DATA_DIR, 'bitcoin.edgelist')
EMBEDDINGS_PATH = os.path.join(DATA_DIR, 'bitcoin.embeddings')
PLOT_PATH = os.path.join(DATA_DIR, 'bitcoin_scores_plot.png')

def preprocess_and_embed():
    print("Reading CSV...")
    # The file is CSV: SOURCE, TARGET, RATING, TIME
    # No header in the snippet I saw, but let's double check. 
    # The snippet "7188,1,10,1407470400" suggests no header.
    df = pd.read_csv(CSV_PATH, header=None, names=['SOURCE', 'TARGET', 'RATING', 'TIME'])
    
    print(f"Loaded {len(df)} interactions.")
    
    # Extract columns
    sources = df['SOURCE'].astype(str)
    targets = df['TARGET'].astype(str)
    ratings = df['RATING'].astype(int)
    
    # Get unique nodes
    unique_nodes = pd.concat([sources, targets]).unique()
    node_to_id = {node: i for i, node in enumerate(unique_nodes)}
    
    print(f"Found {len(unique_nodes)} unique nodes.")
    
    # Map to IDs
    print("Mapping to IDs...")
    df['source_id'] = sources.map(node_to_id)
    df['target_id'] = targets.map(node_to_id)
    
    # Save edge list for DeepWalk
    print(f"Writing structure edge list to {EDGELIST_PATH}...")
    df[['source_id', 'target_id']].to_csv(EDGELIST_PATH, sep=' ', index=False, header=False)
    
    # Run DeepWalk
    print("Loading graph for DeepWalk...")
    G = graph.load_edgelist(EDGELIST_PATH, undirected=True)
    
    print("Number of nodes: {}".format(len(G.nodes())))
    
    num_walks = 10
    walk_length = 40
    window_size = 5
    representation_size = 64
    workers = 4
    seed = 42
    
    print("Walking...")
    walks = graph.build_deepwalk_corpus(G, num_paths=num_walks,
                                        path_length=walk_length, alpha=0, rand=random.Random(seed))
    
    print("Training Word2Vec...")
    model = Word2Vec(walks, vector_size=representation_size, window=window_size, min_count=0, sg=1, hs=1, workers=workers)
    
    print(f"Saving embeddings to {EMBEDDINGS_PATH}...")
    model.wv.save_word2vec_format(EMBEDDINGS_PATH)
    
    return df, node_to_id

def load_embeddings(path):
    print(f"Loading embeddings from {path}...")
    embeddings = {}
    with open(path, 'r') as f:
        header = f.readline().split()
        num_nodes, dim = int(header[0]), int(header[1])
        for line in f:
            parts = line.split()
            node_id = int(parts[0])
            vector = np.array([float(x) for x in parts[1:]])
            embeddings[node_id] = vector
    return embeddings

def evaluate(df, node_to_id):
    embeddings = load_embeddings(EMBEDDINGS_PATH)
    
    print("Preparing edge features for classification...")
    X = []
    y = []
    
    missing_count = 0
    
    src_ids = df['source_id'].values
    tgt_ids = df['target_id'].values
    ratings = df['RATING'].values
    
    for u, v, rating in zip(src_ids, tgt_ids, ratings):
        if u in embeddings and v in embeddings:
            # Feature: Hadamard product
            X.append(embeddings[u] * embeddings[v])
            
            # Label: 1 if Rating > 0 (Trust), 0 if Rating < 0 (Distrust)
            # Ignoring 0 ratings if any (though usually rare in this dataset)
            if rating > 0:
                y.append(1)
            elif rating < 0:
                y.append(0)
            else:
                # Skip neutral/zero ratings
                X.pop() 
                pass
        else:
            missing_count += 1
            
    X = np.array(X)
    y = np.array(y)
    
    print(f"Constructed {len(X)} labeled edge samples. (Missing embeddings for {missing_count} edges)")
    
    # Evaluation Ratios
    ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    num_shuffles = 10
    
    results = {
        'micro': [],
        'macro': [],
        'binary': [],
        'auc': []
    }
    
    print("\nStarting evaluation (Edge Classification: Trust Prediction)...")
    print(f"{'Ratio':<10} {'Micro-F1':<10} {'Macro-F1':<10} {'Binary-F1':<10} {'AUC':<10}")
    print("-" * 55)
    
    for ratio in ratios:
        micro_scores = []
        macro_scores = []
        binary_scores = []
        auc_scores = []
        
        for _ in range(num_shuffles):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=ratio, random_state=None, stratify=y
            )
            
            # Logistic Regression
            clf = LogisticRegression(solver='liblinear', max_iter=1000)
            clf.fit(X_train, y_train)
            
            preds = clf.predict(X_test)
            probs = clf.predict_proba(X_test)[:, 1]
            
            micro_scores.append(f1_score(y_test, preds, average='micro'))
            macro_scores.append(f1_score(y_test, preds, average='macro'))
            binary_scores.append(f1_score(y_test, preds, average='binary'))
            try:
                auc_scores.append(roc_auc_score(y_test, probs))
            except:
                auc_scores.append(0.5)
        
        avg_micro = np.mean(micro_scores)
        avg_macro = np.mean(macro_scores)
        avg_binary = np.mean(binary_scores)
        avg_auc = np.mean(auc_scores)
        
        results['micro'].append(avg_micro)
        results['macro'].append(avg_macro)
        results['binary'].append(avg_binary)
        results['auc'].append(avg_auc)
        
        print(f"{int(ratio*100)}%       {avg_micro:.4f}     {avg_macro:.4f}     {avg_binary:.4f}     {avg_auc:.4f}")
        
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.plot([r*100 for r in ratios], results['micro'], marker='o', label='Micro-F1')
    plt.plot([r*100 for r in ratios], results['macro'], marker='s', label='Macro-F1')
    plt.plot([r*100 for r in ratios], results['binary'], marker='^', label='Binary-F1')
    plt.plot([r*100 for r in ratios], results['auc'], marker='x', label='AUC')
    plt.xlabel('Training Data Percentage (%)')
    plt.ylabel('Score')
    plt.title('DeepWalk Edge Classification (Trust Prediction) on Bitcoin Alpha')
    plt.legend()
    plt.grid(True)
    plt.savefig(PLOT_PATH)
    print(f"\nPlot saved to {PLOT_PATH}")

if __name__ == "__main__":
    df, node_to_id = preprocess_and_embed()
    evaluate(df, node_to_id)

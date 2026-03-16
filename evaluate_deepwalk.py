import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer
from collections import defaultdict

# Define paths
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
RFA_TXT_PATH = os.path.join(os.path.dirname(DATA_DIR), 'wiki-RfA.txt') # It's in the parent dir of codes/
CSV_PATH = os.path.join(DATA_DIR, 'wiki_RfA.csv')
EMBEDDINGS_PATH = os.path.join(DATA_DIR, 'wiki.embeddings')
MAPPING_PATH = os.path.join(DATA_DIR, 'node_mapping.csv')
PLOT_PATH = os.path.join(DATA_DIR, 'f1_scores_plot.png')

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
    return embeddings, dim

def load_labels_from_txt(path):
    print(f"Parsing labels from {path}...")
    user_outcomes = defaultdict(list)
    
    current_tgt = None
    
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if line.startswith('TGT:'):
                current_tgt = line[4:]
            elif line.startswith('RES:'):
                if current_tgt:
                    res = int(line[4:])
                    user_outcomes[current_tgt].append(res)
    
    # Process outcomes: 1 if any success (1), else 0 (failure/other)
    # Original RES: 1 (success), -1 (failure)
    node_labels = {}
    for user, outcomes in user_outcomes.items():
        if 1 in outcomes:
            node_labels[user] = 1
        else:
            node_labels[user] = 0
            
    print(f"Found {len(node_labels)} labeled users.")
    return node_labels

def load_node_mapping(path):
    print(f"Loading node mapping from {path}...")
    df = pd.read_csv(path)
    # Create dictionary: name -> id
    return dict(zip(df['node'].astype(str), df['id']))

def evaluate():
    # 1. Load Data
    if not os.path.exists(RFA_TXT_PATH):
        print(f"Error: {RFA_TXT_PATH} not found. Please ensure wiki-RfA.txt is in the project root.")
        return

    embeddings, dim = load_embeddings(EMBEDDINGS_PATH)
    name_to_id = load_node_mapping(MAPPING_PATH)
    raw_labels = load_labels_from_txt(RFA_TXT_PATH)
    
    # 2. Align Data
    X = []
    y = []
    
    aligned_count = 0
    for user_name, label in raw_labels.items():
        if user_name in name_to_id:
            node_id = name_to_id[user_name]
            if node_id in embeddings:
                X.append(embeddings[node_id])
                y.append(label)
                aligned_count += 1
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Aligned {aligned_count} nodes with both embeddings and labels.")
    
    if aligned_count < 10:
        print("Not enough labeled nodes for evaluation.")
        return

    # 3. Evaluation Loop
    ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    num_shuffles = 10
    
    results = {
        'micro': [],
        'macro': [],
        'binary': [],
        'auc': []
    }
    
    print("\nStarting evaluation...")
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
            
            clf = LogisticRegression(solver='liblinear', max_iter=1000)
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            probs = clf.predict_proba(X_test)[:, 1] # Probability of class 1
            
            micro = f1_score(y_test, preds, average='micro')
            macro = f1_score(y_test, preds, average='macro')
            binary = f1_score(y_test, preds, average='binary')
            try:
                auc = roc_auc_score(y_test, probs)
            except ValueError:
                auc = 0.5 # Handle case where only one class is present in test set (unlikely with stratify)
            
            micro_scores.append(micro)
            macro_scores.append(macro)
            binary_scores.append(binary)
            auc_scores.append(auc)
            
        avg_micro = np.mean(micro_scores)
        avg_macro = np.mean(macro_scores)
        avg_binary = np.mean(binary_scores)
        avg_auc = np.mean(auc_scores)
        
        results['micro'].append(avg_micro)
        results['macro'].append(avg_macro)
        results['binary'].append(avg_binary)
        results['auc'].append(avg_auc)
        
        print(f"{int(ratio*100)}%       {avg_micro:.4f}     {avg_macro:.4f}     {avg_binary:.4f}     {avg_auc:.4f}")

    # 4. Visualization
    plt.figure(figsize=(10, 6))
    plt.plot([r*100 for r in ratios], results['micro'], marker='o', label='Micro-F1')
    plt.plot([r*100 for r in ratios], results['macro'], marker='s', label='Macro-F1')
    plt.plot([r*100 for r in ratios], results['binary'], marker='^', label='Binary-F1')
    plt.plot([r*100 for r in ratios], results['auc'], marker='x', label='AUC')
    plt.xlabel('Percentage of Labeled Nodes (%)')
    plt.ylabel('Score')
    plt.title('DeepWalk Evaluation on wiki-RfA (Admin Prediction)')
    plt.legend()
    plt.grid(True)
    plt.savefig(PLOT_PATH)
    print(f"\nPlot saved to {PLOT_PATH}")

if __name__ == "__main__":
    evaluate()

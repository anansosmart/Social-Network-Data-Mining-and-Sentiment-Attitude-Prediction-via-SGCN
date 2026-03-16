import pandas as pd
import os
import sys
import random
import logging
from deepwalk import graph
from gensim.models import Word2Vec

# Define paths
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(DATA_DIR, 'wiki_RfA.csv')
EDGELIST_PATH = os.path.join(DATA_DIR, 'wiki.edgelist')
EMBEDDINGS_PATH = os.path.join(DATA_DIR, 'wiki.embeddings')

def preprocess():
    print("Reading CSV...")
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found.")
        sys.exit(1)
        
    df = pd.read_csv(CSV_PATH)
    
    print(f"Columns: {df.columns.tolist()}")
    
    # Extract source and target
    if 'source' not in df.columns or 'target' not in df.columns:
        print("Error: 'source' and 'target' columns required.")
        sys.exit(1)

    sources = df['source'].astype(str)
    targets = df['target'].astype(str)
    
    # Get unique nodes
    unique_nodes = pd.concat([sources, targets]).unique()
    node_to_id = {node: i for i, node in enumerate(unique_nodes)}
    
    print(f"Found {len(unique_nodes)} unique nodes.")
    
    # Map to IDs
    print("Mapping to IDs...")
    df['source_id'] = sources.map(node_to_id)
    df['target_id'] = targets.map(node_to_id)
    
    # Write to edge list (space separated)
    print(f"Writing edge list to {EDGELIST_PATH}...")
    df[['source_id', 'target_id']].to_csv(EDGELIST_PATH, sep=' ', index=False, header=False)
    print("Done preprocessing.")
    
    # Save mapping for future reference
    mapping_path = os.path.join(DATA_DIR, 'node_mapping.csv')
    pd.DataFrame(list(node_to_id.items()), columns=['node', 'id']).to_csv(mapping_path, index=False)
    print(f"Saved node mapping to {mapping_path}")

def run_deepwalk_custom():
    print("Loading graph...")
    # deepwalk's graph.load_edgelist expects a filename
    G = graph.load_edgelist(EDGELIST_PATH, undirected=True)
    
    print("Number of nodes: {}".format(len(G.nodes())))
    
    num_walks = 10
    walk_length = 40
    window_size = 5
    representation_size = 64
    workers = 4 # Use 4 workers
    seed = 0
    
    print("Walking...")
    walks = graph.build_deepwalk_corpus(G, num_paths=num_walks,
                                        path_length=walk_length, alpha=0, rand=random.Random(seed))
    
    print("Training Word2Vec...")
    # Fix: use vector_size instead of size for gensim 4.0+
    model = Word2Vec(walks, vector_size=representation_size, window=window_size, min_count=0, sg=1, hs=1, workers=workers)
    
    print(f"Saving embeddings to {EMBEDDINGS_PATH}...")
    model.wv.save_word2vec_format(EMBEDDINGS_PATH)
    print("Done.")

if __name__ == "__main__":
    preprocess()
    run_deepwalk_custom()

import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

# 1. 读取 Embeddings
# DeepWalk 保存的是 Word2Vec 格式：第一行是 "节点数 维度"，后面是 "节点ID 向量..."
def load_embeddings(file_path):
    print("正在加载 DeepWalk 向量...")
    with open(file_path, 'r') as f:
        header = f.readline() # 跳过第一行
        num_nodes, dim = map(int, header.split())
        
        ids = []
        vectors = []
        for line in f:
            parts = line.strip().split()
            ids.append(parts[0]) # 保持 ID 为字符串，方便匹配
            vectors.append([float(x) for x in parts[1:]])
    return ids, np.array(vectors)

ids, vectors = load_embeddings('codes/wiki.embeddings') # 确保路径正确

# 2. 降维可视化 (t-SNE)
print("正在进行 t-SNE 降维 (可能需要几分钟)...")
sample_size = len(vectors) #min(len(vectors), 2000)
indices = np.random.choice(len(vectors), sample_size, replace=False)
vectors_sample = vectors[indices]

tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
vectors_2d = tsne.fit_transform(vectors_sample)

# 3. 画图
plt.figure(figsize=(10, 10))
plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], s=10, alpha=0.6, c='grey')
plt.title("DeepWalk Baseline Visualization (Structure Only)")
plt.axis('off')

# 保存图片
plt.savefig('deepwalk_viz.png', dpi=300)
print("✅ 图片已保存为 deepwalk_viz.png。请查看它是否显示出'混杂'的状态。")
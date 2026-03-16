import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

# 1. 定义路径 (保持和你之前的脚本一致)
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_PATH = os.path.join(DATA_DIR, 'wiki.embeddings')

def visualize():
    print(f"Loading embeddings from {EMBEDDINGS_PATH}...")
    
    # 使用 Gensim 读取 Word2Vec 格式的向量文件
    # binary=False 因为 save_word2vec_format 默认保存的是文本格式
    model = KeyedVectors.load_word2vec_format(EMBEDDINGS_PATH, binary=False)
    
    # 提取所有向量和对应的节点ID
    # model.vectors 是一个 (N, 64) 的 numpy 矩阵
    # model.index_to_key 是对应的节点 ID 列表
    vectors = model.vectors
    indices = model.index_to_key
    
    print(f"Loaded {len(vectors)} vectors of dimension {model.vector_size}")

    # 2. t-SNE 降维
    print("Running t-SNE dimensionality reduction...")
    print("Note: This might take a few moments depending on your dataset size.")
    
    # 参数解释：
    # n_components=2: 降维到 2D 平面
    # perplexity=30: 平衡局部和全局结构，通常在 5-50 之间调整
    # init='pca': 使用 PCA 初始化通常比随机初始化更能保留全局拓扑
    tsne = TSNE(n_components=2, 
                perplexity=30, 
                n_iter=1000, 
                init='pca', 
                learning_rate='auto',
                random_state=42)
    
    embeddings_2d = tsne.fit_transform(vectors)

    # 3. 绘图
    print("Plotting...")
    plt.figure(figsize=(12, 12))
    
    # 绘制散点图
    plt.scatter(embeddings_2d[:, 0], 
                embeddings_2d[:, 1], 
                c=embeddings_2d[:, 1], # 临时用 Y 轴坐标作为颜色渐变，看看分布
                cmap='viridis',        # 使用蓝绿黄渐变色
                alpha=0.6, 
                s=10)
    plt.colorbar(label='Cluster Position')          # 点的大小

    plt.title('DeepWalk Visualization (Wiki RfA)', fontsize=16)
    plt.axis('off') # 去掉坐标轴，模仿你那张图的风格
    
    # 保存图片
    output_img = os.path.join(DATA_DIR, 'wiki_deepwalk_vis.png')
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_img}")
    
    plt.show()

if __name__ == "__main__":
    if not os.path.exists(EMBEDDINGS_PATH):
        print(f"Error: {EMBEDDINGS_PATH} not found. Please run the training script first.")
    else:
        visualize()
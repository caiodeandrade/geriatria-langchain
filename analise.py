import os
import faiss
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# 1. Carregar variáveis de ambiente
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# 2. Função para carregar vetores FAISS
def load_faiss_vectors(index_path):
    db = FAISS.load_local(index_path, OpenAIEmbeddings(openai_api_key=openai_api_key), allow_dangerous_deserialization=True)
    faiss_index = db.index

    num_vectors = faiss_index.ntotal
    dim = faiss_index.d

    if num_vectors == 0:
        raise ValueError("O banco vetorial FAISS está vazio.")

    print(f"Vetores encontrados: {num_vectors}, dimensão: {dim}")

    vectors = np.zeros((num_vectors, dim), dtype=np.float32)
    for i in range(num_vectors):
        vectors[i] = faiss_index.reconstruct(i)

    return vectors, faiss_index

# 3. PCA
def plot_pca(vectors):
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(vectors)

    plt.figure(figsize=(10, 6))
    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], alpha=0.7)
    plt.title("PCA 2D dos Vetores FAISS")
    plt.xlabel("Componente 1")
    plt.ylabel("Componente 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 4. t-SNE
def plot_tsne(vectors):
    sample = vectors[:500]  # usa amostra para performance
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_result = tsne.fit_transform(sample)

    plt.figure(figsize=(10, 6))
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.7, c='green')
    plt.title("t-SNE 2D (Amostra de 500 vetores)")
    plt.xlabel("Componente 1")
    plt.ylabel("Componente 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 5. Histograma de distâncias
def plot_distance_histogram(vectors):
    sample = vectors[:500]
    distances = cdist(sample, sample, metric="euclidean")
    upper = np.triu_indices_from(distances, k=1)
    distance_values = distances[upper]

    plt.figure(figsize=(8,5))
    plt.hist(distance_values, bins=30, color='skyblue', edgecolor='black')
    plt.title("Histograma de Distâncias Euclidianas")
    plt.xlabel("Distância")
    plt.ylabel("Frequência")
    plt.tight_layout()
    plt.show()

# 6. Densidade local por k-NN
def plot_local_density(vectors, faiss_index, k=5):
    D, _ = faiss_index.search(vectors, k+1)  # inclui o próprio, que será descartado
    local_density = D[:, 1:].mean(axis=1)  # ignora a distância do próprio vetor

    plt.figure(figsize=(10,5))
    plt.hist(local_density, bins=30, color='salmon', edgecolor='black')
    plt.title(f"Densidade Local - Distância Média dos {k} Vizinhos Mais Próximos")
    plt.xlabel("Distância Média")
    plt.ylabel("Número de Vetores")
    plt.tight_layout()
    plt.show()

# 7. Executar análises
index_path = "pdf_faiss_index"
vectors, faiss_index = load_faiss_vectors(index_path)

plot_pca(vectors)
plot_tsne(vectors)
plot_distance_histogram(vectors)
plot_local_density(vectors, faiss_index)
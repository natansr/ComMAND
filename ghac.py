import pickle
import os
import re
import numpy as np
import json
import time
import pandas as pd
import networkx as nx
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import mean_squared_log_error, accuracy_score
from scipy.sparse.csgraph import connected_components
from tqdm import tqdm
from tkinter import Tk, Button, Label, Entry, StringVar, filedialog, ttk, messagebox, Frame, Text
from threading import Thread

# Diretórios e variáveis globais
layers = "L2-20000"

# Função para carregar embeddings da GCN
def load_gcn_embeddings(embedding_path):
    try:
        with open(embedding_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except FileNotFoundError:
        print(f"Erro: Arquivo de embeddings {embedding_path} não encontrado.")
        return None

# Função GHAC
def GHAC(mlist, n_clusters=-1):
    distance = []

    for i in range(len(mlist)):
        gtmp = []
        for j in range(len(mlist)):
            if i < j:
                cosdis = np.dot(mlist[i], mlist[j]) / (np.linalg.norm(mlist[i]) * (np.linalg.norm(mlist[j])))
                gtmp.append(cosdis)
            elif i > j:
                gtmp.append(distance[j][i])
            else:
                gtmp.append(0)
        distance.append(gtmp)

    distance = np.array(distance)
    distance = np.multiply(distance, -1)

    if n_clusters == -1:
        best_m = -10000000
        n_components1, labels = connected_components(distance)

        distance[distance <= 0.5] = 0
        G = nx.from_numpy_matrix(distance)
        n_components, labels = connected_components(distance)

        for k in range(n_components, n_components1 - 1, -1):
            model_HAC = AgglomerativeClustering(linkage="average", metric='precomputed', n_clusters=k)
            model_HAC.fit(distance)
            labels = model_HAC.labels_

            mod = nx.algorithms.community.quality.modularity(G, [set(np.where(np.array(labels) == i)[0]) for i in range(len(set(labels)))])
            if mod > best_m:
                best_m = mod
                best_labels = labels
        labels = best_labels
    else:
        model_HAC = AgglomerativeClustering(linkage='average', metric='precomputed', n_clusters=n_clusters)
        model_HAC.fit(distance)
        labels = model_HAC.labels_

    return labels

# Função de Avaliação Pairwise
def pairwise_evaluate(correct_labels, pred_labels):
    if len(correct_labels) != len(pred_labels):
        print("As listas têm tamanhos diferentes. Não é possível realizar a comparação.")
        return 0, 0, 0

    TP = 0.0  # True Positives
    TP_FP = 0.0  # Predicted Same Author Pairs
    TP_FN = 0.0  # Actual Same Author Pairs

    for i in range(len(correct_labels)):
        for j in range(i + 1, len(correct_labels)):
            if correct_labels[i] == correct_labels[j]:
                TP_FN += 1
            if pred_labels[i] == pred_labels[j]:
                TP_FP += 1
            if (correct_labels[i] == correct_labels[j]) and (pred_labels[i] == pred_labels[j]):
                TP += 1

    if TP == 0:
        pairwise_precision = 0
        pairwise_recall = 0
        pairwise_f1 = 0
    else:
        pairwise_precision = TP / TP_FP
        pairwise_recall = TP / TP_FN
        pairwise_f1 = (2 * pairwise_precision * pairwise_recall) / (pairwise_precision + pairwise_recall)
    return pairwise_precision, pairwise_recall, pairwise_f1

# Função para calcular ACP e AAP
def calculate_ACP_AAP(correct_labels, cluster_labels):
    unique_clusters = np.unique(cluster_labels)
    ACP = 0.0
    AAP = 0.0

    # Calcular ACP (Average Cluster Purity)
    for cluster in unique_clusters:
        cluster_indices = np.where(cluster_labels == cluster)[0]
        cluster_author_labels = correct_labels[cluster_indices]
        unique_author_labels, author_counts = np.unique(cluster_author_labels, return_counts=True)

        max_count = np.max(author_counts)
        ACP += max_count / len(cluster_indices)

    # Calcular AAP (Average Author Purity)
    unique_authors = np.unique(correct_labels)
    for author in unique_authors:
        author_indices = np.where(correct_labels == author)[0]
        author_cluster_labels = cluster_labels[author_indices]
        unique_cluster_labels, cluster_counts = np.unique(author_cluster_labels, return_counts=True)

        max_count = np.max(cluster_counts)
        AAP += max_count / len(author_indices)

    ACP /= len(unique_clusters)
    AAP /= len(unique_authors)

    return ACP, AAP

# Função para calcular a métrica K
def calculate_KMetric(ACP, AAP):
    return np.sqrt(ACP * AAP)

def cluster_evaluate(method, embedding_path, json_dir, result_box):
    embeddings = load_gcn_embeddings(embedding_path)
    if embeddings is None:
        result_box.insert("end", f"Erro: Embeddings não encontrados em {embedding_path}.\n")
        return

    results = []
    file_names = [f for f in os.listdir(json_dir) if f.endswith('.json')]

    for fname in tqdm(file_names, desc="Processando arquivos de autores"):
        with open(os.path.join(json_dir, fname), 'r', encoding='utf-8') as file:
            data = json.load(file)

        correct_labels = []
        papers = []

        for entry in data:
            correct_labels.append(int(entry['label']))
            pid = "i" + str(entry['id'])
            if pid in embeddings and not np.all(embeddings[pid] == 0):
                papers.append(pid)

        if len(correct_labels) < 2 or not papers:
            continue

        mlist = [embeddings[pid] for pid in papers if pid in embeddings]
        if len(mlist) == 0:
            continue

        labels = GHAC(mlist) if method == "GHAC_nok" else GHAC(mlist, len(set(correct_labels)))
        correct_labels = np.array(correct_labels)
        labels = np.array(labels)

        pairwise_precision, pairwise_recall, pairwise_f1 = pairwise_evaluate(correct_labels, labels)
        ACP, AAP = calculate_ACP_AAP(correct_labels, labels)
        K = calculate_KMetric(ACP, AAP)

        results.append([fname, pairwise_precision, pairwise_recall, pairwise_f1, ACP, AAP, K])

        result_box.insert("end", f"Processado: {fname}\n")
        result_box.see("end")

    results_df = pd.DataFrame(results, columns=["Author", "Pairwise Precision", "Pairwise Recall", "Pairwise F1", "ACP", "AAP", "K Metric"])
    output_path = os.path.join(json_dir, 'clustering_results.csv')
    results_df.to_csv(output_path, index=False)

    result_box.insert("end", f"Resultados salvos em {output_path}.\n")
    result_box.insert("end", results_df.to_string(index=False) + "\n")
    result_box.see("end")

def main():
    root = Tk()
    root.title("Validação de Embeddings com GCN")
    root.geometry("800x600")

    Label(root, text="Caminho para Embedding Final:").pack(pady=5)
    embedding_path_var = StringVar()
    embedding_path_entry = Entry(root, textvariable=embedding_path_var, width=50)
    embedding_path_entry.pack(pady=5)
    Button(root, text="Selecionar Embedding", command=lambda: embedding_path_entry.insert(0, filedialog.askopenfilename(filetypes=[("Arquivo PKL", "*.pkl")], title="Selecione o arquivo de embedding"))).pack(pady=5)

    Label(root, text="Diretório de Arquivos JSON dos Autores:").pack(pady=5)
    json_dir_var = StringVar()
    json_dir_entry = Entry(root, textvariable=json_dir_var, width=50)
    json_dir_entry.pack(pady=5)
    Button(root, text="Selecionar Diretório", command=lambda: json_dir_entry.insert(0, filedialog.askdirectory(title="Selecione o diretório dos arquivos JSON"))).pack(pady=5)

    Button(root, text="Iniciar Validação", command=lambda: Thread(target=cluster_evaluate, args=("GHAC", embedding_path_var.get(), json_dir_var.get(), result_box)).start()).pack(pady=20)

    result_frame = Frame(root)
    result_frame.pack(pady=10, fill="both", expand=True)

    result_box = Text(result_frame, wrap="word", state="normal")
    result_box.pack(side="left", fill="both", expand=True)

    scrollbar = ttk.Scrollbar(result_frame, command=result_box.yview)
    scrollbar.pack(side="right", fill="y")
    result_box.config(yscrollcommand=scrollbar.set)

    root.mainloop()

if __name__ == "__main__":
    main()

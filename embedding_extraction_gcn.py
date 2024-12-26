import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import numpy as np
import pickle
import os
from tkinter import Tk, Button, Label, Entry, IntVar, StringVar, filedialog, ttk, messagebox, Frame, Text
from threading import Thread

# Verificar se a GPU está disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Funções auxiliares para carregar e preparar os dados
def load_embeddings(data_dir, embedding_type):
    try:
        with open(os.path.join(data_dir, f'{embedding_type}_emb.pkl'), "rb") as file_obj:
            return pickle.load(file_obj)
    except FileNotFoundError:
        print(f"Erro: Arquivo de embeddings {embedding_type} não encontrado.")
        return None

def prepare_features(G, embeddings, device):
    nodes = list(G.nodes)
    node_idx_map = {node: idx for idx, node in enumerate(nodes)}
    edges = [(node_idx_map[u], node_idx_map[v]) for u, v in G.edges]

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)

    if embeddings:
        sample_embedding = next(iter(embeddings.values()))
        embedding_dim = sample_embedding.shape[0]
        features = []
        for node in nodes:
            features.append(embeddings.get(node, np.zeros(embedding_dim)))
    else:
        embedding_dim = 128
        features = np.random.normal(loc=0.0, scale=1.0, size=(len(nodes), embedding_dim))

    features = np.array(features)
    x = torch.tensor(features, dtype=torch.float).to(device)

    return Data(x=x, edge_index=edge_index), nodes, embedding_dim

class GCN(torch.nn.Module):
    def __init__(self, input_dim, num_layers):
        super(GCN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(input_dim, 512))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(512, 512))

        self.convs.append(GCNConv(512, 512))
        self.fc = torch.nn.Linear(512, input_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = self.fc(x)
        return x

def train_gcn(data, input_dim, num_layers, epochs, progress_bar, progress_label, output_box):
    model = GCN(input_dim, num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    def train():
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.mse_loss(out, data.x)
        loss.backward()
        optimizer.step()
        return loss

    for epoch in range(epochs):
        loss = train()
        progress = ((epoch + 1) / epochs) * 100
        progress_bar["value"] = progress
        progress_bar.update()
        progress_label.config(text=f"Treinando: Epoch {epoch + 1}/{epochs} - Loss: {loss.item():.4f}")
        output_box.insert("end", f"Epoch {epoch + 1}, Loss: {loss.item():.4f}\n")
        output_box.see("end")

    model.eval()
    with torch.no_grad():
        new_embeddings = model(data).cpu().numpy()

    return new_embeddings

def save_embeddings(embeddings, nodes, save_path):
    embeddings_dict = {node: embeddings[idx] for idx, node in enumerate(nodes)}
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as file_obj:
        pickle.dump(embeddings_dict, file_obj)

def main():
    root = Tk()
    root.title("Treinamento de Embeddings com GCN")
    root.geometry("700x600")

    # Diretório dos dados
    Label(root, text="Diretório dos Dados:").pack(pady=5)
    data_dir_entry = Entry(root, width=50)
    data_dir_entry.pack(pady=5)
    Button(root, text="Selecionar", command=lambda: data_dir_entry.insert(0, filedialog.askdirectory(title="Selecione o diretório dos dados"))).pack(pady=5)

    # Tipo de embedding
    Label(root, text="Tipo de Embedding:").pack(pady=5)
    embedding_type_var = StringVar(value="scibert")
    embedding_type_entry = Entry(root, textvariable=embedding_type_var, width=30)
    embedding_type_entry.pack(pady=5)

    # Número de camadas
    Label(root, text="Número de Camadas (default: 3):").pack(pady=5)
    num_layers_var = IntVar(value=3)
    num_layers_entry = Entry(root, textvariable=num_layers_var, width=10)
    num_layers_entry.pack(pady=5)

    # Número de épocas
    Label(root, text="Número de Épocas (default: 1000):").pack(pady=5)
    epochs_var = IntVar(value=1000)
    epochs_entry = Entry(root, textvariable=epochs_var, width=10)
    epochs_entry.pack(pady=5)

    # Função para obter o valor de épocas com fallback para o valor padrão
    def get_epochs_value():
        entry_value = epochs_entry.get().strip()
        try:
            if entry_value == "":  # Verifica se a entrada está vazia
                return 1000  # Retorna o valor padrão
            value = int(entry_value)
            if value <= 0:  # Verifica se o valor é positivo
                raise ValueError("Número de épocas deve ser maior que zero.")
            return value
        except (ValueError, TypeError):
            # Exibe uma mensagem de erro e usa o valor padrão se o valor for inválido
            messagebox.showwarning("Aviso", "Entrada inválida para o número de épocas. Usando o valor padrão de 1000.")
            return 1000  # Valor padrão

    # Botão para iniciar o treinamento
    Button(root, text="Iniciar Treinamento", command=lambda: Thread(target=run_gcn_training_gui, args=(data_dir_entry.get(), embedding_type_var.get(), num_layers_var.get(), get_epochs_value(), root)).start()).pack(pady=20)


    # Barra de progresso
    progress_bar = ttk.Progressbar(root, orient="horizontal", length=500, mode="determinate")
    progress_bar.pack(pady=10)

    # Label de progresso
    progress_label = Label(root, text="")
    progress_label.pack(pady=5)

    # Caixa de saída com barra de rolagem
    output_frame = Frame(root)
    output_frame.pack(pady=10, fill="both", expand=True)

    output_box = Text(output_frame, wrap="word", state="normal")
    output_box.pack(side="left", fill="both", expand=True)

    scrollbar = ttk.Scrollbar(output_frame, command=output_box.yview)
    scrollbar.pack(side="right", fill="y")
    output_box.config(yscrollcommand=scrollbar.set)

    root.mainloop()

def run_gcn_training_gui(data_dir, embedding_type, num_layers, epochs, root):
    try:
        # Obter widgets para progresso e saída
        progress_bar = root.nametowidget(".!progressbar")
        progress_label = root.nametowidget(".!label2")
        output_box = root.nametowidget(".!frame.!text")

        with open(os.path.join(data_dir, "HeterogeneousNetwork.pkl"), 'rb') as file:
            G = pickle.load(file)

        embeddings = load_embeddings(data_dir, embedding_type)
        if embeddings is None:
            messagebox.showerror("Erro", f"Arquivo de embeddings {embedding_type} não encontrado.")
            return

        data, nodes, input_dim = prepare_features(G, embeddings, device)
        
        # Verifique se o valor de epochs é válido e faça o treinamento
        if epochs <= 0:
            messagebox.showerror("Erro", "O número de épocas deve ser maior que 0.")
            return

        new_embeddings = train_gcn(data, input_dim, num_layers, epochs, progress_bar, progress_label, output_box)

        save_path = os.path.join(data_dir, f'pemb_final_gcn_{embedding_type}.pkl')
        save_embeddings(new_embeddings, nodes, save_path)

        messagebox.showinfo("Concluído", f"Treinamento concluído e embeddings salvos em {save_path}.")
    except Exception as e:
        messagebox.showerror("Erro", f"Erro durante o treinamento do GCN: {e}")
        print(f"Erro: {e}")


if __name__ == "__main__":
    main()

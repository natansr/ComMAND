import torch
from transformers import AutoTokenizer, AutoModel
import pickle
import os
import numpy as np
from tkinter import simpledialog, ttk, END, messagebox, Label, Text, Scrollbar, Frame, Tk, filedialog, BooleanVar, Checkbutton, Radiobutton, Button
from threading import Thread

class EmbeddingExtractor:
    def __init__(self, model_dir, data_dir, progress_bar, status_label, output_box, progress_label, num_docs=None):
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.progress_bar = progress_bar
        self.status_label = status_label
        self.output_box = output_box
        self.progress_label = progress_label
        self.num_docs = num_docs  # Quantidade de documentos a serem processados
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModel.from_pretrained(model_dir)

    def clean_value(self, value, substitute='.'):
        if value is None or str(value).lower() == 'null':
            return substitute
        return str(value).strip()

    def load_features(self, features_to_include):
        paperid_abstract = {}
        paperid_author_names = {}
        paperid_title = {}
        paperid_venue_name = {}
        paperid_words = {}

        # Carregar abstracts
        if 'abstract' in features_to_include:
            with open(os.path.join(self.data_dir, "paper_abstract.txt"), encoding='utf-8') as abstract_file:
                for line in abstract_file:
                    toks = line.strip().split("\t")
                    if len(toks) == 2:
                        paperid_abstract[toks[0]] = self.clean_value(toks[1], '.')

        # Carregar nomes dos autores
        if 'author_names' in features_to_include:
            with open(os.path.join(self.data_dir, "paper_author_names.txt"), encoding='utf-8') as author_file:
                for line in author_file:
                    toks = line.strip().split("\t")
                    if len(toks) == 2:
                        paperid_author_names[toks[0]] = self.clean_value(toks[1], ' ')

        # Carregar títulos
        if 'title' in features_to_include:
            with open(os.path.join(self.data_dir, "paper_title.txt"), encoding='utf-8') as title_file:
                for line in title_file:
                    toks = line.strip().split("\t")
                    if len(toks) == 2:
                        paperid_title[toks[0]] = self.clean_value(toks[1], '.')

        # Carregar nomes de venues
        if 'venue_name' in features_to_include:
            with open(os.path.join(self.data_dir, "paper_venue_name.txt"), encoding='utf-8') as venue_file:
                for line in venue_file:
                    toks = line.strip().split("\t")
                    if len(toks) == 2:
                        paperid_venue_name[toks[0]] = self.clean_value(toks[1], ' ')

        # Carregar palavras
        if 'word' in features_to_include:
            with open(os.path.join(self.data_dir, "paper_word.txt"), encoding='utf-8') as words_file:
                for line in words_file:
                    toks = line.strip().split("\t")
                    if len(toks) == 2:
                        if toks[0] not in paperid_words:
                            paperid_words[toks[0]] = []
                        paperid_words[toks[0]].append(self.clean_value(toks[1], ' '))

        return paperid_abstract, paperid_author_names, paperid_title, paperid_venue_name, paperid_words

    def combine_features(self, paperid_abstract, paperid_author_names, paperid_title, paperid_venue_name, paperid_words, features_to_include):
        documents = []
        paper_ids = []
        for paperid in paperid_title:
            abstract = self.clean_value(paperid_abstract.get(paperid, '.'), ' ') if 'abstract' in features_to_include else ''
            author_names = self.clean_value(paperid_author_names.get(paperid, ""), ' ') if 'author_names' in features_to_include else ''
            title = self.clean_value(paperid_title.get(paperid, '.'), ' ') if 'title' in features_to_include else ''
            venue_name = self.clean_value(paperid_venue_name.get(paperid, ""), ' ') if 'venue_name' in features_to_include else ''
            words = " ".join([self.clean_value(word, '.') for word in paperid_words.get(paperid, [])]) if 'word' in features_to_include else ''

            combined_text = f"{abstract} {author_names} {title} {venue_name} {words}".strip()

            if combined_text and not combined_text.isspace():
                documents.append(combined_text)
                paper_ids.append(paperid)
            else:
                self.print_output(f"Warning: Empty or invalid document for paper ID {paperid}")

        # Limitar a quantidade de documentos se `self.num_docs` for definida
        if self.num_docs and len(documents) > self.num_docs:
            documents = documents[:self.num_docs]
            paper_ids = paper_ids[:self.num_docs]

        return documents, paper_ids

    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        return embedding

    def extract_embeddings(self, features_to_include):
        self.print_output("Carregando features selecionadas...")
        paperid_abstract, paperid_author_names, paperid_title, paperid_venue_name, paperid_words = self.load_features(features_to_include)

        documents, paper_ids = self.combine_features(paperid_abstract, paperid_author_names, paperid_title, paperid_venue_name, paperid_words, features_to_include)

        paper_vec = {}
        total_docs = len(documents)
        self.print_output(f"Total de documentos a serem processados: {total_docs}")

        for i, doc in enumerate(documents):
            paper_vec[paper_ids[i]] = self.get_embedding(doc)
            self.progress_bar["value"] = ((i + 1) / total_docs) * 100
            self.progress_bar.update()
            self.progress_label.config(text=f"Processando documento {i + 1} de {total_docs}")

        with open(os.path.join(self.data_dir, 'scibert_emb.pkl'), "wb") as file_obj:
            pickle.dump(paper_vec, file_obj)

        self.print_output("Extração de embeddings concluída e embeddings salvos.")
        self.status_label.config(text="Extração de embeddings concluída e embeddings salvos.")



    def print_output(self, message):
        self.output_box.insert(END, message + "\n")
        self.output_box.see(END)
        print(message)

    

def main(model_dir, data_dir, features_to_include):
    root = Tk()
    root.title("Progresso de Extração de Embeddings")
    root.geometry("600x600")

    progress_bar = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate")
    progress_bar.pack(pady=10)

    progress_label = Label(root, text="")
    progress_label.pack(pady=5)

    status_label = Label(root, text="", wraplength=400)
    status_label.pack(pady=5)

    output_frame = Frame(root)
    output_frame.pack(pady=10, fill="both", expand=True)

    # Caixa de texto para saída com barra de rolagem
    scrollbar = Scrollbar(output_frame)
    scrollbar.pack(side="right", fill="y")

    output_box = Text(output_frame, wrap="word", yscrollcommand=scrollbar.set, state="normal")
    output_box.pack(side="left", fill="both", expand=True)
    scrollbar.config(command=output_box.yview)

    # Solicitar a quantidade de documentos a serem processados
    num_docs = simpledialog.askinteger("Quantidade de Documentos", "Digite a quantidade de documentos a serem processados (ou deixe em branco para todos):", parent=root)

    # Inicializar o extrator e iniciar a extração em uma thread separada
    extractor = EmbeddingExtractor(model_dir, data_dir, progress_bar, status_label, output_box, progress_label, num_docs)
    extraction_thread = Thread(target=extractor.extract_embeddings, args=(features_to_include,))
    extraction_thread.start()

    root.mainloop()




if __name__ == "__main__":
    # Exemplo de como chamar o main diretamente
    # Substitua os caminhos de `model_dir` e `data_dir` conforme necessário
    model_dir = filedialog.askdirectory(title="Selecione o diretório do modelo BERT")
    data_dir = filedialog.askdirectory(title="Selecione o diretório dos dados")
    features_to_include = ['abstract', 'author_names', 'title', 'venue_name', 'word']
    
    if model_dir and data_dir:
        main(model_dir, data_dir, features_to_include)
    else:
        print("Diretórios de modelo ou dados não foram selecionados.")


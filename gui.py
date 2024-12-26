import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import filedialog, simpledialog, messagebox, BooleanVar, Checkbutton, Menu
import pre_processing
import network_creation
import embedding_extraction
import embedding_extraction_gcn
import ghac

from threading import Thread
from multiprocessing import Process

from pre_process_ghac import dividir_json_por_autor


class MainApplication(ttk.Window):
    def __init__(self):
        super().__init__(themename="solar")  # Você pode escolher outros temas como "darkly", "litera", etc.
        self.title("ComMAND v.0.1")
        self.geometry("600x500")
        
        self.selected_features = {
            "abstract": BooleanVar(value=True),
            "author_names": BooleanVar(value=True),
            "title": BooleanVar(value=True),
            "venue_name": BooleanVar(value=False),
            "word": BooleanVar(value=False)
        }

        self.setup_ui()
        self.create_menu()

    def setup_ui(self):
        # Título da aplicação
        label_title = ttk.Label(self, text="Welcome to ComMAND!", font=("Helvetica", 16))
        label_title.pack(pady=10)

        # Seção para seleção de features
        label_features = ttk.Label(self, text="Select features for AND task:")
        label_features.pack(pady=5)
        
        for feature, var in self.selected_features.items():
            cb = ttk.Checkbutton(self, text=feature, variable=var, bootstyle="primary")
            cb.pack(anchor='w')



        # Botões de ação
        btn_preprocess = ttk.Button(self, width=50, text="Pre-processing", command=self.run_pre_processing, bootstyle="info")
        btn_preprocess.pack(pady=5)

        btn_network = ttk.Button(self,  width=50, text="Creation of a Heterogeneous Graph", command=self.run_network_creation, bootstyle="success")
        btn_network.pack(pady=5)

        btn_embeddings = ttk.Button(self,width=50, text="Embeddings Extraction with NLP", command=self.run_embedding_extraction, bootstyle="success")
        btn_embeddings.pack(pady=5)

        btn_gcn = ttk.Button(self,       width=50, text="GCN Learning", command=self.run_gcn_extraction, bootstyle="warning")
        btn_gcn.pack(pady=5)

        btn_pre_ghac = ttk.Button(self,       width=50, text="GHAC Pre-processing", command=self.run_dividir_json_por_autor, bootstyle="secondary")
        btn_pre_ghac.pack(pady=5)

        btn_ghac = ttk.Button(self,       width=50, text="GHAC Clustering", command=self.run_clustering_validation, bootstyle="secondary")
        btn_ghac.pack(pady=5)


        # Barra de progresso
        self.progress = ttk.Progressbar(self, orient="horizontal", length=400, mode="determinate", bootstyle="warning")
        self.progress.pack(pady=20)


    def create_menu(self):
        # Criação da barra de menus
        menu_bar = Menu(self)

        # Menu "Arquivo"
        file_menu = Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Open", command=self.open_file)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)
        menu_bar.add_cascade(label="File", menu=file_menu)

        # Menu "View"
        view_menu = Menu(menu_bar, tearoff=0)
        themes = ["solar", "darkly", "litera"]
        for theme in themes:
            view_menu.add_command(label=f"Modo {theme.capitalize()}", command=lambda t=theme: self.change_theme(t))
        menu_bar.add_cascade(label="View", menu=view_menu)

        # Menu "Ajuda"
        help_menu = Menu(menu_bar, tearoff=0)
        help_menu.add_command(label="JSON format", command=self.show_help)
        help_menu.add_command(label="About", command=self.show_about)
        menu_bar.add_cascade(label="Help", menu=help_menu)

        # Adicionar a barra de menus à janela
        self.config(menu=menu_bar)


    def show_about(self):
        about_message = "ComMAND v.1.4\n"
        messagebox.showinfo("Sobre", about_message)



    def show_help(self):
        help_message = (
            "Formato esperado do arquivo JSON:\n\n"
            "[\n"
            "  {\n"
            "    \"id\": <número inteiro>,\n"
            "    \"label\": <número inteiro>,\n"
            "    \"author\": \"Nome do autor\",\n"
            "    \"title\": \"Título do artigo\",\n"
            "    \"venue\": \"Nome do local de publicação\",\n"
            "    \"abstract\": \"Resumo do artigo\",\n"
            "    \"coauthors\": [\"coautor1\", \"coautor2\", ...]\n"
            "  },\n"
            "  ...\n"
            "]\n\n"
            "Certifique-se de que todos os campos obrigatórios estejam preenchidos e sigam o formato correto."
        )
        messagebox.showinfo("Formato do JSON", help_message)


    def open_file(self):
        # Função para abrir um arquivo (pode ser adaptada conforme necessário)
        filedialog.askopenfile

    def change_theme(self, theme_name):
        self.style.theme_use(theme_name)


    def get_selected_features(self):
        selected = [feature for feature, var in self.selected_features.items() if var.get()]
        if not selected:
            messagebox.showwarning("Aviso", "Nenhuma feature selecionada. Por favor, selecione ao menos uma.")
        return selected

    def run_pre_processing(self):
        self.progress.start(10)
        try:
            input_path = filedialog.askopenfilename(title="Selecione o arquivo JSON de entrada", filetypes=[("Arquivos JSON", "*.json")])
            if not input_path:
                messagebox.showwarning("Atenção", "Nenhum arquivo selecionado.")
                return

            output_dir = filedialog.askdirectory(title="Selecione o diretório de saída")
            if not output_dir:
                messagebox.showwarning("Atenção", "Nenhum diretório de saída selecionado.")
                return

            selected_features = self.get_selected_features()
            if not selected_features:
                return

            pre_processing.main(input_path, output_dir, selected_features)
            #messagebox.showinfo("Concluído", "Pré-processamento concluído com sucesso.")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao processar: {e}")
        finally:
            self.progress.stop()

    def run_network_creation(self):
        self.progress.start(10)
        try:
            selected_features = self.get_selected_features()
            if not selected_features:
                messagebox.showwarning("Aviso", "Nenhuma feature selecionada. Por favor, selecione ao menos uma.")
                return

            network_creation.main(selected_features)
            messagebox.showinfo("Concluído", "Rede heterogênea criada com sucesso.")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao criar a rede: {e}")
        finally:
            self.progress.stop()

    def run_embedding_extraction(self):
        self.progress.start(10)
        try:
            use_scibert = messagebox.askyesno("Modelo", "Deseja usar o modelo SciBERT (uncased)?")
            if use_scibert:
                model_dir = "allenai/scibert_scivocab_uncased"
            else:
                model_dir = filedialog.askdirectory(title="Selecione o diretório do modelo")
                if not model_dir:
                    messagebox.showerror("Erro", "Diretório do modelo não selecionado.")
                    return

            data_dir = filedialog.askdirectory(title="Selecione o diretório dos dados")
            if not data_dir:
                messagebox.showerror("Erro", "Diretório dos dados não selecionado.")
                return

            features_to_include = self.get_selected_features()
            if not features_to_include:
                return

            embedding_extraction.main(model_dir, data_dir, features_to_include)
            messagebox.showinfo("Concluído", "Extração de embeddings concluída com sucesso.")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro na extração de embeddings: {e}")
        finally:
            self.progress.stop()

    def run_gcn_extraction(self):
        try:
            # Chamar a função main() do embedding_extraction_gcn diretamente
            embedding_extraction_gcn.main()
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao iniciar o treinamento GCN: {e}")

    def run_dividir_json_por_autor(self):
        try:
            # Chama a função de divisão do JSON em uma nova thread para não travar a interface
            Thread(target=dividir_json_por_autor).start()
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao dividir o JSON: {e}")

    def run_clustering_validation(self):
        try:
            # Chamar a função main() do embedding_extraction_gcn diretamente
            #thread = Thread(target=ghac.main)
            #thread.start()
            ghac.main()
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao iniciar o treinamento GCN: {e}")
        




if __name__ == "__main__":
    app = MainApplication()
    app.mainloop()

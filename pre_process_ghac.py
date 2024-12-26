import json
import os
import re
from tkinter import Tk, filedialog, messagebox, Button, Label

# Função para limpar o nome do autor para ser usado como nome de arquivo
def clean_filename(name):
    if not name or not isinstance(name, str):
        return "unknown_author"
    # Substituir caracteres inválidos por underscore
    return re.sub(r'[\/:*?"<>|]', '_', name.replace(' ', '_'))

# Função principal para dividir o JSON
def dividir_json_por_autor():
    # Janela do tkinter
    root = Tk()
    root.withdraw()  # Oculta a janela principal

    # Selecionar o arquivo JSON de entrada
    json_file = filedialog.askopenfilename(title="Selecione o arquivo JSON de entrada", filetypes=[("Arquivos JSON", "*.json")])
    if not json_file:
        messagebox.showerror("Erro", "Nenhum arquivo JSON selecionado.")
        return

    # Selecionar o diretório de saída
    output_dir = filedialog.askdirectory(title="Selecione o diretório de saída")
    if not output_dir:
        messagebox.showerror("Erro", "Nenhum diretório de saída selecionado.")
        return

    output_dir = os.path.join(output_dir, 'autores_json')
    os.makedirs(output_dir, exist_ok=True)

    # Carregar o JSON principal
    try:
        with open(json_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except Exception as e:
        messagebox.showerror("Erro", f"Erro ao carregar o arquivo JSON: {e}")
        return

    # Separar os dados por autor e salvar em JSONs individuais
    for entry in data:
        author = entry.get('author')

        # Verificar se o campo 'author' existe e é uma string válida
        if author is None or not isinstance(author, str) or author.strip() == '':
            author = 'unknown_author'
        else:
            author = author.strip()

        author_filename = clean_filename(author) + '.json'
        json_output = os.path.join(output_dir, author_filename)

        # Se o arquivo já existe, adiciona a entrada ao arquivo existente
        if os.path.isfile(json_output):
            with open(json_output, 'r', encoding='utf-8') as file:
                existing_data = json.load(file)

            existing_data.append(entry)

            with open(json_output, 'w', encoding='utf-8') as file:
                json.dump(existing_data, file, indent=4, ensure_ascii=False)
        else:
            with open(json_output, 'w', encoding='utf-8') as file:
                json.dump([entry], file, indent=4, ensure_ascii=False)

    messagebox.showinfo("Concluído", "JSONs criados para cada nome de autor.")
    print("JSONs criados para cada nome de autor.")

if __name__ == "__main__":
    dividir_json_por_autor()

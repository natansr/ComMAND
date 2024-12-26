import json
import re
import os
from tkinter import filedialog, Tk, messagebox
from tqdm import tqdm

# Função principal de pré-processamento
def preprocess_data(input_path, output_dir, selected_features):
    if not input_path or not output_dir:
        raise ValueError("Caminhos de entrada e saída não podem ser vazios.")

    os.makedirs(output_dir, exist_ok=True)

    papers = {}
    authors = {}
    venues = {}
    word = {}
    keyid = 0

    r = '[!“”"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~—～]+'
    stopword = ['at', 'based', 'in', 'of', 'for', 'on', 'and', 'to', 'an', 'using', 'with', 'the']

    with open(os.path.join(output_dir, 'paper_author.txt'), 'w', encoding='utf-8') as f1, \
         open(os.path.join(output_dir, 'paper_venue.txt'), 'w', encoding='utf-8') as f2, \
         open(os.path.join(output_dir, 'paper_word.txt'), 'w', encoding='utf-8') as f3, \
         open(os.path.join(output_dir, 'paper_title.txt'), 'w', encoding='utf-8') as f4, \
         open(os.path.join(output_dir, 'paper_author_names.txt'), 'w', encoding='utf-8') as f5, \
         open(os.path.join(output_dir, 'paper_abstract.txt'), 'w', encoding='utf-8') as f6, \
         open(os.path.join(output_dir, 'paper_venue_name.txt'), 'w', encoding='utf-8') as f7:

        with open(input_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)

        if not data:
            raise ValueError("O arquivo JSON está vazio ou não é válido.")

        for entry in tqdm(data, desc="Processando JSON"):
            pid = entry.get('id')
            label = entry.get('label')

            if pid is None or label is None:
                continue

            title = str(entry.get('title', '')).strip()
            author = str(entry.get('author', '')).strip()
            coauthors = entry.get('coauthors', []) if entry.get('coauthors') else []
            abstract = str(entry.get('abstract', '')).strip()
            venue = str(entry.get('conf', '')).strip()

            if 'abstract' in selected_features and abstract:
                f6.write(f'i{pid}\t{abstract}\n')

            all_authors = [author] + coauthors
            for author_name in all_authors:
                author_clean = author_name.replace(" ", "")
                if author_clean and author_clean not in authors:
                    authors[author_clean] = keyid
                    keyid += 1
                if author_clean:
                    f1.write(f'i{pid}\t{authors[author_clean]}\n')

            f5.write(f'i{pid}\t{",".join(all_authors)}\n')

            if 'venue_name' in selected_features and venue:
                if venue and venue not in venues:
                    venues[venue] = keyid
                    keyid += 1
                if venue:
                    f2.write(f'i{pid}\t{venues[venue]}\n')
                    f7.write(f'i{pid}\t{venue}\n')

            title_cleaned = re.sub(r, ' ', title)
            title_cleaned = title_cleaned.replace('\t', ' ').lower()

            f4.write(f'i{pid}\t{title_cleaned}\n')

            if 'word' in selected_features:
                split_cut = title_cleaned.split(' ')
                for word_part in split_cut:
                    if word_part and word_part not in stopword:
                        if word_part in word:
                            word[word_part] += 1
                        else:
                            word[word_part] = 1

        if 'word' in selected_features:
            for entry in tqdm(data, desc="Escrevendo palavras"):
                pid = entry.get('id')
                title = str(entry.get('title', '')).strip()

                title_cleaned = re.sub(r, ' ', title).replace('\t', ' ').lower()
                split_cut = title_cleaned.split(' ')
                for word_part in split_cut:
                    if word_part in word and word[word_part] >= 2:
                        f3.write(f'i{pid}\t{word_part}\n')

    messagebox.showinfo("Concluído", "Processamento concluído com sucesso!")

# Função principal para seleção de caminhos e execução do pré-processamento
def main(input_path, output_dir, selected_features):
    if selected_features is None:
        selected_features = []

    preprocess_data(input_path, output_dir, selected_features)

if __name__ == "__main__":
    # Apenas para fins de teste ou uso independente
    root = Tk()
    root.withdraw()
    input_path = filedialog.askopenfilename(title="Selecione o arquivo JSON de entrada", filetypes=[("Arquivos JSON", "*.json")])
    output_dir = filedialog.askdirectory(title="Selecione o diretório de saída")
    #selected_features = ['abstract', 'venue_name', 'word']  # Exemplo de features padrão
    #main(input_path, output_dir, selected_features)

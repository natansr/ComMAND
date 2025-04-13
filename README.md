# ComMAND: Author Name Disambiguation Framework

ComMAND is a framework for Author Name Disambiguation (AND) that combines SciBERT-based embeddings, heterogeneous graph construction, Graph Convolutional Networks (GCN), and GHAC clustering. The system is accessible via a graphical interface built with ttkbootstrap.

## Project Structure

```
├── data_process/
│   ├── pre_processing.py
│   └── dividir_json_por_autor.py
├── datasets/                    # Input and processed data
├── gcn/
│   └── embedding_extraction_gcn.py
├── ghac/
│   └── ghac.py
├── het_network/
│   └── network_creation.py
├── nlp/
│   └── nlp.py
├── gui.py                       # Main GUI script
└── README.md
```

## Modules Overview

- **Pre-processing**: Filters input JSONs by selected features.
- **NLP**: Extracts contextual embeddings using SciBERT (`allenai/scibert_scivocab_uncased`).
- **Graph Construction**: Builds a heterogeneous graph including papers, authors, venues, and keywords.
- **GCN**: Learns refined node embeddings from the graph structure.
- **GHAC**: Clusters documents into authors and evaluates results using standard AND metrics.

## Running the Application

To start the graphical interface:

```bash
python gui.py
```

This GUI supports feature selection, embedding extraction, graph creation, GCN training, and clustering validation.

## Dependencies

- Python 3.10+
- PyTorch
- PyTorch Geometric
- HuggingFace Transformers
- scikit-learn
- ttkbootstrap
- tqdm
- pandas
- networkx

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Notes

- Only SciBERT is currently supported for embedding extraction.
- Evaluation includes Pairwise Precision, Recall, F1-score, ACP, AAP, and K-Metric.

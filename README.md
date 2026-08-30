# English–Persian Neural Machine Translation with Transformer & FastText

An end-to-end neural machine translation project for translating **English text into Persian** using a custom **Transformer encoder–decoder** implemented with TensorFlow/Keras and initialized with **300-dimensional FastText embeddings**.

The project covers dataset loading, text normalization, subword tokenization, padding, pretrained embedding initialization, Transformer training, and translation-quality evaluation.

## Project Overview

The model is trained on the **ParsInLU English–Persian translation dataset** available through Hugging Face.

The pipeline includes:

```text
ParsInLU Translation Dataset
        ↓
English & Persian Text Cleaning
        ↓
SentencePiece Tokenization
        ↓
Sequence Padding
        ↓
FastText Embedding Initialization
        ↓
Transformer Encoder–Decoder
        ↓
Model Training
        ↓
BLEU / ROUGE-L Evaluation
```

## Model Architecture

The implementation contains:

- English and Persian embedding layers
- 300-dimensional pretrained FastText initialization
- Sinusoidal positional encoding
- Multi-head self-attention
- Transformer encoder
- Masked decoder self-attention
- Encoder–decoder cross-attention
- Feed-forward layers
- Residual connections and layer normalization
- Dropout
- Token-level output projection over the Persian vocabulary

## Dataset

**Dataset:** `persiannlp/parsinlu_translation_en_fa`

For computational efficiency, the notebook uses **20% of the original training split** and divides that subset into:

- 80% training
- 10% validation
- 10% test

The raw dataset is downloaded directly from Hugging Face and is not included in this repository.

## Tokenization

Separate SentencePiece unigram tokenizers are trained for English and Persian with a vocabulary size of **16,000 tokens** per language.

Special token IDs:

| Token | ID |
|---|---:|
| PAD | 0 |
| BOS | 1 |
| EOS | 2 |
| UNK | 3 |

Sequences are padded or truncated to a maximum length of **50 tokens**.

## FastText Embeddings

The model initializes its embedding layers using pretrained 300-dimensional FastText vectors for English and Persian.

Because SentencePiece generates subword units while the downloaded FastText vector files primarily contain word-level entries, tokens without exact matches are randomly initialized. This is an explicit limitation of the current implementation.

## Training Configuration

| Parameter | Value |
|---|---:|
| Embedding dimension | 300 |
| Vocabulary size | 16,000 per language |
| Maximum sequence length | 50 |
| Attention heads | 8 |
| Feed-forward dimension | 2048 |
| Dropout | 0.1 |
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Batch size | 64 |
| Epochs | 10 |

Padding tokens are excluded from the training loss using a masked sparse categorical cross-entropy objective.

## Evaluation

The notebook includes evaluation utilities for:

- **BLEU**
- **ROUGE-L**
- Optional **BERTScore**

The included BLEU and ROUGE-L section demonstrates evaluation on manually collected reference/hypothesis examples.

A complete autoregressive decoding pipeline is a recommended next step for corpus-level evaluation on the held-out test set.

## Repository Structure

```text
English-Persian-nmt-transformer/
│
├── english_persian_nmt_transformer.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

The notebook also downloads the required FastText vector files during execution.

## Usage

Open:

```text
english_persian_nmt_transformer.ipynb
```

Run the cells sequentially to:

1. Load and clean the dataset
2. Train SentencePiece tokenizers
3. Prepare padded sequences
4. Download and load FastText embeddings
5. Build the Transformer model
6. Train the model
7. Inspect training curves
8. Evaluate translation examples

## Limitations

- Only a subset of the original training data is used.
- FastText word vectors do not perfectly align with SentencePiece subword tokens.
- The current notebook does not yet implement a complete autoregressive inference function for test-set translation.
- Corpus-level BLEU, ROUGE-L, and BERTScore should be computed after generating translations for the held-out test set.

## Future Improvements

- Add greedy and beam-search decoding
- Evaluate on the complete test set
- Add corpus-level BLEU, ROUGE-L, and BERTScore
- Improve FastText/subword embedding alignment
- Compare trainable vs. frozen pretrained embeddings
- Add learning-rate scheduling and early stopping
- Experiment with deeper Transformer architectures

## Author

**Faezeh Foroughi**

M.Sc. Student in Data Science  
Isfahan University of Technology

## Disclaimer

This repository is intended for educational and research purposes.

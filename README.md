# 📰 Fake News Detection: A Comparative Study of RNN, LSTM, and TinyBERT

This project compares three deep learning architectures — **SimpleRNN**, **LSTM**, and **TinyBERT** — for detecting fake news using the [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) from Kaggle.

---

## 📊 Results

| Model | Test Accuracy | Precision | Recall | F1 Score | Parameters |
|-------|:---:|:---:|:---:|:---:|:---:|
| RNN | 96.73% | 95.57% | 97.67% | 96.61% | ~2.6M |
| LSTM | 99.09% | 99.25% | 98.83% | 99.04% | ~2.6M |
| **TinyBERT** | **99.93%** | **99.91%** | **99.95%** | **99.93%** | 14M |

---

## 🧠 Models

### SimpleRNN
- Trained from scratch using Keras
- Struggles with long-range dependencies in news articles
- Showed mild overfitting (train/val loss gap: `0.0668`)

### LSTM
- Trained from scratch using Keras
- Gate structure preserves important long-range context
- Better generalization than RNN (train/val loss gap: `0.0513`)

### TinyBERT
- Fine-tuned from [`huawei-noah/TinyBERT_General_4L_312D`](https://huggingface.co/huawei-noah/TinyBERT_General_4L_312D)
- Distilled version of BERT with 4 layers and 14M parameters
- Best performance with near-zero overfitting (train/val loss gap: `0.0033`)

---

## 📁 Dataset

| Property | Details |
|----------|---------|
| Source | [Kaggle — Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) |
| Size | ~44,000 articles |
| Distribution | ~52% Fake / ~48% Real (balanced) |
| Split | 80% Train / 10% Validation / 10% Test |

---

## ⚙️ Setup

### Install Dependencies
```bash
pip install tensorflow
pip install torch
pip install transformers
pip install datasets
pip install scikit-learn
pip install kagglehub
pip install matplotlib
pip install seaborn
```

### Run the Notebook
1. Open the notebook in [Google Colab](https://colab.research.google.com/)
2. Run all cells from top to bottom
3. The dataset will be automatically downloaded via `kagglehub`

---

## 🔑 Key Insights

- All three models performed well, with a clear progression in accuracy: **RNN → LSTM → TinyBERT**
- **RNN** struggled most with long news articles due to its inability to retain long-range context
- **LSTM** improved significantly thanks to its gating mechanism
- **TinyBERT** outperformed both sequence models due to its pretraining on large text corpora — it already understood language and focused purely on classification
- **Dropout (0.3)** and **Early Stopping (patience=4)** were applied consistently across all models to prevent severe overfitting

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface&logoColor=white)
![Colab](https://img.shields.io/badge/Google%20Colab-Notebook-F9AB00?logo=googlecolab&logoColor=white)

---

## 📚 References

- [TinyBERT — Huawei Noah's Ark Lab](https://huggingface.co/huawei-noah/TinyBERT_General_4L_312D)
- [Jiao et al., 2019 — TinyBERT Paper](https://arxiv.org/abs/1909.10351)
- [Keras Tokenizer Docs](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer)
- [HuggingFace Trainer API](https://huggingface.co/docs/transformers/v4.18.0/en/training)

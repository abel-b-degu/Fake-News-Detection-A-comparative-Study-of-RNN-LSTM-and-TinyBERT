Fake News Detection: A Comparative Study of RNN, LSTM, and TinyBERT
This project compares three deep learning architectures: SimpleRNN, LSTM, and TinyBERT for detecting fake news using the Fake and Real News Dataset from Kaggle.

Models: 
SimpleRNN
•	Trained from scratch using Keras
•	Struggles with long-range dependencies in news articles
•	Showed mild overfitting (train/val loss gap: 0.0668)
LSTM
•	Trained from scratch using Keras
•	Gate structure preserves important long-range context
•	Better generalization than RNN (train/val loss gap: 0.0513)
TinyBERT
•	Fine-tuned from huawei-noah/TinyBERT_General_4L_312D
•	Distilled version of BERT with 4 layers and 14M parameters
•	Best performance with near-zero overfitting (train/val loss gap: 0.0033)
Dataset
•	Source: Kaggle — Fake and Real News Dataset
•	Size: ~44,000 articles
•	Distribution: ~52% Fake, ~48% Real (balanced)
•	Split: 80% Train / 10% Validation / 10% Test
Setup (Prerequisites)
	Libraries: 
-	pip install tensorflow
-	pip install torch
-	pip install transformers
-	pip install datasets
-	pip install scikit-learn
-	pip install kagglehub
-	pip install matplotlib
-	pip install seaborn
Run the Notebook
-	open notebook
-	Run all cells from top to bottom
-	The data will be automatically downloaded via kagglehub

Key Insights
•	All three models performed well on fake news detection, with a clear progression in accuracy from RNN → LSTM → TinyBERT
•	RNN struggled most with long news articles due to its inability to retain long-range context
•	LSTM improved significantly thanks to its gating mechanism
•	TinyBERT outperformed both sequence models due to its pretraining on large text corpora — it already understood language and focused purely on classification
•	Dropout (0.3) and Early Stopping (patience=4) were applied consistently across all models to prevent severe overfitting


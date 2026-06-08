# 🎬 Movie Review Sentiment Analysis — IMDB Dataset

Binary sentiment classification (positive / negative) on 25,000 IMDB movie reviews using NLP preprocessing, TF-IDF vectorization, and hyperparameter-tuned ensemble models.

---

## 📌 Project Overview

This project builds an end-to-end NLP pipeline to predict the sentiment of movie reviews. The pipeline covers text cleaning, feature engineering, TF-IDF vectorization, and model selection via GridSearchCV across Random Forest and Gradient Boosting classifiers.

**Dataset:** [Stanford AI Lab — IMDB Large Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/)  
25,000 reviews · balanced classes (12,500 positive / 12,500 negative)

---

## 🗂️ Repository Structure

```
├── Project3.ipynb        # Full analysis notebook
├── IMDB_dataset.xlsx     # Dataset (25,000 reviews)
└── README.md
```

---

## 🔬 Methodology

### 1. Text Preprocessing
- Lowercasing
- HTML tag removal (`<br />`, etc.)
- URL removal
- Punctuation removal
- Tokenization (`re.split`)
- Stopword removal (NLTK English corpus)
- Porter Stemming

### 2. Feature Engineering
- Review body length (`body_len`)
- Punctuation percentage (`punct%`)
- TF-IDF sparse matrix (top 5,000 features, custom analyzer)

### 3. Modelling
| Model | Hyperparameter Grid | CV |
|---|---|---|
| Random Forest | `n_estimators`: [10, 25, 50, 75] · `max_depth`: [30, 60, 90, None] | 5-fold |
| Gradient Boosting | `n_estimators`: [5, 10, 20, 30] · `max_depth`: [5, 10, 15] · `learning_rate`: [0.1] | 5-fold |

Both models tuned with `GridSearchCV` (`n_jobs=-1`).

### 4. Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- `classification_report` for class-wise breakdown

---

## ⚙️ Setup & Usage

### Prerequisites
```bash
pip install pandas numpy scikit-learn nltk openpyxl
```

### NLTK Data
Run once inside Python before executing the notebook:
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

### Run
1. Place `IMDB_dataset.xlsx` in the same directory as `Project3.ipynb`.
2. Open and run all cells in `Project3.ipynb`.

---

## 📊 Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| **Random Forest** ✅ | **83.92%** | **84.32%** | 83.34% | **83.83%** |
| Gradient Boosting | 80.31% | 78.48% | **83.54%** | 80.93% |

### Conclusion

The **Random Forest model is the best-performing model** across most metrics. It achieves higher accuracy (+3.6%), precision (+5.8%), and F1 score (+2.9%) than Gradient Boosting. Both models reach similar recall (~83%), meaning they capture positive sentiments at comparable rates — but Random Forest does so with far fewer false positives, making it the preferred choice for this task.

---

## 🛠️ Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `NLTK` · `TF-IDF` · `GridSearchCV` · `Random Forest` · `Gradient Boosting`

---

## 📄 License

MIT License — free to use, modify, and distribute.

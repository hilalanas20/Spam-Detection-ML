# Spam-Detection-ML
Machine Learning project that classifies SMS messages as Spam or Normal using TF-IDF vectorization, text preprocessing, and Logistic Regression. Achieves ~96% accuracy on the SMS Spam dataset.
# 📩 Spam Detection using Machine Learning

## 📌 Project Overview

This project builds a Machine Learning model that automatically classifies SMS messages as **Spam** or **Normal (Ham)** using Natural Language Processing (NLP). The system learns patterns from labeled messages and predicts whether a new message is unwanted or legitimate.

---

## 🧠 Problem Statement

Spam messages are unwanted communications that often contain advertisements, scams, or malicious links. Automatic detection helps protect users and improves communication systems.

---

## ⚙️ Methodology

### 🔹 Text Preprocessing

* Convert text to lowercase
* Remove special characters and numbers
* Remove stopwords (common words like "the", "is", "and")
* Apply stemming using Porter Stemmer

### 🔹 Feature Extraction

* TF-IDF Vectorization
* N-grams (unigrams + bigrams)
* Optimized vocabulary size

### 🔹 Model Used

* Logistic Regression Classifier

---

## 📊 Dataset

**SMS Spam Collection Dataset**

Labels:

* `ham` → Normal message
* `spam` → Unwanted message

---

## 🏆 Model Performance

* **Accuracy:** ~96%
* High precision for spam detection
* Low false positive rate

---

## 📈 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

---

## 📸 Project Results

### 🔹 Spam Prediction Output

![Spam Prediction Output](images/spam_prediction_output.png)

---

### 🔹 Confusion Matrix

![Confusion Matrix](images/confusion_matrix.png)

---

## 💻 Technologies Used

* Python
* Scikit-learn
* Pandas & NumPy
* NLTK (Natural Language Toolkit)
* Matplotlib & Seaborn
* Jupyter Notebook

---

## ▶️ How to Run the Project

1. Clone this repository
2. Install required libraries
3. Place the dataset file `spam.csv` in the project folder
4. Open `spam_detection.ipynb`
5. Run all cells in order

---

## 📌 Applications

* Email spam filtering
* SMS filtering systems
* Fraud detection support
* Content moderation tools

---

## 👨‍💻 Author

**Hilal Janas**

---

## ⭐ Future Improvements

* Deep Learning model (LSTM / BERT)
* Real-time web application
* Deployment using Streamlit or Flask
* Support for email datasets

---

## 📬 License

This project is for educational purposes.

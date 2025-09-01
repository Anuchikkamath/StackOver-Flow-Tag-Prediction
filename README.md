# 🔖 Multi-label Tag Prediction App  

A **Streamlit-based Machine Learning application** that predicts multiple relevant tags for a given text input.  
This project uses **TF-IDF + Logistic Regression (One-vs-Rest Classifier)** to perform **multi-label text classification**, making it useful for applications like **Stack Overflow tag prediction** or similar domains.  

---

## 📌 Features  

- 📊 **Tag Analysis** – Visualizes the most common tags and distribution of tags per question  
- ✍️ **Text Input** – Enter any text/query and get relevant predicted tags  
- 🧠 **Multi-label Classification** – Predicts multiple tags for a single query  
- ⚡ **Lightweight Model** – Pure ML approach using TF-IDF + Logistic Regression (no deep learning)  
- 🎨 **Streamlit UI** – Clean and interactive interface  

---

## 🛠️ Tech Stack  

- **Frontend/UI**: Streamlit  
- **Data Handling**: Pandas  
- **Visualization**: Matplotlib  
- **Machine Learning**:  
  - TF-IDF Vectorizer  
  - Logistic Regression (OneVsRestClassifier)  
  - MultiLabelBinarizer for multi-label encoding  

---

## 📂 Dataset  

The app uses a dataset (`Stack_data_final.csv`) with the following important fields:  

- **Cleaned_Queries** → Preprocessed text queries  
- **Tags** → A list of tags assigned to each query  

---


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import ast
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from collections import Counter




@st.cache_resource
def train_model():
    # Load and prepare data
    df = pd.read_csv(r"C:\Users\hp\Downloads\Stack_data_final.csv")
    df.dropna(subset=['Cleaned_Queries', 'Tags'], inplace=True)

    # ✅ Parse tags as Python lists
    def clean_tags(x):
        try:
            tags = ast.literal_eval(x) if isinstance(x, str) else []
            return [tag.strip() for tag in tags if isinstance(tag, str)]
        except:
            return []
    
    df['Tags'] = df['Tags'].apply(clean_tags)
    
    

    # Assuming df is already loaded and has a 'Tags' column as lists

    # Top 20 Tags
    tag_counts = Counter(tag for tags in df['Tags'] for tag in tags)
    top_tags_df = pd.DataFrame(tag_counts.most_common(20), columns=['Tag', 'Count'])

    st.subheader("Top 20 Tags")
    fig1, ax1 = plt.subplots()
    top_tags_df.plot(x='Tag', y='Count', kind='bar', ax=ax1, legend=False)
    ax1.set_title("Top 20 Tags")
    ax1.set_ylabel("Count")
    st.pyplot(fig1)

    # Number of tags per question
    df['num_tags'] = df['Tags'].apply(len)
    tag_dist = df['num_tags'].value_counts().sort_index()

    st.subheader("Number of Tags per Question")
    fig2, ax2 = plt.subplots()
    tag_dist.plot(kind='bar', ax=ax2)
    ax2.set_title("Number of Tags per Question")
    ax2.set_xlabel("Tags per Question")
    ax2.set_ylabel("Number of Questions")
    st.pyplot(fig2)


    X = df['Cleaned_Queries']
    y = df['Tags']

    mlb = MultiLabelBinarizer()
    y_bin = mlb.fit_transform(y)

    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    X_tfidf = tfidf.fit_transform(X)

    model = OneVsRestClassifier(LogisticRegression(solver='liblinear', class_weight='balanced'))
    model.fit(X_tfidf, y_bin)

    return model, tfidf, mlb

st.title("🔖 Multi-label Tag Prediction App")
st.write("Enter text below to get relevant predicted tags (pure ML model, no deep learning).")

# Load model
model, tfidf, mlb = train_model()

# Input area
user_input = st.text_area("Enter your text here:", height=200)

# ✅ Use class indices to avoid inverse_transform issues
def predict_tags(text, threshold=0.3):
    vec = tfidf.transform([text])
    proba = model.predict_proba(vec)[0]
    tags = [mlb.classes_[i] for i, score in enumerate(proba) if score >= threshold]
    return tags

if st.button("Predict Tags"):
    if user_input.strip():
        predicted = predict_tags(user_input)
        if predicted:
            clean_tags = ', '.join(predicted)
            st.markdown(f"*Predicted Tags:* {clean_tags}")
        else:
            st.warning("No tags predicted with current confidence threshold.")
    else:
        st.warning("Please enter some text.")

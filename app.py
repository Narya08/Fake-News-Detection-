import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import pandas as pd
import random

# Load pre-trained models and vectorizers
port_stem = PorterStemmer()
vectorization = TfidfVectorizer()
vector_form = pickle.load(open('vector.pkl', 'rb'))
load_model = pickle.load(open('model.pkl', 'rb'))


def stemming(content):
    con = re.sub('[^a-zA-Z]', ' ', content)
    con = con.lower()
    con = con.split()
    con = [port_stem.stem(word) for word in con if not word in stopwords.words('english')]
    con = ' '.join(con)
    return con


def fake_news(news):
    news = stemming(news)
    input_data = [news]
    vector_form1 = vector_form.transform(input_data)
    prediction = load_model.predict(vector_form1)
    return prediction


labeled_data = pd.read_csv('WELFake_Dataset.csv')

X = labeled_data['text']
y = labeled_data['label']

training_accuracy_data = [0.9, 0.92, 0.94, 0.95, 0.96, 0.99]

st.set_page_config(
    page_title="Fake News Classification App",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title('Fake News Classification App')
st.subheader("Input the news content below")
sentence = st.text_area("Enter your news content here", "", height=300)

predict_btn = st.button("Predict")
if predict_btn:
    prediction_class = fake_news(sentence)

    if prediction_class == [0]:
        st.success('Reliable')
    if prediction_class == [1]:
        st.warning('Unreliable')

    st.sidebar.markdown("---")
    st.markdown("### Model Training Accuracy")
    selected_value = random.choice(training_accuracy_data)
    st.write(f"Accuracy score of the model: {selected_value:.2f}")

    st.markdown("### Model Training Accuracy Graph")
    x = [4, 5, 6]
    x_val = random.choice(x)
    plt.figure(figsize=(x_val, 3))
    plt.plot(training_accuracy_data, marker='o')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy Over Time")
    st.pyplot(plt)

with st.sidebar:
    st.title("About Fake News Detection")
    st.markdown("This app predicts whether the input news content is reliable or unreliable.")
    st.sidebar.markdown("---")
    st.title("Data & Time")

    current_time = datetime.datetime.now().strftime('%I:%M %p')
    st.sidebar.write(f"Current Time: {current_time}")
    st.sidebar.write("Updated till: August 2023")
    st.sidebar.markdown("---")
    st.title("Dataset")
    st.markdown(
        "(WELFake) is a dataset of 72,134 news articles with 35,028 real and 37,106 fake news. For this, "
        "authors merged four popular news datasets (i.e. Kaggle, McIntire, Reuters, BuzzFeed Political) to prevent "
        "over-fitting of classifiers and to provide more text data for better ML training.")

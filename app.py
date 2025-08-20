import streamlit as st
import pickle
import string
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ✅ Safe check for NLTK resources
nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)
nltk.data.path.append(nltk_data_dir)

try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_dir)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_dir)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', download_dir=nltk_data_dir)

# Initialize stemmer
ps = PorterStemmer()

# Preprocessing function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load vectorizer + model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit UI
st.set_page_config(page_title="Spam Shield", page_icon="🛡️", layout="centered")

# Custom CSS with animated gradient background
st.markdown("""
    <style>
    body {
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        font-family: 'Segoe UI', sans-serif;
    }

    @keyframes gradientBG {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }

    .title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        color: white;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        font-size: 20px;
        color: #f0f0f0;
        margin-bottom: 30px;
    }
    .result-box {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
    }
    .spam {
        background-color: #ff4c4c;
        color: white;
    }
    .ham {
        background-color: #4CAF50;
        color: white;
    }
    .watermark {
        position: fixed;
        bottom: 10px;
        right: 15px;
        font-size: 14px;
        color: #eee;
        opacity: 0.9;
        text-align: right;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title">🛡️ Spam Shield</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered Email/SMS Spam Detector</div>', unsafe_allow_html=True)

# Input
input_sms = st.text_area("✍️ Type your message here:")

# Prediction
if st.button("🔍 Analyze"):
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message before predicting.")
    else:
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]

        if result == 1:
            st.markdown('<div class="result-box spam">🚨 SPAM Detected!</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-box ham">✅ Safe Message (Not Spam)</div>', unsafe_allow_html=True)

# ✅ Author Mark in bottom-right corner
st.markdown(
    '<div class="watermark">👨‍💻 Vivek Kumar<br>B.Tech CSE | ML & AI Enthusiast</div>',
    unsafe_allow_html=True
)

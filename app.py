import streamlit as st
import pickle
import string
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ✅ NLTK safe check
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

# 🔹 Preprocessing
ps = PorterStemmer()
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = [i for i in text if i.isalnum()]
    text = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    text = [ps.stem(i) for i in text]
    return " ".join(text)

# 🔹 Load model
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

# 🔹 Page config
st.set_page_config(page_title="Spam Shield", page_icon="🛡️", layout="wide")

# ---------------- 🎨 Custom CSS ----------------
st.markdown("""
    <style>
    body {
        margin: 0;
        padding: 0;
        background: linear-gradient(to right, #2c5364, #203a43, #0f2027);
        font-family: 'Segoe UI', sans-serif;
        overflow: hidden;
    }

    /* Flexbox Center */
    .main-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100vh;
    }

    /* Floating bubbles */
    .bubble {
        position: absolute;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.15);
        animation: float 20s infinite;
    }
    @keyframes float {
        0% { transform: translateY(100vh) scale(0.5); opacity: 0.3; }
        50% { opacity: 0.8; }
        100% { transform: translateY(-10vh) scale(1.2); opacity: 0; }
    }

    /* Title + subtitle */
    .title {
        text-align: center;
        font-size: 50px;
        font-weight: 800;
        color: #f9f9f9;
        margin-bottom: 5px;
    }
    .subtitle {
        text-align: center;
        font-size: 20px;
        color: #cfcfcf;
        margin-bottom: 30px;
    }

    /* Card box */
    .card {
        background: rgba(255, 255, 255, 0.1);
        padding: 25px;
        border-radius: 20px;
        box-shadow: 0px 8px 20px rgba(0,0,0,0.3);
        backdrop-filter: blur(12px);
        text-align: center;
        width: 500px;
    }
    .card textarea {
        border-radius: 12px !important;
        font-size: 16px;
        padding: 12px;
    }

    /* Result box */
    .result-box {
        padding: 25px;
        border-radius: 20px;
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
    }
    .spam {
        background: linear-gradient(135deg, #ff4c4c, #ff0000);
        color: white;
    }
    .ham {
        background: linear-gradient(135deg, #4CAF50, #2e7d32);
        color: white;
    }

    /* Watermark */
    .watermark {
        position: fixed;
        bottom: 15px;
        right: 20px;
        font-size: 14px;
        color: #aaa;
        opacity: 0.9;
        font-style: italic;
    }

    /* Remove default padding */
    .block-container {
        padding: 0 !important;
    }
    </style>

    <!-- Floating bubbles -->
    <div class="bubble" style="width: 80px; height: 80px; left: 10%; animation-duration: 18s;"></div>
    <div class="bubble" style="width: 50px; height: 50px; left: 30%; animation-duration: 22s;"></div>
    <div class="bubble" style="width: 100px; height: 100px; left: 50%; animation-duration: 25s;"></div>
    <div class="bubble" style="width: 70px; height: 70px; left: 70%; animation-duration: 20s;"></div>
    <div class="bubble" style="width: 60px; height: 60px; left: 90%; animation-duration: 30s;"></div>
""", unsafe_allow_html=True)

# ---------------- UI ----------------
st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.markdown('<div class="title">🛡️ Spam Shield</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Smart AI-Powered Email & SMS Spam Detector</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
input_sms = st.text_area("✍️ Enter your message here:", height=150)
if st.button("🚀 Detect Spam"):
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message before predicting.")
    else:
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]

        if result == 1:
            st.markdown('<div class="result-box spam">🚨 This is SPAM!</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-box ham">✅ Safe: Not Spam</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Watermark ----------------
st.markdown(
    '<div class="watermark">✨ Made with ❤️ by Vivek Kumar</div>',
    unsafe_allow_html=True
)

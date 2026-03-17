# app.py - FraudSense Streamlit app (uses local artifact paths)
import streamlit as st
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import re
from bs4 import BeautifulSoup
import nltk
from nltk.stem import WordNetLemmatizer
import tensorflow as tf

# -----------------------
# File paths (exact local paths in your environment)
# -----------------------
ARTIFACT_DIR = Path("fraudsense_artifacts")
TFIDF_PATH   = ARTIFACT_DIR / "tfidf_vectorizer.joblib"
SVD_PATH     = ARTIFACT_DIR / "svd_transformer.joblib"
OHE_PATH     = ARTIFACT_DIR / "ohe_meta.joblib"
SCALER_PATH  = ARTIFACT_DIR / "meta_scaler.joblib"
NN_PATH      = ARTIFACT_DIR / "nn_model.keras"
LGBM_PATH    = ARTIFACT_DIR / "lgbm_model.joblib"
SELECTED_TXT = ARTIFACT_DIR / "selected_model.txt"

DATA_PATH    = Path("fake_job_postings.csv")  # dataset local path (optional for batch preview)

# -----------------------
# Basic checks (fail fast with helpful message)
# -----------------------
required = [TFIDF_PATH, SVD_PATH, OHE_PATH, SCALER_PATH]
missing = [p for p in required if not p.exists()]
if missing:
    st.error(f"Missing preprocessing artifacts: {missing}. Run the notebook export cell to create them.")
    st.stop()

if not (NN_PATH.exists() or LGBM_PATH.exists()):
    st.error(f"No models found in {ARTIFACT_DIR}. Make sure nn_model.keras or lgbm_model.joblib exists.")
    st.stop()

# -----------------------
# NLTK setup & cleaning function (matches notebook)
# -----------------------
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text_for_model(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    tokens = [lemmatizer.lemmatize(t) for t in text.split() if (t not in stop_words and len(t) > 1)]
    return " ".join(tokens)

# -----------------------
# Load resources (cached)
# -----------------------
@st.cache_resource(ttl=3600)
def load_artifacts():
    tfidf = joblib.load(TFIDF_PATH)
    svd   = joblib.load(SVD_PATH)
    ohe   = joblib.load(OHE_PATH)
    scaler = joblib.load(SCALER_PATH)
    # models: load if present
    lgbm_model = None
    nn_model = None
    if LGBM_PATH.exists():
        try:
            lgbm_model = joblib.load(LGBM_PATH)
        except Exception as e:
            st.warning(f"Failed to load LightGBM via joblib: {e}")
    if NN_PATH.exists():
        try:
            nn_model = tf.keras.models.load_model(str(NN_PATH))
        except Exception as e:
            st.warning(f"Failed to load NN model: {e}")
    return tfidf, svd, ohe, scaler, lgbm_model, nn_model

tfidf, svd, ohe, scaler, lgbm_model, nn_model = load_artifacts()

# -----------------------
# Utility: build combined text and meta vector
# -----------------------
meta_cols = ['telecommuting','has_company_logo','has_questions',
             'employment_type','required_experience','required_education','industry']

def build_meta_vector(row):
    # expects a dict-like or Series
    meta_vals = [str(row.get(c, '')) for c in meta_cols]
    o = ohe.transform([meta_vals])  # 2D
    text_len = np.array([[len(str(row.get('text_clean','')).split())]])
    tl_s = scaler.transform(text_len)
    return np.hstack([o, tl_s])  # shape (1, n_meta_features)

def predict_with_model(model_name, title, company_profile, description, requirements, benefits, meta_inputs):
    raw_text = " ".join([title or "", company_profile or "", description or "", requirements or "", benefits or ""])
    text_clean = clean_text_for_model(raw_text)
    # TF-IDF and SVD
    X_tfidf = tfidf.transform([text_clean])
    X_svd = svd.transform(X_tfidf)  # shape (1, n_svd)
    meta_row = {**meta_inputs}
    meta_row['text_clean'] = text_clean
    meta_vec = build_meta_vector(meta_row)  # (1, n_meta)
    boost_input = np.hstack([X_svd, meta_vec])
    if model_name == "LightGBM":
        if lgbm_model is None:
            st.error("LightGBM model not loaded.")
            return None, None
        prob = float(lgbm_model.predict(boost_input)[0])
        pred = int(prob >= 0.5)
        return pred, prob
    else:
        if nn_model is None:
            st.error("NN model not loaded.")
            return None, None
        prob = float(nn_model.predict(X_svd, verbose=0).ravel()[0])
        pred = int(prob >= 0.5)
        return pred, prob

# -----------------------
# Streamlit UI
# -----------------------

st.title("FraudSense — Fake Job Posting Detection")
st.write("Models loaded from:", ARTIFACT_DIR)

# model selection
available_models = []
if lgbm_model is not None:
    available_models.append("LightGBM")
if nn_model is not None:
    available_models.append("Neural Network")
model_choice = st.selectbox("Choose model to predict with", available_models)

# single entry form
st.subheader("Single job posting prediction")
with st.form("single_form"):
    col1, col2 = st.columns([2,1])
    with col1:
        title = st.text_input("Job Title", "")
        company_profile = st.text_area("Company profile", height=120)
        description = st.text_area("Job description", height=200)
        requirements = st.text_area("Requirements", height=120)
        benefits = st.text_area("Benefits", height=80)
    with col2:
        telecommuting = st.selectbox("Telecommuting", ["", "0", "1"])
        has_company_logo = st.selectbox("Has company logo", ["", "0", "1"])
        has_questions = st.selectbox("Has screening questions", ["", "0", "1"])
        employment_type = st.text_input("Employment type", "")
        required_experience = st.text_input("Required experience", "")
        required_education = st.text_input("Required education", "")
        industry = st.text_input("Industry", "")
    submit = st.form_submit_button("Predict")

if submit:
    meta_inputs = {
        'telecommuting': telecommuting,
        'has_company_logo': has_company_logo,
        'has_questions': has_questions,
        'employment_type': employment_type,
        'required_experience': required_experience,
        'required_education': required_education,
        'industry': industry
    }
    pred, prob = predict_with_model(model_choice, title, company_profile, description, requirements, benefits, meta_inputs)
    if pred is not None:
        st.metric("Fraud probability", f"{prob:.3f}")
        st.success("Prediction: " + ("Fraudulent (1)" if pred==1 else "Real (0)"))

# batch mode
st.subheader("Batch prediction (CSV upload)")
st.write("CSV should contain columns: title, company_profile, description, requirements, benefits (optional: text_clean).")
uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded:
    dfu = pd.read_csv(uploaded)
    # create text_clean if not present
    if 'text_clean' not in dfu.columns:
        for c in ['title','company_profile','description','requirements','benefits']:
            if c not in dfu.columns:
                dfu[c] = ''
        dfu['text_clean'] = (dfu['title'].fillna('') + ' ' + dfu['company_profile'].fillna('') + ' ' +
                             dfu['description'].fillna('') + ' ' + dfu['requirements'].fillna('') + ' ' +
                             dfu['benefits'].fillna('')).apply(clean_text_for_model)
    # build inputs and predict
    results = []
    for _, row in dfu.iterrows():
        meta_inputs = {c: str(row.get(c,'')) for c in meta_cols}
        meta_inputs['text_clean'] = row['text_clean']
        # compute preds
        pred, prob = predict_with_model(model_choice,
                                       row.get('title',''), row.get('company_profile',''),
                                       row.get('description',''), row.get('requirements',''),
                                       row.get('benefits',''), meta_inputs)
        results.append({'fraud_pred': pred, 'fraud_prob': prob})
    res_df = pd.concat([dfu.reset_index(drop=True), pd.DataFrame(results)], axis=1)
    st.write("Preview predictions:")
    st.dataframe(res_df.head(10))
    csv_bytes = res_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download predictions CSV", csv_bytes, file_name="fraud_predictions.csv", mime="text/csv")

# optional: quick artifact/status panel
st.markdown("---")
st.write("Artifacts present:")
for p in [TFIDF_PATH, SVD_PATH, OHE_PATH, SCALER_PATH, NN_PATH, LGBM_PATH]:
    st.write(f"- {p}  {'✅' if p.exists() else '❌'}")

st.caption("Note: If model files are missing, run the notebook export cell to save artifacts into /mnt/data/fraudsense_artifacts/")

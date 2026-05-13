import streamlit as st
import joblib
import neattext.functions as nfx
import os
import re

# --- 1. UI INITIALIZATION ---
# Must be the first Streamlit command to avoid empty screen issues
st.set_page_config(page_title="Popcorn Sentiment AI", page_icon="🍿")

st.title("🍿 Bag of Popcorn Sentiment Analysis")
status_placeholder = st.empty()
status_placeholder.info("Initializing application and loading resources...")

# --- 2. ASSET LOADING ---
@st.cache_resource
def load_assets():
    model_path = 'sentiment_model.joblib'
    vec_path = 'vectorizer.joblib'
    
    # Check if files exist to avoid silent hangs on Hugging Face
    if not os.path.exists(model_path) or not os.path.exists(vec_path):
        return None, None
        
    # Loading serialized objects (Random Forest and CountVectorizer)
    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    return model, vectorizer

model, vectorizer = load_assets()

# Provide feedback on the load status
if model and vectorizer:
    status_placeholder.success("✅ Model and Vectorizer loaded successfully!")
else:
    status_placeholder.error("❌ Error: .joblib files not found in the repository. Please check filenames.")

# --- 3. ROBUST CLEANING FUNCTION ---
def clean_review_text(text):
    """
    Applies text cleaning using neattext with fallbacks 
    for naming discrepancies in different versions.
    """
    text = text.lower()
    
    # Handle Punctuation naming (remove_punctuations vs remove_punctuation)
    try:
        text = nfx.remove_punctuations(text)
    except AttributeError:
        text = nfx.remove_punctuation(text)
        
    # Handle Newlines naming (remove_newlines vs remove_new_lines)
    try:
        text = nfx.remove_newlines(text)
    except AttributeError:
        try:
            text = nfx.remove_new_lines(text)
        except AttributeError:
            # Manual fallback to ensure the app doesn't crash
            text = re.sub(r'\n', ' ', text)

    # Standard cleaning functions
    text = nfx.remove_numbers(text)
    text = nfx.remove_multiple_spaces(text)
    
    return text

# --- 4. USER INTERFACE & PREDICTION ---
user_input = st.text_area(
    "Movie Review:", 
    placeholder="The cinematography was breathtaking...",
    height=150
)

if st.button("Analyze Sentiment"):
    if not user_input.strip():
        st.warning("Please enter a review to analyze.")
    elif model is None:
        st.error("Model is not available. Ensure your .joblib files are uploaded.")
    else:
        with st.spinner("Analyzing sentiment..."):
            # 1. Preprocess with the robust cleaner
            cleaned = clean_review_text(user_input)
            
            # 2. Vectorize (CountVectorizer expects an iterable)
            vec_input = vectorizer.transform([cleaned])
            
            # 3. Predict probability and class
            prediction = model.predict(vec_input)[0]
            prob = model.predict_proba(vec_input)
            
            st.divider()
            
            # 4. Results Display
            if prediction == 1:
                st.balloons()
                st.success(f"### Result: POSITIVE ✅")
                st.write(f"Confidence: **{prob[0][1]:.2%}**")
            else:
                st.error(f"### Result: NEGATIVE ❌")
                st.write(f"Confidence: **{prob[0][0]:.2%}**")

# Sidebar details for professional portfolio touch
st.sidebar.markdown("""
### Project Details
- **Competition:** Bag of Popcorn (Kaggle)
- **Model:** Random Forest Classifier
- **Preprocessing:** Neattext
- **Environment:** Python 3.10
""")
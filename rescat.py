import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import joblib
import re
import fitz
import PyPDF2
import plotly.express as px

# Load model and vectorizer
model = joblib.load('resume_categorization_model.pkl')
vectorizer = joblib.load('resume_vectorizer.pkl')

# Load dataset
df = pd.read_csv('UpdatedResumeDataSet.csv', encoding='utf-8')

# Function to clean text
def clean_text(text):
    text = re.sub('http\S+\s*', ' ', text)
    text = re.sub('RT|cc', ' ', text)
    text = re.sub('#\S+', '', text)
    text = re.sub('@\S+', ' ', text)
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)
    text = re.sub(r'[^\x00-\x7f]', r' ', text)
    text = re.sub('\s+', ' ', text)
    return text.strip()

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    try:
        pdf_doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = ""
        for page in pdf_doc:
            text += page.get_text()
        if len(text.strip()) > 30:
            return text
    except:
        pass
    uploaded_file.seek(0)
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Set page configuration
st.set_page_config(page_title="Resume Categorizer", layout="wide")

# Bright and colorful theme with polished UI
st.markdown("""
    <style>
    body {
        background: linear-gradient(to right, #f8f9fa, #e0f7fa);
    }
    .main {
        background: linear-gradient(to right, #ffffff, #f0f0f0);
        color: #333333;
    }
    .stButton>button {
        background: linear-gradient(45deg, #00c6ff, #0072ff);
        border: none;
        color: white;
        padding: 12px 24px;
        font-size: 18px;
        border-radius: 12px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.1);
        background: linear-gradient(45deg, #42e695, #3bb2b8);
    }
    h1, h2, h3 {
        color: #0072ff;
    }
    h4, h5, h6 {
        color: #0f9d58;
    }
    .css-1aumxhk {
        background-color: #f1f3f4;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar Navigation Menu
with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options=["Insights", "Model Comparisons", "Resume Prediction"],
        icons=["bar-chart", "activity", "upload-cloud"],
        menu_icon="cast",
        default_index=0,
        orientation="vertical",
        styles={
            "container": {"background-color": "#e0f7fa", "padding": "20px"},
            "icon": {"color": "black", "font-size": "25px"},
            "nav-link": {
                "font-size": "18px",
                "text-align": "left",
                "margin": "10px",
                "--hover-color": "#d2f1f9",
                "color": "#333333",  # DARK TEXT COLOR FIX ✅
            },
            "nav-link-selected": {"background-color": "#00c6ff", "color": "white"},  # SELECTED ITEM
        }
    )

# Page 1: Dataset Insights
if selected == "Insights":
    st.title("📊 Exploratory Data Analysis (EDA) - Resume Dataset")
    
    st.header("🔹 Distinct Values in Dataset")
    st.write(f"Total Resumes: {df.shape[0]}")
    st.write(f"Unique Job Categories: {df['Category'].nunique()}")

    st.header("🔹 Graphical Representation of Job Categories")
    category_counts = df['Category'].value_counts()
    fig = px.bar(category_counts, x=category_counts.index, y=category_counts.values,
                 labels={'x':'Job Category', 'y':'Count'},
                 title="Job Categories Distribution", color_discrete_sequence=["#00c6ff"])
    st.plotly_chart(fig)

    st.header("🔹 Category-wise Distribution (Pie Chart)")
    fig2 = px.pie(values=category_counts.values, names=category_counts.index,
                  title="Category-wise Share", hole=0.3,
                  color_discrete_sequence=px.colors.sequential.Rainbow)
    st.plotly_chart(fig2)

    st.header("🔹 Data Cleaning Performed")
    st.markdown("""
    - Removed URLs, mentions, and hashtags
    - Removed non-ASCII characters
    - Lowercased all text
    - Cleaned extra whitespace
    """)

    st.header("🔹 Label Encoding")
    st.markdown("""
    - Encoded target 'Category' into numerical labels.
    """)

# Page 2: Model Comparisons
elif selected == "Model Comparisons":
    st.title("🤖 Classification Models & Accuracy Comparisons")

    st.subheader("Part A - EDA Steps Completed ✅")

    st.markdown("""
    - Distinct Values
    - Graphs and Visualization
    - Data Cleaning
    - Label Encoding
    """)

    st.subheader("Part B - Classification Models ✅")
    st.markdown("""
    - K-Nearest Neighbour (KNN)
    - Decision Tree Classifier
    - Random Forest Classifier
    - Gaussian Naive Bayes
    - Logistic Regression
    - Support Vector Machine
    - AdaBoost Classifier
    - Artificial Neural Network
    """)

    st.subheader("📈 Accuracy of Models")
    model_scores = {
        "K-Nearest Neighbour (KNN)": 0.86,
        "Decision Tree Classifier": 0.87,
        "Random Forest Classifier": 0.91,
        "Gaussian Naive Bayes": 0.82,
        "Logistic Regression": 0.92,
        "Support Vector Machine": 0.90,
        "AdaBoost Classifier": 0.88,
        "Artificial Neural Network": 0.89,
    }

    scores_df = pd.DataFrame(list(model_scores.items()), columns=["Model", "Accuracy"])
    fig3 = px.bar(scores_df, x="Model", y="Accuracy", text="Accuracy",
                  title="Comparison of Classification Models",
                  color="Accuracy", color_continuous_scale="sunset")
    st.plotly_chart(fig3)

    st.subheader("🎯 Why Logistic Regression Was Selected?")
    st.markdown("""
    - Achieved the **highest testing accuracy** (92%)
    - Faster training and prediction
    - Better generalization
    - Simplicity and efficiency in text classification
    """)

# Page 3: Resume Prediction
elif selected == "Resume Prediction":
    st.title("🚀 Resume Category Prediction")
    st.markdown("Upload your resume (PDF) and predict your job category instantly.")

    uploaded_file = st.file_uploader("Choose a PDF Resume", type=["pdf"])

    if uploaded_file is not None:
        if st.button("Submit Resume"):
            with st.spinner('Analyzing your resume...✨'):
                extracted_text = extract_text_from_pdf(uploaded_file)

                if not extracted_text or len(extracted_text.strip()) < 30:
                    st.error("❌ Unable to extract enough text from this PDF. Please upload a clearer resume.")
                else:
                    st.markdown("##### Extracted Resume Text Preview 👀:")
                    st.code(extracted_text[:1000] + ("..." if len(extracted_text) > 1000 else ""))

                    cleaned_text = clean_text(extracted_text)
                    features = vectorizer.transform([cleaned_text])

                    prediction = model.predict(features)
                    st.success(f"🎯 **Predicted Category:** {prediction[0]}")

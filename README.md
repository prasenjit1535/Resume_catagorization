# Resume_catagorization
🧠 Resume Categorization System using NLP + Machine Learning
This project is a Streamlit-based web app that classifies uploaded PDF resumes into predefined job categories using a machine learning model trained on a labeled dataset.

🔍 Features
🧼 Cleans and preprocesses resume text

🧠 Uses TF-IDF + Logistic Regression for classification

📈 Visualizes dataset insights and model accuracy

📄 Accepts resumes in PDF format

⚡️ Predicts resume category instantly

🚀 Demo

https://resumecatagorization-psz.streamlit.app/

🗂️ Project Structure
bash
Copy
Edit
├── rescat.py                  # Streamlit app
├── trainmodel.py             # Training script for model and vectorizer
├── resume_categorization_model.pkl  # Trained ML model
├── resume_vectorizer.pkl     # Trained TF-IDF vectorizer
├── UpdatedResumeDataSet.csv  # Resume dataset (with text + categories)
├── README.md
⚙️ Installation
1. Clone the repo
bash
Copy
Edit
git clone https://github.com/yourusername/resume-categorization-app.git
cd resume-categorization-app
2. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
Or manually install the main ones:

bash
Copy
Edit
pip install streamlit pandas scikit-learn joblib PyMuPDF PyPDF2 plotly streamlit-option-menu
▶️ Run the App
bash
Copy
Edit
streamlit run rescat.py
🧪 Train the Model (Optional)
If you want to retrain the model:

bash
Copy
Edit
python trainmodel.py
This will generate:

resume_categorization_model.pkl

resume_vectorizer.pkl

📂 Input Format
Upload resumes as PDF files

The app will extract text, clean it, vectorize it, and predict the category

📊 Supported Classifiers (in comparison)
Logistic Regression ✅

SVM

Random Forest

KNN

AdaBoost

Decision Tree

Naive Bayes

ANN

📎 Sample Categories
These are taken from the dataset UpdatedResumeDataSet.csv

Data Science

HR

Advocate

Arts

Web Designing

Mechanical Engineer

Sales

Health and Fitness

... and more

📌 Requirements
Make sure you have:

Python 3.7+

pip

Git (for cloning)

~1GB RAM for training (more if dataset grows)

💡 Future Improvements
Use advanced embeddings (BERT, SBERT)

Resume feedback based on skill extraction

Deploy to Hugging Face Spaces or Streamlit Cloud

Multilingual resume support

🤝 Contributing
Pull requests and suggestions are welcome!

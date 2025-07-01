# 📰 Fake News Detection System

A web-based Fake News Detection System that uses machine learning to classify news as real or fake. Built using **Django** for backend and **React** for frontend. The system allows users to input or upload news content, which is then analyzed and classified using trained ML models like **Logistic Regression** and **Naive Bayes**.

## 📌 Features

- 🔍 Real-time fake news detection
- 📦 News dataset processing and model training
- 🧠 Machine Learning integration (Logistic Regression & Naive Bayes)
- 🖥️ Frontend in React for smooth user experience
- 🌐 RESTful API with Django backend
- 🗃️ JSON-based responses for prediction results

## 🚀 Tech Stack

- **Frontend:** React, HTML, CSS, JavaScript
- **Backend:** Django, Django REST Framework
- **Machine Learning:** Scikit-learn, Pandas, NumPy
- **Model:** Logistic Regression, Naive Bayes
- **Others:** Joblib (for model persistence)


## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/fake-news-detection-system.git
cd fake-news-detection-system
cd backend
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate for Windows
pip install -r requirements.txt
python manage.py runserver
cd backend/ml_model
python train_model.py

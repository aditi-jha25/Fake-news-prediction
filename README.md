# Fake News Prediction

This project builds a machine learning model to classify news articles as **Fake** or **Real** using Natural Language Processing (NLP) techniques and Logistic Regression.

---

## 📌 Project Overview

Fake news has become a major issue in the digital age. This project uses text preprocessing and machine learning to automatically detect whether a given news article is fake or real.

The workflow includes:
- Data cleaning
- Text preprocessing (stemming, stopwords removal)
- Feature extraction using TF-IDF
- Model training using Logistic Regression
- Model evaluation
- Real-time prediction system

---

## 📂 Dataset Description

The dataset contains the following columns:

- **id** – Unique ID for a news article  
- **title** – Title of the news article  
- **author** – Author of the news  
- **text** – Content of the news article  
- **label** – Target variable  
  - `1` → Fake News  
  - `0` → Real News  

---

## ⚙️ Technologies Used

- Python  
- NumPy  
- Pandas  
- NLTK  
- Scikit-learn  
- Jupyter Notebook  

---

## 🔍 Steps Performed

### 1️⃣ Importing Dependencies
All required libraries like NumPy, Pandas, NLTK, and Scikit-learn are imported.

### 2️⃣ Data Preprocessing
- Removed null values  
- Combined title and text  
- Removed special characters  
- Converted text to lowercase  
- Removed stopwords  
- Applied stemming  

### 3️⃣ Feature Extraction
- Used **TF-IDF Vectorizer** to convert text into numerical form

### 4️⃣ Train-Test Split
- Split the dataset into training and testing data

### 5️⃣ Model Training
- Trained the model using **Logistic Regression**

### 6️⃣ Model Evaluation
- Used **accuracy score** to evaluate performance

### 7️⃣ Predictive System
- Built a simple system to predict whether a given news article is fake or real

---

## ✅ Results

The Logistic Regression model gives good accuracy in detecting fake news and works well for text classification problems.

---

## ▶️ How to Run the Project

1. Clone the repository  
```bash
git clone <https://github.com/aditi-jha25/Fake-news-prediction.git>

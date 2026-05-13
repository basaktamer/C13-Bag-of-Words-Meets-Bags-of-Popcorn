---
title: Bag of Popcorn Sentiment Analysis
emoji: 🍿
colorFrom: yellow
colorTo: red
sdk: streamlit
sdk_version: 1.31.0
app_file: app.py
pinned: false
---

# Bag of Words Meets Bags of Popcorn: Movie Sentiment AI

This repository contains a binary sentiment classification application built for the classic Kaggle competition: **"Bag of Words Meets Bags of Popcorn."** The project demonstrates a transition from basic text preprocessing to professional machine learning deployment.

## 🚀 Project Overview
The goal of this project was to build a model capable of distinguishing between positive and negative movie reviews using the IMDb dataset. By leveraging NLP techniques, the application identifies sentiment patterns in paragraph-length text.

- **Problem Type:** Binary Classification (NLP)
- **Evaluation Metric:** ROC-AUC
- **Model Used:** Random Forest Classifier
- **Key Feature:** CountVectorizer with N-Gram range (1, 2) to capture context and phrases.

## 🛠️ Tech Stack
- **Language:** Python 3.10
- **Libraries:** Scikit-learn, Pandas, Joblib, Neattext
- **Deployment:** Streamlit & Hugging Face Spaces

## 📝 Preprocessing Workflow
To maintain high performance and clean data, the following **neattext** operations were applied:
1.  **Lowercasing:** Standardizing text to lowercase.
2.  **Punctuation Removal:** Stripping special characters using `nfx.remove_punctuation`.
3.  **Digit Removal:** Eliminating numbers using `nfx.remove_numbers`.
4.  **Whitespace Management:** Cleaning up multiple spaces and newlines.

## 💻 Local Installation & Setup
To run this project locally on your machine (specifically configured for **Python 3.10**):


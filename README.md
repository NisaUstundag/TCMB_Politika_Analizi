# NLP-Based Financial Early Warning and Decision Support System 🚀

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white) ![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white) ![NLP](https://img.shields.io/badge/NLP-005571?style=for-the-badge) ![Machine Learning](https://img.shields.io/badge/Machine_Learning-8E75B2?style=for-the-badge)

This project is a FinTech solution aiming to predict inflation risk (Low/Medium/High) by analyzing the Monetary Policy Committee (MPC) announcement texts published by the Central Bank of the Republic of Turkey (CBRT) using **Natural Language Processing (NLP)** and **Machine Learning (ML)** methods.

---

## Project Purpose and Core Problem
In the economic world, the language used by the Central Bank is the most critical factor shaping market expectations. However, manual analysis of these texts is:
1. **Subjective:** It is open to individual interpretation and has a high margin of error.
2. **Time-Consuming:** It makes it difficult to take fast and algorithmic actions at the moment of decision.

**This system** provides a data-driven, objective, and fast-acting **Early Warning System** by mathematically scoring the "Hawkish" (strict/combative) and "Dovish" (mild/supportive) tones in the texts.

---

## Technical Methodology
The project is built on an end-to-end machine learning pipeline from raw data to the final risk prediction.

### 1. Data Source and Reliability
* **Official Source:** MPC texts and inflation data covering the years 2006-2025 were pulled directly from the **CBRT EVDS** (Electronic Data Delivery System).
* **Data Processing:** The model's accuracy was maximized by training it entirely on real and official government data.

### 2. Algorithm Selection (Model Comparison)
Within the scope of the project, 5 different supervised learning algorithms were subjected to performance testing on the same dataset:
* **Naive Bayes**
* **Decision Tree**
* **K-Nearest Neighbors (KNN)**
* **Support Vector Machines (SVM)**
* **Random Forest**

Based on the analyses and cross-validation results, **Random Forest**, which demonstrated the most stable performance, was selected as the main prediction engine.

### 3. Explainable AI (XAI)
Our model is not a "Black Box". It acts as a **Decision Support System** by transparently reporting the reasoning behind its risk prediction (word weights, tone score, and historical data patterns).

---

## Key Features
* **Live Inference:** The ability to instantly analyze newly published texts.
* **Dynamic Synthesis:** Providing consistent reporting by cross-checking word signals with the machine learning prediction.
* **Interactive Dashboard:** A user-friendly, enterprise-standard interface based on Streamlit.

---

## Developer
**Nisa Üstündağ** *Software Engineering Department 

---

## Installation and Setup
The system is Python-based and runs via the Streamlit library. You can follow the steps below to run the project in your local environment:

```bash
# Install required libraries
pip install streamlit pandas scikit-learn seaborn matplotlib

# Run the application
streamlit run main.py
```

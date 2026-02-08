# 🩺 Diabetes Prediction System

A Machine Learning application built with Python to predict whether a patient has diabetes based on diagnostic measures. This system uses a Support Vector Machine (SVM) model for high-accuracy classification.

## 🚀 Overview
This project implements a complete Machine Learning pipeline, including:
* **Data Cleaning:** Handling missing values and removing inconsistent data.
* **Preprocessing:** Standardization of medical data using `StandardScaler` to improve model performance.
* **Model Training:** Using a Support Vector Machine (SVM) with a linear kernel.
* **Prediction System:** A real-time interface where users can input health data and get an instant diagnosis.

## 🛠️ Technologies Used
* **Language:** Python 3.x
* **Libraries:** * `pandas` (Data Manipulation)
    * `numpy` (Numerical Computation)
    * `scikit-learn` (Machine Learning & Metrics)

## 📂 Dataset
The dataset used is the **PIMA Indians Diabetes Database**. It contains 768 rows with 8 diagnostic attributes:
1.  Pregnancies
2.  Glucose
3.  BloodPressure
4.  SkinThickness
5.  Insulin
6.  BMI
7.  DiabetesPedigreeFunction
8.  Age

## ⚙️ How to Run
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git](https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git)
    ```
2.  **Install dependencies:**
    ```bash
    pip install numpy pandas scikit-learn
    ```
3.  **Run the script:**
    ```bash
    python main.py
    ```

## 📊 Model Accuracy
* **Training Accuracy:** ~78%
* **Test Accuracy:** ~77%
*(Note: Accuracy may vary slightly based on the random state split)*

## 🔮 Future Improvements
* Add a GUI using Streamlit or Tkinter.
* Test other algorithms like Random Forest or Logistic Regression for comparison.
* Deploy the model as a web API using Flask.

---
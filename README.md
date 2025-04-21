# Heart-Disease-Prediction-KNN

This project is a machine learning pipeline built to **predict the presence of heart disease** based on patient health attributes. It includes exploratory data analysis, feature engineering, model building, and hyperparameter tuning.

---

## Dataset

The dataset was sourced from https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction. It contains anonymized patient health records with features such as:

- Age, Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Resting ECG Results
- Exercise-Induced Angina
- ST Depression (Oldpeak)
- ST Slope
- Target: `HeartDisease` (0 = No, 1 = Yes)

---

##  Exploratory Data Analysis (EDA)

- Count plots for categorical features
- Distribution of target variable across key features
- Correlation heatmaps to understand relationships

---

##  Data Cleaning & Preprocessing

- Removed zero values from `RestingBP`
- Replaced zero `Cholesterol` values using median imputation (stratified by heart disease)
- One-hot encoded categorical variables
- Scaled numerical features using `MinMaxScaler`

---

##  Model Building

Used **k-Nearest Neighbors (k-NN)** for classification.

### Steps:
- Trained baseline models with individual features
- Selected key features based on correlation and performance
- Tuned hyperparameters using **GridSearchCV**:
  - `n_neighbors` from 1 to 20
  - Distance metrics: `'minkowski'` and `'manhattan'`

### ✅ Best Model:
- Accuracy on validation: **~83%**
- Accuracy on test set: **~87%**
- Evaluated using confusion matrix

---

## 🛠 Tech Stack

- Python
- Pandas, NumPy
- Seaborn, Matplotlib
- Scikit-learn
- Jupyter Notebook



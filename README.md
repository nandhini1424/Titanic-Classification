# 🚢 Titanic Survival Prediction 

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Status](https://img.shields.io/badge/status-Prototype-yellow)

A simple machine learning project that predicts whether a Titanic passenger survived or not, using linear regression.  
Includes data preprocessing, model training, evaluation, and a command-line interface for user input.

---

## 📦 Dataset
- Requires `titanic.csv` in the same directory.
- Uses the following features:
  - `Pclass`: Passenger class (1st, 2nd, or 3rd)
  - `Gender`: Encoded as `0` (male) or `1` (female)
  - `Age`: Age of the passenger
  - `Embarked`: Encoded as:
    - `1`: Southampton
    - `2`: Queenstown
    - `3`: Cherbourg

---

## 🛠️ How It Works
✅ Reads and cleans data (removes missing values)  
✅ Encodes categorical features into numeric labels  
✅ Splits data into training and testing sets  
✅ Scales features using `StandardScaler`  
✅ Trains a `LinearRegression` model  
✅ Evaluates the model with common metrics  
✅ Prompts user for passenger details and predicts survival

---

## 📊 Evaluation Metrics
After training, the script displays:
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **R² Score** (explains how well the model fits)


---

## ▶️ How to Run
1. Install required dependencies:
   ```bash
   pip install pandas numpy scikit-learn

2. Make sure titanic.csv is in the project folder.

3. Run the script:
   ```bash
   python main1.py

4. Enter passenger details when prompted to see the survival prediction.

## ✅ Example CLI Prediction
  ```bash
Enter Class: 2
Enter Gender(enter 0 for male and 1 for female): 1
Enter Age: 28
Enter Place Embarked(enter 1 for Southampton 2 for Queenstown and 3 for Cherbourg): 3
Saved
```

## ⚠️ Note
This project uses linear regression for a classification task (survived or not).
For better accuracy, you could replace it with a classifier like Logistic Regression, Random Forest, or XGBoost.
The script drops rows with missing data, which may reduce the dataset size.





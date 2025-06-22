
# 🚗 Car Sales Prediction using Machine Learning

This project predicts the **car purchase amount** a customer might spend using a **Linear Regression** model. It uses a real-world dataset and provides a simple **Streamlit** web interface for predictions.

The dataset includes the following fields:

- `Customer Name`
- `Customer Email`
- `Country`
- `Gender`
- `Age`
- `Annual Salary`
- `Credit Card Debt`
- `Net Worth`
- `Car Purchase Amount` (target)

## 📈 Model Details

- **Model Used:** Linear Regression
- **Scaler:** StandardScaler (from sklearn)
- **Performance:**
  - R² Score: ~1.00
  - Mean Squared Error: ~2.07

## 🚀 How to Run the Project

🔧 1. Clone the Repository
git clone https://github.com/Maniteja0611/car-sales-prediction.git
cd car-sales-prediction

📦 2. Install Dependencies
pip install -r requirements.txt

▶️ 3. Run the App
streamlit run app.py

🌟 Features
Clean UI built with Streamlit

Realtime predictions using trained model

Heatmap and statistics for EDA

Saves model and scaler using joblib

🔮 Future Improvements

Add support for multiple models (e.g., Random Forest, XGBoost)

Store user predictions for analysis

Deploy on Streamlit Cloud or HuggingFace Spaces

Feel free to fork the repo and make changes or improvements related to the project. Contributions are welcome!

⭐ Star the repo if you find it helpful!







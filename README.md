This project aims to predict stock prices using a hybrid deep learning model that combines 1D Convolutional Neural Networks (CNNs) for feature extraction and Long Short-Term Memory (LSTM) networks for sequential learning. Additionally, SHAP (SHapley Additive exPlanations) is employed for explainability, offering insights into the model's predictions.

Features
Hybrid Model:
CNN layers to extract local patterns from financial time series.
LSTM layers to capture long-term dependencies and trends.
Explainable AI (XAI):
SHAP values to interpret the importance of features.
Custom Prediction:
Interactive user input for real-time stock price predictions.
Visualization:
Detailed plots to compare actual vs. predicted prices, residual errors, and explainability insights.
Evaluation Metrics:
Includes RMSE, MAE, R2, and residual analysis.
Workflow
Data Collection:
Historical stock prices for the S&P 500 Index (^GSPC) from 2000 to 2024 are fetched using Yahoo Finance.
Feature Engineering:
Key features include Open, High, Low, Close, and Volume.
Data is normalized using Min-Max scaling.
Data Preparation:
Sliding window technique for time series data to create input-output pairs.
Split data into training and testing sets.
Model Architecture:
CNN Layers: Two Conv1D layers with MaxPooling for feature extraction.
LSTM Layers: Single LSTM layer for sequence modeling.
Fully connected Dense layers for prediction.
Training:
Model is trained with early stopping to prevent overfitting.
Validation:
Model performance is evaluated on test data.
Explainability:
SHAP values provide insights into how features impact predictions.
Interactive Prediction:
Accepts recent prices as input and predicts the next stock price.
Visualizations
Actual vs. Predicted Prices: Line plot showing performance.
Residual Analysis: Error plots and histograms for model evaluation.
SHAP Summary Plot: Feature importance visualization.
Installation and Requirements
Libraries:
yfinance: Fetch stock data.
tensorflow: Deep learning framework.
pandas and numpy: Data manipulation.
matplotlib and seaborn: Visualization.
shap: Explainable AI.
scikit-learn: Data preprocessing and metrics.

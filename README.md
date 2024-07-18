## README for Microcap Stock Predictions with GRU Neural Network

## Project Overview
This project utilizes a Gated Recurrent Unit (GRU) neural network to predict the closing prices of five selected microcap stocks. The GRU model is designed to handle time-series data efficiently and is employed to forecast the closing prices for the current day, next day, and previous day. The project also incorporates various financial indicators and news sentiment scores to enhance the model's predictive accuracy.

## Project Structure
Report.pdf: Comprehensive report detailing the project methodology, analysis, and results.
Data collection and Analysis.ipynb: Jupyter notebook for data collection and preprocessing.
Model and Analysis.ipynb: Jupyter notebook for model training, evaluation, and analysis.

## Requirements

Python 3.x
Jupyter Notebook
Pandas
Matplotlib
scikit-learn
yfinance
VADER Sentiment Analysis

## Setup and Usage

# Clone the Repository:
git clone <repository_url>
cd <repository_directory>

# Install Dependencies:
Install the required Python libraries using pip:

pip install pandas matplotlib scikit-learn yfinance vaderSentiment

# Run the Jupyter Notebooks:
Open and run the Jupyter notebooks to replicate the data collection, preprocessing, model training, and analysis:

jupyter notebook Data\ collection\ and\ Analysis.ipynb
jupyter notebook Model\ and\ Analysis.ipynb


## Methodology

# Data Collection and Preprocessing
Stock Data: Historical financial data for five microcap stocks (SAVE, CLNE, LAZR, AMWL, GEO) were collected from Yahoo Finance using the yfinance library.

News Data: Relevant news headlines were gathered from Yahoo Finance.

Feature Engineering: Computed daily returns, volatility, SMA, RSI, stochastic oscillator, and sentiment scores using the VADER sentiment analysis tool.

Data Merging: Combined the historical stock data and news data into a single dataset for analysis.


# Model Implementation
# GRU (Gated Recurrent Unit) Model
Architecture: The GRU model consists of an update gate and a reset gate to manage information flow, effectively capturing long-term dependencies in the data.

Training and Validation: The dataset was split into training (60%), validation (20%), and test (20%) sets. The model was trained using the training and validation sets.

Prediction Tasks: The GRU model predicts the closing prices for the current day, next day, and previous day.

# Baseline Models
Decision Tree Regression: A non-linear model that creates a tree structure for regression tasks.

K-Nearest Neighbor (KNN) Regression: A non-parametric method that predicts stock prices based on the average of the nearest neighbors.

## Results

GRU Model Performance: The GRU model demonstrated high predictive accuracy, with R-squared values consistently above 0.98 for all stocks and prediction tasks.

Comparison with Baseline Models: The GRU model outperformed both Decision Tree and KNN models in terms of Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared metrics.

K-Fold Cross Validation: Confirmed the robustness of the GRU model across different folds, maintaining high R-squared values and low error rates.

## Future Scope

Incorporate Additional Market Indicators: Include macroeconomic variables, sector indices, and global financial trends to improve model accuracy.

Hybrid Models: Explore models that combine the interpretability of Random Forest or XGBoost with the temporal strengths of GRU.

Real-Time Trading Systems: Implement the model in real-time trading environments to evaluate its operational performance.

Model Interpretability and Efficiency: Enhance the interpretability and computational efficiency of the GRU model for broader application.

## Conclusion
The GRU neural network model demonstrates superior predictive capabilities for microcap stock prices, outperforming traditional regression models. This project highlights the potential of GRU models in financial forecasting, setting the stage for future enhancements and applications in algorithmic trading.

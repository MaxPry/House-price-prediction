# House Price Prediction

This project uses machine learning to predict house prices based on various features like the number of rooms, location, and more. We used the California Housing dataset and trained a Linear Regression model to make price predictions.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Visualizations](#visualizations)
- [License](#license)

## 🛠️ Installation
1. Clone this repository to your local machine:
   
   git clone https://github.com/MaxPry/House-price-prediction.git
   
2. Install the required libraries:

   pip install -r requirements.txt
   Note: If you don't have requirements.txt, you can manually install dependencies:
   pip install scikit-learn pandas matplotlib seaborn
   
3. Run the project: python main.py

## Usage
Open the House-price-prediction.ipynb notebook in Google Colab.

Run the cells to load the data, train the model, and visualize the results.

Alternatively, you can run the Python script main.py locally: python main.py

## Results
The model was trained using Linear Regression, and here are the evaluation results:

Mean Squared Error (MSE): 0.56
R² Score: 0.58

These results indicate a moderate correlation between predicted and actual house prices.

## Visualizations
This first scatter plot - Actual vs Predicted Prices 
Shows how close the model's predictions are to the actual house prices.
The blue dots represent predictions, and the red dashed line represents perfect predictions (where predicted = actual)

The second plot - Distribution of Prediction Errors (Residuals) 
Represents the distribution of residuals (errors), which helps us assess how well the model is performing.

The  third plot - Feature Importance
After training, we analyzed which features had the most impact on the predicted house price. This is visualized using the model’s learned coefficients

## Model Saving
After training, the linear regression model is saved as model.pkl using joblib. This allows you to reuse the trained model without retraining.

## Run the notebook on Google Colab
[Open the notebook in Google Colab] https://colab.research.google.com/drive/11oPrBhSmcu_bX5ygD9zA5CdugzhvSd74?usp=sharing



   

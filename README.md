# Linear Regression: Predicting Daily Electricity Usage

## Introduction  
Linear regression is a statistical method used to model the relationship between one dependent variable (target) and one or more independent variables (features). It predicts the output as a linear combination of the input features.

## Problem Statement  
We aim to predict **daily electricity usage (kWh)** based on features such as:  
- Indoor temperature (°C)  
- Humidity levels (%)  
- Hourly appliance energy consumption  

**Dataset Used**: [Appliances Energy Prediction Dataset (UCI)](https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction)

---

## Mathematical Foundations  

### Hypothesis Function  
The model assumes a linear relationship between input features \( X \) and the output \( y \):  
\[
y = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \ldots + \theta_n x_n
\]  
Where:  
- \( y \): Predicted output (electricity usage)  
- \( x_i \): Input features (e.g., temperature, humidity)  
- \( \theta_i \): Model coefficients (weights)  

---

### Cost Function  
To measure the error between the predicted values (\( \hat{y} \)) and actual values (\( y \)), we use the cost function:  
\[
J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}_i - y_i)^2
\]  
Where \( m \) is the number of data samples.

---

### Gradient Descent  
To minimize the cost function, the model updates the weights iteratively using gradient descent:  
\[
\theta_j := \theta_j - \alpha \frac{\partial J(\theta)}{\partial \theta_j}
\]  
Where:  
- \( \alpha \): Learning rate  
- \( \frac{\partial J(\theta)}{\partial \theta_j} \): Partial derivative of the cost function with respect to \( \theta_j \).  

---

## Steps to Implement  

1. **Preprocessing the Data**:  
   - Load the dataset and select relevant features.  
   - Normalize features for better convergence.  

2. **Building the Model**:  
   - Define the hypothesis function.  
   - Compute the cost function.  
   - Update weights using gradient descent.  

3. **Evaluating the Model**:  
   - Use metrics such as:  
     - **Mean Squared Error (MSE)**  
     - **R² Score**  

4. **Visualizing Results**:  
   - Plot the cost function convergence.  
   - Compare predicted and actual values.  

---

## Dataset Information  

- **Name**: Appliances Energy Prediction Dataset  
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction)  
- **Description**: This dataset contains features like temperature, humidity, and hourly energy consumption data for appliances.

---

## Implementation  

- **Option 1**: Implement linear regression **from scratch** using NumPy for matrix computations.  
- **Option 2**: Use **scikit-learn**'s `LinearRegression` for quick model building and evaluation.  

For complete code, check the `implementation.py` file in this repository.

---

## Results  

- **Final Model Coefficients**:  
  - \( \theta_0 \): Bias term  
  - \( \theta_1, \theta_2, \ldots \): Coefficients for features  

- **Performance Metrics**:  
  - Mean Squared Error (MSE): Measures the average error.  
  - R² Score: Indicates the proportion of variance explained by the model.  

---

## How to Run  

1. Clone this repository:  
   ```bash
   git clone https://github.com/ML_PROGRESS.git
   cd ML_PROGRESS

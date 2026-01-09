# 📈 Simple Linear Regression

A foundational implementation of Simple Linear Regression for predictive modeling and understanding linear relationships. This project demonstrates how to build, train, and evaluate regression models with a single independent variable to predict continuous target values.

## 📋 Description

This project implements Simple Linear Regression, the most fundamental machine learning algorithm used to model the linear relationship between a single independent variable (feature) and a dependent variable (target). The implementation covers the entire machine learning pipeline including data preprocessing, model training, evaluation, and visualization to understand how one variable influences another.

## ✨ Features

- 📉 **Single Variable Analysis**: Models relationship between one feature and target
- 📊 **Visual Insights**: Clear scatter plots with regression line visualization
- 🔧 **Data Preprocessing**: Handles data cleaning and preparation
- 🧠 **Model Training**: Implements the Ordinary Least Squares (OLS) method
- 📈 **Performance Evaluation**: R² score, MSE, RMSE metrics
- 📊 **Prediction Capability**: Make predictions on new data points
- 📝 **Coefficient Analysis**: Understanding slope and intercept
- ⚖️ **Best-Fit Line**: Calculates optimal line through data points

## 🛠️ Technologies Used

- **Python 3.x**
- **NumPy**: Numerical operations and calculations
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning implementation
- **Matplotlib**: Data visualization and plotting
- **Seaborn**: Enhanced statistical visualizations
- **Jupyter Notebook**: Interactive development

## 📊 How Simple Linear Regression Works

Simple Linear Regression finds the best-fitting straight line through the data points:

**Formula**: `y = mx + b`

Or in statistical notation: `y = β₀ + β₁x + ε`

Where:
- `y` = Dependent variable (what we're predicting)
- `x` = Independent variable (what we're using to predict)
- `β₀` (b) = Intercept (y-axis crossing point)
- `β₁` (m) = Slope (rate of change)
- `ε` = Error term (residuals)

### The Algorithm

1. **Calculate the slope (m)**: `m = Σ[(xᵢ - x̄)(yᵢ - ȳ)] / Σ[(xᵢ - x̄)²]`
2. **Calculate the intercept (b)**: `b = ȳ - m * x̄`
3. **Make predictions**: `ŷ = mx + b`

## 📁 Project Structure

```
simple-linear-regression/
│
├── main                     # Jupyter Notebook with full implementation
├── main.py                  # Python script version
├── LICENSE                  # MIT License
└── README.md                # Project documentation
```

## 🚀 Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/lakumsaicharan/simple-linear-regression.git
   cd simple-linear-regression
   ```

2. **Install dependencies**:
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn jupyter
   ```

3. **Run Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

4. **Open and explore**:
   - Open `main` notebook in Jupyter
   - Execute cells to see the implementation
   - Alternatively, run `main.py` for command-line execution

## 📚 Usage Example

### Basic Implementation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 5])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Model coefficients
print(f'Slope (m): {model.coef_[0]}')
print(f'Intercept (b): {model.intercept_}')

# Model evaluation
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'R² Score: {r2}')
print(f'RMSE: {rmse}')

# Visualize
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
```

## 📈 Key Concepts

### Evaluation Metrics

- **R² Score (Coefficient of Determination)**: 
  - Range: 0 to 1 (higher is better)
  - Measures how well the model explains variance
  - R² = 1 means perfect fit
  
- **MSE (Mean Squared Error)**:
  - Average of squared differences
  - Penalizes larger errors more
  
- **RMSE (Root Mean Squared Error)**:
  - Square root of MSE
  - Same units as the target variable
  - Easier to interpret

### Assumptions

1. **Linearity**: Relationship between X and y is linear
2. **Independence**: Observations are independent
3. **Homoscedasticity**: Constant variance of residuals
4. **Normality**: Residuals are normally distributed
5. **No outliers**: Extreme values can skew the line

## 🎓 Learning Objectives

This project demonstrates:
- ✅ Simple Linear Regression implementation
- ✅ Data visualization techniques
- ✅ Model training and evaluation
- ✅ Understanding slope and intercept
- ✅ Calculating best-fit line
- ✅ Interpreting regression coefficients
- ✅ Making predictions on new data

## 💼 Real-World Applications

- **Sales Forecasting**: Predicting sales based on advertising spend
- **Temperature Conversion**: Converting Celsius to Fahrenheit
- **Salary Prediction**: Estimating salary based on years of experience
- **Stock Prices**: Basic trend analysis
- **Height vs Weight**: Understanding body mass relationships
- **Study Hours vs Grades**: Academic performance prediction
- **House Size vs Price**: Real estate valuation

## 📉 When to Use Simple Linear Regression

**Best Used When:**
- ✅ You have one independent variable
- ✅ Relationship appears linear
- ✅ Quick exploratory analysis needed
- ✅ Baseline model for comparison

**Consider Alternatives When:**
- ❌ Multiple independent variables (use Multiple Linear Regression)
- ❌ Non-linear relationships (use Polynomial Regression)
- ❌ Categorical predictions (use Classification)

## 🔧 Extending the Model

- Add data preprocessing pipelines
- Implement residual analysis
- Add confidence intervals
- Create interactive visualizations
- Handle outliers detection
- Add cross-validation

## 🤝 Contributing

Contributions welcome! Feel free to:
- 🐛 Report bugs or issues
- 💡 Suggest improvements
- 🔧 Submit pull requests
- 📊 Add new visualizations

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Lakum Sai Charan**
- GitHub: [@lakumsaicharan](https://github.com/lakumsaicharan)
- Part of the 100 Days of Code Challenge
- Machine Learning & Data Science Journey

## 🙏 Acknowledgments

- Foundation of supervised learning
- Built as part of ML fundamentals
- Thanks to the scikit-learn team
- Inspired by classical statistics

## 📚 Resources

- [Scikit-learn Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- [Understanding Linear Regression](https://en.wikipedia.org/wiki/Simple_linear_regression)
- [Statistics and ML Connection](https://www.statlearning.com/)

---

⭐ **Found this useful? Star the repo!** ⭐

*Building predictive models one variable at a time* 📈

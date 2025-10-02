# COM-404-E-assignments
# Bank Customer Churn Prediction

## Overview
A machine learning project using Artificial Neural Networks to predict whether bank customers will leave based on their demographics and banking behavior.

## Dataset
- **Source**: `Churn_Modelling.csv`
- **Size**: 10,000 customer records
- **Features**: 11 input features, 1 target variable (Exited)

### Key Features
- Credit Score, Geography, Gender, Age, Tenure
- Balance, Number of Products, Credit Card Status
- Active Member Status, Estimated Salary

## Installation

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow keras
```

## Quick Start

```python
# Load data
import pandas as pd
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Preprocess (encode, scale, split)
# ... (see notebook for details)

# Build model
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(6, activation='relu', input_dim=11))
classifier.add(Dense(6, activation='relu'))
classifier.add(Dense(1, activation='sigmoid'))

# Compile and train
classifier.compile(optimizer='adam', loss='binary_crossentropy', 
                   metrics=['accuracy'])
classifier.fit(X_train, y_train, batch_size=10, epochs=100)

# Predict
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
```

## Model Architecture
- Input: 11 features
- Hidden Layer 1: 6 neurons (ReLU)
- Hidden Layer 2: 6 neurons (ReLU)
- Output: 1 neuron (Sigmoid)

## Results
- **Accuracy**: 84.55%
- **Precision**: 71.82%
- **Recall**: 39.01%
- **F1 Score**: 50.56%

### Confusion Matrix
```
[[1533   62]   # True Negative, False Positive
 [ 247  158]]  # False Negative, True Positive
```

## Interpretation
- Model is conservative: high precision, low recall
- Good at identifying churners when it predicts them (72% accurate)
- Misses 61% of actual churners (low recall)
- Best for high-confidence, targeted retention efforts

## Limitations
- Low recall (misses many churners)
- Class imbalance not addressed
- No validation set or cross-validation
- Simple architecture may underfit

## Improvements
1. Use SMOTE or class weights for imbalance
2. Add dropout for regularization
3. Implement cross-validation
4. Try deeper/wider architectures
5. Feature engineering (ratios, interactions)

## Files
- `ML_assignment 1.ipynb` - Main code
- `Churn_Modelling.csv` - Dataset
- `README.md` - This file

## License
Educational use only

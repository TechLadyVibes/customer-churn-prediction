# ============================================================
# CUSTOMER CHURN PREDICTION USING AN ARTIFICIAL NEURAL NETWORK
# UoPeople - Machine Learning Programming Assignment
# Mohpheth Ekhaguere
# Part 1
# - Imports
# - Load Dataset
# - Dataset Exploration
# - Logic Gates (AND, OR, NOT)
# ============================================================

# ===========================
# IMPORT LIBRARIES
# ===========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

random.seed(42)
np.random.seed(42)

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

print("=" * 70)
print("CUSTOMER CHURN PREDICTION USING ARTIFICIAL NEURAL NETWORK")
print("=" * 70)

# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading Customer Churn Dataset...\n")

# Replace the file name if yours is different
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Dataset Loaded Successfully.")

# ============================================================
# DATASET EXPLORATION
# ============================================================

print("\n" + "=" * 70)
print("FIRST FIVE ROWS")
print("=" * 70)

print(df.head())

print("\n" + "=" * 70)
print("LAST FIVE ROWS")
print("=" * 70)

print(df.tail())

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print(df.info())

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

print(df.isnull().sum())

print("\n" + "=" * 70)
print("DATASET SHAPE")
print("=" * 70)

print("Rows :", df.shape[0])
print("Columns :", df.shape[1])

print("\n" + "=" * 70)
print("COLUMN NAMES")
print("=" * 70)

for column in df.columns:
    print("-", column)

print("\n" + "=" * 70)
print("CUSTOMER CHURN DISTRIBUTION")
print("=" * 70)

print(df["Churn"].value_counts())

print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)

print(df.describe(include="all"))

print("\nDataset exploration completed successfully.")

# ============================================================
# LOGIC GATES USING A SINGLE NEURON
# ============================================================

print("\n" + "=" * 70)
print("IMPLEMENTING LOGIC GATES")
print("=" * 70)


# ------------------------------------------
# Step Function (Activation Function)
# ------------------------------------------

def step_function(value):
    if value >= 0:
        return 1
    else:
        return 0


# ------------------------------------------
# Single Neuron
# ------------------------------------------

def neuron(x1, x2, w1, w2, bias):

    total = (x1 * w1) + (x2 * w2) + bias

    return step_function(total)


# ============================================================
# AND GATE
# ============================================================

print("\nAND GATE")

print("------------------------------")
print("Input1  Input2  Output")
print("------------------------------")

and_inputs = [
    (0, 0),
    (0, 1),
    (1, 0),
    (1, 1)
]

for x1, x2 in and_inputs:

    output = neuron(
        x1,
        x2,
        w1=1,
        w2=1,
        bias=-1.5
    )

    print(f"{x1:^7}{x2:^8}{output:^8}")


# ============================================================
# OR GATE
# ============================================================

print("\nOR GATE")

print("------------------------------")
print("Input1  Input2  Output")
print("------------------------------")

for x1, x2 in and_inputs:

    output = neuron(
        x1,
        x2,
        w1=1,
        w2=1,
        bias=-0.5
    )

    print(f"{x1:^7}{x2:^8}{output:^8}")


# ============================================================
# NOT GATE
# ============================================================

print("\nNOT GATE")

print("------------------")
print("Input   Output")
print("------------------")

not_inputs = [0, 1]

for x in not_inputs:

    output = step_function((-1 * x) + 0.5)

    print(f"{x:^7}{output:^8}")

print("\nLogic gate implementation completed successfully.")

print("\nEND OF PART 1")
print("=" * 70)

# ============================================================
# PART 2
# DATA PREPROCESSING
# XOR USING A NEURAL NETWORK
# ============================================================

print("\n" + "=" * 70)
print("DATA PREPROCESSING")
print("=" * 70)

# ============================================================
# DATA PREPROCESSING
# ============================================================

print("\n" + "=" * 70)
print("DATA PREPROCESSING")
print("=" * 70)

# ------------------------------------------------------------
# Convert TotalCharges to numeric
# ------------------------------------------------------------

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# ------------------------------------------------------------
# Fill missing values
# ------------------------------------------------------------

df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

print("\nMissing Values After Cleaning\n")
print(df.isnull().sum())

# ------------------------------------------------------------
# Remove Customer ID
# ------------------------------------------------------------

if "customerID" in df.columns:
    df = df.drop("customerID", axis=1)

# ------------------------------------------------------------
# Convert Target Variable
# ------------------------------------------------------------

df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

# ------------------------------------------------------------
# Encode Categorical Variables
# ------------------------------------------------------------

print("\nEncoding Categorical Variables...")

df = pd.get_dummies(df, drop_first=True)

print("\nEncoding Completed Successfully.\n")
print(df.head())

# ------------------------------------------------------------
# Create Feature Matrix and Target Variable
# ------------------------------------------------------------

X = df.drop("Churn", axis=1)
y = df["Churn"]

print("\nFeature Matrix Shape :", X.shape)
print("Target Shape :", y.shape)

# ------------------------------------------------------------
# Scale Features
# ------------------------------------------------------------

print("\nScaling Numerical Features...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Scaling Completed Successfully.")

# ------------------------------------------------------------
# Train Test Split
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 70)
print("TRAIN TEST SPLIT")
print("=" * 70)

print("Training Samples :", len(X_train))
print("Testing Samples  :", len(X_test))
# ============================================================
# XOR USING A SMALL NEURAL NETWORK
# ============================================================

print("\n" + "=" * 70)
print("XOR USING A NEURAL NETWORK")
print("=" * 70)

xor_inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

xor_outputs = np.array([
    0,
    1,
    1,
    0
])


xor_model = MLPClassifier(
    hidden_layer_sizes=(4,),
    activation="tanh",
    solver="lbfgs",
    random_state=42,
    max_iter=10000
)


xor_model.fit(xor_inputs, xor_outputs)


xor_predictions = xor_model.predict(xor_inputs)


print("\nTruth Table\n")

print("-------------------------------")
print("Input1   Input2   Prediction")
print("-------------------------------")


for values, prediction in zip(xor_inputs, xor_predictions):

    print(
        f"{values[0]:^8}{values[1]:^9}{prediction:^11}"
    )


print("\nExpected Outputs")
print(xor_outputs)


print("\nPredicted Outputs")
print(xor_predictions)


print("\nExplanation")
print("-------------------------------------------")
print("AND, OR and NOT gates can be implemented")
print("using one neuron because their outputs")
print("can be separated using a straight line.")

print()

print("XOR cannot be solved using one neuron")
print("because its output pattern is not")
print("linearly separable.")

print("A hidden layer allows the neural network")
print("to create additional representations")
print("and learn the complex XOR relationship.")


print("\nPart 2 Completed Successfully.")
print("=" * 70)

# ============================================================
# PART 3
# BUILD, TRAIN AND EVALUATE THE NEURAL NETWORK
# ============================================================

print("\n" + "=" * 70)
print("BUILDING THE ARTIFICIAL NEURAL NETWORK")
print("=" * 70)

# ------------------------------------------------------------
# Define the Neural Network Architecture
# ------------------------------------------------------------

ann_model = MLPClassifier(
    hidden_layer_sizes=(16, 8),   # Two hidden layers
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,
    max_iter=500,
    random_state=42
)

print("\nNeural Network Architecture")
print("--------------------------------")
print("Input Layer  : Number of input features")
print("Hidden Layer : 16 neurons")
print("Hidden Layer : 8 neurons")
print("Output Layer : 1 neuron (Churn Prediction)")
print("Activation   : ReLU")
print("Optimizer    : Adam")

# ------------------------------------------------------------
# Train the Model
# ------------------------------------------------------------

print("\nTraining Neural Network...\n")

ann_model.fit(X_train, y_train)

print("Training Completed Successfully.")

# ------------------------------------------------------------
# Generate Predictions
# ------------------------------------------------------------

print("\nGenerating Predictions...\n")

y_pred = ann_model.predict(X_test)

print("First 20 Predictions\n")

for actual, predicted in zip(y_test.iloc[:20], y_pred[:20]):
    print(f"Actual : {actual}    Predicted : {predicted}")

# ============================================================
# MODEL EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy : {accuracy:.4f}")

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix")
print(cm)

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# ------------------------------------------------------------
# Display Confusion Matrix
# ------------------------------------------------------------

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()

plt.title("Customer Churn - Confusion Matrix")

plt.tight_layout()

plt.show()

# ============================================================
# PERFORMANCE SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PERFORMANCE SUMMARY")
print("=" * 70)

print(f"Training Samples : {len(X_train)}")
print(f"Testing Samples  : {len(X_test)}")
print(f"Accuracy         : {accuracy:.4f}")

print("\nModel Performance")

if accuracy >= 0.90:
    print("- Excellent performance.")
elif accuracy >= 0.80:
    print("- Good performance.")
elif accuracy >= 0.70:
    print("- Acceptable performance.")
else:
    print("- Model needs improvement.")

print("\nObservations")
print("--------------------------------")
print("1. The neural network learned patterns")
print("   from the customer data.")

print("2. Feature scaling improved training.")

print("3. The confusion matrix shows")
print("   correct and incorrect predictions.")

print("4. Higher accuracy indicates better")
print("   customer churn prediction.")

print("5. This model can help a telecom")
print("   company identify customers")
print("   who are likely to leave.")

print("\n" + "=" * 70)
print("PROGRAM COMPLETED SUCCESSFULLY")
print("=" * 70)

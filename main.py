#Version 3
#Machine Learning S5 Project

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url, sep=';')

# Phase 1: Exploratory Data Analysis (EDA)

# Visualize data distributions
plt.figure(figsize=(12, 8))
sns.histplot(data['quality'], kde=True)
plt.title('Distribution of Wine Quality')
plt.xlabel('Quality')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(12, 8))
sns.histplot(data['alcohol'], kde=True)
plt.title('Distribution of Alcohol Content')
plt.xlabel('Alcohol Content')
plt.ylabel('Frequency')
plt.show()

# Visualize relationships between features
plt.figure(figsize=(12, 8))
sns.scatterplot(x='alcohol', y='quality', data=data)
plt.title('Alcohol Content vs Wine Quality')
plt.xlabel('Alcohol Content')
plt.ylabel('Quality')
plt.show()

plt.figure(figsize=(12, 8))
sns.boxplot(x='quality', y='volatile acidity', data=data)
plt.title('Volatile Acidity by Wine Quality')
plt.xlabel('Quality')
plt.ylabel('Volatile Acidity')
plt.show()

# Identify missing values
missing_values = data.isnull().sum()
print("Missing values per feature:\n", missing_values)

# Calculate summary statistics
summary_stats = data.describe()
print("Summary statistics:\n", summary_stats)

# Identify outliers using boxplots
plt.figure(figsize=(12, 8))
sns.boxplot(data=data)
plt.title('Boxplot of All Features')
plt.xticks(rotation=90)
plt.show()

# Additional visualization: Correlation heatmap
plt.figure(figsize=(12, 8))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Data preprocessing

# Define numerical and categorical columns
numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = data.select_dtypes(include=['object']).columns

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data using Label Encoding
def label_encode(df, cols):
    for col in cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df

# Apply preprocessing
data_imputed = data.copy()
data_imputed[numerical_cols] = numerical_transformer.fit_transform(data[numerical_cols])
data_imputed = label_encode(data_imputed, categorical_cols)

# Convert the target variable 'quality' to discrete categories
data_imputed['quality'] = data_imputed['quality'].apply(lambda x: int(x))

# Save the preprocessed dataset to a CSV file
data_imputed.to_csv('preprocessed_wine_quality.csv', index=False)

# Split the data into features and target variable
X = data_imputed.drop('quality', axis=1)
y = data_imputed['quality']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Phase 2: Model Implementation and Evaluation

# Decision Tree Classifier with hyperparameter tuning
dt_params = {'max_depth': [5, 10, 15, 20, None]}
dt_classifier = GridSearchCV(DecisionTreeClassifier(random_state=42), dt_params, cv=5)
dt_classifier.fit(X_train, y_train)
y_pred_dt = dt_classifier.predict(X_test)
dt_accuracy = accuracy_score(y_test, y_pred_dt)
print("Decision Tree Classifier:")
print(classification_report(y_test, y_pred_dt, zero_division=0))
print("Best Parameters:", dt_classifier.best_params_)
print("Accuracy:", dt_accuracy)

# Random Forest Classifier with hyperparameter tuning
rf_params = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 15, 20, None]}
rf_classifier = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5)
rf_classifier.fit(X_train, y_train)
y_pred_rf = rf_classifier.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print("\nRandom Forest Classifier:")
print(classification_report(y_test, y_pred_rf, zero_division=0))
print("Best Parameters:", rf_classifier.best_params_)
print("Accuracy:", rf_accuracy)

# K-Nearest Neighbors Classifier with hyperparameter tuning
knn_params = {'n_neighbors': [3, 5, 7, 9, 11]}
knn_classifier = GridSearchCV(KNeighborsClassifier(), knn_params, cv=5)
knn_classifier.fit(X_train, y_train)
y_pred_knn = knn_classifier.predict(X_test)
knn_accuracy = accuracy_score(y_test, y_pred_knn)
print("\nK-Nearest Neighbors Classifier:")
print(classification_report(y_test, y_pred_knn, zero_division=0))
print("Best Parameters:", knn_classifier.best_params_)
print("Accuracy:", knn_accuracy)

# Print summary of accuracies
print("\nSummary of Accuracies:")
print(f"KNN: {knn_accuracy:.4f}")
print(f"Decision Tree: {dt_accuracy:.4f}")
print(f"Random Forest: {rf_accuracy:.4f}")

# Binarize the classes for multi-class ROC curve
classes = sorted(y.unique())
y_test_binarized = label_binarize(y_test, classes=classes)

# Decision Tree Classifier with hyperparameter tuning
dt_classifier = GridSearchCV(DecisionTreeClassifier(random_state=42), dt_params, cv=5)
dt_classifier.fit(X_train, y_train)
y_pred_dt = dt_classifier.predict(X_test)
dt_accuracy = accuracy_score(y_test, y_pred_dt)
print("Decision Tree Classifier:")
print(classification_report(y_test, y_pred_dt, zero_division=0))
print("Best Parameters:", dt_classifier.best_params_)
print("Accuracy:", dt_accuracy)

# Probability scores for ROC curve
y_score_dt = dt_classifier.predict_proba(X_test)

# Compute ROC curve for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(classes)):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score_dt[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves
plt.figure()
for i in range(len(classes)):
    plt.plot(fpr[i], tpr[i], label='Class {0} (AUC = {1:0.2f})'.format(classes[i], roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Decision Tree')
plt.legend(loc='lower right')
plt.show()

# Repeat for the Random Forest Classifier
rf_classifier = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5)
rf_classifier.fit(X_train, y_train)
y_pred_rf = rf_classifier.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print("\nRandom Forest Classifier:")
print(classification_report(y_test, y_pred_rf, zero_division=0))
print("Best Parameters:", rf_classifier.best_params_)
print("Accuracy:", rf_accuracy)

y_score_rf = rf_classifier.predict_proba(X_test)
for i in range(len(classes)):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score_rf[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure()
for i in range(len(classes)):
    plt.plot(fpr[i], tpr[i], label='Class {0} (AUC = {1:0.2f})'.format(classes[i], roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest')
plt.legend(loc='lower right')
plt.show()

# Repeat for the K-Nearest Neighbors Classifier
knn_classifier = GridSearchCV(KNeighborsClassifier(), knn_params, cv=5)
knn_classifier.fit(X_train, y_train)
y_pred_knn = knn_classifier.predict(X_test)
knn_accuracy = accuracy_score(y_test, y_pred_knn)
print("\nK-Nearest Neighbors Classifier:")
print(classification_report(y_test, y_pred_knn, zero_division=0))
print("Best Parameters:", knn_classifier.best_params_)
print("Accuracy:", knn_accuracy)

y_score_knn = knn_classifier.predict_proba(X_test)
for i in range(len(classes)):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score_knn[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure()
for i in range(len(classes)):
    plt.plot(fpr[i], tpr[i], label='Class {0} (AUC = {1:0.2f})'.format(classes[i], roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - KNN')
plt.legend(loc='lower right')
plt.show()

# Summary of accuracies
print("\nSummary of Accuracies:")
print(f"KNN: {knn_accuracy:.4f}")
print(f"Decision Tree: {dt_accuracy:.4f}")
print(f"Random Forest: {rf_accuracy:.4f}")

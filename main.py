import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, label_binarize

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

# The "Wine Quality" dataset is suited for classification. The reasoning is as follows:
# - The target variable in the dataset is 'quality', which is an integer value representing the quality of the wine on a scale from 0 to 10.
# - Classification problems involve predicting a discrete label (in this case, the quality score) based on input features.
# - The goal is to classify each wine sample into one of the quality categories based on its chemical properties.
# Since we have predefined labels (quality scores) and the goal is to predict these labels, classification is the appropriate approach.

# Phase 2: Data Preprocessing and Model Training

# Handle missing values using mean imputation
imputer = SimpleImputer(strategy='mean')
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Scale numerical features using StandardScaler
scaler = StandardScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data_imputed), columns=data.columns)

# Convert the target variable 'quality' to discrete categories
data_scaled['quality'] = data_scaled['quality'].apply(lambda x: int(x))

# Split the data into features and target variable
X = data_scaled.drop('quality', axis=1)
y = data_scaled['quality']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

# Explanation:
# - For the Decision Tree, we tuned the 'max_depth' parameter to control the depth of the tree, which helps prevent overfitting.
# - For the Random Forest, we tuned the 'n_estimators' (number of trees) and 'max_depth' parameters to balance between bias and variance.
# - For K-Nearest Neighbors, we tuned the 'n_neighbors' parameter to find the optimal number of neighbors for classification.

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

# Compare the performance of each model and discuss why some performed better or worse on the dataset.
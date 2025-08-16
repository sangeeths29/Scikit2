'''
Problem 3 - Implementing PCA in Python with Scikit-Learn (https://www.geeksforgeeks.org/implementing-pca-in-python-with-scikit-learn/))
'''
# Step 1 - Import necessary libraries
import numpy as np #type: ignore
import pandas as pd #type: ignore
import matplotlib.pyplot as plt #type: ignore
import seaborn as sns #type: ignore

from sklearn.preprocessing import StandardScaler #type: ignore
from sklearn.decomposition import PCA #type: ignore
from sklearn.model_selection import train_test_split #type: ignore
from sklearn.linear_model import LogisticRegression #type: ignore
from sklearn.metrics import confusion_matrix, classification_report #type: ignore

# Step 2 - Load the Data
df = pd.read_csv('data.csv')
df.head()

# Step 3 - Data Cleaning and Preprocessing
df.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Step 4 - Separate Features and Target
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# Step 5 - Standardize the Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(X_scaled[:2])

# Step 6 - Apply PCA Algorithm
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(X_pca[:2])

# Step 7 - Explained Variance
print("Explained variance:", pca.explained_variance_ratio_)
print("Cumulative:", np.cumsum(pca.explained_variance_ratio_))

# Step 8 - Visualization Before vs After PCA
plt.figure(figsize=(8,6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='coolwarm', edgecolor='k')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Original Data (First Two Features)")
plt.colorbar(label="Diagnosis")
plt.show()

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', edgecolor='k')
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Transformed Data")
plt.colorbar(label="Diagnosis")
plt.show()

# Step 9 - Train a Model on PCA Data
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))

# Step 10 - Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Benign', 'Malignant'],
            yticklabels=['Benign', 'Malignant'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Step 11 - Reconstruct Data and Check Information Loss
X_reconstructed = pca.inverse_transform(X_pca)
reconstruction_loss = np.mean((X_scaled - X_reconstructed) ** 2)
print(f"Reconstruction Loss: {reconstruction_loss:.4f}")
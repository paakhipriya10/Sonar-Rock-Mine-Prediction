# train_models.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = pd.read_csv("sonar.csv", header=None)

# Separate features and label
X = data.drop(60, axis=1)
y = data[60]

# Add target to the dataset for EDA
data['Label'] = y

# ----------- 📊 Exploratory Data Analysis (EDA) -----------

# Plot the average signal for each class
mean_rock = data[data.Label == 'R'].drop([60, 'Label'], axis=1).mean()
mean_mine = data[data.Label == 'M'].drop([60, 'Label'], axis=1).mean()

plt.figure(figsize=(14, 6))
plt.plot(mean_rock.values, label='Rock (R)', color='red')
plt.plot(mean_mine.values, label='Mine (M)', color='blue')
plt.title('Average Sonar Signal Intensity per Feature (0–59)')
plt.xlabel('Feature Index')
plt.ylabel('Average Signal')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("sonar_signal_comparison.png")  # Save plot as image
plt.show()

# ----------- 🔍 Model Training & Evaluation -----------

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=1
)

# Define models
models = {
    "Logistic Regression(Batch Learning)": LogisticRegression(max_iter=1000),
    "K-Nearest Neighbors (k=3,Batch Learning)": KNeighborsClassifier(n_neighbors=3),
    "Random Forest(Batch Learning)": RandomForestClassifier(),
    "SGD Classifier (Online Learning)": SGDClassifier()
}

best_model_name = ""
best_accuracy = 0
print("\nModel Accuracies:\n")

# Train, predict, compare accuracy
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name}: {acc:.4f}")
    
    # Save model
    safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "")
    joblib.dump(model, f"{safe_name}.pkl")
    
    if acc > best_accuracy:
        best_accuracy = acc
        best_model_name = name

print(f"\n✅ Best model: {best_model_name} with accuracy: {best_accuracy:.4f}")

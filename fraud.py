import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import numpy as np

df = pd.read_csv("creditcard.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(class_weight="balanced", max_iter=1000)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))

ConfusionMatrixDisplay.from_estimator(model, X_test_scaled, y_test)
plt.title("Fraud Detection Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()
plt.close()

feature_importance = pd.Series(
    np.abs(model.coef_[0]),
    index=X.columns
)

feature_importance.sort_values(ascending=False).head(10).plot(kind="bar", figsize=(10,5))
plt.title("Top 10 Features for Fraud Detection")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()
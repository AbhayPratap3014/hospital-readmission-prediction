import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

from google.colab import files
uploaded = files.upload()

df = pd.read_csv("hospital_readmissions_30k.csv")

# Remove patient ID
df = df.drop("patient_id",axis=1)

# Separate features and targets
X = df.drop("readmitted_30_days",axis=1)
y = df["readmitted_30_days"]

# Convert categorical columns into numerical columns
X = pd.get_dummies(X,drop_first=True)

# split dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

# Logistic Regression with L2 regularization
model = LogisticRegression(
  penalty="l2",
  C=1.0,
  max_iter=1000
)

# Train the model
model.fit(X_train, y_train)

# Predict Probability of class 1
y_prob = model.predict_proba(X_test)[:,1]

# ROC-AUC score
auc = roc_auc_score(y_test, y_prob)
print("ROC-AUC:",auc)

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob, pos_label='Yes')

plt.plot(fpr, tpr, label=f"ROC-AUC={auc:.2f}")
plt.plot([0,1],[0,1],linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

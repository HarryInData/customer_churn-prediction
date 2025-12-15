import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import PrintTime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
file_path=r"C:\Users\HP\Downloads\archive\WA_Fn-UseC_-Telco-Customer-Churn.csv"
df=pd.read_csv(file_path)
print(df.head(10))# top 10 data shown
print(df.columns)# it will show all the column in the dataset
print(df.shape)
print(df.info())          # data types + non‑null counts by this
print(df.isnull().sum())  # missing values per column
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce") #convert to numaric and if anu null value fill with NULL
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)#if null value fill the median etry into them
print(df["Churn"].value_counts())#count the  churn value
print(df["Churn"].value_counts(normalize=True) * 100)  # percentages the chun yes or No

sns.countplot(x="Churn", data=df)
plt.title("Churn Distribution")
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Contract", hue="Churn")
plt.title("Contract Type vs Churn")
plt.xticks(rotation=15)
plt.show()

# Tenure distribution by churn
plt.figure(figsize=(8, 4))
sns.histplot(data=df, x="tenure", hue="Churn", multiple="stack", bins=30)
plt.title("Tenure distribution by Churn")
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Contract", hue="Churn")
plt.title("Contract Type vs Churn")
plt.xticks(rotation=15)
plt.show()

plt.figure(figsize=(8, 4))
sns.boxplot(data=df, x="Churn", y="MonthlyCharges")
plt.title("Monthly Charges by Churn")
plt.show()
# Drop ID column (not useful for prediction)
df = df.drop("customerID", axis=1)
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
print(df["Churn"].value_counts())

X = df.drop("Churn", axis=1)
y = df["Churn"]

# Get dummy variables for all categorical columns
X = pd.get_dummies(X, drop_first=True)

print(X.shape)
print(X.head())
# assuming X and y already defined
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = LogisticRegression(max_iter=7000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

print("Classification report:")
print(classification_report(y_test, y_pred))


feature_names = X.columns
coeffs = model.coef_[0]

feature_importance = pd.DataFrame({
    "feature": feature_names,
    "coef": coeffs,
    "abs_coef": np.abs(coeffs)
}).sort_values("abs_coef", ascending=False)

print(feature_importance.head(15))
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print(classification_report(y_test, y_pred_rf))

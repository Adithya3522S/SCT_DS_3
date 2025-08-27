#Block 1 – Load and Inspect Data

import pandas as pd
from sklearn.datasets import fetch_openml

bank = fetch_openml(name="BankMarketing", version=1, as_frame=True)
df = bank.frame

print("Dataset shape:", df.shape)
df.head()

#Block 2 – Preprocessing

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

X = df.drop("y", axis=1)
y = df["y"]

for col in X.select_dtypes(include=["category", "object"]).columns:
    X[col] = LabelEncoder().fit_transform(X[col])

y = LabelEncoder().fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training and testing sets created successfully.")

#Block 3 – Train Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

print("Full tree depth:", clf.get_depth())
print("Number of leaves:", clf.get_n_leaves())

#Block 4 – Full Tree Visualization (Save as PDF)

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(50, 30))  # large figure for readability
plot_tree(clf, feature_names=X.columns, class_names=["No", "Yes"], filled=True)
plt.savefig("full_decision_tree.pdf", bbox_inches="tight")
plt.close()

print("Full decision tree saved as 'full_decision_tree.pdf'")


# Task 3 – Decision Tree Classifier on Bank Marketing Dataset

This project is part of my **Data Science Internship at SkillCraft Technology**.  
The objective is to **train and visualize a Decision Tree Classifier** to predict whether a customer will purchase a product or service, based on their demographic and behavioral data.

---

## 📊 Dataset

- **Source:** [Bank Marketing Dataset – OpenML](https://www.openml.org/d/1461)
- **Description:** The dataset contains marketing campaign data from a Portuguese banking institution.
- **Target variable (`y`):** Whether the client subscribed to a term deposit (Yes/No).

---

## 🧩 Project Workflow

### 🔹 Block 1 – Load & Inspect Data

- Loaded dataset using `fetch_openml`
- Printed dataset shape and previewed first few rows

### 🔹 Block 2 – Preprocessing

- Encoded categorical variables using `LabelEncoder`
- Split dataset into **training** and **testing** sets

### 🔹 Block 3 – Train Classifier

- Trained a **Decision Tree Classifier** without depth limit
- Checked model complexity (depth, number of leaves)

### 🔹 Block 4 – Full Tree Visualization

- Generated and saved the **entire decision tree** as a **high-resolution PDF**
- Reason: full tree is too large to display inline, PDF ensures better readability

---

## 📂 Files

- `task3_decision_tree.py` → Python script containing all blocks of code
- `full_decision_tree.pdf` → High-resolution visualization of the trained decision tree
- `README.md` → Documentation
- `requirements.txt` → Dependencies list

---

## 🚀 Key Learnings

- Handling categorical data with encoders is essential for ML models
- Decision Trees are interpretable but can grow very complex
- Proper visualization techniques are crucial to analyze model decisions

---

## 🛠 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

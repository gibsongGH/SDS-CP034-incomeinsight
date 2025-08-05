# 📄 IncomeInsight – Project Report - 🟢 **Beginner Track**

Welcome to your personal project report!  
Use this file to answer the key reflection questions for each phase of the project. This report is designed to help you think like a data scientist, guide AI tools more effectively, and prepare for real-world job interviews.

---

## ✅ Week 1: Setup & Exploratory Data Analysis (EDA)

> Answer the EDA questions provided in the project materials here. Focus on data quality, trends, anomalies, and relationships.

### 🔑 Question 1: What features show the strongest correlation with earning >$50K?

### 🔑 Question 2: How does income vary with education, marital status, or hours worked per week?

### 🔑 Question 3: Are there disparities across race, sex, or native country?

### 🔑 Question 4: Do capital gains/losses strongly impact the income label?

---

## ✅ Week 2: Feature Engineering & Data Preprocessing

### 🔑 Question 1:
**Which features in the dataset appear to have the strongest relationship with the income label (>50K), and how did you determine this?**  
🎯 *Purpose: Tests ability to identify influential predictors through EDA.*

💡 **Hint:**  
Use `.groupby('income')` to compare mean values of numeric features.  
Use bar plots or violin plots for categorical features vs. income.  
Check chi-squared test or information gain if desired.

✏️ *Your answer here...*

---

### 🔑 Question 2:
**Did you engineer any new features from existing ones? If so, explain the new feature(s) and why you think they might help your classifier.**  
🎯 *Purpose: Tests creativity and business-driven reasoning in feature creation.*

💡 **Hint:**  
Consider grouping `education_num` into bins, creating a `has_capital_gain` flag, or interaction terms like `hours_per_week * education_num`.

✏️ *Your answer here...*

---

### 🔑 Question 3:
**Which continuous features required scaling or transformation before modeling, and which method did you use?**  
🎯 *Purpose: Connects feature scaling to model compatibility.*

💡 **Hint:**  
Use `df.describe()` and `hist()` to evaluate spread.  
Logistic Regression is sensitive to feature scale; Random Forest is not.  
Apply `StandardScaler` or `MinMaxScaler` accordingly.

✏️ *Your answer here...*

---

### 🔑 Question 4:
**Is the target variable (`income`) imbalanced? How did you check, and what will you do (if anything) to handle it?**  
🎯 *Purpose: Tests understanding of classification imbalances and impact on metrics.*

💡 **Hint:**  
Use `.value_counts(normalize=True)`.  
If imbalance exists, consider using class weights, SMOTE, or stratified splits.  
Mention implications for precision, recall, and F1.

✏️ *Your answer here...*

---

### 🔑 Question 5:
**What does your final cleaned dataset look like before modeling? Include shape, types of features (numerical/categorical), and a summary of the preprocessing steps applied.**  
🎯 *Purpose: Encourages documentation and preparation for modeling.*

💡 **Hint:**  
Use `df.shape`, `df.dtypes`, and summarize what was dropped, encoded, scaled, or engineered.

✏️ *Your answer here...*

---


---

### ✅ Week 3: Model Development & Experimentation

### 🔑 Question 1:
**Which classification models did you train for predicting income, and what are the strengths or assumptions of each model?**  
🎯 *Purpose: Tests understanding of algorithm selection and fit for the problem.*

💡 **Hint:**  
Train Logistic Regression (baseline, interpretable), Random Forest (handles non-linearities), and XGBoost (boosted performance).  
Explain what each model assumes (e.g., linearity in Logistic Regression) or does well (e.g., handling missing values, feature interactions).

✏️ *Your answer here...*

---

### 🔑 Question 2:
**How did each model perform based on your evaluation metrics (accuracy, precision, recall, F1-score, ROC-AUC)? Which performed best, and why?**  
🎯 *Purpose: Tests ability to evaluate and compare classifiers fairly.*

💡 **Hint:**  
Use `classification_report`, `confusion_matrix`, and `roc_auc_score`.  
Show results in a table or chart.  
Explain model strengths (e.g., better recall = catches more high-income earners).

✏️ *Your answer here...*

---

### 🔑 Question 3:
**Is your model biased toward one class (>$50K or ≤$50K)? How did you detect this, and what might you do to fix it?**  
🎯 *Purpose: Tests understanding of class imbalance and metric interpretation.*

💡 **Hint:**  
Inspect confusion matrix, precision/recall per class.  
Use `.value_counts()` on the `income` label to see imbalance.  
Consider using `class_weight='balanced'` or resampling techniques.

✏️ *Your answer here...*

---

### 🔑 Question 4:
**What features were most important in your best-performing model, and do they align with expectations about income prediction?**  
🎯 *Purpose: Tests interpretability and domain reasoning.*

💡 **Hint:**  
Use `.feature_importances_` for tree models or `.coef_` for Logistic Regression.  
Do features like `education`, `occupation`, or `hours_per_week` appear at the top?  
Visualize using bar plots.

✏️ *Your answer here...*

---

### 🔑 Question 5:
**How did you use MLflow to track your model experiments, and what comparisons did it help you make?**  
🎯 *Purpose: Tests reproducibility and experiment tracking skills.*

💡 **Hint:**  
Log model name, hyperparameters, evaluation metrics, and notes.  
Use MLflow’s comparison view to track which run performed best.  
Share screenshots or describe insights gained.

✏️ *Your answer here...*

---

## ✅ Week 4: Model Selection & Hyperparameter Tuning

### 🔑 Question 1:
**Which hyperparameters did you tune for your best-performing model, and how did you decide which ones to adjust?**

💡 **Hint:**
For Logistic Regression: C, solver.
For Random Forest: n_estimators, max_depth, min_samples_split.
For XGBoost: learning_rate, max_depth, n_estimators.

✏️ Your answer here...

---

### 🔑 Question 2:
**What method did you use for hyperparameter tuning (e.g., GridSearchCV, RandomizedSearchCV), and why?**

💡 **Hint:**
GridSearch = exhaustive but slow.
RandomizedSearch = faster, good for large search spaces.

✏️  *Your answer here...*


---

### 🔑 Question 3:
**How did the tuned model’s performance compare to the baseline version, and what does that tell you about the value of tuning?**

💡 **Hint:**
Compare metrics (accuracy, F1, AUC) from Week 3 and Week 4 side-by-side.
Small improvements may still matter in real-world deployment.

✏️  *Your answer here...*

---

### 🔑 Question 4:
**What risk of overfitting did you observe during tuning, and how did you mitigate it?**

💡 **Hint:**
Use cross-validation and monitor gap between train/test metrics.
Apply early stopping (XGBoost), pruning (trees), or reduce model complexity.

✏️  *Your answer here...*

---

### 🔑 Question 5:

**Which model and hyperparameters did you ultimately select for deployment, and why?**

💡 **Hint:**
Explain the tradeoff between interpretability, performance, and generalization.
Justify final parameter choices and their business relevance.

✏️  *Your answer here...*

---

## ✅ Week 5: Model Deployment

### 🔑 Question 1:
**How did you design the user interface and user experience of your Streamlit app? What considerations did you make to ensure usability for non-technical users?**  
🎯 *Purpose: Tests ability to translate technical models into accessible tools.*

💡 **Hint:**  
Discuss layout, input forms, instructions, and any visualizations included.  
Mention how you handled user errors or invalid inputs.

---

### 🔑 Question 2:
**Describe the steps you took to deploy your Streamlit app. What challenges did you encounter during deployment, and how did you resolve them?**  
🎯 *Purpose: Evaluates practical deployment skills and troubleshooting.*

💡 **Hint:**  
List steps from local testing to deployment (e.g., requirements.txt, Streamlit Cloud setup).  
Mention issues like dependency conflicts, environment variables, or app crashes.

---

### 🔑 Question 3:
**How does your deployed app handle new or unexpected user inputs? What measures did you implement to ensure robustness and reliability?**  
🎯 *Purpose: Assesses defensive programming and error handling.*

💡 **Hint:**  
Discuss input validation, default values, and error messages.  
Explain how you prevent the app from crashing or producing misleading results.

---

### 🔑 Question 4:
**How did you ensure that your deployed model remains consistent with your training environment? What steps did you take to manage dependencies and model artifacts?**  
🎯 *Purpose: Tests understanding of reproducibility and environment management.*

💡 **Hint:**  
Mention use of requirements files, MLflow model export, or version control.  
Discuss how you loaded the trained model in the app.

---

### 🔑 Question 5:
**If you were to extend this app for real-world business use, what additional features or improvements would you prioritize? Why?**  
🎯 *Purpose: Encourages product thinking and awareness of business needs.*

💡 **Hint:**  
Consider user authentication, logging, explainability features, or integration with databases.  
Discuss how these changes would add value for end users or stakeholders.

## ✨ Final Reflections

> What did you learn from this project? What would you do differently next time? What did AI tools help you with the most?

✏️ *Your final thoughts here...*

---

# 📄 IncomeInsight – Project Report - 🔴 **Advanced Track**

Welcome to your personal project report!  
Use this file to answer the key reflection questions for each phase of the project. This report is designed to help you think like a data scientist, guide AI tools more effectively, and prepare for real-world job interviews.

---

## ✅ Week 1: Setup & Exploratory Data Analysis (EDA)

> Answer the EDA questions provided in the project materials here. Focus on data quality, trends, anomalies, and relationships.

### 🔑 Question 1: What features show the strongest correlation with earning >$50K?

### 🔑 Question 2: How does income vary with education, marital status, or hours worked per week?M

### 🔑 Question 3: Are there disparities across race, sex, or native country?

### 🔑 Question 4: Do capital gains/losses strongly impact the income label?

---

## ✅ Week 2: Feature Engineering & Data Preprocessing

### 🔑 Question 1:
**Which high-cardinality categorical features (e.g., `occupation`, `native_country`) are best suited for embeddings, and how did you determine the embedding dimensions for each?**

💡 **Hint:**  
Use `.nunique()` to assess cardinality.  
Use heuristics like `min(50, (n_unique + 1) // 2)` for embedding dimension.  
Consider category frequency: are there rare classes that may cause overfitting?

✏️ *Your answer here...*

---

### 🔑 Question 2:
**What preprocessing steps did you apply to the numerical features before feeding them into your FFNN, and why are those steps important for deep learning models?**

💡 **Hint:**  
Inspect `df.describe()` and histograms.  
Apply `StandardScaler`, `MinMaxScaler`, or log transformations based on spread and skew.  
Avoid scaling label-encoded categorical values.

✏️ *Your answer here...*

---

### 🔑 Question 3:
**Did you create any new features or interactions, and what evidence suggests they might improve predictive performance?**

💡 **Hint:**  
Visualize combinations of features across income brackets.  
Use correlation with the target, separation by class, or logic from social/economic context.  
Try binary flags or ratios.

✏️ *Your answer here...*

---

### 🔑 Question 4:
**Which features (if any) did you decide to exclude from the model input, and what was your reasoning?**

💡 **Hint:**  
Drop features with very low variance, high missingness, or high correlation to others.  
Ask: Does this feature introduce noise or offer little predictive power?

✏️ *Your answer here...*

---

### 🔑 Question 5:
**What is the distribution of the target class in your dataset, and how might this class imbalance affect your model’s learning and evaluation?**

💡 **Hint:**  
Use `.value_counts(normalize=True)` to check balance of >50K vs ≤50K.  
Class imbalance may require:
- Stratified sampling  
- Weighted loss functions  
- Evaluation via precision, recall, F1, and AUC instead of just accuracy.

✏️ *Your answer here...*


---

### ✅ Week 3: Model Development & Experimentation

### 🔑 Question 1:
**What architecture did you design for your neural network (layers, activations, embeddings, etc.), and how did you choose the embedding sizes for categorical features?**  
🎯 *Purpose: Tests understanding of FFNN design and embedding layer logic.*

💡 **Hint:**  
Describe your architecture, e.g., `[inputs → embeddings → dense layers → dropout → sigmoid output]`.  
Use rules of thumb for embedding sizes like `min(50, (n_unique + 1) // 2)`.  
Justify choices based on cardinality and model complexity.

✏️ *Your answer here...*

---

### 🔑 Question 2:
**What loss function, optimizer, and evaluation metrics did you use for training, and how did your model perform on the validation set?**  
🎯 *Purpose: Tests alignment between loss, task type, and evaluation strategy.*

💡 **Hint:**  
Use `binary_crossentropy` (or BCEWithLogits), `Adam` optimizer, and track metrics like F1-score and AUC.  
Plot learning curves and confusion matrix.  
Summarize validation performance across metrics.

✏️ *Your answer here...*

---

### 🔑 Question 3:
**Did your model show signs of overfitting or underfitting during training? How did you detect this, and what adjustments did you make?**  
🎯 *Purpose: Tests ability to read learning curves and apply regularization.*

💡 **Hint:**  
Plot training vs. validation loss.  
Use early stopping, dropout, or batch normalization to control overfitting.  
Underfitting may require deeper/wider models or longer training.

✏️ *Your answer here...*

---

### 🔑 Question 4:
**How did your neural network's performance compare to a traditional baseline model (e.g., Logistic Regression or XGBoost), and what does that tell you about model suitability for this problem?**  
🎯 *Purpose: Tests comparative model reasoning and suitability of deep learning for tabular data.*

💡 **Hint:**  
Train and evaluate a traditional model using the same features.  
Compare AUC, F1, accuracy.  
Reflect on what your FFNN captured that the baseline didn’t — or vice versa.

✏️ *Your answer here...*

---

### 🔑 Question 5:
**What experiments did you track using MLflow, and how did that help you evaluate and iterate on your model?**  
🎯 *Purpose: Tests reproducibility and experimentation discipline.*

💡 **Hint:**  
Log model parameters (e.g., learning rate, dropout), metrics, and training duration.  
Use MLflow’s comparison UI to track the best run.  
Share how this process helped you debug or improve your architecture.

✏️ *Your answer here...*

---

## ✅ Week 4: Model Selection & Hyperparameter Tuning

### 🔑 Question 1:

**Which neural network hyperparameters did you experiment with, and how did you narrow down the search space?**
🎯 *Purpose: Tests familiarity with DL tuning components and experiment scope.*

💡 **Hint:**
Tune learning rate, hidden layers, neurons per layer, dropout, batch size, epochs.
Use validation performance and early stopping to limit overfitting.

✏️ *Your answer here...*

---

### 🔑 Question 2:

**What tuning strategy did you follow (e.g., manual tuning, learning rate scheduler, Optuna), and why did you choose it?**
🎯 *Purpose: Tests awareness of tuning methodologies and their applicability.*

💡 **Hint:**
Manual = intuitive but slow.
Grid/Random = more systematic.
Optuna = automated and efficient for deep learning.

✏️ *Your answer here...*

---

### 🔑 Question 3:

**How did tuning impact your validation metrics (e.g., F1, ROC-AUC), and which configuration performed best?**
🎯 *Purpose: Tests metric-based model comparison and performance insight.*

💡 **Hint:**
Present before/after tuning metrics in a table or chart.
Highlight improvements and justify which config you picked.

✏️ *Your answer here...*

---

### 🔑 Question 4:

**How did you use MLflow to track tuning experiments, and what insights did you gain from visualizing the logs?**
🎯 *Purpose: Tests use of tooling for iterative development and reproducibility.*

💡 **Hint:**
Log hyperparameters, metrics, training time, and run notes.
Use MLflow comparison dashboard to pick best model.

✏️ *Your answer here...*

---

### 🔑 Question 5:

**Which model architecture and hyperparameter combination did you finalize for deployment, and how confident are you in its robustness?**
🎯 *Purpose: Tests model selection logic and confidence in generalization.*

💡 **Hint:**
Justify final selection based on validation stability, test performance, and interpretability.

✏️ *Your answer here...*

---

## ✅ Week 5: Model Deployment

### 🔑 Question 1:
**How did you architect the Streamlit app to support deep learning inference and SHAP-based interpretability? What design choices did you make for the user interface and experience?**  
🎯 *Purpose: Tests ability to translate complex models and explanations into accessible tools.*

💡 **Hint:**  
Describe how you structured the app (input forms, prediction output, SHAP visualizations).  
Discuss layout, instructions, and how you made interpretability features understandable for users.

---

### 🔑 Question 2:
**Describe the process you followed to deploy your Streamlit app with the deep learning model and SHAP integration. What technical challenges did you face, and how did you resolve them?**  
🎯 *Purpose: Evaluates practical deployment skills and troubleshooting with advanced dependencies.*

💡 **Hint:**  
List steps from local testing to deployment (requirements, model serialization, SHAP setup).  
Mention issues like large model files, dependency conflicts, or resource limits.

---

### 🔑 Question 3:
**How does your deployed app handle unexpected or invalid user inputs, and what steps did you take to ensure reliability and security?**  
🎯 *Purpose: Assesses robust error handling and secure deployment practices.*

💡 **Hint:**  
Discuss input validation, default values, and error messages.  
Explain how you prevent crashes, misleading results, or exposure of sensitive information.

---

### 🔑 Question 4:
**How did you ensure consistency between your training and deployment environments, especially regarding deep learning and SHAP dependencies?**  
🎯 *Purpose: Tests understanding of reproducibility and environment management for advanced ML.*

💡 **Hint:**  
Mention use of requirements files, Docker, or environment.yml.  
Describe how you exported and loaded the trained model and SHAP explainer.

---

### 🔑 Question 5:
**If you were to extend this app for production use, what advanced features or improvements would you prioritize, and why?**  
🎯 *Purpose: Encourages product thinking and awareness of advanced deployment needs.*

💡 **Hint:**  
Consider authentication, logging, batch prediction, model monitoring, or scalable hosting.  
Discuss how these would benefit users or stakeholders in a real-world scenario.

## ✨ Final Reflections

> What did you learn from this project? What would you do differently next time? What did AI tools help you with the most?

✏️ *Your final thoughts here...*

---

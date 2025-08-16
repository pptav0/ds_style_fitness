# 🚀 Candidate Fit Prediction with XGBoost & Bayesian Hyperparameter Tuning

## 🔍 Business Case
Organizations in health, wellness, and fitness industries face a **dual challenge when screening individuals for “fitness”**:

- **Missed opportunities**: Potential “Fit” candidates are incorrectly rejected (false negatives), meaning valuable individuals are overlooked.
- **Wasted resources**: Unsuitable “Not Fit” candidates are misclassified as “Fit” (false positives), leading to inefficiencies and unnecessary costs.

Both outcomes directly affect **business performance** — lowering efficiency, increasing operational risks, and misallocating resources.

A reliable predictive model addresses these challenges by:
- Enabling **smarter prioritization** of candidates,
- Reducing **false classifications**, and
- Ensuring resources are directed where they generate the most value.

Ultimately, this strengthens decision-making, minimizes risk, and maximizes the **business impact** of fitness programs, health screenings, or insurance assessments.

---

## 🛠️ Methodology - Approach
This project applies a combination of **advanced machine learning** and **Bayesian optimization**:

1. **Modeling**
   - Used **XGBoost**, a state-of-the-art gradient boosting classifier, well-suited for structured/tabular data.

2. **Hyperparameter Optimization**
   - Applied **Bayesian Probabilistic Hyperparameter Tuning with Optuna**.
   - Efficiently searches the parameter space, avoiding wasteful grid/random searches.

3. **Objective Function**
   - Custom-defined to balance **AUC, precision, recall, and F1**.
   - Ensures alignment with **business priorities** (capturing true “Fits” while filtering out “Not Fits”).

4. **Evaluation**
   - **Cross-validation** for robust estimates.
   - **ROC-AUC curves** to assess classification power.
   - **Precision-Recall curves** to evaluate trade-offs under class imbalance.

---

## 📊 Results
The tuned XGBoost model delivered **clear improvements** over the baseline:

- **AUC** and **Accuracy** increased → confirming stronger generalization.
- **Class 1 (“Fit”)**: Recall and F1 improved → fewer missed opportunities.
- **Class 0 (“Not Fit”)**: Precision increased → fewer wasted resources.
- **Macro F1** and **Weighted F1** gains → balanced improvements across both classes.

**In summary**:
By combining **XGBoost** with **Bayesian hyperparameter tuning**, the project demonstrates how data science can deliver **direct business value**:
- Higher efficiency
- Lower risk
- Smarter, more confident decision-making

---

## 📂 Repository Structure
style_fitness/
│── data/ 										# Datasets (raw/processed)
│
│── style_fitness/
│ ├── helpers/
│ │ ├── data_handler.py 			# Dataset handling and transformations
│ │ ├── data_preparation.py 	# Preprocessing pipeline
│ │ ├── plots.py 							# Custom visualization helpers (ROC, PR curves)
│ │ └── init.py
│ │
│ ├── notebooks/ 									# Jupyter notebooks for experimentation
│ │ ├── project_analysis.ipynb 		# Model analysis
│ │ └── init.py
│ │
│ └── init.py
│
│── .gitignore
│── poetry.lock 				# Dependency lockfile
│── pyproject.toml 			# Project dependencies & configuration
│── README.md 					# Project documentation


---

## 📈 Visuals
- **ROC Curves** → better separation of Fit vs Not Fit
- **Precision-Recall Curves** → higher precision across thresholds
- **Comparison Tables** → percentage improvements vs. baseline

---

## ✅ Conclusion
This project shows how combining **machine learning (XGBoost)** with **Bayesian optimization (Optuna)** enables the development of **robust, business-aligned predictive models**.

The tuned model improves both efficiency and confidence:
- More true “Fit” candidates are captured.
- “Not Fit” candidates are filtered with higher precision.
- Resources are directed to the right opportunities with **lower misclassification risk**.

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/style_fitness.git
cd style_fitness

# Install dependencies (using Poetry)
poetry install

# **Obesity Level Estimation Using Machine Learning**  
### [Statistical Learning](https://unibg.coursecatalogue.cineca.it/insegnamenti/2024/38091-MOD2/2021/8865/89?coorte=2024&adCodRadice=38091)   Project – Master's in Computer Engineering (Data Science & Data Engineering Pathway)  

## **Overview**  
This project aims to estimate obesity levels in individuals from **Mexico, Peru, and Colombia** based on their **eating habits and physical condition**. Using **machine learning classification models**, we analyze patterns in the dataset to predict obesity levels with high accuracy.  

## **Dataset Information**  
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition)  
- **Instances:** 2111  
- **Features:** 16  
- **Target Variable:** *NObesity* (Obesity Level)  
- **Classes:**  
  - Insufficient Weight  
  - Normal Weight  
  - Overweight Level I  
  - Overweight Level II  
  - Obesity Type I  
  - Obesity Type II  
  - Obesity Type III  
- **Data Generation:**  
  - **77% Synthetic** (via Weka & SMOTE)  
  - **23% Collected directly** from users via a web platform  

## **Models Applied & Results**  

We implemented several **classification models** to evaluate their predictive performance, leveraging **GridSearchCV for hyperparameter tuning** and **5-fold cross-validation** to ensure generalization.  

### **Performance Summary**  

| Model                | Best Hyperparameters | Mean CV Accuracy | Test Accuracy (%) | Train Accuracy (%) |
|----------------------|----------------------|------------------|-------------------|-------------------|
| **Logistic Regression** | ElasticNet (L1/L2) | **87%** | **87.36%** | **91.05%** |
| **Decision Tree** | Depth = 10, Entropy | **91%** | **93.68%** | **98.79%** |
| **Random Forest** | 400 Estimators, Entropy | **91%** | **91.95%** | **96.10%** |
| **SVM** | Linear Kernel, C=5 | **95%** | **96.74%** | **97.76%** |
| **AdaBoost** | Learning Rate = 0.5, 500 Estimators | **96%** | **95.02%** | **99.89%** |

**Insights:**  
- **SVM** achieved the **highest test accuracy** (96.74%).  
- **AdaBoost** had **strong generalization**, balancing precision and recall effectively.  
- **Decision Tree & Random Forest** performed well but showed signs of overfitting.  
- **Logistic Regression** had lower accuracy compared to ensemble-based models, but remained interpretable.  

## Methodology
1- **Data Preprocessing** → Standardization, PCA for dimensionality reduction, SMOTE for class balancing  
2- **Model Training** → Decision Tree, Random Forest, AdaBoost, SVM, Logistic Regression  
3- **Hyperparameter Optimization** → GridSearchCV for best parameter selection  
4- **Cross-Validation** → 5-Fold CV to ensure robustness  
5- **Model Evaluation** → Accuracy, F1 Score, Confusion Matrix, Precision, Recall  

##  Installation & Dependencies
To replicate the project, install the required dependencies:

```bash
pip install -r requirements.txt
```
**Dependencies:**  
```
numpy  
pandas  
matplotlib  
plotly  
seaborn  
scikit-learn  
imbalanced-learn  
xgboost  
```

## Running the Project
1- Clone the repository  

2- Install dependencies  
```bash
pip install -r requirements.txt
```
3- Run the Jupyter Notebook  
```bash
jupyter notebook
```
4- Execute the analysis  

##  Conclusion
This project demonstrates the effectiveness of **machine learning** in predicting obesity levels based on dietary and physical behavior data. The **ensemble models (AdaBoost & SVM) outperform traditional methods**, highlighting the impact of **boosting techniques** in classification problems.  



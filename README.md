# **Obesity Level Estimation Using Machine Learning**  
### [Statistical Learning](https://unibg.coursecatalogue.cineca.it/insegnamenti/2024/38091-MOD2/2021/8865/89?coorte=2024&adCodRadice=38091) - Master's in Computer Engineering (Data Science & Data Engineering Pathway)  

## **Overview**  
This project aims to estimate obesity levels in individuals from **Mexico, Peru, and Colombia** based on their **eating habits and physical condition**. Using **machine learning classification models**, patterns in the dataset are analyzed to predict obesity levels with high accuracy.  

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

Several **classification models** were implemented to evaluate predictive performance, leveraging **GridSearchCV for hyperparameter tuning** and **5-fold cross-validation** to ensure generalization.  

### **Performance Summary**  

| Model                | Best Hyperparameters | CV Accuracy (%) | Test Accuracy (%) | Zero-One Loss | F1 Score (%) |
|----------------------|----------------------|----------------|----------------|--------------|-------------|
| **Logistic Regression** | ElasticNet (L1/L2) | **87%** | **87.36%** | **66.0** | **87.36%** |
| **Decision Tree** | Depth = 10, Entropy | **91%** | **93.68%** | **33.0** | **93.68%** |
| **Random Forest** | 400 Estimators, Entropy | **91%** | **91.95%** | **42.0** | **91.95%** |
| **SVM** | Linear Kernel, C=5 | **95%** | **96.74%** | **17.0** | **96.74%** |
| **AdaBoost** | Learning Rate = 0.6, 300 Estimators | **92%** | **88.12%** | **62.0** | **88.12%** |

### **Key Observations**  
- **SVM achieved the highest test accuracy (96.74%)**, indicating superior generalization.  
- **Decision Tree performed well (93.68%)**, but showed minor overfitting in training.  
- **Random Forest provided stable predictions (91.95%)**, with lower variance compared to Decision Tree.  
- **Logistic Regression**, serving as the baseline, performed moderately (87.36%).  
- **AdaBoost**, despite strong cross-validation accuracy (92%), exhibited lower test accuracy (88.12%), suggesting generalization challenges.  

## **Methodology**
1- **Data Preprocessing** → Standardization, PCA for dimensionality reduction, SMOTE for class balancing  
2- **Model Training** → Decision Tree, Random Forest, AdaBoost, SVM, Logistic Regression  
3- **Hyperparameter Optimization** → GridSearchCV for best parameter selection  
4- **Cross-Validation** → 5-Fold CV to ensure robustness  
5- **Model Evaluation** → Accuracy, F1 Score, Confusion Matrix, Precision, Recall  

## **Installation & Dependencies**
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



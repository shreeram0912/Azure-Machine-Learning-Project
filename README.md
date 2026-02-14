# Credit Card Fraud Detection using Machine Learning

### **📌 Project Overview**
Credit card fraud is a growing concern for financial institutions and customers worldwide. Fraudulent transactions are rare compared to legitimate ones, making detection challenging due to highly imbalanced 
datasets. This project leverages machine learning models to classify transactions as either fraudulent or legitimate, integrating cloud-based services for scalability, security, and monitoring.

### **🎯 Problem Statement**
Credit card fraud detection requires building a system that can accurately identify fraudulent transactions while minimizing false positives and false negatives. The challenge lies in handling imbalanced 
datasets and ensuring real-time detection capabilities.

### **💡 Proposed Solution**
The system uses machine learning algorithms to analyze transaction patterns and classify them as fraudulent or legitimate.
1. **Data Collection:**
  * Historical transaction data from kaggle.com, https://drive.google.com/file/d/1T2P7HmG7aH0tDAY41_dhhhkpldjyu550/view?usp=sharing.
  * Secure storage of datasets in Azure Blob Storage.

2. **Data Preprocessing**
  * Cleaning and handling missing values, outliers, and inconsistencies.
  * Feature engineering to derive attributes such as unusual spending patterns.
  * Normalization and scaling for consistent input features.

3. **Machine Learning Algorithm**
  * Azure ML Designer: Two-Class Decision Forest.
  * Google Colab: Random Forest Classifier (scikit-learn).

4. **Deployment**
  * Model is ready to deployed on Azure Machine Learning Designer as a REST API.
  * Credentials secured using Azure Key Vault.
  * Monitoring with Azure Log Analytics for performance and anomaly detection.

5. **Evaluation**
  * Metrics: Accuracy, Precision, Recall, F1-Score, MCC.
  * Confusion matrix visualization for classification effectiveness.

### **⚙️ System Development Approach (Technology Used)**
* **Azure Machine Learning Designer** – Model design, training, and deployment.
* **Azure Log Analytics** – Monitoring and logging.
* **Azure Blob Storage** – Secure dataset storage.
* **Azure Key Vault** – Credential and key management.
* **Google Colab** – Experimentation and prototyping with Python libraries.

### **🔍 Algorithm & Deployment**
1. **Algorithm Selection**
  * Two-Class Decision Forest (Azure ML Designer): Ensemble method chosen for robustness and handling imbalanced datasets.
  * Random Forest Classifier (scikit-learn in Colab): Effective classification algorithm resistant to overfitting.

2. **Training Process**
  * Cross-validation and hyperparameter tuning in Azure ML Designer.
  * Stratified sampling and GridSearchCV tuning in Google Colab.

3. **Prediction Process**
  * Real-time predictions via Azure ML REST API after deploying the model to endpoint.
  * Batch predictions and validation in Google Colab.

### **📊 Results**
1. **Azure Machine Learning Designer - Two-Class Decision Forest**
   * Azure Machine Learning Resource Group: <img width="940" height="358" alt="image" src="https://github.com/user-attachments/assets/647aeed7-5ae2-4a37-adad-c112e0416cc2" />
   * Azure Machine Learning Designer Pipeline: <img width="940" height="630" alt="image" src="https://github.com/user-attachments/assets/50b72c00-6885-48c6-8f23-3a34ab05295e" />
   * Result: <img width="940" height="403" alt="image" src="https://github.com/user-attachments/assets/2c7e51ec-c9cc-4eb6-9bdd-19135849a9eb" />
   * Profiling: <img width="940" height="562" alt="image" src="https://github.com/user-attachments/assets/a7006b6a-a9dc-4dd0-b70f-5990a61348b8" />
2. **Google Colab - Random Forest Classifier**
   * <img width="851" height="830" alt="image" src="https://github.com/user-attachments/assets/e4d3e09c-c546-41dd-876a-16440a44d621" />
   * Correlation between features: <img width="774" height="652" alt="image" src="https://github.com/user-attachments/assets/1e81004a-f804-417b-8d6a-8c2d4d744193" />
3. **Azure Machine Learning Designer Video:** https://drive.google.com/file/d/1p_A5l3-YYjYnrSeCLaotxbYlMJ9U2GwC/view?usp=sharing
   * Confusion matrix confirms effective classification, though recall highlights missed fraud cases.

### **✅ Conclusion**
The project demonstrates the successful application of machine learning for fraud detection. While accuracy is high, recall indicates the need for further improvement to capture more fraudulent cases. 
Integration with Azure ensures secure, scalable, and monitored deployment.

### **🚀 Future Scope**
* Deployment: Production-ready APIs for real-time fraud detection.
* Consumption: Integration into financial transaction monitoring systems.
* Enhancements: Explore advanced algorithms (XGBoost, deep learning) to improve recall.
* Monitoring: Continuous refinement using Azure Log Analytics.
* Security: Enhanced credential management with Azure Key Vault.

### **📚 References**
* AICTE Azure Internship Team
* Azure Machine Learning Documentation: https://learn.microsoft.com/en-us/azure/machine-learning/concept-designer?view=azureml-api-2

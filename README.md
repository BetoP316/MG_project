# 📊 Predictive Modeling for Debt Recovery Profile
🚀 **Enhancing Debt Collection Strategies with Machine Learning & Statistical Models**

## 📌 Project Overview
In an economic landscape marked by **high uncertainty and rising default rates**, financial institutions need **predictive tools** to optimize debt recovery strategies. This study develops a **logistic regression model** enhanced by **machine learning techniques** to identify **high-risk clients** and refine collection strategies.

🔹 **Key Takeaways:**

✅ Predictive modeling **reduces collection costs** and improves efficiency 📉  
✅ Machine learning helps **prioritize high-risk debtors** 🎯  
✅ **Interest rates and installment size** are key factors affecting repayment probability 💰  
✅ Logistic regression and Random Forest models offer **~96% accuracy** in default prediction 🔍  


## 📊 Methodology & Approach

### **1️⃣ Data Collection**
I analyzed **credit portfolios, macroeconomic indicators, and socioeconomic factors** affecting debt repayment behavior.

📌 **Key Data Sources:**
- **Credit Portfolio Data** (Microgestión S.A.) 📑  
- **Macroeconomic Trends** (Ecuadorian National Statistics Institute) 📈  
- **Customer Socioeconomic Profiles** 👥  

| Data Type | Variables |
|------------|-----------|
| Customer Info | Age, Gender, Civil Status, Income Level |
| Loan Details | Interest Rate, Installment Amount, Credit Type |
| Economic Factors | Employment Rate, Industry Type, Inflation |


### **2️⃣ Predictive Modeling: Logistic Regression & Machine Learning**

I developed a **logistic regression model** to estimate **default probability**, minimizing **risk exposure** while improving collection efficiency.

📌 **Model Equation:**
```math
P(IMPAGO = 1 | X) = \frac{1}{1 + e^{-\sum \beta_i X_i}}
```

* Where  X represents loan and customer characteristics affecting default probability.* 

🔹 **Key Features Impacting Default Risk:**

✅ **Interest on active & overdue installments** (22-24% impact on default probability)  
✅ **Installment capital amount** (Lower capital reduces default risk)  
✅ **Age & credit cycle** (New clients are higher risk)  

📊 **Performance Metrics:**

| Model | Accuracy | Precision | Recall |
|------------|-----------|-----------|-----------|
| **Logistic Regression** | 96% ✅ | 94% | 95% |
| **Random Forest** | 90% | 91% | 89% |
| **Decision Tree** | 91% | 86% | 90% |

📌 **Key Takeaway:** Machine learning models **enhance debt collection accuracy** by filtering high-risk clients.  

## 🎯 Key Insights & Results

📌 **Main Findings:**

✅ Clients **with higher overdue interest rates** are at greater risk of default  
✅ **First-time borrowers have a 40% default rate**, while repeat borrowers have almost zero risk  
✅ **Clients in urban areas default more frequently** than rural borrowers  
✅ **Machine learning models improve risk identification by 10%** over traditional statistical models  

📊 **Chi-Square Tests confirm significant differences** between repayment behaviors across **age, credit cycle, and financial stability**.  

## 🔧 Implementation & Reproducibility

### **📂 Repository Structure**
```yaml
├── CODE/              # Python scripts for modeling & predictions
├── GRAPHS/            # Visualizations & analytical charts
├── Corr_table.xlsx    # Correlation table for feature analysis
└── README.md          # Project Overview
```

## 📌 Future Enhancements
🔹 **Incorporating deep learning models (Neural Networks) for better predictions** 🧠  
🔹 **Integrating real-time financial market data** to refine credit risk scoring 📊  
🔹 **Enhancing customer segmentation** for personalized debt recovery strategies 🎯  

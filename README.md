****Breast Cancer Diagnosis using WDBC Dataset****  
  
**Overview**  
This project analyzes the Wisconsin Diagnostic Breast Cancer (WDBC) dataset to identify significant relationships 
between characteristics of a cell nucei and the diagnosis of that tumor. Using methods such as statistical 
analysis and multiple charts and graphs. The features we focus on within this data set include how features such
as radius, texture, and perimeter relate to the diagnosis of the tumor.  

**Key Features**  
**Radius Mean**:  Measures the average distance from tumor to center to its edge.  
**Perimeter Mean**: Represents the total length around the tumor's edge.  
**Area Mean**: Indicates the size of the tumor.  

  Classification of Diagnosis:  
  - **M**: Malignant (Harmful)
  - **B**: Benign (Non-Harmful)

**Project Hypothesis**  
  1. **Null Hypothesis**: the features have no statisticallyt significant relationship to the diagnosis of the tumor.
  2. **Alternate Hypothesis**: One or more of the variables have a statisically significant relationship to the diagnosis of the tumor.


**Dataset**  
The dataset was downloaded from Kaggle from the Wisconsin Diagnostic Breast Cancer (WDBC) dataset. Key Features of the dataset include:  
- **Radius**
- **Perimeter**
- **Area**
- **Diagnosis**  
    
**Tools & Libraries**  
- **Python**:
1. **pandas** for data manipulation
2. **seaborn** and **matplotlib** for data visuals
**Jupyter Notebook**

**Visualizations**  
The scatter plot that was generated using the code below provides information pertaining to the relationship into our features and their correlation with the tumor diagnosis. For ease of viewing, Malignant and Benign cases are color-coded:  
- **Red**: Malignant
- **Blue**: Benign  
  
**Conclusion**  
The statistical analysis confirms the alternate hypothesis that certain features outlined in our dataset, such as perimeter, area, and radius all have strong correlation with the diagnosis of a tumor.  

**Root**  
├── data.csv             # The cleaned dataset  
├── wdbc.py              # Analysis and visualization script  
├── Breast Cancer Dataset.pptx  # Supporting presentation  
└── README.md            # Project documentation    
  
**How to Run**  
  
**Clone the repository**: Open your Python environment and clone the repository:  
import os  
os.system("git clone https://github.com/hgonzalez542/Wisconsin-Breast-Cancer-Diagnostics")  
  
**Navigate to the directory**:  
os.chdir("breast-cancer-diagnosis")  
  
**Install dependencies**:  
import os  
os.system("pip install pandas seaborn matplotlib")  

**Run the script:**  
exec(open("wdbc.py").read())  

**Authors:**
- Bryce Costa
- Hector Gonzalez 

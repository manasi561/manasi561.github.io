---
layout: post
title: Credit Risk Management
image: "/posts/crm-title.jpg"
tags: [Python, Classification, Logistic Regression, Random Forest]
---

Our client, a banking and financial service company, hired an analytics consultancy to get insights and to evaluate the performance of their credit management system and determine a borrower's ability to meet their debt obligation. Let's do this!

- [00. Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Actions](#overview-actions)
    - [Results](#overview-results)
    - [Growth/Next Steps](#overview-growth)
- [01. Data Overview and Preprocessing](#data-overview)
    - [Missing Values](#missing-values)
    - [Categorical Features](#categorical-features)
    - [Train-Test Split](#train-test)
    - [Feature Scaling](#feature-scaling)
- [02. Random Forest](#rf-title)
- [03. Performace Metrics - Random Forest](#rf-metrics)
- [04. Logistic Regression](#logreg-title)
- [05. Performace Metrics - Logistic Regression](#logreg-metrics)
- [06. Finding Optimal Threshold - Logistic Regression](#logreg-threshold)
- [07. Conclusion](#conclusion)


# Project Overview  <a name="overview-main"></a>

### Context <a name="overview-context"></a>

The overall aim of this project is to help the lender bank or financial institute to accurately determine if borrower(s) will default on their contractual obligations, inorder to cushion itself from loss due to borrower's failure to repay loan ammount. Credit Risk Management is essential to various financial organizations as it enables the business to maximise sales while carefully managing its risk exposure.

To ascertain this, we build a classification predictive model that will find relationships between customer details and their credit worthiness for our customers with historical data, and use this to predict the future scope for those who are not.
<br>
<br>

### Actions <a name="overview-actions"></a>

We first perform necessary Data Exploration and Data preprocessing steps to prepare our data to be passed for model training. As we are predicting a binary classification output, we tested two classification modelling approaches, namely:

* Logistic Regression
* Random Forest Classification
<br>
<br>

### Results <a name="overview-results"></a>

Our testing found that the Random Forest had the highest predictive accuracy. However, as we have imbalanced class data, accuracy is not a good metric 
to judge our model performance.

In case of imbalance data, F1 score would be our choice to assess model performace.
We tested many classification thresholds, and found that the optimal f1-score came at a threshold of 0.12.



**Metric 1: Accuracy**

* Random Forest = 99.74%
* Logistic Regression = 99.33%

**Metric 2: Precision**

* Random Forest = 99.96%
* Logistic Regression = 99.67%

**Metric 2: Recall**

* Random Forest = 96.64%
* Logistic Regression = 91.55%

**Metric 4: F1 score**

* Random Forest = 95.44%
* Logistic Regression = 98.25%
<br>
<br>

### Growth/Next Steps <a name="overview-growth"></a>

While predictive accuracy was relatively high - other classification modelling approaches could be tested, example:
Decision Tree, KNN, SVM, Naive Bayes etc.

From a data point of view, further customer details could be collected, and further feature engineering could be undertaken to ensure that we have as much useful information available for predicting whether a customer will default.
<br>
<br>

# Data Overview  <a name="data-overview"></a>

Our data set consists of customer details from a banking institute. We would be looking after key variables pertaining to customers which help us determine their default status. We will be predicting the default_ind metric which belongs to eith 1 or 0 class.

### Importing required Libraries


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
pd.set_option('display.max_columns',None)
```

### Reading the Dataset


```python
data = pd.read_csv('bank_data.csv', header=0)
```


```python
print(f"There are about {data.shape[0]} rows and {data.shape[1]} features in the dataset")
```

    There are about 598978 rows and 38 features in the dataset
    
We have 38 features in our data set. You can find what each one represents [here](/docs/data_information.txt)

```python
data = shuffle(data)
```

Lets take an overview of our data


```python
data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>loan_amnt</th>
      <th>funded_amnt</th>
      <th>funded_amnt_inv</th>
      <th>int_rate</th>
      <th>installment</th>
      <th>annual_inc</th>
      <th>dti</th>
      <th>delinq_2yrs</th>
      <th>open_acc</th>
      <th>pub_rec</th>
      <th>revol_bal</th>
      <th>revol_util</th>
      <th>total_acc</th>
      <th>out_prncp</th>
      <th>out_prncp_inv</th>
      <th>total_pymnt</th>
      <th>total_pymnt_inv</th>
      <th>total_rec_prncp</th>
      <th>total_rec_int</th>
      <th>total_rec_late_fee</th>
      <th>recoveries</th>
      <th>collection_recovery_fee</th>
      <th>last_pymnt_amnt</th>
      <th>collections_12_mths_ex_med</th>
      <th>acc_now_delinq</th>
      <th>tot_coll_amt</th>
      <th>tot_cur_bal</th>
      <th>default_ind</th>
      <th>rand no</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5.989780e+05</td>
      <td>598978.000000</td>
      <td>598978.000000</td>
      <td>598978.000000</td>
      <td>598978.000000</td>
      <td>598978.000000</td>
      <td>5.989780e+05</td>
      <td>598978.000000</td>
      <td>598978.000000</td>
      <td>598978.000000</td>
      <td>598978.000000</td>
      <td>5.989780e+05</td>
      <td>598637.000000</td>
      <td>598978.000000</td>
      <td>598978.000000</td>
      <td>598978.000000</td>
      <td>598978.000000</td>
      <td>598978.000000</td>
      <td>598978.000000</td>
      <td>598978.000000</td>
      <td>598978.000000</td>
      <td>598978.000000</td>
      <td>598978.000000</td>
      <td>598978.000000</td>
      <td>598922.000000</td>
      <td>598978.000000</td>
      <td>5.316650e+05</td>
      <td>5.316650e+05</td>
      <td>598978.000000</td>
      <td>598978.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.044157e+07</td>
      <td>14540.642519</td>
      <td>14521.789031</td>
      <td>14478.973948</td>
      <td>13.520911</td>
      <td>434.562058</td>
      <td>7.397280e+04</td>
      <td>17.638408</td>
      <td>0.297171</td>
      <td>11.357197</td>
      <td>0.177768</td>
      <td>1.650743e+04</td>
      <td>55.838410</td>
      <td>25.226733</td>
      <td>5946.808215</td>
      <td>5945.054198</td>
      <td>10073.087967</td>
      <td>10029.098206</td>
      <td>7730.781476</td>
      <td>2274.617695</td>
      <td>0.439511</td>
      <td>67.249287</td>
      <td>7.075547</td>
      <td>2815.793083</td>
      <td>0.011255</td>
      <td>0.004625</td>
      <td>2.089273e+02</td>
      <td>1.389905e+05</td>
      <td>0.077058</td>
      <td>0.499938</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.619182e+07</td>
      <td>8336.476483</td>
      <td>8327.529212</td>
      <td>8337.695086</td>
      <td>4.370961</td>
      <td>242.892709</td>
      <td>5.676163e+04</td>
      <td>8.070746</td>
      <td>0.828659</td>
      <td>5.138351</td>
      <td>0.547331</td>
      <td>2.093414e+04</td>
      <td>23.711238</td>
      <td>11.717682</td>
      <td>7330.142895</td>
      <td>7328.397671</td>
      <td>8091.548244</td>
      <td>8070.839362</td>
      <td>6923.288534</td>
      <td>2270.974523</td>
      <td>4.245093</td>
      <td>491.327816</td>
      <td>74.587986</td>
      <td>5398.379951</td>
      <td>0.119935</td>
      <td>0.075951</td>
      <td>1.268440e+04</td>
      <td>1.525544e+05</td>
      <td>0.266684</td>
      <td>0.288469</td>
    </tr>
    <tr>
      <th>min</th>
      <td>5.473400e+04</td>
      <td>500.000000</td>
      <td>500.000000</td>
      <td>0.000000</td>
      <td>5.320000</td>
      <td>15.690000</td>
      <td>3.000000e+03</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>5.937838e+06</td>
      <td>8000.000000</td>
      <td>8000.000000</td>
      <td>8000.000000</td>
      <td>10.160000</td>
      <td>259.642500</td>
      <td>4.500000e+04</td>
      <td>11.610000</td>
      <td>0.000000</td>
      <td>8.000000</td>
      <td>0.000000</td>
      <td>6.462000e+03</td>
      <td>38.800000</td>
      <td>17.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4242.350000</td>
      <td>4219.600000</td>
      <td>2764.050000</td>
      <td>821.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>305.080000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>2.919700e+04</td>
      <td>0.000000</td>
      <td>0.250400</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.564996e+07</td>
      <td>12600.000000</td>
      <td>12525.000000</td>
      <td>12500.000000</td>
      <td>13.330000</td>
      <td>381.840000</td>
      <td>6.300000e+04</td>
      <td>17.210000</td>
      <td>0.000000</td>
      <td>10.000000</td>
      <td>0.000000</td>
      <td>1.187000e+04</td>
      <td>57.000000</td>
      <td>24.000000</td>
      <td>3032.270000</td>
      <td>3031.615000</td>
      <td>7631.015000</td>
      <td>7593.020000</td>
      <td>5393.340000</td>
      <td>1573.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>507.380000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>8.098600e+04</td>
      <td>0.000000</td>
      <td>0.499900</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.671239e+07</td>
      <td>20000.000000</td>
      <td>20000.000000</td>
      <td>20000.000000</td>
      <td>16.290000</td>
      <td>568.900000</td>
      <td>9.000000e+04</td>
      <td>23.290000</td>
      <td>0.000000</td>
      <td>14.000000</td>
      <td>0.000000</td>
      <td>2.061800e+04</td>
      <td>74.300000</td>
      <td>32.000000</td>
      <td>10070.040000</td>
      <td>10066.860000</td>
      <td>13403.250000</td>
      <td>13349.895000</td>
      <td>10150.382500</td>
      <td>2871.087500</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1506.602500</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>2.084020e+05</td>
      <td>0.000000</td>
      <td>0.749500</td>
    </tr>
    <tr>
      <th>max</th>
      <td>5.095230e+07</td>
      <td>35000.000000</td>
      <td>35000.000000</td>
      <td>35000.000000</td>
      <td>28.990000</td>
      <td>1409.990000</td>
      <td>8.706582e+06</td>
      <td>39.990000</td>
      <td>39.000000</td>
      <td>90.000000</td>
      <td>63.000000</td>
      <td>2.568995e+06</td>
      <td>892.300000</td>
      <td>162.000000</td>
      <td>34073.890000</td>
      <td>34073.890000</td>
      <td>57777.579870</td>
      <td>57777.580000</td>
      <td>35000.030000</td>
      <td>24205.620000</td>
      <td>358.680000</td>
      <td>33520.270000</td>
      <td>7002.190000</td>
      <td>36475.590000</td>
      <td>20.000000</td>
      <td>14.000000</td>
      <td>9.152545e+06</td>
      <td>8.000078e+06</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



### Since default_ind is our target feature, so lets plot its count plot
#### Here '1' indicates applicant has defaulted and is a credit risk and '0' indicates an applicant is not a credit risk


```python
import warnings
with warnings.catch_warnings():
    warnings.simplefilter(action='ignore', category=FutureWarning)
    plt.figure(figsize = (8,6))
    sns.countplot(data['default_ind'])
    plt.title("Count plot for the Target Variable", fontsize = 15)
    plt.xlabel("Target Variable (default_ind)",fontsize = 12)
    plt.ylabel("Count",fontsize = 12)

```


    

    


![alt text](/img/posts/data_distribution_0_1.png "Logistic Regression Feature Selection Plot")

#### We observe that the number of defaulter's count is extremely less as compared to the non-defaulters. This indeed would be the case as we would have more number of people paying their installments on time as compared to those who don't. The banks too will provide loans  only to elligle candidates, which would lead to less number of defaulters.

## Data Preprocessing  <a name="data-preprocessing"></a>

We will perform certain data preprocessing steps to prep our data for model building. This includes:

* Dealing with missing values in the data 
* Encoding categorical variables to numeric form
* Feature Scaling

### A) Dealing with missing values <a name="missing-values"></a>


```python
data.isna().sum()
```




    id                                0
    loan_amnt                         0
    funded_amnt                       0
    funded_amnt_inv                   0
    term                              0
    int_rate                          0
    installment                       0
    grade                             0
    emp_length                    28105
    home_ownership                    0
    annual_inc                        0
    verification_status               0
    issue_d                           0
    pymnt_plan                        0
    dti                               0
    delinq_2yrs                       0
    open_acc                          0
    pub_rec                           0
    revol_bal                         0
    revol_util                      341
    total_acc                         0
    out_prncp                         0
    out_prncp_inv                     0
    total_pymnt                       0
    total_pymnt_inv                   0
    total_rec_prncp                   0
    total_rec_int                     0
    total_rec_late_fee                0
    recoveries                        0
    collection_recovery_fee           0
    last_pymnt_amnt                   0
    collections_12_mths_ex_med       56
    application_type                  0
    acc_now_delinq                    0
    tot_coll_amt                  67313
    tot_cur_bal                   67313
    default_ind                       0
    rand no                           0
    dtype: int64



#### Converting missing values into percentages of the total missing values


```python
(100 * data.isna().sum()) / len(data)
```




    id                             0.000000
    loan_amnt                      0.000000
    funded_amnt                    0.000000
    funded_amnt_inv                0.000000
    term                           0.000000
    int_rate                       0.000000
    installment                    0.000000
    grade                          0.000000
    emp_length                     4.692159
    home_ownership                 0.000000
    annual_inc                     0.000000
    verification_status            0.000000
    issue_d                        0.000000
    pymnt_plan                     0.000000
    dti                            0.000000
    delinq_2yrs                    0.000000
    open_acc                       0.000000
    pub_rec                        0.000000
    revol_bal                      0.000000
    revol_util                     0.056930
    total_acc                      0.000000
    out_prncp                      0.000000
    out_prncp_inv                  0.000000
    total_pymnt                    0.000000
    total_pymnt_inv                0.000000
    total_rec_prncp                0.000000
    total_rec_int                  0.000000
    total_rec_late_fee             0.000000
    recoveries                     0.000000
    collection_recovery_fee        0.000000
    last_pymnt_amnt                0.000000
    collections_12_mths_ex_med     0.009349
    application_type               0.000000
    acc_now_delinq                 0.000000
    tot_coll_amt                  11.237975
    tot_cur_bal                   11.237975
    default_ind                    0.000000
    rand no                        0.000000
    dtype: float64



The missing values in 'emp_lenght' , revol_util and collections_12_mths_ex_med columns are extremely low( < 5%), so instead of applying any imputation, we will eliminate these rows



```python
data = data.dropna(subset=['emp_length'])
data = data.dropna(subset=['revol_util'])
data = data.dropna(subset=['collections_12_mths_ex_med'])
```

We replace the missing values in ""tot_cur_bal" and "tot_coll_amt" by mean values of their respective columns


```python
data["tot_cur_bal"] = data["tot_cur_bal"].fillna(value =  data["tot_cur_bal"].mean())

data["tot_coll_amt"] = data["tot_coll_amt"].fillna(value =  data["tot_coll_amt"].mean())
```


```python
data.isna().sum()
```




    id                            0
    loan_amnt                     0
    funded_amnt                   0
    funded_amnt_inv               0
    term                          0
    int_rate                      0
    installment                   0
    grade                         0
    emp_length                    0
    home_ownership                0
    annual_inc                    0
    verification_status           0
    issue_d                       0
    pymnt_plan                    0
    dti                           0
    delinq_2yrs                   0
    open_acc                      0
    pub_rec                       0
    revol_bal                     0
    revol_util                    0
    total_acc                     0
    out_prncp                     0
    out_prncp_inv                 0
    total_pymnt                   0
    total_pymnt_inv               0
    total_rec_prncp               0
    total_rec_int                 0
    total_rec_late_fee            0
    recoveries                    0
    collection_recovery_fee       0
    last_pymnt_amnt               0
    collections_12_mths_ex_med    0
    application_type              0
    acc_now_delinq                0
    tot_coll_amt                  0
    tot_cur_bal                   0
    default_ind                   0
    rand no                       0
    dtype: int64



After veryfying there are no missing values in our data, we also drop the id column as it is not significant for our model building and training.


```python
data.drop('id',axis=1,inplace=True)
```


### B) Dealing with categorical features <a name="categorical-features"></a>

First, we will find out our categorical features


```python
categorical_variables = data.select_dtypes(['object']).columns
print(f"Features that are Categorical are : {categorical_variables}")
```

    Features that are Categorical are : Index(['term', 'grade', 'emp_length', 'home_ownership', 'verification_status',
           'issue_d', 'pymnt_plan', 'application_type'],
          dtype='object')
    

In our dataset, we have several categorical variables.

The Logistic Regression algorithm can’t deal with data in categorical format as it can’t assign any numerical meaning to it when looking to assess the relationship between the variable and the dependent variable. Hence, we use OneHot Encoding to convert these categorical variables to numeric form. 

One Hot Encoding can be thought of as a way to represent categorical variables as binary vectors, in other words, a set of new columns for each categorical value with either a 1 or a 0 saying whether that value is true or not for that observation. These new columns would go into our model as input variables, and the original column is discarded.

We also drop one of the new columns using the parameter drop = “first”. We do this to avoid the dummy variable trap where our newly created encoded columns perfectly predict each other - and we run the risk of breaking the assumption that there is no multicollinearity, a requirement or at least an important consideration for some models, Logistic Regression being one of them! 

Multicollinearity occurs when two or more input variables are highly correlated with each other, it is a scenario we attempt to avoid as in short, while it won’t neccessarily affect the predictive accuracy of our model, it can make it difficult to trust the statistics around how well the model is performing, and how much each input variable is truly having.



```python
#Term

term_dummies = pd.get_dummies(data['term'],drop_first=True)

data = pd.concat([data.drop('term',axis=1),term_dummies],axis=1)
```


```python
#grade

grade_dummies = pd.get_dummies(data['grade'],drop_first=True)

data = pd.concat([data.drop('grade',axis=1),grade_dummies],axis=1)
```


```python
#emp_length

emp_length = pd.get_dummies(data['emp_length'],drop_first=True)

data = pd.concat([data.drop('emp_length',axis=1),emp_length],axis=1)
```


```python
# home_ownership

#We can merge NONE & ANY types to OTHER, to represent data in a better and concise manner

data['home_ownership'] = data['home_ownership'].replace(['NONE','ANY'],'OTHER')
home_dummies = pd.get_dummies(data['home_ownership'],drop_first=True)
data = pd.concat([data.drop('home_ownership',axis=1),home_dummies],axis=1)
```


```python
#verification_status

verification_dummies = pd.get_dummies(data['verification_status'],drop_first=True)
data = pd.concat([data.drop('verification_status',axis=1),verification_dummies],axis=1)
```


```python
#application_type

data['application_type'].value_counts()

#We observe that there is only 1 value for it.Hence, we drop the column

data.drop('application_type',axis=1,inplace=True)
```


```python

#issue_d feature

#This is the target leak feature. As our aim to identify credit risk applicant, issue date will be a futuristic value and has no impact on current model. Hence we drop the column#

data.drop('issue_d',axis=1,inplace=True)
```


```python
#pymnt_plan

data['pymnt_plan'] = data['pymnt_plan'].map({'y':1,'n':0})
```

Verifying none of our feature variables are in categorical form


```python
data.dtypes
```




    loan_amnt                       int64
    funded_amnt                     int64
    funded_amnt_inv               float64
    int_rate                      float64
    installment                   float64
    annual_inc                    float64
    pymnt_plan                      int64
    dti                           float64
    delinq_2yrs                     int64
    open_acc                        int64
    pub_rec                         int64
    revol_bal                       int64
    revol_util                    float64
    total_acc                       int64
    out_prncp                     float64
    out_prncp_inv                 float64
    total_pymnt                   float64
    total_pymnt_inv               float64
    total_rec_prncp               float64
    total_rec_int                 float64
    total_rec_late_fee            float64
    recoveries                    float64
    collection_recovery_fee       float64
    last_pymnt_amnt               float64
    collections_12_mths_ex_med    float64
    acc_now_delinq                  int64
    tot_coll_amt                  float64
    tot_cur_bal                   float64
    default_ind                     int64
    rand no                       float64
     60 months                      uint8
    B                               uint8
    C                               uint8
    D                               uint8
    E                               uint8
    F                               uint8
    G                               uint8
    10+ years                       uint8
    2 years                         uint8
    3 years                         uint8
    4 years                         uint8
    5 years                         uint8
    6 years                         uint8
    7 years                         uint8
    8 years                         uint8
    9 years                         uint8
    < 1 year                        uint8
    OTHER                           uint8
    OWN                             uint8
    RENT                            uint8
    Source Verified                 uint8
    Verified                        uint8
    dtype: object





### C) Split Out Data For Modelling <a name="train-test"></a>


Next, we first split our data into an X object which contains only the predictor variables, and a Y object that contains only our dependent variable.

Once we have done this, we split our data into training and test sets to ensure we can fairly validate the accuracy of the predictions on data that was not used in training. In this case, we have allocated 75% of the data for training, and the remaining 25% for validation. We make sure to add in the stratify parameter to ensure that both our training and test sets have the same proportion of default and non-default customers.


```python
from sklearn.model_selection import train_test_split,cross_val_score
```


```python
X = data.drop(['default_ind'],axis=1)
Y = data['default_ind']
```


```python
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state = 50, stratify = Y)
```



### D) Feature Scaling <a name="feature-scaling"></a>

It is essential to perform feature scaling on our data as we have varied data scaling from 0 to ten's of thousands. 

Feature Scaling is where we force the values from different columns to exist on the same scale, in order to enchance the learning capabilities of the model. There are two common approaches for this, Standardisation, and Normalisation.

Standardisation rescales data to have a mean of 0, and a standard deviation of 1 - meaning most datapoints will most often fall between values of around -4 and +4.

Normalisation rescales datapoints so that they exist in a range between 0 and 1.

Here, we will look to apply normalisation as this will ensure all variables will end up having the same range, fixed between 0 and 1, and therefore the logistic regression algorithm can judge each variable in the same context. Standardisation can result in different ranges, variable to variable, and this is not so useful (although this isn’t explcitly true in all scenarios).

Another reason for choosing Normalisation over Standardisation is that our scaled data will all exist between 0 and 1, and these will then be compatible with any categorical variables that we have encoded as 1’s and 0’s.

The below code uses the in-built MinMaxScaler functionality from scikit-learn to apply Normalisation to all of our variables.


```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
```

In the above code, we also make sure to apply fit_transform to the training set, but only transform to the test set. This means the feature scaling will learn and apply the “rules” from the training data, but only apply them to the test data. 

This is important in order to avoid data leakage where the test set learns information about the training data, and means we can’t fully trust model performance metrics!

We are done with our data preparation. Next, we move to train our model on prepared data

# Random Forest <a name="rf-title"></a>

## Model Training - Random Forest


```python
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.metrics import classification_report,confusion_matrix,recall_score,precision_score,roc_auc_score, f1_score, accuracy_score
```

We use the default parameter values for our random forest model


```python
#Instantiating model and fitting the data

clf = rf()
clf.fit(x_train, y_train)
```




    RandomForestClassifier()



We the predict our target variable using the trained model object (here called clf) and ask it to predict the default class for the test set


```python
y_pred_rf = clf.predict(x_test)
```

## Model Performance Assessment <a name="rf-metrics"></a>

We can assess our model performce by a number of metrics

#### Confusion Matrix


A Confusion Matrix provides us a visual way to understand how our predictions match up against the actual values for those test set observations


```python
conf_matrix = confusion_matrix(y_test, y_pred_rf)

plt.style.use("seaborn-poster")
plt.matshow(conf_matrix, cmap = "coolwarm")
plt.gca().xaxis.tick_bottom()
plt.title("Confusion Matrix")
plt.xlabel("Actual class")
plt.ylabel("Predicted Class")
for (i,j), corr_value in np.ndenumerate(conf_matrix):
    plt.text(j, i, corr_value, ha = "center", va = "center", fontsize = 20)
plt.show()
```


    

    


![alt text](/img/posts/confusion_matrix_rf.png)

The aim is to have a high proportion of observations falling into the top left cell (predicted default and actual default) and the bottom right cell (predicted non-default and actual predicted non-default).

Since the proportion of target feature in our data was heavily skewed towards the non-defaulters, we will next analyse not only Classification Accuracy, but also Precision, Recall, and F1-Score which will help us assess how well our model has performed in reality

#### Accuracy

Classification Accuracy is a metric that tells us of all predicted observations, what proportion did we correctly classify. This is very intuitive, but when dealing with imbalanced classes, can be misleading.


```python
print("Accuracy using Random Forest is",accuracy_score(y_test,y_pred_rf)*100,"%")
```

    Accuracy using Random Forest is 99.74267824965118 %
    

#### Precision

Precision is a metric that tells us of all observations that were predicted as positive, how many actually were positive.


```python
print("Precision value using Random Forest is",precision_score(y_test,y_pred_rf)*100,"%")
```

    Precision value using Random Forest is 99.96199524940617 %
    

#### Recall

Recall is a metric that tells us of all positive observations, how many did we predict as positive


```python
print("Recall value using Random Forest is",recall_score(y_test,y_pred_rf)*100,"%")
```

    Recall value using Random Forest is 96.66482910694597 %
    

#### F1-Score

F1-Score is a metric that essentially “combines” both Precision & Recall. Technically speaking, it is the harmonic mean of these two metrics. A good, or high, F1-Score comes when there is a balance between Precision & Recall, rather than a disparity between them.


```python
print("F1 score using Random Forest is",f1_score(y_test,y_pred_rf)*100,"%")
```

    F1 score using Random Forest is 98.28576766780324 %
    

To judge if Random Forest model is a better classification model to predict our cutomer default class, we will run Logistic Regression as well and compare the metrics.



# Logistic Regression <a name="logreg-title"></a>

## Model Training - Logistic Regression 


```python
from sklearn.linear_model import LogisticRegression
```


```python
#Instantiating model and fitting the data

LR = LogisticRegression(solver = "liblinear")
LR.fit(x_train,y_train)
```




    LogisticRegression(solver='liblinear')




```python
#Predicting target variable

LR_pred = LR.predict(x_test)
```


```python
# Prediction probabilities for the positive class

LR_pred_prob = LR.predict_proba(x_test)[:, 1]
```


## Model Performance Assesment <a name="logreg-metrics"></a>

Just like we did with Random Forest, we will find out different metrics to determine the performace of our Logistic Regression Model

#### Confusion Matrix


```python
conf_matrix = confusion_matrix(y_test, LR_pred)

plt.matshow(conf_matrix, cmap = "coolwarm")
plt.gca().xaxis.tick_bottom()
plt.title("Confusion Matrix")
plt.xlabel("Actual class")
plt.ylabel("Predicted Class")
for (i,j), corr_value in np.ndenumerate(conf_matrix):
    plt.text(j, i, corr_value, ha = "center", va = "center", fontsize = 20)
plt.show()
```


    

    


![alt text](/img/posts/confusion_matrix_logreg.png "Logistic Regression Confusion Matrix")


```python
print("Recall value using Logistic Regression is",recall_score(y_test,LR_pred)*100,"%")
print("Precision value using Logistic Regression is",precision_score(y_test,LR_pred)*100,"%")
print("Accuracy using Logistic Regression is",accuracy_score(y_test,LR_pred)*100,"%")
print("F1 score using Logistic Regression is",f1_score(y_test,LR_pred)*100,"%")
```

    Recall value using Logistic Regression is 91.55641308342521 %
    Precision value using Logistic Regression is 99.67990397119135 %
    Accuracy using Logistic Regression is 99.3332071264803 %
    F1 score using Logistic Regression is 95.44562042047795 %
    


## Finding The Optimal Classification Threshold <a name="logreg-threshold"></a>

By default, most pre-built classification models & algorithms will just use a 50% probability to discern between a positive class prediction (non-default) and a negative class prediction (default).

Just because 50% is the default threshold does not mean it is the best one for our task.

We will iterate through many potential classification thresholds, and plot the Precision, Recall & F1-Score, and find an optimal solution


```python
# set up the list of thresholds to loop through

thresholds = np.arange(0, 1, 0.01) #numbers from 0 to 1 in creament of 0.01

# create empty lists to append the results to

precision_scores = []
recall_scores = []
f1_scores = []

# loop through each threshold - fit the model - append the results

for threshold in thresholds:
    pred_class = (LR_pred_prob >= threshold) * 1
    
    precision = precision_score(y_test, pred_class, zero_division = 0)
    precision_scores.append(precision)
    
    recall = recall_score(y_test, pred_class)
    recall_scores.append(recall)
    
    f1 = f1_score(y_test, pred_class)
    f1_scores.append(f1)
```


```python
# extract the optimal f1-score, precision and recall (and it's threshold value)

```


```python
# Optimal F1 score   
max_f1 = max(f1_scores)
max_f1_idx = f1_scores.index(max_f1)

print(f"Maximum f1 score achived by model is {max_f1} at threshold {max_f1_idx/100}")
```

    Maximum f1 score achived by model is 0.9736842105263159 at threshold 0.12
    


```python
# Optimal precision score

max_precision = max(precision_scores)
max_precision_idx = precision_scores.index(max_precision)

print(f"Maximum precision score achived by model is {max_precision} at threshold {max_precision_idx/100}")
```

    Maximum precision score achived by model is 0.9992457573852923 at threshold 0.99
    


```python
# Optimal recall score 

max_recall = max(recall_scores)
max_recall_idx = recall_scores.index(max_recall)

print(f"Maximum recall score achived by model is {max_recall} at threshold {max_recall_idx/100}")
```

    Maximum recall score achived by model is 1.0 at threshold 0.0
    

#### For better understanding of achived results, we visualize a plot


```python
plt.style.use("seaborn-poster")
plt.plot(thresholds, precision_scores, label = "Precision", linestyle = "--") 
plt.plot(thresholds, recall_scores, label = "Recall", linestyle = "--")  
plt.plot(thresholds, f1_scores, label = "F1", linewidth = 5)
plt.title("Finding Optimal Threshold")
plt.xlabel("Thresholds")
plt.ylabel("Assesment Score")
plt.legend(loc = "lower left")
plt.tight_layout()
plt.show()

```


    

    


![alt text](/img/posts/optimal_threshold.png)

## Conclusion <a name="conclusion"></a>

We now have a model that can readily predict default class of a customer. We can provide the neccessary customer information and the model will provide us the target output of whether or not the customer's application should be approved.

The goal for the project was to build a model that would accurately predict the customers that would default on their payment. Based upon these, model the models performed well. 

The chosen the model is the Random Forest as it was the most consistently performant on the test set across classication accuracy, precision, recall, and f1-score.

Based upon this, the bank can further examine the application and avoid giving out loans or limit the loan amount and apply stringent checks for such customers. This will drastically reduce loss and improve performance.


```python

```

## $$CREDIT$$  $$RISK$$  $$ANALYSIS$$

Credit Risk Management is essential to various financial organizations as it helps them boost their business as well provide. Our client, a banking and financial service company, hired an analytics consultancy to get insights and to evaluate the performance of their credit management system and determine a borrower's ability to meet their debt obligation. Let's do this!

- [00. Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Actions](#overview-actions)
    - [Results](#overview-results)
    - [Growth/Next Steps](#overview-growth)
- [01. Data Overview](#data-overview)
    - [Data exploration](#data-exploration)
    - [Data preprocessing](#data-preprocessing)
    - [Dealing with categorial features](#data-categorical)
    - [Splitting data into train test](#data-split)
- [02. Modelling Overview](#modelling-overview)
- [03. Logistic Regression](#logreg-title)
- [04. Random Forest](#rf-title)
- [05. Model Test](#model-test)
- [06. Conclusion](#conclusion)


# Project Overview  <a name="overview-main"></a>

### Context <a name="overview-context"></a>

The overall aim of this project is to help the lender bank or financial institute to accurately determine if borrower(s) will default on their contractual obligations, inorder to cushion itself from loss due to borrower's failure to repay loan ammount .

To ascertain this, we build a classification predictive model that will find relationships between customer details and their credit worthiness for our customers with historical data, and use this to predict the future scope for those who are not.
<br>
<br>
### Actions <a name="overview-actions"></a>

As we are predicting a binary classification output, we tested three classification modelling approaches, namely:

* Logistic Regression
* Random Forest Classification
<br>
<br>

### Results <a name="overview-results"></a>

Our testing found that the Logistic Regression had the highest predictive accuracy.

<br>

**Metric 1: Accuracy**

* Random Forest = 99.02%
* Logistic Regression = 99.32%

**Metric 2: Precision**

* Random Forest = 100%
* Logistic Regression = 99.57%

**Metric 2: Recall**

* Random Forest = 86.81%
* Logistic Regression = 91.68%
<br>
<br>

### Growth/Next Steps <a name="overview-growth"></a>

While predictive accuracy was relatively high - other classification modelling approaches could be tested, example:
Decision Tree, KNN, SVM, Naive Bayes etc.

From a data point of view, further customer details could be collected, and further feature engineering could be undertaken to ensure that we have as much useful information available for predicting whether a customer will default.
<br>
<br>

# Data Overview  <a name="data-overview"></a>

Our data set consists of customer details from a banking institute. We would be looking after key variables pertaining to customers which help us determine their default status.

### Importing required Libraries


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    


```python
data.head()
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
      <th>term</th>
      <th>int_rate</th>
      <th>installment</th>
      <th>grade</th>
      <th>emp_length</th>
      <th>home_ownership</th>
      <th>annual_inc</th>
      <th>verification_status</th>
      <th>issue_d</th>
      <th>pymnt_plan</th>
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
      <th>application_type</th>
      <th>acc_now_delinq</th>
      <th>tot_coll_amt</th>
      <th>tot_cur_bal</th>
      <th>default_ind</th>
      <th>rand no</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>26859567</td>
      <td>3600</td>
      <td>3600</td>
      <td>3600.0</td>
      <td>36 months</td>
      <td>15.61</td>
      <td>125.88</td>
      <td>D</td>
      <td>6 years</td>
      <td>RENT</td>
      <td>43000.0</td>
      <td>Source Verified</td>
      <td>Sep-14</td>
      <td>n</td>
      <td>15.41</td>
      <td>0</td>
      <td>12</td>
      <td>0</td>
      <td>13007</td>
      <td>72.3</td>
      <td>22</td>
      <td>1599.40</td>
      <td>1599.40</td>
      <td>2588.20</td>
      <td>2588.20</td>
      <td>2000.60</td>
      <td>587.60</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>125.88</td>
      <td>0.0</td>
      <td>INDIVIDUAL</td>
      <td>0</td>
      <td>0.0</td>
      <td>24812.0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>46134366</td>
      <td>12275</td>
      <td>12275</td>
      <td>12275.0</td>
      <td>60 months</td>
      <td>15.61</td>
      <td>295.97</td>
      <td>D</td>
      <td>2 years</td>
      <td>MORTGAGE</td>
      <td>45000.0</td>
      <td>Not Verified</td>
      <td>Apr-15</td>
      <td>n</td>
      <td>28.37</td>
      <td>3</td>
      <td>7</td>
      <td>1</td>
      <td>7694</td>
      <td>74.0</td>
      <td>13</td>
      <td>11133.70</td>
      <td>11133.70</td>
      <td>2357.11</td>
      <td>2357.11</td>
      <td>1141.30</td>
      <td>1215.81</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>295.97</td>
      <td>0.0</td>
      <td>INDIVIDUAL</td>
      <td>0</td>
      <td>0.0</td>
      <td>154494.0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8266388</td>
      <td>18000</td>
      <td>18000</td>
      <td>18000.0</td>
      <td>36 months</td>
      <td>10.99</td>
      <td>589.22</td>
      <td>B</td>
      <td>&lt; 1 year</td>
      <td>OWN</td>
      <td>75000.0</td>
      <td>Not Verified</td>
      <td>Oct-13</td>
      <td>n</td>
      <td>12.58</td>
      <td>0</td>
      <td>10</td>
      <td>0</td>
      <td>17921</td>
      <td>66.4</td>
      <td>16</td>
      <td>5611.25</td>
      <td>5611.25</td>
      <td>15312.04</td>
      <td>15312.04</td>
      <td>12388.75</td>
      <td>2923.29</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>589.22</td>
      <td>0.0</td>
      <td>INDIVIDUAL</td>
      <td>0</td>
      <td>0.0</td>
      <td>155032.0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2365918</td>
      <td>20000</td>
      <td>20000</td>
      <td>20000.0</td>
      <td>36 months</td>
      <td>8.90</td>
      <td>635.07</td>
      <td>A</td>
      <td>10+ years</td>
      <td>RENT</td>
      <td>86000.0</td>
      <td>Verified</td>
      <td>Dec-12</td>
      <td>n</td>
      <td>25.40</td>
      <td>2</td>
      <td>15</td>
      <td>0</td>
      <td>46826</td>
      <td>69.3</td>
      <td>26</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>15876.75</td>
      <td>15876.75</td>
      <td>13306.79</td>
      <td>2559.04</td>
      <td>0.0</td>
      <td>10.92</td>
      <td>0.0</td>
      <td>635.07</td>
      <td>0.0</td>
      <td>INDIVIDUAL</td>
      <td>0</td>
      <td>0.0</td>
      <td>72223.0</td>
      <td>1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8244941</td>
      <td>34475</td>
      <td>34475</td>
      <td>34475.0</td>
      <td>60 months</td>
      <td>23.40</td>
      <td>979.81</td>
      <td>E</td>
      <td>10+ years</td>
      <td>MORTGAGE</td>
      <td>76785.0</td>
      <td>Verified</td>
      <td>Oct-13</td>
      <td>n</td>
      <td>16.21</td>
      <td>0</td>
      <td>13</td>
      <td>0</td>
      <td>24095</td>
      <td>89.9</td>
      <td>23</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>39050.73</td>
      <td>39050.73</td>
      <td>34475.00</td>
      <td>4575.73</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>33171.87</td>
      <td>0.0</td>
      <td>INDIVIDUAL</td>
      <td>0</td>
      <td>0.0</td>
      <td>51364.0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



## 1 : Data Exploration  <a name="data-exploration"></a>

### First lets check what are the types of the features that are present in our dataset


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
#### Here '1' indicates applicant is credit risk and '0' indicates an applicant is not a credit risk


```python
plt.figure(figsize = (8,6))
sns.countplot(data['default_ind'])
plt.title("Count plot for the Target Variable", fontsize = 15)
plt.xlabel("Target Variable (default_ind)",fontsize = 12)
plt.ylabel("Count",fontsize = 12);
```

    C:\Users\Manasi\anaconda3\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(
    


    
![png](output_16_1.png)
    


### Create a histogram of loan_amnt, to see the distribution of loan amount


```python
plt.figure(figsize=(12,4))
sns.distplot(data['loan_amnt'],kde=False,bins=30)
plt.xlim(0,36000)
plt.title("Distribution Plot for Loan Amount", fontsize = 15)
plt.xlabel("Loan Amount",fontsize = 12)
plt.ylabel("Amount",fontsize = 12);
```

    C:\Users\Manasi\anaconda3\lib\site-packages\seaborn\distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    


    
![png](output_18_1.png)
    


***This is the distribution plot for loan Amount (The listed amount of the loan applied by the borrower).From this we can infer that the maximum amount of loan applied by the borrower is between range of 10000-12500***

### Displaying summary of default_ind with loan_amount


```python
data.groupby('default_ind')['loan_amnt'].describe()
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
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>default_ind</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>552822.0</td>
      <td>14538.024536</td>
      <td>8332.212394</td>
      <td>500.0</td>
      <td>8000.0</td>
      <td>12600.0</td>
      <td>20000.0</td>
      <td>35000.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>46156.0</td>
      <td>14571.998765</td>
      <td>8387.407333</td>
      <td>900.0</td>
      <td>8000.0</td>
      <td>12700.0</td>
      <td>20000.0</td>
      <td>35000.0</td>
    </tr>
  </tbody>
</table>
</div>



***The mean amount of loan is almost same for people who defaulted and those who did not***

### Lets explore the grades that the bank attributes to judge customer behaviour


```python
grades_sorted_order = sorted(data['grade'].unique())
```


```python
plt.figure(figsize=(12,4))
sns.countplot(x='grade',data=data,hue='default_ind',order=grades_sorted_order,palette='coolwarm');
plt.title("", fontsize = 15)
plt.xlabel("Grade",fontsize = 12)
plt.ylabel("Count",fontsize = 12)
plt.title('Count plot for default_ind across different Grades');
```


    
![png](output_25_0.png)
    


## 2 :  Data Preprocessing  <a name="data-preprocessing"></a>

### Checking null values in our dataset


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



### Converting them into percantages of the total missing values


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



#### The missing values in 'emp_lenght' , revol_util and collections_12_mths_ex_med columns are extremely low( < 5%), so instead of applying any imputation, we will eliminate these row


```python
data.dtypes
```




    id                              int64
    loan_amnt                       int64
    funded_amnt                     int64
    funded_amnt_inv               float64
    term                           object
    int_rate                      float64
    installment                   float64
    grade                          object
    emp_length                     object
    home_ownership                 object
    annual_inc                    float64
    verification_status            object
    issue_d                        object
    pymnt_plan                     object
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
    application_type               object
    acc_now_delinq                  int64
    tot_coll_amt                  float64
    tot_cur_bal                   float64
    default_ind                     int64
    rand no                       float64
    dtype: object




```python
data['collections_12_mths_ex_med']    
```




    0         0.0
    1         0.0
    2         0.0
    3         0.0
    4         0.0
             ... 
    598973    0.0
    598974    0.0
    598975    0.0
    598976    0.0
    598977    0.0
    Name: collections_12_mths_ex_med, Length: 598978, dtype: float64




```python
data = data.dropna(subset=['emp_length'])
data = data.dropna(subset=['revol_util'])
data = data.dropna(subset=['collections_12_mths_ex_med'])
```

### For 'total_coll_amt' and 'tot_cur_bal', we replace null values by their respective mean


```python
data["tot_cur_bal"] = data["tot_cur_bal"].fillna(value =  data["tot_cur_bal"].mean())

data["tot_coll_amt"] = data["tot_coll_amt"].fillna(value =  data["tot_coll_amt"].mean())
```

### Verifying no null values remain


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



###  There are no null values left in our dataset.

## 3: Dealing with Categorical features  <a name="data-categorical"></a>

**We're done working with the NULL values! Now we deal with the data types of various categorical columns.**


```python
categorical_variables = data.select_dtypes(['object']).columns
print(f"Features that are Categorical are : {categorical_variables}")
```

    Features that are Categorical are : Index(['term', 'grade', 'emp_length', 'home_ownership', 'verification_status',
           'issue_d', 'pymnt_plan', 'application_type'],
          dtype='object')
    



### term feature

***Convert the term feature into either a 36 or 60 integer numeric data type***


```python
data['term'].value_counts()
```




     36 months    404069
     60 months    166420
    Name: term, dtype: int64




```python
data['term'] = data['term'].apply(lambda term : int(term[:3]))
```

### grade feature


```python
data['grade'].value_counts()
```




    B    165355
    C    155116
    A     96440
    D     91279
    E     43674
    F     15005
    G      3620
    Name: grade, dtype: int64



***Convert the grade into dummy variables and drop one of the newly generated dummy variable to deal with dummy variable effect. Then concatenate these new columns to the original dataframe.***


```python
grade_dummies = pd.get_dummies(data['grade'],drop_first=True)
```


```python
data = pd.concat([data.drop('grade',axis=1),grade_dummies],axis=1)
```


```python
emp_length = pd.get_dummies(data['emp_length'],drop_first=True)
```


```python
data = pd.concat([data.drop('emp_length',axis=1),emp_length],axis=1)
```


```python

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
      <th>term</th>
      <th>int_rate</th>
      <th>installment</th>
      <th>annual_inc</th>
      <th>pymnt_plan</th>
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
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
      <th>F</th>
      <th>G</th>
      <th>OTHER</th>
      <th>OWN</th>
      <th>RENT</th>
      <th>10+ years</th>
      <th>2 years</th>
      <th>3 years</th>
      <th>4 years</th>
      <th>5 years</th>
      <th>6 years</th>
      <th>7 years</th>
      <th>8 years</th>
      <th>9 years</th>
      <th>&lt; 1 year</th>
      <th>Source Verified</th>
      <th>Verified</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>26859567</td>
      <td>3600</td>
      <td>3600</td>
      <td>3600.0</td>
      <td>36</td>
      <td>15.61</td>
      <td>125.88</td>
      <td>43000.0</td>
      <td>0</td>
      <td>15.41</td>
      <td>0</td>
      <td>12</td>
      <td>0</td>
      <td>13007</td>
      <td>72.3</td>
      <td>22</td>
      <td>1599.40</td>
      <td>1599.40</td>
      <td>2588.20</td>
      <td>2588.20</td>
      <td>2000.60</td>
      <td>587.60</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>125.88</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>24812.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>46134366</td>
      <td>12275</td>
      <td>12275</td>
      <td>12275.0</td>
      <td>60</td>
      <td>15.61</td>
      <td>295.97</td>
      <td>45000.0</td>
      <td>0</td>
      <td>28.37</td>
      <td>3</td>
      <td>7</td>
      <td>1</td>
      <td>7694</td>
      <td>74.0</td>
      <td>13</td>
      <td>11133.70</td>
      <td>11133.70</td>
      <td>2357.11</td>
      <td>2357.11</td>
      <td>1141.30</td>
      <td>1215.81</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>295.97</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>154494.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8266388</td>
      <td>18000</td>
      <td>18000</td>
      <td>18000.0</td>
      <td>36</td>
      <td>10.99</td>
      <td>589.22</td>
      <td>75000.0</td>
      <td>0</td>
      <td>12.58</td>
      <td>0</td>
      <td>10</td>
      <td>0</td>
      <td>17921</td>
      <td>66.4</td>
      <td>16</td>
      <td>5611.25</td>
      <td>5611.25</td>
      <td>15312.04</td>
      <td>15312.04</td>
      <td>12388.75</td>
      <td>2923.29</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>589.22</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>155032.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2365918</td>
      <td>20000</td>
      <td>20000</td>
      <td>20000.0</td>
      <td>36</td>
      <td>8.90</td>
      <td>635.07</td>
      <td>86000.0</td>
      <td>0</td>
      <td>25.40</td>
      <td>2</td>
      <td>15</td>
      <td>0</td>
      <td>46826</td>
      <td>69.3</td>
      <td>26</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>15876.75</td>
      <td>15876.75</td>
      <td>13306.79</td>
      <td>2559.04</td>
      <td>0.0</td>
      <td>10.92</td>
      <td>0.0</td>
      <td>635.07</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>72223.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8244941</td>
      <td>34475</td>
      <td>34475</td>
      <td>34475.0</td>
      <td>60</td>
      <td>23.40</td>
      <td>979.81</td>
      <td>76785.0</td>
      <td>0</td>
      <td>16.21</td>
      <td>0</td>
      <td>13</td>
      <td>0</td>
      <td>24095</td>
      <td>89.9</td>
      <td>23</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>39050.73</td>
      <td>39050.73</td>
      <td>34475.00</td>
      <td>4575.73</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>33171.87</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>51364.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>598973</th>
      <td>32348923</td>
      <td>10000</td>
      <td>10000</td>
      <td>10000.0</td>
      <td>36</td>
      <td>13.98</td>
      <td>341.68</td>
      <td>36840.0</td>
      <td>0</td>
      <td>15.70</td>
      <td>2</td>
      <td>9</td>
      <td>0</td>
      <td>3875</td>
      <td>49.7</td>
      <td>14</td>
      <td>6597.27</td>
      <td>6597.27</td>
      <td>4767.99</td>
      <td>4767.99</td>
      <td>3402.73</td>
      <td>1365.26</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>341.68</td>
      <td>0.0</td>
      <td>0</td>
      <td>269.0</td>
      <td>8940.0</td>
      <td>0</td>
      <td>1.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>598974</th>
      <td>41149911</td>
      <td>1550</td>
      <td>1550</td>
      <td>1550.0</td>
      <td>36</td>
      <td>11.53</td>
      <td>51.14</td>
      <td>38800.0</td>
      <td>0</td>
      <td>25.83</td>
      <td>0</td>
      <td>10</td>
      <td>0</td>
      <td>3002</td>
      <td>18.6</td>
      <td>29</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1662.33</td>
      <td>1662.33</td>
      <td>1550.00</td>
      <td>112.33</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>1306.83</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>30003.0</td>
      <td>0</td>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>598975</th>
      <td>43370018</td>
      <td>7500</td>
      <td>7500</td>
      <td>7500.0</td>
      <td>36</td>
      <td>9.17</td>
      <td>239.10</td>
      <td>53000.0</td>
      <td>0</td>
      <td>10.08</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>5675</td>
      <td>82.2</td>
      <td>14</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>7948.09</td>
      <td>7948.09</td>
      <td>7500.00</td>
      <td>448.09</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>6282.03</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>8890.0</td>
      <td>0</td>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>598976</th>
      <td>15479925</td>
      <td>30000</td>
      <td>30000</td>
      <td>30000.0</td>
      <td>60</td>
      <td>18.92</td>
      <td>776.90</td>
      <td>95000.0</td>
      <td>0</td>
      <td>15.45</td>
      <td>0</td>
      <td>11</td>
      <td>1</td>
      <td>15801</td>
      <td>72.9</td>
      <td>23</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>37702.77</td>
      <td>37702.77</td>
      <td>30000.00</td>
      <td>7702.77</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>25272.37</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>184654.0</td>
      <td>0</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>598977</th>
      <td>39590935</td>
      <td>6400</td>
      <td>6400</td>
      <td>6400.0</td>
      <td>36</td>
      <td>12.69</td>
      <td>214.69</td>
      <td>70000.0</td>
      <td>0</td>
      <td>14.57</td>
      <td>0</td>
      <td>8</td>
      <td>1</td>
      <td>9983</td>
      <td>52.8</td>
      <td>11</td>
      <td>4694.61</td>
      <td>4694.61</td>
      <td>2350.31</td>
      <td>2350.31</td>
      <td>1705.39</td>
      <td>644.92</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>214.69</td>
      <td>0.0</td>
      <td>0</td>
      <td>7683.0</td>
      <td>19580.0</td>
      <td>0</td>
      <td>1.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>570489 rows × 53 columns</p>
</div>



### home_ownership feature


```python
data['home_ownership'].value_counts()
```




    MORTGAGE    288080
    RENT        231654
    OWN          50573
    OTHER          139
    NONE            42
    ANY              1
    Name: home_ownership, dtype: int64



***We can merge NONE & ANY types to OTHER, to represent data in a better and concise manner***


```python
data['home_ownership'] = data['home_ownership'].replace(['NONE','ANY'],'OTHER')
```


```python
data['home_ownership'].value_counts()
```




    MORTGAGE    288080
    RENT        231654
    OWN          50573
    OTHER          182
    Name: home_ownership, dtype: int64




```python
home_dummies = pd.get_dummies(data['home_ownership'],drop_first=True)
data = pd.concat([data.drop('home_ownership',axis=1),home_dummies],axis=1)
```

### verification_status feature


```python
data['verification_status'].value_counts()
```




    Source Verified    204653
    Not Verified       183347
    Verified           182489
    Name: verification_status, dtype: int64




```python
verification_dummies = pd.get_dummies(data['verification_status'],drop_first=True)
data = pd.concat([data.drop('verification_status',axis=1),verification_dummies],axis=1)
```

### application_type  feature


```python
data['application_type'].value_counts()
```




    INDIVIDUAL    570489
    Name: application_type, dtype: int64



***We observe that there is only 1 value for it.Hence, we drop the column***


```python
data.drop('application_type',axis=1,inplace=True)
```

### issue_d feature

***This is the target leak feature. As our aim to identify credit risk applicant, issue date will be a futuristic value and has no impact on current model. Hence we drop the column***


```python
data.drop('issue_d',axis=1,inplace=True)
```

### pymnt_plan feature


```python
data['pymnt_plan'].value_counts()
```




    n    570484
    y         5
    Name: pymnt_plan, dtype: int64




```python
data['pymnt_plan'] = data['pymnt_plan'].map({'y':1,'n':0})
```

### Now check the data types of the features


```python
data.dtypes
```




    id                              int64
    loan_amnt                       int64
    funded_amnt                     int64
    funded_amnt_inv               float64
    term                            int64
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



***Now we can see that all of them have been converted to Integer or float***

## 4 : Train-test Split  <a name="data-split"></a>


```python
from sklearn.model_selection import train_test_split,cross_val_score
```


```python
X = data.drop('default_ind',axis=1).values
y = data['default_ind'].values
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)
```


```python
print("Shape of X_train is :",X_train.shape)
print("Shape of X_test is :",X_test.shape)
print("Shape of y_train is :",y_train.shape)
print("Shape of y_test is :",y_test.shape)
```

    Shape of X_train is : (427866, 52)
    Shape of X_test is : (142623, 52)
    Shape of y_train is : (427866,)
    Shape of y_test is : (142623,)
    

### As our dataset consists of a varying scale of values for different features, we normalize our  data set values, i.e we scale our values between 0-1. We can use the standardscaler library of scikit-learn to implement the same


```python
from sklearn.preprocessing import MinMaxScaler
```


```python
scaler = MinMaxScaler()
```


```python
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

## Model Overview <a name="modelling-overview"></a>

***This is a classification problem, we will be applying Logistic Regression, Random Forest and validate the one that will give us the highest recall/precision***

#### Import all the libraries


```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.metrics import classification_report,confusion_matrix,recall_score,precision_score,roc_auc_score, accuracy_score
from sklearn import metrics
```

### 1. Logistic Regression <a name="logreg-title"></a>


```python
LR = LogisticRegression(solver = "liblinear")
LR.fit(X_train,y_train)
```




    LogisticRegression(solver='liblinear')




```python
LR_pred = LR.predict(X_test)
```

#### Checking the evaluation Metrics


```python
print("Recall value of the dataset is :",recall_score(y_test,LR_pred)*100,"%")
```

    Recall value of the dataset is : 92.16326530612244 %
    


```python
print("Precision of the dataset is :",precision_score(y_test,LR_pred)*100,"%")
```

    Precision of the dataset is : 99.54932889193691 %
    


```python
print("Accuracy of the dataset is :",accuracy_score(y_test,LR_pred)*100,"%")
```

    Accuracy of the dataset is : 99.36195424300428 %
    


```python
print("Confusion Matrix is :")
confusion_matrix(y_test,LR_pred)
```

    Confusion Matrix is :
    




    array([[131552,     46],
           [   864,  10161]], dtype=int64)




```python
print("Classification Report :")
print(classification_report(y_test,LR_pred))
```

    Classification Report :
                  precision    recall  f1-score   support
    
               0       0.99      1.00      1.00    131598
               1       1.00      0.92      0.96     11025
    
        accuracy                           0.99    142623
       macro avg       0.99      0.96      0.98    142623
    weighted avg       0.99      0.99      0.99    142623
    
    


```python
ra_score = roc_auc_score(y_test,LR_pred)
print("ROC AUC score:", ra_score)
```

    ROC AUC score: 0.9606415518379877
    


```python
fpr, tpr, thresholds = metrics.roc_curve(y_test, LR_pred)
```


```python
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b')
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')# the base line
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.show()
```

    No handles with labels found to put in legend.
    


    
![png](output_100_1.png)
    


### 2. Random Forest Classifier <a name="rf-title"></a>

#### We will be using the max_depth=8  and n_estimators to be 100 for our Random Forest classifier


```python
rf(n_estimators=100,max_depth=8).fit(X_train,y_train)
```




    RandomForestClassifier(max_depth=8)




```python
y_pred_rf = rf(n_estimators=100,max_depth=8).fit(X_train,y_train).predict(X_test)
```


```python
print("Recall value using Random Forest is",recall_score(y_test,y_pred_rf)*100,"%")
print("Precision value using Random Forest is",precision_score(y_test,y_pred_rf)*100,"%")
print("Accuracy using Random Forest is",accuracy_score(y_test,y_pred_rf)*100,"%")
```

    Recall value using Random Forest is 85.25170068027211 %
    Precision value using Random Forest is 100.0 %
    Accuracy using Random Forest is 98.85993142760985 %
    


```python
print("Confusion Matrix is :")
confusion_matrix(y_test,y_pred_rf)
```

    Confusion Matrix is :
    




    array([[131598,      0],
           [  1626,   9399]], dtype=int64)




```python
print("Classification Report :")
print(classification_report(y_test,y_pred_rf))
```

    Classification Report :
                  precision    recall  f1-score   support
    
               0       0.99      1.00      0.99    131598
               1       1.00      0.85      0.92     11025
    
        accuracy                           0.99    142623
       macro avg       0.99      0.93      0.96    142623
    weighted avg       0.99      0.99      0.99    142623
    
    


```python
ra_score = roc_auc_score(y_test,y_pred_rf)
print("ROC AUC score:", ra_score)
```

    ROC AUC score: 0.9262585034013606
    


```python
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_rf)
```


```python
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b')
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')# the base line
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.show()
```

    No handles with labels found to put in legend.
    


    
![png](output_110_1.png)
    


**Comparing the Recall ,ROC_AUC and Precision values of both the models, LOGISTIC REGRESSION  model is the best model to implement **

### Testing the model  <a name="model-test"></a>


```python
import random
random.seed(10)
```


```python
rand_num = random.randint(0,len(data))
print(rand_num)
```

    34167
    


```python
test_data = data.drop('default_ind',axis=1).iloc[rand_num]
print(test_data)
```

    id                            1.472063e+06
    loan_amnt                     2.005000e+04
    funded_amnt                   2.005000e+04
    funded_amnt_inv               2.005000e+04
    term                          6.000000e+01
    int_rate                      2.247000e+01
    installment                   5.591400e+02
    annual_inc                    1.120000e+05
    pymnt_plan                    0.000000e+00
    dti                           1.793000e+01
    delinq_2yrs                   0.000000e+00
    open_acc                      1.200000e+01
    pub_rec                       0.000000e+00
    revol_bal                     2.326700e+04
    revol_util                    8.310000e+01
    total_acc                     1.700000e+01
    out_prncp                     0.000000e+00
    out_prncp_inv                 0.000000e+00
    total_pymnt                   2.635328e+04
    total_pymnt_inv               2.635328e+04
    total_rec_prncp               2.005000e+04
    total_rec_int                 6.303280e+03
    total_rec_late_fee            0.000000e+00
    recoveries                    0.000000e+00
    collection_recovery_fee       0.000000e+00
    last_pymnt_amnt               5.909000e+02
    collections_12_mths_ex_med    0.000000e+00
    acc_now_delinq                0.000000e+00
    tot_coll_amt                  0.000000e+00
    tot_cur_bal                   6.210600e+04
    rand no                       6.010000e-02
    B                             0.000000e+00
    C                             0.000000e+00
    D                             0.000000e+00
    E                             1.000000e+00
    F                             0.000000e+00
    G                             0.000000e+00
    10+ years                     0.000000e+00
    2 years                       1.000000e+00
    3 years                       0.000000e+00
    4 years                       0.000000e+00
    5 years                       0.000000e+00
    6 years                       0.000000e+00
    7 years                       0.000000e+00
    8 years                       0.000000e+00
    9 years                       0.000000e+00
    < 1 year                      0.000000e+00
    OTHER                         0.000000e+00
    OWN                           0.000000e+00
    RENT                          1.000000e+00
    Source Verified               0.000000e+00
    Verified                      1.000000e+00
    Name: 35869, dtype: float64
    

#### Now we will take this test data and pass this value in our model and predict the value


```python
print("Predicted value is :",LR.predict(test_data.values.reshape(1,-1)))
```

    Predicted value is : [0]
    


```python
print("Actual value is :",data.iloc[rand_num]['default_ind'])
```

    Actual value is : 0.0
    

***We observe that our Model predicted the correct value and implies that our model is able to identify target applicant adeptly***

### Conclusion  <a name="conclusion"></a>

Implementing Machine Learning models, we are able to identify credit risk customers for various financial institue. The correct identification of loan applicants helps these institutions to mitigate the loss and ensures that the customers will pay for the products and servives they rendered. Features like the principal loan amount, the employee grade, ownership of the property, annual income, interest rate and many other features influence the model and also impact the decision greatly.

## $$Thank you$$


```python

```

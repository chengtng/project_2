# Project 2: Regression Modelling of HDB resale prices in Singapore

This project aims to apply linear regression techniques on the price of homes at sale for a Singapore public housing dataset from kaggle. [DSI-SG Project 2 Regression Challenge (HDB Price)](https://www.kaggle.com/competitions/dsi-sg-project-2-regression-challenge-hdb-price/overview). 

### Background

![HDB Resale Price Index (RPI)](images/RPI.png)
Source: [HDB Statistics](https://www.hdb.gov.sg/residential/selling-a-flat/overview/resale-statistics)

There is an upward trend in resale prices, with prices increasing about 2.8 % every quarter for the past 2 years. There is also increasing demand for resale flats driven my factors such as:

1. Lesser waiting time compared to BTO flats
2. Mature estates established infrastructures, amenitites and good accessibility
3. More spacious than BTO flats
4. Increments in resale grants for familieis buying 4-Room and 5-Room resale flats for the first time
5. Lesser costs on renovation needed
6. More options for location and flat types

Whereas, the downsides are:

1. Higher price than BTO flats
2. Lesser remaining lease

### Problem Statement

With rising demand and prices in resale flats, we attempt on a regression model to based on variables selected from a combination of manual shortlisting and EDA analysis. For model prediction, it helps home buyers have a alternative metric to compare against the plethora of pricing tools available online.

### Datasets

#### Training Data 
* [HDB resale prices from 2012 to 2022 Training](./data/train.csv): <br>

#### Test Data 
* [HDB resale prices from 2012 to 2022 Testing](./data/test.csv): <br>

#### Kaggle Submission Predictions 
* [HDB resale prices from 2012 to 2022 Prediction](./data/sub_reg.csv): <br>

We have selected the following features and target:

|Feature|Description|Type|
|---|---|---|
|flat_type|type of the resale flat unit e.g. 3 ROOM|Categorical|
|floor_area_sqm|floor area of the resale flat unit in square metres|numerical|
|flat_model|HDB model of the resale flat, e.g. Multi Generation|Categorical|
|Tranc_Year|year of resale transaction|Categorical|
|mid|middle value of storey_range|numerical|
|max_floor_lvl|highest floor of the resale flat|numerical|
|planning_area|Government planning area that the flat is located|Categorical|
|mrt_nearest_distance|distance (in metres) to the nearest MRT station|numerical|
|hdb_age_at_tranc|hdb age during the transaction date|numerical|
|resale_price|price of resale flat|target|

### EDA Process

```mermaid
graph TD
    A-->B-->C-->D-->E-->F1-->G1-->H-->I-->J
    E-->F2-->G2-->H
    A[Data Preparation]
    B(Check for Nulls, Zeroes)
    C(Impute Nulls with mean)
    D(Create new feature)
    E{Manually <br> flag relevant features}
    F1(fa:fa-calculator Numerical <br> Features)
    F2(fa:fa-box Categorical <br> Features)
    G1(EDA: Correlation Heat Map)
    G2(EDA: Bar Chart)
    H(EDA: Scatter plot <br> Planning Area vs resale price)
    I(EDA: Scatter plot <br> floor Area vs resale price)
    J(Split dataset into 2 subset of planning areas <br> to train 2 different models)
```

Findings:
1. Features with many empty or null values are not selected to train the model.
2. A new HDB age at transaction feature is created as the age during transaction affects the price.
3. 9 features are shortlisted manually based on meaning of the feature and put into either categorical or numerical basket.
4. Correlation of numerical features indicate to use "floor_area_sqm", "hdb_age_at_tranc" and "mid" features are they are more correlated with resale price. "max_floor_level" will be excluded as it is correlated with both selected "hdb_age_at_tranc" and "mid" features at 0.44 and 0.56 respectively. "floor_area_sqm" is most correlated feature with resale price at 0.65.
![Numerical Features Correlation](images/Correlation.png)
5. Scatter plot shows more linearity when dataset is grouped by "planning_area"
![Data grouped by Planning Area](images/PlanningArea.png)
6. Splitting dataset into 2 subsets, we can see the first dataset is more linearly correlated. Though the second dataset still looks fan-shaped.
* First dataset contains planning areas with lower spread in prices
![Dataset 1](images/dataset1.png)
* Second dataset contains planning areas with higher spread in prices
![Dataset 2](images/dataset2.png)

### Model
I am building two models, one for planning areas with lower spread of prices, another for planning areas with higher spread in prices.

Both models go through a pipeline:
1. Numerical features are transformed by StandardScaler
2. Categorical features are one hot encoded
3. Estimator is chosen to be Ridge Regression and the best alpha is selected by gridsearchcv

### Results
* The first model is a ridge regression with a better cross validated train and test RMSE score of 37503 ± 194 and 37833.
* The second model is a ridge regression with a worse cross validated train and test RMSE score of 53092 ± 413 and 53024.

Submission score:
![Submission RMSE Score](images/submit1.png)

### Additional Exploration 

Additionally I also explored using 1 model for each flat type. However, The root mean square error is higher.

### Notebooks:

![Notebook for HDB resale Regression based on Planning Area](code/hdb_prices.ipynb) 
![Additional Notebook for HDB resale Regression based on Flat Type](code/hdb_prices_additional.ipynb)

---



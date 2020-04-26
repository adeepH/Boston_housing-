# Boston_housing-
This dataset contains information collected by the U.S Census Service concerning housing in the area of Boston Mass. It was obtained from the StatLib archive (http://lib.stat.cmu.edu/datasets/boston), and has been used extensively throughout the literature to benchmark algorithms. However, these comparisons were primarily done outside of Delve and are thus somewhat suspect. The dataset is small in size with only 506 input data.
we attempt to predict the correct prices using certain machine learning and deep learning algorithms.

### Install 
This Project Requires **Python 3** and the following libraries installed:
- [NumPy](http://www.numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [keras](https://keras.io/)
- [XGBoost](https://xgboost.readthedocs.io/en/latest/)


If you do not have Python installed yet, it is highly recommended that you install the [Anaconda](http://continuum.io/downloads) distribution of Python, which already has the above packages and more included. 
## About the dataset
To know more about the features of the 13 feature variables, execute the following code `boston_dataset.DESCR`
It is expected to show the following output:

- CRIM: Per capita crime rate by town
- ZN: Proportion of residential land zoned for lots over 25,000 sq. ft
- INDUS: Proportion of non-retail business acres per town
- CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
- NOX: Nitric oxide concentration (parts per 10 million)
- RM: Average number of rooms per dwelling
- AGE: Proportion of owner-occupied units built prior to 1940
- DIS: Weighted distances to five Boston employment centers
- RAD: Index of accessibility to radial highways
- TAX: Full-value property tax rate per $10,000
- PTRATIO: Pupil-teacher ratio by town
B: 1000(Bk — 0.63)², where Bk is the proportion of [people of African American descent] by town
- LSTAT: Percentage of lower status of the population
- MEDV: Median value of owner-occupied homes in $1000s

## Code
- Please refer `Boston_Housing_comparisions.ipynb` for the 
- Please refer `LinearRegresson.py` for the predictions done using Linear Regression.
- Please refer `Decision_Tree.py` for the predictions done using Decision Tree Regression.
- Please refer `XGBoost.py` for the predictions done using XGBoost Regressor.
- please refer `Neural_Network.py` for the model who's predictions are done using Keras 

# **Target Variable**
 - `MEDV`: median value of owner-occupied homes

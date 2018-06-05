
# coding: utf-8

# <img src="http://imgur.com/1ZcRyrc.png" style="float: left; margin: 20px; height: 55px">
# 
# # Project 3: Linear Regression and KNN - Train/Test Split
# 
# ---

# # Introduction
# 
# To evaluate how our models would perform on new data, we split our data into a training set and a test set, train only on the training set, and evaluate on the test set. In $k$-fold cross validation we repeat this process $k$ times, using a different subset of our data as the test set each time.
# 
# We use this process to compare multiple models on the same data set. Those models could be variations on a single type (e.g. linear regression models with and without a particular feature), or they could be of completely different types.

# Recall that k-fold cross-validation creates a hold portion of your data set for each iteration of training and validating:
# 
# ![](http://i.imgur.com/0PFrPXJ.png)

# ## Linear Regression Practice
# 
# In this given task, you will be asked to model the median home price of various houses across U.S. Census tracts in the city of Boston.

# In[1]:


import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import pandas as pd
import numpy as np
from sklearn.datasets import load_boston

boston = load_boston()

X = pd.DataFrame(boston.data,
                 columns=boston.feature_names)
y = pd.DataFrame(boston.target,
                 columns=['MEDV'])

print(boston['DESCR'])


# - Clean Up Data and Perform Exporatory Data Analysis

# Boston data is from scikit-learn, so it ought to be pretty clean, but we should always perform exploratory data analysis.

# **Items to include:**
# 
# - Data shape
# - Data types
# - Count of null values by column
# - Basic summary statistics
# - Number of unique values for each column
# - Plot of the distribution of each column (e.g. histogram, bar plot, or box plot)
# - Scatterplot of each column with median price

# In[3]:


#added back y to the datafram to explore data in full, I drop it later
X.loc[:,'MEDV'] = y

#explore
X.head()
X.shape
X.dtypes
X.describe()
X.nunique()

#set the rows and column since there were 14 variables
fig,ax = plt.subplots(nrows = 7, ncols = 2, figsize = (8,8))
X.hist(ax=ax);
X.columns


# In[4]:


fig, axes = plt.subplots(nrows=13, ncols=1, figsize=(3,30))

column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
       'PTRATIO', 'B', 'LSTAT']

#in order to avoid typing out each variable, created  a loop to do scatter chart for each variable against y
for xcol, ax in zip(column_names, axes):
    X.plot(kind='scatter', x=xcol, y='MEDV', ax=ax)


# - Get the MSE of a null model that simply predicts the mean value of MEDV. (You could do a train/test split here, but it won't make much difference for such a simple model.)

# In[5]:


X.loc[:,'Prediction'] = X.MEDV.mean()
X.head()
X.MEDV.mean()

from sklearn import metrics
metrics.mean_squared_error(X.MEDV, X.Prediction)


# - Develop a linear regression model to predict MEDV. Train it on 70% of the data. Gets its MSE on both that 70% and the other 30%.

# In[6]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#drop y now that we are ready to do regression
X = X.drop('MEDV', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,y)

lr = LinearRegression()

lr.fit(X_train,y_train)
y_pred_train = lr.predict(X_train)
y_pred_test = lr.predict(X_test)

print(metrics.mean_squared_error(y_train,y_pred_train))
print(metrics.mean_squared_error(y_test, y_pred_test))

#MSE went down quite a bit, the model has improved


# - K-fold cross-validation is slower than a simple train/test split, but it gives more reliable estimates of generalization error. Use ten-fold cross-validation to evaluate your model's MSE on both training and test data. Use this result to answer the following questions.

# In[7]:


from sklearn.model_selection import KFold

kf = KFold(n_splits=10,shuffle=True)
kf.split(X,y)

list(kf.split(X,y))
n = 0
mse_values_test = 0
mse_values_train = 0

#cycle through all 10 splits and perform LR on each, storing results into containers above
for train_index, test_index in kf.split(X,y):
    
    lr = LinearRegression()
    X_train = X.loc[train_index,:]
    y_train = y.loc[train_index]
    lr.fit(X_train,y_train)
    
    X_test = X.loc[test_index,:]
    y_test = y.loc[test_index]
    
    mse_test = metrics.mean_squared_error(y_test, lr.predict(X_test))
    mse_values_test += mse_test
    
    mse_train = metrics.mean_squared_error(y_train, lr.predict(X_train))
    mse_values_train += mse_train
    
    n +=1
    print('MSE test: ',mse_test)
    print('MSE train: ',mse_train)


print(mse_values_test/10)
print(mse_values_train/10)

#want to print the MSE for both the test and train data sets


# - How well did your model perform on the training set compared to the null model? What does this result tell you about the bias and/or variance of your model?

# It did slightly better

# - How well did your model perform on the test set compared to how well it performed on the training set? What does this result tell you about the bias and/or variance of your model?

# the test set performed worse than the training set, means there is more of a variance problem

# - Does your model appear to have more of a bias problem or more of a variance problem? Why?

# variance - because when we were able to use more data to test with (k fold) the test MSE was lowered, so more data did improve the model

# - Add or remove variables from your model to address this problem, and get its test-set MSE using ten-fold cross validation. Did your model get better or worse? By how much?

# In[8]:


#dropped B, ZN and CHAS but model actually got worse
X = X.drop(['B'],axis=1)

from sklearn.model_selection import KFold

kf = KFold(n_splits=10,shuffle=True)
kf.split(X,y)

list(kf.split(X,y))
n = 0
mse_values_test = 0
mse_values_train = 0
    
for train_index, test_index in kf.split(X,y):
    
    lr = LinearRegression()
    X_train = X.loc[train_index,:]
    y_train = y.loc[train_index]
    lr.fit(X_train,y_train)
    
    X_test = X.loc[test_index,:]
    y_test = y.loc[test_index]
    
    mse_test = metrics.mean_squared_error(y_test, lr.predict(X_test))
    mse_values_test += mse_test
    
    mse_train = metrics.mean_squared_error(y_train, lr.predict(X_train))
    mse_values_train += mse_train
    
    n +=1
    print('MSE test: ',mse_test)
    print('MSE train: ',mse_train)

#get the average of all 10 MSEs as the overall number to compare
print(mse_values_test/10)
print(mse_values_train/10)


# - **Bonus:** Find a *transformation* of one of more of your feature variables that allows you to create a model that performs better on the test set than your previous model. 
# 
# Possible examples:
#     - Square a column
#     - Take the square root of a column
#     - Take the logarithm of a column
#     - Multiply two columns together
# 
# Tip: Look at scatterplots of MEDV against your column(s) before and after applying the transformation. The transformation should help if it makes the relationship more linear.

# # KNN Practice

# In[9]:


# Read the iris data into a DataFrame
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris = pd.read_csv(url, header=None, names=col_names)


# In[10]:


iris.head()

# Increase the default figure and font sizes for easier viewing
plt.rcParams['figure.figsize'] = (6, 4)
plt.rcParams['font.size'] = 14


# In[11]:


# Create a custom colormap
from matplotlib.colors import ListedColormap

cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])


# - Map each iris species to a number. Let's use Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2 and assign the result to a column called 'species_num'.

# In[12]:


iris.loc[:,'species_num'] = (iris.loc[:,'species'].map({'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2}))


# - Clean Up Data and Perform Exporatory Data Analysis

# **Items to include:**
# 
# - Data shape
# - Data types
# - Count of null values by column
# - Basic summary statistics
# - Number of unique values for each column
# - Plot of the distribution of each column (e.g. histogram, bar plot, or box plot) grouped by species

# In[13]:


iris.shape
iris.dtypes
iris.isnull().sum()
iris.describe()
iris.nunique()

fig,ax = plt.subplots(nrows = 5, ncols = 1)
iris.groupby('species').hist(ax=ax);



# - Get the accuracy of a null model that simply predicts the most common species.

# In[14]:


#They are all equally common so it would just be 30% correct

iris.head()


# - Develop a KNN model to predict species. Use ten-fold cross-validation to evaluate your model's accuracy on both training and test data. Remember to standardize your feature variables!

# In[25]:


XX = iris.loc[:,['sepal_length','sepal_width','petal_length','petal_width']]
yy = iris.loc[:,'species_num']


from sklearn.neighbors import KNeighborsClassifier


kf = KFold(n_splits=10,shuffle=True,random_state=99)
kf.split(XX,yy)

list(kf.split(XX,yy))
n = 0
as_values_test = 0
as_values_train = 0
    
for train_index, test_index in kf.split(XX,yy):
    
    knn = KNeighborsClassifier(n_neighbors=5)
    XX_train = XX.loc[train_index,:]
    yy_train = yy.loc[train_index]
    knn.fit(XX_train,yy_train)
    
    XX_test = XX.loc[test_index,:]
    yy_test = yy.loc[test_index]
    
    as_test = metrics.accuracy_score(yy_test, knn.predict(XX_test))
    as_values_test += as_test
    
    as_train = metrics.accuracy_score(yy_train, knn.predict(XX_train))
    as_values_train += as_train
    
    n +=1
    print('AS test: ',as_test)
    print('AS train: ',as_train)


print(as_values_test/10)
print(as_values_train/10)


# - How well did your model perform on the training set compared to the null model? What does this result tell you about the bias and/or variance of your model?

# Much better! It tells me that variance is the key issue because as we added more features, the model got more accurate

# - How well did your model perform on the test set compared to how well it performed on the training set? What does this result tell you about the bias and/or variance of your model?

# slightly better on the training data but hardly. the model is well fitted

# - Does your model appear to have more of a bias problem or more of a variance problem? Why?

# variance
# 

# - Adjust $k$ to address this problem, and get the resulting test-set accuracy using ten-fold cross validation. Did your model get better or worse? By how much?

# In[26]:


kf = KFold(n_splits=10,shuffle=True,random_state=99)
n = 0
as_values_test = 0
as_values_train = 0
    
for train_index, test_index in kf.split(XX,yy):
    
    knn = KNeighborsClassifier(n_neighbors=2)
    XX_train = XX.loc[train_index,:]
    yy_train = yy.loc[train_index]
    knn.fit(XX_train,yy_train)
    
    XX_test = XX.loc[test_index,:]
    yy_test = yy.loc[test_index]
    
    as_test = metrics.accuracy_score(yy_test, knn.predict(XX_test))
    as_values_test += as_test
    
    as_train = metrics.accuracy_score(yy_train, knn.predict(XX_train))
    as_values_train += as_train
    
    n +=1
    print('AS test: ',as_test)
    print('AS train: ',as_train)


print(as_values_test/10)
print(as_values_train/10)

#not really able to make the test % over 96% , model got worse


# - Plot your model's test-set accuracy across a range of $k$ values using ten-fold cross validation. Use a large enough range of $k$ values to provide reasonable assurance that making $k$ larger would not help.
# 
# Tip: Use functions and loops to avoid writing duplicate code.

# In[40]:


scores = []
as_values_test = 0 
n = 0

for k in range(1,100): 
    
    knn = KNeighborsClassifier(n_neighbors=k)
    
    for train_index, test_index in kf.split(XX,yy):
       
        kf = KFold(n_splits=10,shuffle=True, random_state=99)
        
        XX_train = XX.loc[train_index,:]
        yy_train = yy.loc[train_index]
        knn.fit(XX_train,yy_train)

        XX_test = XX.loc[test_index,:]
        yy_test = yy.loc[test_index]

        as_test = metrics.accuracy_score(yy_test, knn.predict(XX_test))
        as_values_test += as_test

        n +=1
        
    test_accuracy = (as_values_test / 10)
    scores.append([k,test_accuracy])
    as_values_test = 0

data = pd.DataFrame(scores,columns=['k','score'])
fig, ax = plt.subplots()
data.plot(kind='line',x='k',y='score',ax=ax);

#need a nested for loop to iterate first on k values then on the k-folds


# In[41]:


scores


# - **Bonus:** Find a *transformation* of one of more of your feature variables that allows you to create a model that performs better on the test set than your previous model. 
# 
# Possible examples:
#     - Square a column
#     - Take the square root of a column
#     - Take the logarithm of a column
#     - Multiply two columns together

# ## Bonus
# 
# `scikit-learn` is the most popular machine learning library in Python, but there are alternative packages that have different strengths. 

# ### Example: Using the Statsmodels Formula

# In[18]:


# First, format our data in a DataFrame

df = pd.DataFrame(boston.data, columns=boston.feature_names)
df.loc[:, 'MEDV'] = boston.target
df.head()


# In[19]:


# Set up our new statsmodel.formula handling model
import statsmodels.formula.api as smf

# You can easily swap these out to test multiple versions/different formulas
formulas = {
    "case1": "MEDV ~ RM + LSTAT + RAD + TAX + NOX + INDUS + CRIM + ZN - 1", # - 1 = remove intercept
    "case2": "MEDV ~ NOX + RM",
    "case3": "MEDV ~ RAD + TAX"
}

model = smf.ols(formula=formulas['case1'], data=df)
result = model.fit()

result.summary()


# ### Bonus Challenge #1:
# 
# Can you optimize your R2, selecting the best features and using either test-train split or k-folds?

# ### Bonus Challenge #2:
# 
# Given a combination of predictors, can you find another response variable that can be accurately predicted through the exploration of different predictors in this data set?
# 
# _Tip: Check out pairplots, coefficients, and Pearson scores._

# In[20]:


# Check out variable relations
import seaborn as sns

sns.pairplot(X);


# In[21]:


# Check out Pearson scores


# ## Demo: Patsy

# In[22]:


import patsy

# Add response to the core DataFrame
df.loc[:, 'MEDV'] = y


# In[23]:


from sklearn.model_selection import train_test_split #If you didn't import it earlier, do so now

# Easily change your variable predictors without reslicing your DataFrame
y, X = patsy.dmatrices("MEDV ~ AGE + RM", data=df, return_type="dataframe")

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.7)


# In[24]:


from sklearn import metrics
from sklearn.linear_model import LinearRegression

# Rerun your model, iteratively changing your variables and train_size from the previous cell

lm = LinearRegression()
model = lm.fit(X_train, y_train)

predictions = model.predict(X_test)
print("R^2 Score: {}".format(metrics.r2_score(y_test, predictions)))


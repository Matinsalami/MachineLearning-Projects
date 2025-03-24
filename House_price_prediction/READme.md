The project is following the book Hands_on machine learning. I try to go through multiple steps. Then I train my data with different regression models and finally decide which model is the best for this project

First let's break down the project into 8 different steps.
The first step is "Look at the big picture" where we try to frame the problem, select a performance measure and check the assumptions of the project.

Let's get started....


-------------------------------------------------------------------------------------------

# 1. Look at the big picture

## a. Frame the problem

The dataset we are working on is California_housing_prices which is a dataset from 1990. The data are somehow old but it is a really well-structured datasset for educational purposes

Firstly, we need a model to learn from the data. And given all the other metrics predict the median housing in any district in California
The goal of the project is to give insight if a specific district worth investing or not for housing companies. So the model we work on directly effects the revenue of these companies. 

Secondly we want to know what type of machine leaning model we should work on. We clearly have a **Supervised** learning since it is a labeled training. It is a **Univariate regression** problem, since the predicted data is continuous and the number of value to predict is one. Also the data is small enough to fit in memory, so **batch learning** is enough.

## b. Select a Performance Measure

A typical performance measure for regression models is **Root Mean Square Error(RMSE)**. It gives a idea of how much error we have in our data with bigger weight with higher weight for large errors. 
If in the data we have many outliers then we would have used **Mean Absolute Error**. Both of these functions use the distance between two vectors: the predictions and the target values. 

Now we can start coding. And explore the data...

-------------------------------------------------------------------------------------------

# 2. Get the Data

## a. Download the data

The first step is to get the data. In this case I downloaded the .tgz file from repo of the writer. There are a few modifications for educational purposes.
In other cases the data is more available to us.

## b. Take a quick look at the Data Structure

After making the dataFrame, we use `housing.head()` to get insight about the data itself. As can be seen each row represents one district. and we have 10 columns which represents 10 different features they have. 

Then we use `housing.info()` to get insight about the type of features and number of entries. We have 20640 entries in the dataset which make the dataset a small one. Also we note that there are 20 missing values for `total_bedrooms` attribute which e should take care of.

All other objects are numerical except for `ocean_approximity` field. We use `value_counts()` method for this attribute and we find out that there are 5 different type of objects(<1H OCEAN, INLAND, NEAR OCEAN etc.). 

With `housing.describe()` method we get a summary of the numerical attributes of the data.

Another way to get a feel of the data e are working with is to use histograms. A histogram shows the number of instances in each value range. with `hist()` method we can draw all the histograms related to attributes. The attribute values are on the X axis.

With the histograms we can notice a few things: 
 
 1. The median income is represented as 10,000 of dollars. For example, 3 means 30,000 dollars
 2. The `housing_median` and `median_house_value` are capped. Meaning that we consider that the values never go beyond the one in the histogram.
 3. The sacles of the attributes are very different from each other. We surely need feature scaling.
 4. Also many hisograms are tail heavy. Meaning that the mean extends to the right of the median than to the left. We should transform these histogramsa later.

## c. Create a Test Set

For splitting the dataset into train and test set we can use two approaches. One is easy to implement; We just take the data set and using `train_test_split` function we split the data into 2 other datasets, namely train and test set. This method is excellent when the dataset is large. If not we are in danger of **Sampling bias** .

The other one is more specific and is more suitable for the data to be a representative of the whole data. In this method we split the **target values** and split it in a arbitrary number of intervals. Then based on the size of the interval we split the data. I will use the second method because the dataset is relatively small. (Quick example: The population of US is 51.3% female and 48.7% male. If we want to choose 1000 people ramdomly we should try to have 513 female and 487 male).

In the case of this project we want to make sure the test set is representative of the variuos categories of incomes. We use `cut()` method in numpy to cut the incomes in 5 different intervals and label them. Then we use `StratifiedshuffleSplit` classs to split the dataset. 

-----------------------------------------------------------------------------------------
# 3. Discover and Visualize the Data to Gain Insights

Now that we splitted the data. We set aside the test set and work on train set solely. 

## a. Visualizing Geographical data

Since we have longitude and latitude we can have a scatterplot of all the districts to visualize the data.

The first scatterplot represents California. The high density areas can be easily seen.

The second scatterplot take into account also the house prices with different colors. As seen the house prices in cities like Los Angeles or San Fransisco is much higher than other cities. The image shows that the housing price is pretty much related to the location and the population density. 

## b. Looking for correlations

Since the dataset is small we can compute the **standard correlation coefficient** between every pair of attributes using the `corr()` method. 
The correlation coefficient ranges from -1 to 1. When it is close to 1, it means that there is a positive relation correlation. For example the median house value tends to go up as the median income goes up.

Another way to check for correlations between the atrributes is the `scatter_matrix` function, which plots every numerical attribute against every other numerical attribute. Here we plotted only some promising attributes that seem most correlated with the median housing value.   

The most promising attribute is the **median_house_value**. So we will focus on this attribute mainly. By plotting a scatterplot some details can be revealed. 

 1. The correlation is very strong. The upward trend is easily seen
 2. The price cap is visible at some values like 450000.0, 350000.0, 280000.0. We may want to delete these values to prevent the algorithms from learning to reproduce these data quirks.

The number of quirks in the house value is 900 rows which is a huge number. However as this is an educational project, I removed the rows to get better results when training the data.

## c. Experimenting with Attribute Combinations

One last thing to do is to try out various attribute combinations. For example the total number of rooms in a district is not important if you do not know how many households are living there. Also the number of bedrooms is not important. You should compare it with the number of rooms. And finally we will consider population per household.
In the correlation matrix we can see that the new attributes are much more related to the median house value compared to the original values.

----------------------------------------------------------------------------

# 4. Prepare the Data for Machine Learning Algorithms

It is time to split the predictors and labels. First I split the data to numerical and categorical data. What is done here is done in basically 4 parts. Let me break them down. I use a pipeline transformer for all the numerical attributs. I use 3 transformers for numerical data. First one which is already created is called `CombinedAttributeAdder` and is used to make new attributes which are more related to median_house_value. Second one is `SimpleImputer` which is used to fill the missing gaps with the median of each column. Finally, third one is `StandarScalar` which is used for feature scaling.
Also there is one transformer for categorical data and it is `OneHotEncoder` and is used to turn the categorical data into numerical one. 
Last step taken is combining again the numerical and categorical data. Here again we use  

## a. Data Cleaning

Reffering to the dataset, We can see we have two attributes that have some missing values. We should take care of them. We use `SimpleImputer` function of sicket_learn. It fills all the missing values with a desired value. In this case we use the median of each column. 

## b. Handling Text and Categorical Attributes

Now it is time to take care of `"ocaen_proximity"` attribute. The problem is that this attribute is categorical and as most Machine Learning algorithms prefer to work with numbers, we will try to convert it to a numerical attribute somehow. sicket_learn provides a great tool called `OneHotEncoder` which creates a binary attribute per category. One attribute equals 1 while all others equal 0. 

## c. Feature scaling

The last transformer we are working with is used for standardization of the data. The transformer used is `StandardScalar`

## d. Final transformation pipeline

Here I used the `ColumnTransformer` class to have a pipeline taking care of all the data preparing for numerical and categorical data.

--------------------------------------------------------------------------------

# 5. Select and Train a Model













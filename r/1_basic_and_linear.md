
```r
# RECITATION 1 of 2025: 15.072 • Adam Deng, TA
# Some parts are adapted from previous years

# Section 1: R Basics

# Use the pound key to create a line comment

print("Hello World") # similar to Python

# Arithmetic: like punching in numbers to a calculator

1+1 # = 2 (add)

1-1 # = 0 (subtract)
2*2 # = 4 (multiply)
6/2 # = 3 (divide)
3**2 # = 9 (exponent)
3^2 # = 9 (exponent) note BOTH are acceptable!
8%%3 # = 2 (modulus); 8 is 2 mod 3. a mod n goes from 0 to n-1 for all a

# Variables

a = 3 # assign variable
print(a)

b <- "variable" # <- and = both do variable assignment
print(b)

# Booleans and Logic

a1 = 2
b1 = 2
c1 = 3

a2 = TRUE
b2 = FALSE

a1 == b1 # TRUE, note the double-equals for check equality NOT the single for assignment
a1 < c1 # TRUE
a1 <= b1 #TRUE

a2 | b2 # OR operator, TRUE
a2 & b2 # AND operator, FALSE

# Vectors

# c() operator concatenates elements into a vector
vec1 = c('element 1', 'element 2', 'element 3','element 4')
print(vec1)
length(vec1) # gets length of vector: 4

# accessing data
# IMPORTANT: R is 1 base index, Python is 0 base index. 1 good for position, 0 for offset

vec1[1] # 'element 1' first elements
head(vec1,1) # 'element 1' first elements

vec1[length(vec1)] # 'element 4' last element
tail(vec1,1) # 'element 4' last element


#vector slicing, vec1[ind_start:ind_end] 
#IMPORTANT: slicing is inclusive BOTH

vec1[2:3] #'element 2' 'element3' gets middle 2 elements
vec1[-(2:3)] #'element 1' 'element4' gets everything EXCEPT middle 2 elements

vec2 = c(1:10) #create a vector of integers from 1-10
print(vec2)

# Loading Dataframes: Built-In Example

data(mtcars) #load built-in dataset in R called "mtcars" (Motor Trend (car road tests))
head(mtcars) # display first few rows
tail(mtcars) # display last few rows


# Dataframes Basic Commands
dim(mtcars)  # get dimensions of the dataframe
nrow(mtcars) # get number of rows
ncol(mtcars) # get number of columns
mean(mtcars$mpg) #get average of column: $ used to access column


# Section 2: Manipulating Data

# Loading Data

install.packages("tidyverse") # first you have to install tidyverse
# Best not to run more than once—once you have it, comment out the above line
library(tidyverse) # now, you import it


# We will load a bluebike dataset stored as a CSV file. Go to Canvas and download the file "bluebikes.csv".

# First, since you're reading from local (your computer), set the working directory
# to the folder where the csv is located, probably a downloads folder.
getwd()
setwd('/Users/hannaz/Documents/mit/mit_class/15.072 Analytics Edge') # whatever it is for you

df <- read_csv('./rec1/bluebikes.csv', na="N\\A") 

?read_csv #is a help function to find more about the function

# N\\A means N\A should be a missing value. Need \\
# \ is known as a reserved character, like how you can't name files with : or $

# Getting Data Information

head(df) # see start information on df. <dbl> = double: number with decimals.
summary(df) # SUMMARY command. Variable avg, quantiles, min/max, missing.
dim(df) # dimension: rows, then cols

names(df) # all column names

# SELECTING DATA

# select as you would with slicing
# View the first 3 rows
df[1:3,]
# View first 3 columns 
df[,1:3]
# View the element at row 1 and column 1
df[1,1]

# select columns 1,3,5 
df[,c(1,3,5)] # c() creates a vector and provides values to this vector explicitly 
# select the first 5 rows with the specified columns
df[1:5,names(df) %in% c("rentals","temp")]
# %in% operator is used to check if an element belongs to a data frame or vector


# selecting columns by name

# Select a column by column name, using $ 
df$rentals # see all values…sometimes, not quite

# Select(data frame name, col name variables) keeps only the variables that you mention
selected_df_1 <- select(df, temp, rentals, rel_humidity)

# Select columns using starts_with(), ends_with(), contains() in select()
selected_df_2 <- select(df, starts_with("temp"))


# SELECTING ALL EXCEPT
# -columnName to exclude the column from the selection
selected_df_3 <- select(df, -month,-day, -hour) # minus sign is what not to select
# you may notice that this looks a bit like a mix of SQL and Python
view(selected_df_3)

# do not select 1,3,5 column. Useful when you want to remove certain columns from df 
df[,-c(1,3,5)]


# FILTERING
# Let's consider only examples or observations with precipitation equal to zero
# To do this we use the filter function, which takes the data table (name of data frame) as 
# the first argument and then the conditional statement (criteria for filtering) as the second. 

#Two Ways

#option 1
df_no_precip <- filter(df, precipitation == 0)
view(df_no_precip)

#option 2
df_filtered <- df[df$precipitation==0,] # don't forget the , at the end
view(df_filtered)

# select rows that have relative humidity greater than 50 
df[df$rel_humidity>50,]

# A Few More Functions: mutate, summarize, group_by, 

# mutate: What if you wanted to have a label that represented how interesting the weather was?
# weather_interest = 0.3*humidity + precipitation + windspeed
# mutate(), a tidyverse function, creates, modifies, and deletes columns
# Here we add a new variable, interest, to the df data frame

df <- mutate(df, interest = 0.3*rel_humidity + precipitation + windspeed)

#Alternatively
df$interest = df$rel_humidity*0.3 + df$precipitation + df$windspeed

# summarize
# We can compute the mean interestingness of the weather.
# summarize() takes a dataset and computes the statistics you specify on the columns you specify.
# we can also use median, min, max, sd, etc.  
summarize(df, 
             interestingness = mean(interest))

#Alternatively
interestingness = mean(df$interest)

# group_by makes group summaries.
# summarize(group_by()) creates grouped summaries
# For example, we can compute the avg temp of non-precipitation days & precip. days

no_precip_days_avg_temp <- summarize(group_by(df,precipitation=0), 
          avg_temp = mean(temp))
no_precip_days_avg_temp

precip_days_avg_temp <- summarize(group_by(df,precipitation>0), 
                            avg_temp = mean(temp))
precip_days_avg_temp # Note: it interprets it as a boolean! Group by whether or not…

# more sophisticated numbers: group by day of week and whether precipitation
dow_precip_df = summarize(group_by(df,day_of_week,precipitation>0), 
          rental_info = mean(rentals))
dow_precip_df # grouped by day of week and whether precip > 0, # of rentals
# note: rental_info was just the name of the column in dow_precip_df



# Section 3: Linear Regression.

library(tidyverse) # it's ok if you repeat import, nothing bad happens

rm(list = ls()) ##clear your current environment, get rid of all variables
# can save memory if your computer is running low on that

# now you need to get back the dataset!
df <- read_csv('rec1/bluebikes.csv', na="N\\A")

# The 'glimpse()' function outputs a quick summary of the dataset
glimpse(df)

## Single Variable Regression

# We will first build a regression model using only a single independent variable.
# Regression models in R are built using the 'lm' command, 'lm' is short for linear model.
# The code below builds a single variable linear model predicting 'rentals' with 'temp'.
# The command uses 'df' as the dataframe, 
# 'rentals' as the INDEPENDENT variable, and 'temp' as the DEPENDENT variable.

mod1 = lm(data = df, rentals ~ temp) # y ~ x, dep ~ indep

# The 'summary()' command shows the output of the model.
summary(mod1)

# The command 'confint()' gives confidence intervals for the variables in the model.
confint(mod1) # intercept and temperature: rentals = intercept + b * temp

# Simple scatterplot showing the 'rentals', 'temp' and the regression line for mod1

plot(df$temp,df$rentals)
abline(mod1, col ='red')

# To do regression analysis in R, two important commands 'lm()' and 'summary()'

# Let's build a second regression model on the independent variable 'windspeed'
mod2 = lm(data = df, rentals ~ windspeed)
summary(mod2)

# Scatterplot showing the 'rentals', 'windspeed' and the regression line for mod2

plot(df$windspeed,df$rentals)
abline(mod2, col ='red')

## Regression outputs

# Let's look at the output of mod1, regression with 'temp' as the independent variable.

summary(mod1)

# There is a ton of information to look at in the summary output. 
# We'll start with just a few:
#  - the coefficient 'Estimate'
#  - the fit to the training set as measured by R2
# Coefficient estimates: Higher temp is associated with more rentals
# The in-sample R2 (pronounced R-squared; a.k.a coefficient of determination) is 0.235 for mod1.

## Comparing models:

summary(mod2)

# The in-sample R^2 for mod2 is 0.0003742.

# R^2 is the proportion of the variation in the dependent variable explained by variation in the independent variable.
# The value ranges from 0 - 100 percent.

## Calculating R^2

# The formula for R^2 is 1 - SSR/SST where SSR is the sum of squared residuals and SST is the total sum of squares.
# Residuals are the differences between the predicted values and the actual values of a model.

#To get the predictions of a model in R (called "infer" in ML), use the 'predict()' function.
# For example for mod1.

predictions = predict(mod1, newdata = df)
head(predictions) # note: the 1, 2, 3... refers to the temp.

# The sum of squared residuals can be calculated from this code chunk.

SSR = sum((df$rentals - predictions)^2)
SSR

# The total sum of squares SST measures the sum of squared residuals between the dependent variable and the baseline model (the mean of the dependent variable).

# OPTIONAL: plot that shows the the baseline model: baseline literally means every predicted value is the mean

plot(df$temp, df$rentals)
abline( a=  mean(df$rentals), b = 0, col = 'red')


# To compute the SST use this code chunk.

SST = sum((df$rentals - mean(df$rentals))^2)
SST

# Then the R^2 is:

R_squared = 1 - SSR/SST
R_squared

# Which is equal to the R^2 value in the output of mod1.

# you can also delete variables from the model if they are useless
df <- select(df,-day)
df

# Categorical variables / factors

# In R, a categorical variable is called a 'factor'.
# We use the 'factor' and 'mutate' commands to convert 'month', hour' and
# 'day_of_week' to factors.

# Note: Categorical variables can have numeric type i.e. zipcodes, area codes, day-of-week, etc.

df = mutate(df, 
            month = factor(month), 
            hour = factor(hour),
            day_of_week = factor(day_of_week)
            )
glimpse(df)

# 'fct' denotes a factor column in the output
# A good example of importance of categorical variable is the variable "Hour". 
# If we considered this variable numeric instead of categorical, then we would interpret the model as "when the hour increases by 1, the number of rental bikes increases(or decreases) by x". 
# However, hour doesn't increase from 24 to 25. But by using hour as numeric values, we are forcing our model to predict results under this assumption, which creates inaccuracies in the model.

# We can build regression models on factor columns as well.

mod3 = lm(data = df, rentals ~ day_of_week)
summary(mod3)

# Out of order? Force order!
df$day_of_week <- factor(df$day_of_week, 
                         levels = c("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"))
mod3 = lm(data = df, rentals ~ day_of_week)
summary(mod3)

# Let's interpret "estimate"…intercept day is one of them
plot(df$day_of_week, df$rentals)

# It is important to note that each factor level (Mon, Tues, Wed, etc..) has its own coefficient.

# Bonus: Common Functions in R
# Random sampling
set.seed(1) # choose a random seed to draw results from - the same seed always gives the same output
sample(1:10, 4) # note: WITHOUT REPLACEMENT by default!
sample(1:10, replace = TRUE) # allows for repeats

#sample normal distribution data with mean = 20, standard deviation = 2
x <- rnorm(3, 20, 2) # 3 datapoints, mu = 20, sigma = 2
x
summary(x)

#sample uniform distribution (every outcome is equally likely) data with min = -1, max = 5
runif(10,min = -1,max = 5)

# sample follows binomial distribution, sample size = 20, number of trials = 100, probability of trial success = 0.5
rbinom(n=20, size = 100, p = 0.5) 

# sample follows poisson distribution, sample size = 10, lambda(expected value)=1
rpois(n=110, lambda = 1)
# poisson: e^-lambda * lambda^n / n! and is the number of times an event will occur within an interval


```
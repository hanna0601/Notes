

```r
# RECITATION 2: Linear Regression Computations + Logistic Regression: 15.072 • Adam Deng, TA
# Some parts are adapted from previous years


## multicolinearity
## if variables are very similar to each other (correlated), remove one of them

# cor()

# which one with higher corr with dependent


# SECTION 1: LINEAR REGRESSION ANALYSIS - 2nd half

library(tidyverse)
library(dplyr)

# setwd if you need
# setwd('../')
# getwd()

df <- read_csv("rec2/bluebikes.csv")

## Categorical Features (Factors)
# In R, a categorical variable is called a 'factor'.
# We use the 'factor' and 'mutate' commands to convert 'month', hour' and 'day_of_week' to factors.
# Note: Categorical variables can have numeric type i.e. zipcodes, area codes, day-of-week, etc.

df = mutate(df, 
            month = factor(month), 
            hour = factor(hour),
            day_of_week = factor(day_of_week)
            )

glimpse(df)

# 'fct' denotes a factor column in the output
# Why is "Hour" categorical and not numerical? Because you can't draw a linear association between 23 to 24 = 0

# We can build regression models on factor columns as well.

mod3 = lm(data = df, rentals ~ day_of_week)
summary(mod3)

# It is important to note that each factor level (Mon, Tues, Wed, etc..) has its own coefficient.

## Multicollinearity

# Multicollinearity is the presence of high correlation between independent variables. 
# There is no fixed rule of what constitutes "high" correlation, but typically correlations higher than 0.75. 
# We look at the correlation of numerical variables to identify potential multicollinearity issues.
# Question: WHY does this matter?

## simplify the model and make things faster and avoid redundency

numeric_vars <- select(df,
                       "rentals","temp",
                       "temp_wb","rel_humidity",
                       "windspeed","precipitation")
glimpse(numeric_vars)


coreMatrix <- round(cor(numeric_vars),2)
coreMatrix

# Note temp and temp_wb are highly correlated (0.98), so we should remove one of them.
# The rule-of-thumb for deciding which variable to keep and which to drop is that **when there is a multicollinearity issue is to keep the one with the higher absolute correlation to the dependent variable** 
# UNLESS there's some business/contextual reason to keep the other variable.
# Since temp_wb's correlation with rentals is lower than temp's correlation with rentals (0.43 vs 0.48), and we don't see any compelling business reasons in favor of temp_wb, we remove temp_wb from the dataframe.

df <- select(df, -temp_wb)

## Multiple Linear Regression with Train/Test Data

## Random Train/Test Split

# First, we set a seed. Recall that the `set.seed()` function ensures that you get the same result if you start with that same seed each time you run the same process.
# This process is fundamental in ML.

set.seed(42)

# We will use `sample()` to split dataset into train & test set using row number. The **`sample()`** function in R allows you to take a random sample of elements from a dataset or a vector, either with or without replacement.

train_rowIndex <- sample(row.names(df), 0.7*nrow(df))  
test_rowIndex  <- setdiff(row.names(df), train_rowIndex)  

df_train <- df[train_rowIndex, ]
df_test <- df[test_rowIndex, ]

### Establish a Baseline Model

baseline_train <- mean(df_train$rentals)
mod_baseline_train <- lm(data = df_train, rentals ~ 1) # literally just a line
summary(mod_baseline_train)

### Build the regression model

mod <- lm(data = df_train, rentals ~.)

# We are building a linear regression model where the training dataset is `loans_train`, the dependent variable is "rentals", and the independent variables are all other available variables, which is what the `.` stands for.
# Because we are doing multivariate regression, we can't plot a nice line of best fit. :(
# But we can look at the summary table to try to understand the output of our model.

summary(mod)

# Residuals: the distance from the actual data to the fitted lines; ideally we want them to be symmetrically distributed on both sides o the line, which means you want the min and max to roughly cancel out each other. And the 1st quantile and 3rd quantile to be equal distance to 0.

# Multiple R-squared = r-squared
# Adjusted R-squared: R-squared by the number of parameters in the model

### Examine your model

# Now, let's look at the output. Should we keep this model?
# Generally, we would examine our model by looking at the following:
# 1.  The coefficient 'Estimate' yielding the best fit (What does the coefficient for temp mean?)
# 2.  The stars/p-values indicating variable 'significance': In general, we want to keep variables with **at least 1 star**. These non-significant variables are then removed.
# 3.  The fit to the training set as measured by R^2

# What if there's a categorical variable? **If some of the categories are significant and others are not, then we generally preserve the categorical variable**
# We say that we are "95% confident" that the true coefficients for the variables with at least 1 star are different from zero. That is, we are confident these variable play a role in predicting rentals. We consider these to be significant variables (at the 95% level).

### Make Predictions Using the Model

# To generate model's predictions, we use the `pred()` function.

pred_test <- predict(mod, newdata=df_test)

### Out-of-sample R^2

resid_test <- df_test$rentals - pred_test
SSR_test <- sum((resid_test)^2)
SST_test <- sum((df_test$rentals - baseline_train)^2)
OSR <- 1 - SSR_test / SST_test
OSR
# 0.67…

### From Point Prediction to Interval Prediction

# Prediction interval returns you a range, where we have reasonable confidence (usually 95%) the true predicted value will fall in. It also gives an indication of how well this model performs.

# Let's make a prediction for an observation in the test dataset.

x0 <- df_test[900,]
# here we are taking one single row out of the dataset
x0

point_prediction_x0 <- predict(mod, newdata = x0)
# we make a prediction for this one row of data
point_prediction_x0

interval_prediction_x0 <- predict(mod, 
                                  newdata = x0, 
                                  interval = "prediction", 
                                  level = 0.95)
# here we are still using the predict() function 
# but we are adding additional parameters to predict an interval, 
# i.e. we are 95% confidence that the prediction 
# will be between this interval

interval_prediction_x0

### Remove variables based on p-value automatically - backwards selection - remove variables by significance
# install.packages("olsrr")
library(olsrr) # tools for building OLS regression models: OLS = ordinary least squares

ols_step_backward_p(mod, p_val = 0.05, progress = TRUE)

# `ols_step_backward_p()` builds a regression model from a set of candidate predictor variables by removing predictors based on p values, in a stepwise manner until there is no variable left to remove any more.
# `p_val = 0.05` specifies the p-value threshold where variables with p more than p_val will be removed from the model.
# `progress = TRUE` setting it to TRUE will display variable selection progress.

# check model
mod_check = lm(data = df_train, rentals ~ . - day)
summary(mod_check)

# SECTION 2: LOGISTIC REGRESSION

install.packages("ROCR") # "evaluating and visualizing classifier performance"
library(tidyverse)
library(ROCR)
library(skimr)

### 2. Loading data
loans <- read_csv("rec2/loans.csv")
# variable naming: make sure it makes sense!

# What is the overall mean of the installment variable?
glimpse(loans)

head(loans, 12)

# Explanation on the dataset:
# Dependent variable: 1 (Default) loan was not repaid, 0 (Repaid) otherwise.
# Independent variables:
# 1. Monthly loan installment ($)
# 2. log(annual income)
# 3. FICO score
# 4. Revolving balance (portion of credit card spending that goes unpaid at the end of a billing cycle)
# 5. Number of inquiries in the past six months
# 6. Number of derogatory public records

### 3. Exploratory analysis

# Run a multicollinearity check

cor(loans) # too many decimals
round(cor(loans), 2) # force 2 decimals
# What is the largest correlation between any two variables?

# As always, we should initially review the available variables and remove those that:
# 1. Do not make managerial sense -> Not applicable to this dataset
# 2. Present multicollinearity issues -> Not applicable to this dataset

### 4. Train / test split

# We are splitting the data into a train set and a test set, with 70% of the data going into the train set and 30% going to the test set.

set.seed(42)

train_rowIndex <- sample(row.names(loans), 0.7*nrow(loans))  
test_rowIndex  <- setdiff(row.names(loans), train_rowIndex)  

loans_train <- loans[train_rowIndex, ]
loans_test <- loans[test_rowIndex , ]

### 5. Logistic regression model


library(tidyverse)
library(ROCR)
library(skimr)


# While linear regression models are constructed with `lm()` (linear models), logistic regression models are constructed with `glm()` (generalized linear models). You can think of `glm` as generalization of the `lm` command that works for logistic regression and a variety of other models.
# The naming is suboptimal…
# We can specify the type of regression with the "family" argument.

mod <- glm(default ~., family="binomial",data=loans_train)
# Write the command with installment and log-income
summary(mod)

# pr(Y=1) = 1/(1+exp(-(B_0 + B_1X_1 +,..., B_PX_P)))
# log(p/(1-p)) = B_0 + B_1X_1 +,..., B_PX_P
# Reduce the model and re-fit

# Recall:  `~.` means that we want R to use all of the other variables as independent variables. You can also remove variables by writing `~ . - rev_balance, -installment`

# Coefficient can be interpreted as if the explanatory (independent variable) increases by 1 unit, the odds of (Y = 1, i.e. someone defaulting) increases by exp(coefficient).

### 6. Making predictions

# To use the model to predict for an individual data point, we can create a single-row data.frame as shown below

new_obs <- data.frame(installment=300, 
                      log_income=5, 
                      fico_score=600, 
                      rev_balance=7, 
                      inquiries=1, 
                      records=0)

# Going to show how to calculate the prediction directly
coef <- coef(mod)
coef

# Create a vector of predictor values, including 1 for the intercept
predictors <- c(1, new_obs$installment, new_obs$log_income, new_obs$fico_score, 
                new_obs$rev_balance, new_obs$inquiries, new_obs$records)

# Calculate the logit Z by multiplying coefficients with predictor values and summing
Z <- sum(coef * predictors)

# Calculate probability
prob <- 1 / (1 + exp(-Z))

# Print probability
print(prob)

# simpler method: predict
predict(mod, newdata = new_obs, type="response")
predict(mod, newdata=loans_test, type="response")

loans_test$pred_prob <- predict(mod, newdata=loans_test, type="response")

head(loans_test, 5)

# First we will create a new "actual_default" column so that it's easier to read.

loans_test$actual_default <- ifelse(loans_test$default == 1,"Defaulted","Repaid")
loans_test

# As we learned in class,to convert the predicted probabilities to a "Will Default" or "Will Repay" prediction, we need to use a cutoff. Let's start with a cutoff of 0.5.
cutoff <- 0.5

# Now we will compare to see if our prediction and the actual results in the test data are the same. 
# To do this, we will create a `prediction` column in `loans_test`

loans_test$prediction <- ifelse(loans_test$pred_prob > cutoff,"Predict_Default","Predict_Repay")
loans_test

# Once we have decided on a cutoff and added the additional columns, we can go ahead and calculate the key metrics we need to examine.

### 7. Key metrics calculation manually

#### 7.1 Confusion Matrix

# Now we are ready to create the "Confusion Matrix" and calculate key metrics contingency table displays the (multivariate) frequency distribution of the variables

confusion_matrix <- table(loans_test$prediction, loans_test$actual_default)
# Be careful -- sometimes columns can be switched and to read column names
confusion_matrix

#### 7.2 Accuracy

# Accuracy = # of correct predictions / total # of observations in the test set
# For a prediction to be correct, it has to agree with what actually happened, so we add up the diagonal elements of the confusion matrix to get the \# of correct predictions

accuracy <- (confusion_matrix[1,1] + confusion_matrix[2,2]) / nrow(loans_test)
accuracy

## WILL BE DONE NEXT RECITATION SEP 9/26

#### 7.3 TPR Sensitivity / TNR

# True Positive Rate (TPR): true positive / actual positive
# TPR = true positives / (true positives + false negatives)
TPR <- confusion_matrix[1,1] / (confusion_matrix[1,1] + confusion_matrix[2,1])
TPR
# very bad, <5% of predicted-positives are actually positive

# True Negative Rate (TNR): true negative / actual negatives
# TNR = true negatives / (true negatives + false positives)
TNR <- confusion_matrix[2,2] / (confusion_matrix[1,2] + confusion_matrix[2,2])
TNR
# TNR: 0.99 - if it says negative it's almost always negative

# "Predict_default" means we predicted a default, and "Repaid" means they did not default - a false positive!
  
### 8. ROC
# back to ROCR.

# The "prediction" command below calculates metrics like accuracy, TPR, FPR etc, for a range of cutoffs from 0.0 to 1.0. We store the results in the variable 'pred'

pred_object <- prediction(loans_test$pred_prob, loans_test$default)
# here it's critical that we use the original column of "default" rather than the human-readable column "actual_default"
str(pred_object)

# Now we can "extract" the TPR and FPR and plot them together on one chart. This is called the ROC curve.

plot(performance(pred_object,"tpr","fpr"))

### 9. AUC

# Now, let's calculate the Area Under the Curve (AUC) and store it in the "AUC" variable. In general we want AUC to be as large as possible.

AUC <- performance(pred_object, "auc")@y.values[[1]]

AUC

# 0.65

# Note: the `@` operator is similar to the `$` operator for accessing named elements of a list or data.frame but `@` is used to access "Slots" in an R object, which is just jargon for "components of (certain types of) R objects".

# The partial area under the ROC curve up to a given false positive rate can be calculated by passing the optional parameter `fpr.stop=0.5` (or any other value between 0 and 1) to performance.
```
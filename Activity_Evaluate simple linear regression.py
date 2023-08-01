#!/usr/bin/env python
# coding: utf-8

# # Activity: Evaluate simple linear regression

# ## Introduction

# In this activity, you will use simple linear regression to explore the relationship between two continuous variables. To accomplish this, you will perform a complete simple linear regression analysis, which includes creating and fitting a model, checking model assumptions, analyzing model performance, interpreting model coefficients, and communicating results to stakeholders.
# 
# For this activity, you are part of an analytics team that provides insights about marketing and sales. You have been assigned to a project that focuses on the use of influencer marketing, and you would like to explore the relationship between marketing promotional budgets and sales. The dataset provided includes information about marketing campaigns across TV, radio, and social media, as well as how much revenue in sales was generated from these campaigns. Based on this information, leaders in your company will make decisions about where to focus future marketing efforts, so it is critical to have a clear understanding of the relationship between the different types of marketing and the revenue they generate.
# 
# This activity will develop your knowledge of linear regression and your skills evaluating regression results which will help prepare you for modeling to provide business recommendations in the future.

# ## Step 1: Imports

# ### Import packages

# Import relevant Python libraries and packages. In this activity, you will need to use `pandas`, `pyplot` from `matplotlib`, and `seaborn`.

# In[1]:


# Import pandas, pyplot from matplotlib, and seaborn.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ### Import the statsmodel module and the ols function
# 
# Import the `statsmodels.api` Python module using its common abbreviation, `sm`, along with the `ols()` function from `statsmodels.formula.api`. To complete this, you will need to write the imports as well.

# In[2]:


# Import the statsmodel module.

# Import the ols function from statsmodels.

from statsmodels.formula.api import ols


# ### Load the dataset

# `Pandas` was used to load the provided dataset `modified_marketing_and_sales_data.csv` as `data`, now display the first five rows. This is a fictional dataset that was created for educational purposes. The variables in the dataset have been kept as is to suit the objectives of this activity. As shown in this cell, the dataset has been automatically loaded in for you. You do not need to download the .csv file, or provide more code, in order to access the dataset and proceed with this lab. Please continue with this activity by completing the following instructions.

# In[3]:


# RUN THIS CELL TO IMPORT YOUR DATA. 

### YOUR CODE HERE ###
data = pd.read_csv('modified_marketing_and_sales_data.csv')

# Display the first five rows.

data.head()


# ## Step 2: Data exploration

# ### Familiarize yourself with the data's features
# 
# Start with an exploratory data analysis to familiarize yourself with the data and prepare it for modeling.
# 
# The features in the data are:
# * TV promotion budget (in millions of dollars)
# * Social media promotion budget (in millions of dollars)
# * Radio promotion budget (in millions of dollars)
# * Sales (in millions of dollars)
# 
# Each row corresponds to an independent marketing promotion where the business invests in `TV`, `Social_Media`, and `Radio` promotions to increase `Sales`.
# 
# The business would like to determine which feature most strongly predicts `Sales` so they have a better understanding of what promotions they should invest in in the future. To accomplish this, you'll construct a simple linear regression model that predicts sales using a single independent variable. 

# **Question:** What are some reasons for conducting an EDA before constructing a simple linear regression model?

# [Write your response here. Double-click (or enter) to edit.]

# ### Explore the data size

# Calculate the number of rows and columns in the data.

# In[4]:


# Display the shape of the data as a tuple (rows, columns).

data.shape


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# There is an attribute of a pandas DataFrame that returns the dimension of the DataFrame.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# The `shape` attribute of a DataFrame returns a tuple with the array dimensions.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Use `data.shape`, which returns a tuple with the number of rows and columns.
# 
# </details>

# ### Explore the independent variables

# There are three continuous independent variables: `TV`, `Radio`, and `Social_Media`. To understand how heavily the business invests in each promotion type, use `describe()` to generate descriptive statistics for these three variables.

# In[5]:


# Generate descriptive statistics about TV, Radio, and Social_Media.

data[['TV', 'Radio', 'Social_Media']].describe()


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Subset `data` to only include the columns of interest.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Select the columns of interest using `data[['TV','Radio','Social_Media']]`.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Apply `describe()` to the data subset.
# 
# </details>

# ### Explore the dependent variable

# Before fitting the model, ensure the `Sales` for each promotion (i.e., row) is present. If the `Sales` in a row is missing, that row isn't of much value to the simple linear regression model.
# 
# Display the percentage of missing values in the `Sales` column in the DataFrame `data`.

# In[6]:


# Calculate the average missing rate in the sales column.

avg_missing_sales = data['Sales'].isna().mean()

# Convert the missing_sales from a decimal to a percentage and round to 2 decimal place.

avg_missing_sales = round(avg_missing_sales * 100, 2)

# Display the results (missing_sales must be converted to a string to be concatenated in the print statement).

print('Percentage of Promotions Mising Sales:' + str(avg_missing_sales) + '%')


# **Question:** What do you observe about the percentage of missing values in the `Sales` column?

# [Write your response here. Double-click (or enter) to edit.]

# ### Remove the missing data

# Remove all rows in the data from which `Sales` is missing.

# In[7]:


# Subset the data to include rows where Sales is present.

data = data.dropna(subset=['Sales'], axis=0)


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Refer to [the content about removing missing values from a DataFrame](https://www.coursera.org/learn/go-beyond-the-numbers-translate-data-into-insight/lecture/rUXcJ/work-with-missing-data-in-a-python-notebook).
# 
# </details>
# 

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# The `dropna()` function may be helpful.
# 
# </details>
# 

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Apply `dropna()` to `data` and use the `subset` and `axis` arguments to drop rows where `Sales` is missing. 
# 
# </details>
# 

# ### Visualize the sales distribution

# Create a histogram to visualize the distribution of `Sales`.

# In[8]:


# Create a histogram of the Sales.

sns.histplot(data['Sales'])

# Add a title

plt.title('Distribution of Sales')


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Use the function in the `seaborn` library that allows you to create a histogram.
# 
# </details>
# 

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Call the `histplot()` function from the `seaborn` library and pass in the `Sales` column as the argument.
# 
# </details>
# 

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# To get a specific column from a DataFrame, use a pair of single square brackets and place the name of the column, as a string, in the brackets. Be sure that the spelling, including case, matches the data exactly.
# 
# </details>
# 

# **Question:** What do you observe about the distribution of `Sales` from the preceding histogram?

# [Write your response here. Double-click (or enter) to edit.]

# ## Step 3: Model building

# Create a pairplot to visualize the relationships between pairs of variables in the data. You will use this to visually determine which variable has the strongest linear relationship with `Sales`. This will help you select the X variable for the simple linear regression.

# In[9]:


# Create a pairplot of the data.

sns.pairplot(data)


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Refer to [the video where creating a pairplot is demonstrated](https://www.coursera.org/learn/regression-analysis-simplify-complex-data-relationships/lecture/dnjWm/explore-linear-regression-with-python).
# 
# </details>
# 

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Use the function in the `seaborn` library that allows you to create a pairplot that shows the relationships between variables in the data.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Use the [`pairplot()`](https://seaborn.pydata.org/generated/seaborn.pairplot.html) function from the `seaborn` library and pass in the entire DataFrame.
# 
# </details>
# 

# **Question:** Which variable did you select for X? Why?

# I selected "TV" as the variable for X because it has the strongest linear relationship with "Sales". The scatterpot of "TV" and "Sales" clearly shows a strong and confident linear pattern, allowing us to estimate "Sales" effectively using "TV". Although "Radio" and "Sales" also show a linear relationship, the variance is larger compared to "TV" and "Sales".

# ### Build and fit the model

# Replace the comment with the correct code. Use the variable you chose for `X` for building the model.

# In[10]:


# Define the OLS formula.

ols_formula = "Sales ~ TV"

# Create an OLS model.

OLS  = ols(formula = ols_formula, data=data)

# Fit the model.

model = OLS.fit() 

# Save the results summary.

model.summary()

# Display the model results.


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Refer to [the video where an OLS model is defined and fit](https://www.coursera.org/learn/regression-analysis-simplify-complex-data-relationships/lecture/Gi8Dl/ordinary-least-squares-estimation).
# 
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Use the [`ols()`](https://www.statsmodels.org/devel/generated/statsmodels.formula.api.ols.html) function imported earlier— which creates a model from a formula and DataFrame—to create an OLS model.
# 
# </details>
# 

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Replace the `X` in `'Sales ~ X'` with the independent feature you determined has the strongest linear relationship with `Sales`. Be sure the string name for `X` exactly matches the column's name in `data`.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 4</strong></h4></summary>
# 
# Obtain the model results summary using `model.summary()` and save it. Be sure to fit the model before saving the results summary. 
# 
# </details>

# ### Check model assumptions

# To justify using simple linear regression, check that the four linear regression assumptions are not violated. These assumptions are:
# 
# * Linearity
# * Independent Observations
# * Normality
# * Homoscedasticity

# ### Model assumption: Linearity

# The linearity assumption requires a linear relationship between the independent and dependent variables. Check this assumption by creating a scatterplot comparing the independent variable with the dependent variable. 
# 
# Create a scatterplot comparing the X variable you selected with the dependent variable.

# In[11]:


# Create a scatterplot comparing X and Sales (Y).

sns.scatterplot(x=data['TV'], y=data['Sales'])


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Use the function in the `seaborn` library that allows you to create a scatterplot to display the values for two variables.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Use the [`scatterplot()`](https://seaborn.pydata.org/generated/seaborn.scatterplot.html) function in `seaborn`.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Pass the X and Y variables you chose for your simple linear regression as the arguments for `x` and `y`, respectively, in the `scatterplot()` function.
# 
# </details>

# **QUESTION:** Is the linearity assumption met?

# Yes, the linearity assumption is met as there is a clear and evident linear relationship between "TV" and "Sales."

# ### Model assumption: Independence

# The **independent observation assumption** states that each observation in the dataset is independent. As each marketing promotion (i.e., row) is independent from one another, the independence assumption is not violated.

# ### Model assumption: Normality

# The normality assumption states that the errors are normally distributed.
# 
# Create two plots to check this assumption:
# 
# * **Plot 1**: Histogram of the residuals
# * **Plot 2**: Q-Q plot of the residuals

# In[14]:


import statsmodels.api as sm

# Calculate the residuals.
residuals = model.resid

# Create a 1x2 plot figures.
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

# Create a histogram with the residuals. 
sns.histplot(residuals, ax=axes[0])

# Set the x label of the residual plot.
axes[0].set_xlabel('Residual Value')

# Set the title of the residual plot.
axes[0].set_title('Histogram of Residuals')

# Create a Q-Q plot of the residuals.
sm.qqplot(residuals, line='s', ax=axes[1])

# Set the title of the Q-Q plot.
axes[1].set_title('Normal Q-Q Plot')

# Use matplotlib's tight_layout() function to add space between plots for a cleaner appearance.
plt.tight_layout()

# Show the plot.
plt.show()


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Access the residuals from the fit model object.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Use `model.resid` to get the residuals from the fit model.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# For the histogram, pass the residuals as the first argument in the `seaborn` `histplot()` function.
#     
# For the Q-Q plot, pass the residuals as the first argument in the `statsmodels` [`qqplot()`](https://www.statsmodels.org/stable/generated/statsmodels.graphics.gofplots.qqplot.html) function.
# 
# </details>

# **Question:** Is the normality assumption met?

# The normality assumption appears to be met for this model. Both the histogram of the residuals and the Q-Q plot show that the residuals are approximately normally distributed, indicating that the errors follow a normal distribution. This suggests that the normality assumption is valid for the model.

# ### Model assumption: Homoscedasticity

# The **homoscedasticity (constant variance) assumption** is that the residuals have a constant variance for all values of `X`.
# 
# Check that this assumption is not violated by creating a scatterplot with the fitted values and residuals. Add a line at $y = 0$ to visualize the variance of residuals above and below $y = 0$.

# In[16]:


# Create a scatterplot with the fitted values from the model and the residuals.

fig = sns.scatterplot(x=model.fittedvalues, y=model.resid)

# Set the x-axis label.
fig.set_xlabel("Fitted Values")

# Set the y-axis label.
fig.set_ylabel("Residuals")

# Set the title.
fig.set_title("Fitted Values v. Residuals")

# Add a line at y = 0 to visualize the variance of residuals above and below 0.
fig.axhline(0)

# Show the plot.
plt.show()


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Access the fitted values from the `model` object fit earlier.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Use `model.fittedvalues` to get the fitted values from the fit model.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Call the `scatterplot()` function from the `seaborn` library and pass in the fitted values and residuals.
#     
# Add a line to the figure using the `axline()` function.
# 
# </details>

# **QUESTION:** Is the homoscedasticity assumption met?

# Yes, the homoscedasticity assumption is met for this model. The variance of the residuals appears to be consistent across all values of the independent variable 𝑋, indicating that the spread of the residuals is constant. Therefore, the assumption of homoscedasticity is satisfied.

# ## Step 4: Results and evaluation

# ### Display the OLS regression results
# 
# If the linearity assumptions are met, you can interpret the model results accurately.
# 
# Display the OLS regression results from the fitted model object, which includes information about the dataset, model fit, and coefficients.

# In[17]:


# Display the model_results defined previously.

model_results = model.summary()
model_results


# **Question:** The R-squared on the preceding output measures the proportion of variation in the dependent variable (Y) explained by the independent variable (X). What is your intepretation of the model's R-squared?
# 

# The interpretation of the model's R-squared is as follows: When using TV as the independent variable (X), the simple linear regression model explains approximately 99.9% of the variation in the dependent variable (Sales). This means that TV is highly effective in predicting Sales, accounting for a significant portion of the variability in Sales. However, it's important to note that the R-squared value will vary depending on the variable selected for X in the model.

# ### Interpret the model results

# With the model fit evaluated, assess the coefficient estimates and the uncertainty of these estimates.

# **Question:** Based on the preceding model results, what do you observe about the coefficients?

# Based on the preceding model results, when TV is used as the independent variable (X):
# 
# The coefficient for the Intercept is -0.1263. This means that when the TV advertising budget is zero, the model predicts a negative sales value of approximately -0.1263. However, this interpretation might not be practically meaningful since TV advertising budget can't be zero in real-world scenarios.
# 
# The coefficient for TV is 3.5614. This means that for every one unit increase in the TV advertising budget, the model predicts an increase in Sales by approximately 3.5614 units. This coefficient indicates the estimated change in Sales associated with a one-unit change in TV advertising spending while holding other variables constant.
# 
# It's important to consider the context and scale of the variables when interpreting the coefficients.

# **Question:** How would you write the relationship between X and `Sales` in the form of a linear equation?

# When using TV as the independent variable (X), the relationship between TV and Sales can be expressed in the form of a linear equation as follows:
# 
# Sales (in millions) = -0.1263 + 3.5614 * TV (in millions)
# 
# This equation represents the estimated linear relationship between the TV advertising budget (TV) and the Sales in millions. It implies that for every one unit increase in TV advertising spending, Sales are estimated to increase by 3.5614 million units, while holding other variables constant. The intercept term (-0.1263) represents the estimated Sales when the TV advertising budget is zero, but this might not be practically meaningful as TV advertising budget is unlikely to be zero in real-world scenarios.

# **Question:** Why is it important to interpret the beta coefficients?

# Interpreting the beta coefficients is crucial because it provides valuable insights into the relationship between the independent variables and the dependent variable. The beta coefficients allow us to estimate the size and direction (positive or negative) of the impact each independent variable has on the dependent variable. By interpreting these coefficients, we gain a deeper understanding of how changes in the independent variables affect the outcome.
# 
# In the context of the model using TV as the independent variable, the beta coefficient of 3.5614 indicates that, on average, for each additional unit of TV advertising spending (in millions), there is an estimated increase of 3.5614 million units in sales, assuming other factors remain constant. This interpretation helps us quantify the influence of TV promotional budgets on sales and make informed decisions based on the model's findings. It allows businesses to understand which factors have the most significant impact on the outcome, helping them optimize their strategies for better results.

# ### Measure the uncertainty of the coefficient estimates

# Model coefficients are estimated. This means there is an amount of uncertainty in the estimate. A p-value and $95\%$ confidence interval are provided with each coefficient to quantify the uncertainty for that coefficient estimate.
# 
# Display the model results again.

# In[18]:


# Display the model_results defined previously.

model_results


# **Question:** Based on this model, what is your interpretation of the p-value and confidence interval for the coefficient estimate of X?

# In this model, when TV is used as the independent variable, the coefficient estimate of X has a p-value of 0.000, indicating that the estimate is highly statistically significant. The p-value essentially tells us the probability of observing such a strong relationship between TV and Sales if there were no true association between them. Since the p-value is very close to zero, it suggests that the relationship between TV and Sales is highly unlikely to be due to chance.
# 
# Additionally, the 95% confidence interval for the coefficient estimate of X is [3.558, 3.565]. This interval represents the range of values within which we can be 95% confident that the true parameter value of the slope lies. The narrowness of this confidence interval suggests that the estimate is precise, and there is little uncertainty in the estimation of the slope of X.
# 
# In summary, the low p-value and narrow confidence interval indicate that there is strong evidence to support the relationship between TV and Sales. The business can be highly confident in the impact TV advertising has on sales, as the model's findings are highly statistically significant and precise.

# **Question:** Based on this model, what are you interested in exploring?

# Based on this model, there are several areas of interest for exploration:
# 
# Estimating Sales: One key aspect of interest is providing the business with estimated sales values based on different TV promotional budgets. By inputting different values for TV advertising expenditures into the linear equation, we can estimate the corresponding sales figures. This can help the business make informed decisions about their advertising budget allocation and potential revenue generation.
# 
# Multiple Independent Variables: While this model focused on using TV as the sole independent variable, there is an opportunity to explore the impact of multiple variables on Sales. Including other variables such as Radio advertising expenditure or any other relevant factors could provide a more comprehensive understanding of the factors influencing sales.
# 
# Visualizing Results: Adding visualizations, such as seaborn's regplot(), can enhance the interpretation of the results. By plotting the data points along with a best-fit regression line, it becomes easier to visualize the relationship between TV advertising and Sales. This visual representation can help stakeholders grasp the insights more effectively.
# 
# Model Validation: It is essential to validate the model's performance and generalization ability. Exploring metrics like Mean Squared Error (MSE) or Cross-Validation can provide insights into how well the model predicts Sales based on TV expenditures and how it might perform on unseen data.
# 
# Overall, these areas of exploration can contribute to a more comprehensive understanding of the relationship between TV advertising and Sales, enabling the business to make data-driven decisions and optimize their marketing strategies for better performance.

# **Question:** What recommendations would you make to the leadership at your organization?

# Based on the analysis, I would recommend the following to the leadership at our organization:
# 
# Allocate Budget to TV Promotion: Given that TV advertising has the strongest positive linear relationship with sales, it is advisable to prioritize and allocate a significant portion of the marketing budget to TV promotions. The model estimates that for each one million dollars increase in the TV promotional budget, sales are expected to rise by approximately 3.56 million dollars. This confident estimate indicates that investing in TV advertising can yield substantial returns in terms of increased sales.
# 
# Monitor and Optimize Results: While TV advertising shows a strong association with sales, it is essential to continuously monitor and assess the actual impact of TV promotions on sales performance. Regularly analyzing the sales data in correlation with advertising efforts can help identify trends and potential opportunities for optimization.
# 
# Consider Complementary Strategies: While TV promotions have proven to be influential, it's also worth considering complementary marketing strategies that include radio and social media promotions. Although their impact might be relatively smaller in comparison to TV, a well-rounded marketing approach that leverages multiple channels can reinforce brand visibility and broaden the customer reach.
# 
# Data-Driven Decision Making: It's important for the leadership to continue making data-driven decisions. Regularly analyze and update the model as new data becomes available to ensure the marketing strategies align with real-time sales trends and customer behavior.
# 
# Competitive Analysis: Keep an eye on competitors' marketing activities, especially in the TV advertising space. Understand the market dynamics and adjust the TV promotion strategies accordingly to maintain a competitive edge.
# 
# By implementing these recommendations, the organization can leverage the insights from the model to optimize marketing efforts, increase sales, and make informed decisions that contribute to the overall growth and success of the business.

# ## Considerations
# 
# **What are some key takeaways that you learned from this lab?**
# Key Takeaways from this Lab:
# 
# Exploratory Data Analysis: Exploring the data and visualizing the relationships between variables are crucial steps in selecting the most appropriate X variable for a simple linear regression model. Understanding the patterns and correlations in the data helps in making informed decisions about which variable to use as the predictor.
# 
# Assumption Checking: Before interpreting the results of a simple linear regression model, it is essential to check that the assumptions of linearity, independence, normality, and homoscedasticity are met. These assumptions ensure that the model is valid and reliable for making predictions.
# 
# R-squared for Model Evaluation: R-squared is a valuable metric for understanding how well the independent variable explains the variation in the dependent variable. It provides insights into the proportion of variance in the target variable that is accounted for by the predictor, offering a measure of prediction error.
# 
# Uncertainty Measures: When interpreting coefficient estimates, it is essential to provide measures of uncertainty, such as p-values and confidence intervals. These measures help assess the reliability of the coefficient estimates and determine if they are statistically significant.
# 
# Business Insights: The results of the regression model can provide valuable insights for decision-making and strategic planning. Identifying the most influential variable and understanding its impact on the outcome variable can guide resource allocation and marketing strategies for optimizing business performance.
# 
# Overall, this lab highlights the importance of thorough data exploration, assumption checking, and providing meaningful interpretations with uncertainty measures to draw reliable conclusions from simple linear regression models. These skills are essential for making data-driven decisions and leveraging data analysis to drive business success.
# 
# 
# **What findings would you share with others?**
# 
# Findings to Share:
# 
# Sales Distribution: Sales data is relatively evenly distributed across a range of $25 million to $350 million for all promotions.
# 
# Strong Linear Relationship: Among the available promotions, TV shows the strongest linear relationship with sales. There is a clear positive trend between TV spending and sales, indicating that increasing TV promotional budgets is likely to result in higher sales. On the other hand, the relationship between radio and sales is moderate, while social media has a weaker impact on sales.
# 
# High R-squared Value: When using TV as the independent variable, the simple linear regression model explains a significant proportion of the variation in sales, with an impressive R-squared value of 0.999. This indicates that TV effectively accounts for 99.9% of the variance in sales, making it a powerful predictor.
# 
# Coefficient Estimates: The regression model yields the following coefficient estimates when TV is used as the independent variable: The intercept is -0.1263, and the coefficient for TV is 3.5614. These coefficients help estimate the relationship between TV spending and sales.
# 
# Statistical Significance: The coefficient for TV's slope is highly statistically significant, with a p-value of 0.000. Additionally, the 95% confidence interval for the TV coefficient lies between [3.558, 3.565]. These results indicate that the impact of TV spending on sales is robust and reliable.
# 
# These findings suggest that the organization should prioritize allocating resources to TV promotions to maximize sales. With a strong linear relationship and high explanatory power of TV spending on sales, the business can confidently invest in TV advertising campaigns to drive revenue growth.
# 
# **How would you frame your findings to stakeholders?**
# 
# We are pleased to share our findings from the analysis of promotional budgets and their impact on sales. Among the various promotional channels (TV, social media, and radio), TV stands out as the most influential driver of sales.
# 
# Our linear regression model indicates that a whopping 99.9% of the variation in sales can be attributed to the TV promotional budget alone. This means that nearly all fluctuations in sales can be explained by changes in TV spending, making it an exceptionally reliable predictor of sales performance.
# 
# According to the model's estimates, a one million dollar increase in the TV promotional budget is associated with a substantial increase of 3.5614 million dollars in sales. The statistical analysis confirms the robustness of this relationship, with a narrow 95% confidence interval ranging from 3.558 to 3.565 million dollars.
# 
# We are confident in these findings and recommend that the organization prioritize allocating resources to TV promotions to drive revenue growth. The strong positive linear relationship and high explanatory power of TV spending on sales make it a strategic and rewarding investment.
# 
# Thank you for your attention to these important results, and we are eager to discuss further insights and implications with you in our upcoming meeting.

# #### **References**
# 
# Saragih, H.S. (2020). [*Dummy Marketing and Sales Data*](https://www.kaggle.com/datasets/harrimansaragih/dummy-advertising-and-sales-data).
# 
# Dale, D.,Droettboom, M., Firing, E., Hunter, J. (n.d.). [*Matplotlib.Pyplot.Axline — Matplotlib 3.5.0 Documentation*](https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.axline.html). 

# **Congratulations!** You've completed this lab. However, you may not notice a green check mark next to this item on Coursera's platform. Please continue your progress regardless of the check mark. Just click on the "save" icon at the top of this notebook to ensure your work has been logged.

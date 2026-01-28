## Introduction
The Amazon Kindle Books Dataset 2023 is a comprehensive collection featuring around 130,000 Kindle books available on Amazon. This dataset offers rich metadata for each book, including titles, authors, publishers, star ratings, number of reviews, pricing, accolades, publication dates, and genre classifications. Compiled to enable detailed analysis, it serves as a valuable resource for exploring trends in digital book publishing, consumer preferences, and market behaviors within the Kindle ecosystem. The dataset supports various applications such as market research, recommendation systems, and machine learning projects aimed at understanding factors that influence book popularity and sales dynamics in the digital reading landscape. This dataset was collected and made available in 2023, reflecting current trends and data points pivotal for timely insights and decision-making in the digital book market.

## Dataset justification
The dataset includes explicit bestseller flags ("isBestSeller") for books, creating a clear binary target variable required for logistic regression classification. The dataset offers multiple predictor variables potentially influencing bestseller status, including author, category, star ratings, number of reviews, price, and publisher/seller information. These features allow for comprehensive modeling. With approximately 130,000 books spanning multiple categories and genres, the dataset provides sufficient observations and variability to train a robust model and generalize well. The dataset is structured, cleaned, and contains standardized identifiers like ASIN plus consistent formats for dates and categories, which facilitate feature engineering and model training. As the dataset is from 2023, it reflects recent market trends and consumer behavior in digital book sales, making the predictions relevant for current business decisions. The dataset is publicly available on Kaggle and easily downloadable via the kagglehub Python package, simplifying implementation and reproducibility.

## Steps Taken
Step 1: Import the neccesary pyhton libraries
 Import the python libraries that we need to execute the logistic regression model. These libraries provide pre-built, optimized tools and functions to handle data manipulation, visualization, modelling, and evaluation efficiently.  These are the librarises: 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

The libraries I worked with is the following:
- Sklearn: It provides the logistic regression model, tools to split data, and evaluation metrics.
- NumPy: NumPy it supports numerical operations needed for calculations. It is important to have a basic understanding of NumPy arrays and operations.
- Pandas: Pandas allow for data manipulation and analysis in Python. It is used to read and preprocess data for use in scikit-learn.
- Data visualization: (Seaborn and Matplotlib) facilitate creating insightful visualizations during exploratory data analysis. Matplotlib and Seaborn are popular data visualization packages in Python.
    
Step 2: Load the data into a dataframe.
This dataset can be found on Kaggle.com.
This step allows for structuring, inspecting and manipulating raw data for analysis and modeling.  DataFrames organizes the data in rows and columns, making it easy to perform operations. Loading data into a DataFrame allows me to quickly review column names, data types, and the presence of missing values

Step 3: Data cleaning
Data cleaning is essential because it ensures the dataset is accurate, consistent, and reliable, directly impacting the quality and trustworthiness of the model. Clean the data by handling missing values, correcting errors and removing duplicates (Datacamp, 2025).
Typical data cleaning steps (Datacamp, 2025):
1.	Check for missing or null values
2.	Variable types
3.	Change categorical values to binary
4.	Look for outliers


Step 4: Exploratory Data Analysis
This step is about understanding the dataset in depth before building any machine learning model. EDA helps to summarize the main characteristics of the data, detect patterns, spot anomalies or outliers, test hypotheses, and select the best features for analysis (Biswal, 2025).
EDA Steps:
1.	Univariate analysis: Inspect one variable at a time using histograms, box plots, scatter plot and summary statistics like mean and median.
2.	Bivariate:  Investigate relationships between two or more variables with scatter plots, correlation matrices, and pair plots.
3.	Remove outliers
4.	Normalizing using sklearn
Step 5: Modelling
Build the logistic regression model by splitting the data into training and testing sets, selecting relevant features, and tuning hyperparameters. Feature selection reduces the number of input variables, which helps to avoid overfitting by minimizing noise and irrelevant information (Belcic, 2024).
Step 6: Evaluate the model
Assess model performance using metrics like accuracy, precision, or F1-score.
Step 7: Conclusion

## Data Overview
This project's dataset comprises retail sales transaction logs that detail customer purchases. Every entry corresponds to one transaction, featuring fields like transaction ID, customer information, product category, item quantity, price per unit, and total transaction amount.
Offering a structured snapshot of sales operations, the data is ideal for examining buying patterns, revenue shifts, and performance by category. Its transactional format supports aggregation and slicing to address core business queries, including top categories, valuable customers, and aggregate sales metrics.
Prior to analysis, the dataset underwent validation and preprocessing to guarantee result accuracy, uniformity, and trustworthiness.

## Data Cleaning
The data cleaning process was conducted to improve data quality and ensure the dataset was ready for analysis. The following steps were performed:

1. Handling missing values: Records with missing or null values in critical fields such as sales amount, quantity, or customer identifiers were identified and addressed to prevent inaccurate aggregations.
2. Data type validation: Numeric fields such as quantity, price, and total sales were checked and corrected to ensure they were stored in the appropriate numeric formats.
3. Duplicate checks: The dataset was examined for duplicate transaction records, which could distort sales totals and customer counts.
4. Standardization: Categorical fields, such as product categories, were reviewed for consistency in naming conventions to avoid fragmented groupings during analysis.
5.Logical validation: Sales values were verified to ensure consistency between quantity, unit price, and total sales where applicable.

## Exploratory Data Analysis

 Category count

 The bar chart depicts the number of books in different categories, showing that Biographies & Memoirs, Children’s eBooks, and Parenting & Relationships are the leading genres, each with nearly 5,000 titles. Categories like Science & Math, Computers & Technology, and Self-Help have a moderate number of books, whereas genres such as Romance, Foreign Language, and Comics have substantially fewer entries. Overall, the chart highlights a notable unevenness in the distribution of books, with a few genres dominating the dataset, indicating differing degrees of popularity or accessibility among the various categories.
 <img width="940" height="545" alt="image" src="https://github.com/user-attachments/assets/219c7bff-31c5-4b08-906c-5308a167a97a" />

Numeric Variables

The bar chart contrasts the quantity of best sellers and non-best sellers across multiple book genres. It shows that in every genre, non-best sellers (depicted in blue) greatly exceed best sellers (depicted in orange). Genres such as Biographies & Memoirs, Children’s eBooks, and Parenting & Relationships feature the largest total book counts, but only a small share of these are best sellers. Likewise, categories with fewer titles, like Romance, Fremdsprachen, and Comics also have very few best sellers. Overall, the chart emphasizes a marked disparity between best sellers and non-best sellers throughout all categories, implying that while certain genres have many publications, only a small percentage reach best-seller status.
<img width="940" height="467" alt="image" src="https://github.com/user-attachments/assets/37928d2d-df2f-4a11-89ca-225eadd7e8c0" />

## Conclusion
this project explored the Amazon Kindle Books Dataset (2023) to understand key patterns and insights within a large collection of over 130,000 book records. The process involved importing and cleaning the data, performing exploratory data analysis, and examining trends in book pricing, ratings, categories, and publication years. Through this analysis, it became evident that customer preferences and pricing play a significant role in book performance on Amazon. The findings provide a strong foundation for future work, such as applying machine learning models for sentiment analysis or recommendation systems, to better predict and enhance book success in the digital marketplace.
 



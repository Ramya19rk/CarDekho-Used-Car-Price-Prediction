# CarDekho-Used-Car-Price-Prediction

Problem Statement: This project focuses on predicting used car prices accurately by analyzing a diverse dataset obtained from CarDekho, encompassing details from six different locations. The primary objective is to develop a machine learning model that offers users precise valuations for used cars.

NAME : RAMYA KRISHNAN A

BATCH: DW75DW76

DOMAIN : DATA SCIENCE

Linked in URL : www.linkedin.com/in/ramyakrishnan19

# Data Preprocessing:

Six separate Excel files were initially processed, each representing car details from a specific location.

Cleaned and organized the data in a Jupyter Notebook to create a clear DataFrame.

Concatenated all DataFrames into one comprehensive dataset.

# Exploratory Data Analysis (EDA):

Conducted EDA to understand the distribution of the target variable (used car prices) and explore relationships between relevant features.

# Feature Engineering:

Extracted valuable information from features like age, mileage, etc., to enhance the model's predictive power.

# Data Encoding and Outlier Handling:

Encoded categorical values to numeric for model compatibility.

Applied a log transformation to handle outliers effectively.

# Model Selection:

Tested various regression models, including Linear Regression, Decision Trees, and Random Forests.

Selected the ExtraTreeRegressor as the final model based on superior performance.

# Model Serialization:

Serialized the trained model using the pickle library for easy deployment.

# Prediction:

Conducted predictions using the trained ExtraTreeRegressor model, achieving accurate results.


# Import necessary Libraries

import streamlit as st
import pandas as pd
import numpy as np
from ast import literal_eval
from pandas import json_normalize
from pprint import pprint
import ast
from sklearn.preprocessing import LabelEncoder
import re
import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
import pickle

# Data Import

# Kolkata Cars

df_1 = pd.read_csv('/Users/arul/Documents/CarDheko/kolkata_cars.csv')

df_11 = pd.DataFrame(df_1['new_car_detail'])

#converting string representations of Python literals or containers in the 'new_car_detail' column of the DataFrame into actual Python objects 
df_11['new_car_detail'] = df_11['new_car_detail'].apply(lambda x: ast.literal_eval(x.strip('"')))

# Data Cleaning

data_list = []
for i in df_11['new_car_detail']:
    data = dict(Ignition_Type = i['it'],
                Fuel_Type = i['ft'],
               Body_type = i['bt'],
               Kilomerters_driven = i['km'],
               Transmission_Type = i['transmission'],
               No_Of_Owners = i['ownerNo'],
               Ownership_detail = i['owner'],
               OEM = i['oem'],
               Model = i['model'],
               Manufacture_Year = i['modelYear'],
               Central_Variant_Id = i['centralVariantId'],
               Variant_Name = i['variantName'],
               Price = i['price'],
               Actual_Price = i['priceActual'],
               Saving_Price = i['priceSaving'],
               Fixed_Price = i['priceFixedText'],
               Trending_Car = i['trendingText'])
    data_list.append(data)

dd_1 = pd.DataFrame(data_list)
#print(dd_1)

df_12 =pd.DataFrame(df_1['new_car_overview'])

data_list_12 = []
for i in df_12['new_car_overview']:
    overview_dict = literal_eval(i)     # literal_eval is used here to safely evaluate the string representation of a Python literal (like a dictionary) to an actual Python object.
    
    data12 = {'Car_heading': overview_dict.get('heading', '')}

    # Iterate over the 'top' list
    for item in overview_dict.get('top', []):
        key = item.get('key', '')
        value = item.get('value', '')
        data12[key] = value

    data_list_12.append(data12)

dd_2 = pd.DataFrame(data_list_12)
#print(dd_2)

df_13 = pd.DataFrame(df_1['new_car_feature'])

# Function to extract values from the 'new_car_feature' column
def extract_features(row):
    result_dict = {
        'Features': None,
        'Comfort': None,
        'Interior': None,
        'Exterior': None,
        'Safety': None,
        'Entertainment': None,
    }
    
    try:
        # Convert the string to a dictionary
        data = json.loads(row.replace("'", "\""))
        
        # Extract values under 'value' key for each section
        if 'top' in data:
            features_values = [item['value'] for item in data['top']]
            result_dict['Features'] = ', '.join(features_values)
            
        if 'data' in data and len(data['data']) >= 5:
            comfort_values = [item['value'] for item in data['data'][0]['list']]
            interior_values = [item['value'] for item in data['data'][1]['list']]
            exterior_values = [item['value'] for item in data['data'][2]['list']]
            safety_values = [item['value'] for item in data['data'][3]['list']]
            entertainment_values = [item['value'] for item in data['data'][4]['list']]
            
            result_dict['Comfort'] = ', '.join(comfort_values)
            result_dict['Interior'] = ', '.join(interior_values)
            result_dict['Exterior'] = ', '.join(exterior_values)
            result_dict['Safety'] = ', '.join(safety_values)
            result_dict['Entertainment'] = ', '.join(entertainment_values)
            
    except Exception as e:
        # Handle potential errors during extraction
        pass

    return pd.Series(result_dict)

# Apply the extraction function to the 'new_car_feature' column for all rows
dd_3 = df_13['new_car_feature'].apply(extract_features)
#print(dd_3)

df_14 = pd.DataFrame(df_1['new_car_specs'])

data_list_14 = []

for i in df_14['new_car_specs']:
    overview_dict = literal_eval(i)
    
    data14 = {'Car_heading': overview_dict.get('heading', '')}

    # Iterate over the 'top' list
    for item in overview_dict.get('top', []):
        key = item.get('key', '')
        value = item.get('value', '')
        data14[key] = value  # Assuming you want to mark the presence of each feature with True

    # Iterate over 'data' list
    for section in overview_dict.get('data', []):
        for item in section.get('list', []):
            key = item.get('key', '')
            value = item.get('value', '')
            data14[key] = value  # Assuming you want to mark the presence of each feature with True

    data_list_14.append(data14)

# Convert the list of dictionaries to a DataFrame
dd_4 = pd.DataFrame(data_list_14)

# Display the DataFrame
# print(dd_4)

# Jaipur Cars

df_2 = pd.read_csv('/Users/arul/Documents/CarDheko/jaipur_cars.csv')

df_21 = pd.DataFrame(df_2['new_car_detail'])

df_21['new_car_detail'] = df_21['new_car_detail'].apply(lambda x: ast.literal_eval(x.strip('"')))

data_list_2 = []
for i in df_21['new_car_detail']:
    data2 = dict(Ignition_Type = i['it'],
                Fuel_Type = i['ft'],
               Body_type = i['bt'],
               Kilomerters_driven = i['km'],
               Transmission_Type = i['transmission'],
               No_Of_Owners = i['ownerNo'],
               Ownership_detail = i['owner'],
               OEM = i['oem'],
               Model = i['model'],
               Manufacture_Year = i['modelYear'],
               Central_Variant_Id = i['centralVariantId'],
               Variant_Name = i['variantName'],
               Price = i['price'],
               Actual_Price = i['priceActual'],
               Saving_Price = i['priceSaving'],
               Fixed_Price = i['priceFixedText'],
               Trending_Car = i['trendingText'])
    data_list_2.append(data)

dd_11 = pd.DataFrame(data_list_2)
# print(dd_11)

df_22 =pd.DataFrame(df_2['new_car_overview'])

data_list_22 = []
for i in df_22['new_car_overview']:
    overview_dict = literal_eval(i)
    
    data22 = {'Car_heading': overview_dict.get('heading', '')}

    # Iterate over the 'top' list
    for item in overview_dict.get('top', []):
        key = item.get('key', '')
        value = item.get('value', '')
        data22[key] = value

    data_list_22.append(data22)

dd_12 = pd.DataFrame(data_list_22)
# print(dd_12)

df_23 = pd.DataFrame(df_2['new_car_feature'])

# Function to extract values from the 'new_car_feature' column
def extract_features(row):
    result_dict = {
        'Features': None,
        'Comfort': None,
        'Interior': None,
        'Exterior': None,
        'Safety': None,
        'Entertainment': None,
    }
    
    try:
        # Convert the string to a dictionary
        data = json.loads(row.replace("'", "\""))
        
        # Extract values under 'value' key for each section
        if 'top' in data:
            features_values = [item['value'] for item in data['top']]
            result_dict['Features'] = ', '.join(features_values)
            
        if 'data' in data and len(data['data']) >= 5:
            comfort_values = [item['value'] for item in data['data'][0]['list']]
            interior_values = [item['value'] for item in data['data'][1]['list']]
            exterior_values = [item['value'] for item in data['data'][2]['list']]
            safety_values = [item['value'] for item in data['data'][3]['list']]
            entertainment_values = [item['value'] for item in data['data'][4]['list']]
            
            result_dict['Comfort'] = ', '.join(comfort_values)
            result_dict['Interior'] = ', '.join(interior_values)
            result_dict['Exterior'] = ', '.join(exterior_values)
            result_dict['Safety'] = ', '.join(safety_values)
            result_dict['Entertainment'] = ', '.join(entertainment_values)
            
    except Exception as e:
        # Handle potential errors during extraction
        pass

    return pd.Series(result_dict)

# Apply the extraction function to the 'new_car_feature' column for all rows
dd_13 = df_23['new_car_feature'].apply(extract_features)

# print(dd_13)

df_24 = pd.DataFrame(df_2['new_car_specs'])

data_list_24 = []

for i in df_24['new_car_specs']:
    overview_dict = literal_eval(i)
    
    data24 = {'Car_heading': overview_dict.get('heading', '')}

    # Iterate over the 'top' list
    for item in overview_dict.get('top', []):
        key = item.get('key', '')
        value = item.get('value', '')
        data24[key] = value  # Assuming you want to mark the presence of each feature with True

    # Iterate over 'data' list
    for section in overview_dict.get('data', []):
        for item in section.get('list', []):
            key = item.get('key', '')
            value = item.get('value', '')
            data24[key] = value  # Assuming you want to mark the presence of each feature with True

    data_list_24.append(data24)

# Convert the list of dictionaries to a DataFrame
dd_14 = pd.DataFrame(data_list_24)

# Display the DataFrame
# print(dd_14)

# Hyderabad Cars

df_3 = pd.read_csv('/Users/arul/Documents/CarDheko/hyderabad_cars.csv')

df_31 = pd.DataFrame(df_3['new_car_detail'])

df_31['new_car_detail'] = df_31['new_car_detail'].apply(lambda x: ast.literal_eval(x.strip('"')))

data_list_31 = []
for i in df_31['new_car_detail']:
    data31 = dict(Ignition_Type = i['it'],
                Fuel_Type = i['ft'],
               Body_type = i['bt'],
               Kilomerters_driven = i['km'],
               Transmission_Type = i['transmission'],
               No_Of_Owners = i['ownerNo'],
               Ownership_detail = i['owner'],
               OEM = i['oem'],
               Model = i['model'],
               Manufacture_Year = i['modelYear'],
               Central_Variant_Id = i['centralVariantId'],
               Variant_Name = i['variantName'],
               Price = i['price'],
               Actual_Price = i['priceActual'],
               Saving_Price = i['priceSaving'],
               Fixed_Price = i['priceFixedText'],
               Trending_Car = i['trendingText'])
    data_list_31.append(data)

dd_21 = pd.DataFrame(data_list_31)
# print(dd_21)

df_32 =pd.DataFrame(df_3['new_car_overview'])

data_list_32 = []
for i in df_32['new_car_overview']:
    overview_dict = literal_eval(i)
    
    data32 = {'Car_heading': overview_dict.get('heading', '')}

    # Iterate over the 'top' list
    for item in overview_dict.get('top', []):
        key = item.get('key', '')
        value = item.get('value', '')
        data32[key] = value

    data_list_32.append(data32)

dd_22 = pd.DataFrame(data_list_32)
# print(dd_22)

df_33 = pd.DataFrame(df_3['new_car_feature'])

# Function to extract values from the 'new_car_feature' column
def extract_features(row):
    result_dict = {
        'Features': None,
        'Comfort': None,
        'Interior': None,
        'Exterior': None,
        'Safety': None,
        'Entertainment': None,
    }
    
    try:
        # Convert the string to a dictionary
        data = json.loads(row.replace("'", "\""))
        
        # Extract values under 'value' key for each section
        if 'top' in data:
            features_values = [item['value'] for item in data['top']]
            result_dict['Features'] = ', '.join(features_values)
            
        if 'data' in data and len(data['data']) >= 5:
            comfort_values = [item['value'] for item in data['data'][0]['list']]
            interior_values = [item['value'] for item in data['data'][1]['list']]
            exterior_values = [item['value'] for item in data['data'][2]['list']]
            safety_values = [item['value'] for item in data['data'][3]['list']]
            entertainment_values = [item['value'] for item in data['data'][4]['list']]
            
            result_dict['Comfort'] = ', '.join(comfort_values)
            result_dict['Interior'] = ', '.join(interior_values)
            result_dict['Exterior'] = ', '.join(exterior_values)
            result_dict['Safety'] = ', '.join(safety_values)
            result_dict['Entertainment'] = ', '.join(entertainment_values)
            
    except Exception as e:
        # Handle potential errors during extraction
        pass

    return pd.Series(result_dict)

# Apply the extraction function to the 'new_car_feature' column for all rows
dd_23 = df_33['new_car_feature'].apply(extract_features)
# print(dd_23)

df_34 = pd.DataFrame(df_3['new_car_specs'])

data_list_34 = []

for i in df_34['new_car_specs']:
    overview_dict = literal_eval(i)
    
    data34 = {'Car_heading': overview_dict.get('heading', '')}

    # Iterate over the 'top' list
    for item in overview_dict.get('top', []):
        key = item.get('key', '')
        value = item.get('value', '')
        data34[key] = value  # Assuming you want to mark the presence of each feature with True

    # Iterate over 'data' list
    for section in overview_dict.get('data', []):
        for item in section.get('list', []):
            key = item.get('key', '')
            value = item.get('value', '')
            data34[key] = value  # Assuming you want to mark the presence of each feature with True

    data_list_34.append(data34)

# Convert the list of dictionaries to a DataFrame
dd_24 = pd.DataFrame(data_list_34)

# Display the DataFrame
# print(dd_24)

# Delhi Cars

df_4 = pd.read_csv('/Users/arul/Documents/CarDheko/delhi_cars.csv')

df_41 = pd.DataFrame(df_4['new_car_detail'])

df_41['new_car_detail'] = df_41['new_car_detail'].apply(lambda x: ast.literal_eval(x.strip('"')))

data_list_41 = []
for i in df_41['new_car_detail']:
    data41 = dict(Ignition_Type = i['it'],
                Fuel_Type = i['ft'],
               Body_type = i['bt'],
               Kilomerters_driven = i['km'],
               Transmission_Type = i['transmission'],
               No_Of_Owners = i['ownerNo'],
               Ownership_detail = i['owner'],
               OEM = i['oem'],
               Model = i['model'],
               Manufacture_Year = i['modelYear'],
               Central_Variant_Id = i['centralVariantId'],
               Variant_Name = i['variantName'],
               Price = i['price'],
               Actual_Price = i['priceActual'],
               Saving_Price = i['priceSaving'],
               Fixed_Price = i['priceFixedText'],
               Trending_Car = i['trendingText'])
    data_list_41.append(data)

dd_31 = pd.DataFrame(data_list_41)
# print(dd_31)

df_42 =pd.DataFrame(df_4['new_car_overview'])

data_list_42 = []
for i in df_42['new_car_overview']:
    overview_dict = literal_eval(i)
    
    data42 = {'Car_heading': overview_dict.get('heading', '')}

    # Iterate over the 'top' list
    for item in overview_dict.get('top', []):
        key = item.get('key', '')
        value = item.get('value', '')
        data42[key] = value

    data_list_42.append(data42)

dd_32 = pd.DataFrame(data_list_42)
# print(dd_32)

df_43 = pd.DataFrame(df_4['new_car_feature'])

# Function to extract values from the 'new_car_feature' column
def extract_features(row):
    result_dict = {
        'Features': None,
        'Comfort': None,
        'Interior': None,
        'Exterior': None,
        'Safety': None,
        'Entertainment': None,
    }
    
    try:
        # Convert the string to a dictionary
        data = json.loads(row.replace("'", "\""))
        
        # Extract values under 'value' key for each section
        if 'top' in data:
            features_values = [item['value'] for item in data['top']]
            result_dict['Features'] = ', '.join(features_values)
            
        if 'data' in data and len(data['data']) >= 5:
            comfort_values = [item['value'] for item in data['data'][0]['list']]
            interior_values = [item['value'] for item in data['data'][1]['list']]
            exterior_values = [item['value'] for item in data['data'][2]['list']]
            safety_values = [item['value'] for item in data['data'][3]['list']]
            entertainment_values = [item['value'] for item in data['data'][4]['list']]
            
            result_dict['Comfort'] = ', '.join(comfort_values)
            result_dict['Interior'] = ', '.join(interior_values)
            result_dict['Exterior'] = ', '.join(exterior_values)
            result_dict['Safety'] = ', '.join(safety_values)
            result_dict['Entertainment'] = ', '.join(entertainment_values)
            
    except Exception as e:
        # Handle potential errors during extraction
        pass

    return pd.Series(result_dict)

# Apply the extraction function to the 'new_car_feature' column for all rows
dd_33 = df_43['new_car_feature'].apply(extract_features)
# print(dd_33)

df_44 = pd.DataFrame(df_4['new_car_specs'])

data_list_44 = []

for i in df_44['new_car_specs']:
    overview_dict = literal_eval(i)
    
    data44 = {'Car_heading': overview_dict.get('heading', '')}

    # Iterate over the 'top' list
    for item in overview_dict.get('top', []):
        key = item.get('key', '')
        value = item.get('value', '')
        data44[key] = value  # Assuming you want to mark the presence of each feature with True

    # Iterate over 'data' list
    for section in overview_dict.get('data', []):
        for item in section.get('list', []):
            key = item.get('key', '')
            value = item.get('value', '')
            data44[key] = value  # Assuming you want to mark the presence of each feature with True

    data_list_44.append(data44)

# Convert the list of dictionaries to a DataFrame
dd_34 = pd.DataFrame(data_list_44)

# Display the DataFrame
# print(dd_34)

# Chennai Cars

df_5 = pd.read_csv('/Users/arul/Documents/CarDheko/chennai_cars.csv')

df_51 = pd.DataFrame(df_5['new_car_detail'])

df_51['new_car_detail'] = df_51['new_car_detail'].apply(lambda x: ast.literal_eval(x.strip('"')))

data_list_51 = []
for i in df_51['new_car_detail']:
    data51 = dict(Ignition_Type = i['it'],
                Fuel_Type = i['ft'],
               Body_type = i['bt'],
               Kilomerters_driven = i['km'],
               Transmission_Type = i['transmission'],
               No_Of_Owners = i['ownerNo'],
               Ownership_detail = i['owner'],
               OEM = i['oem'],
               Model = i['model'],
               Manufacture_Year = i['modelYear'],
               Central_Variant_Id = i['centralVariantId'],
               Variant_Name = i['variantName'],
               Price = i['price'],
               Actual_Price = i['priceActual'],
               Saving_Price = i['priceSaving'],
               Fixed_Price = i['priceFixedText'],
               Trending_Car = i['trendingText'])
    data_list_51.append(data51)

dd_41 = pd.DataFrame(data_list_51)
# print(dd_41)

df_52 =pd.DataFrame(df_5['new_car_overview'])

data_list_52 = []
for i in df_52['new_car_overview']:
    overview_dict = literal_eval(i)
    
    data52 = {'Car_heading': overview_dict.get('heading', '')}

    # Iterate over the 'top' list
    for item in overview_dict.get('top', []):
        key = item.get('key', '')
        value = item.get('value', '')
        data52[key] = value

    data_list_52.append(data52)

dd_42 = pd.DataFrame(data_list_52)
# print(dd_42)

df_53 = pd.DataFrame(df_5['new_car_feature'])

# Function to extract values from the 'new_car_feature' column
def extract_features(row):
    result_dict = {
        'Features': None,
        'Comfort': None,
        'Interior': None,
        'Exterior': None,
        'Safety': None,
        'Entertainment': None,
    }
    
    try:
        # Convert the string to a dictionary
        data = json.loads(row.replace("'", "\""))
        
        # Extract values under 'value' key for each section
        if 'top' in data:
            features_values = [item['value'] for item in data['top']]
            result_dict['Features'] = ', '.join(features_values)
            
        if 'data' in data and len(data['data']) >= 5:
            comfort_values = [item['value'] for item in data['data'][0]['list']]
            interior_values = [item['value'] for item in data['data'][1]['list']]
            exterior_values = [item['value'] for item in data['data'][2]['list']]
            safety_values = [item['value'] for item in data['data'][3]['list']]
            entertainment_values = [item['value'] for item in data['data'][4]['list']]
            
            result_dict['Comfort'] = ', '.join(comfort_values)
            result_dict['Interior'] = ', '.join(interior_values)
            result_dict['Exterior'] = ', '.join(exterior_values)
            result_dict['Safety'] = ', '.join(safety_values)
            result_dict['Entertainment'] = ', '.join(entertainment_values)
            
    except Exception as e:
        # Handle potential errors during extraction
        pass

    return pd.Series(result_dict)

# Apply the extraction function to the 'new_car_feature' column for all rows
dd_43 = df_53['new_car_feature'].apply(extract_features)
# print(dd_43)

df_54 = pd.DataFrame(df_5['new_car_specs'])

data_list_54 = []

for i in df_54['new_car_specs']:
    overview_dict = literal_eval(i)
    
    data54 = {'Car_heading': overview_dict.get('heading', '')}

    # Iterate over the 'top' list
    for item in overview_dict.get('top', []):
        key = item.get('key', '')
        value = item.get('value', '')
        data54[key] = value  # Assuming you want to mark the presence of each feature with True

    # Iterate over 'data' list
    for section in overview_dict.get('data', []):
        for item in section.get('list', []):
            key = item.get('key', '')
            value = item.get('value', '')
            data54[key] = value  # Assuming you want to mark the presence of each feature with True

    data_list_54.append(data54)

# Convert the list of dictionaries to a DataFrame
dd_44 = pd.DataFrame(data_list_54)

# Display the DataFrame
# print(dd_44)

# Bangalore Cars

df_6 = pd.read_csv('/Users/arul/Documents/CarDheko/bangalore_cars.csv')

df_61 = pd.DataFrame(df_6['new_car_detail'])

df_61['new_car_detail'] = df_61['new_car_detail'].apply(lambda x: ast.literal_eval(x.strip('"')))

data_list_61 = []
for i in df_61['new_car_detail']:
    data61 = dict(Ignition_Type = i['it'],
                Fuel_Type = i['ft'],
               Body_type = i['bt'],
               Kilomerters_driven = i['km'],
               Transmission_Type = i['transmission'],
               No_Of_Owners = i['ownerNo'],
               Ownership_detail = i['owner'],
               OEM = i['oem'],
               Model = i['model'],
               Manufacture_Year = i['modelYear'],
               Central_Variant_Id = i['centralVariantId'],
               Variant_Name = i['variantName'],
               Price = i['price'],
               Actual_Price = i['priceActual'],
               Saving_Price = i['priceSaving'],
               Fixed_Price = i['priceFixedText'],
               Trending_Car = i['trendingText'])
    data_list_61.append(data61)

dd_51 = pd.DataFrame(data_list_61)
# print(dd_51)

df_62 =pd.DataFrame(df_6['new_car_overview'])

data_list_62 = []
for i in df_62['new_car_overview']:
    overview_dict = literal_eval(i)
    
    data62 = {'Car_heading': overview_dict.get('heading', '')}

    # Iterate over the 'top' list
    for item in overview_dict.get('top', []):
        key = item.get('key', '')
        value = item.get('value', '')
        data62[key] = value

    data_list_62.append(data62)

dd_52 = pd.DataFrame(data_list_62)
# print(dd_52)

df_63 = pd.DataFrame(df_6['new_car_feature'])

# Function to extract values from the 'new_car_feature' column
def extract_features(row):
    result_dict = {
        'Features': None,
        'Comfort': None,
        'Interior': None,
        'Exterior': None,
        'Safety': None,
        'Entertainment': None,
    }
    
    try:
        # Convert the string to a dictionary
        data = json.loads(row.replace("'", "\""))
        
        # Extract values under 'value' key for each section
        if 'top' in data:
            features_values = [item['value'] for item in data['top']]
            result_dict['Features'] = ', '.join(features_values)
            
        if 'data' in data and len(data['data']) >= 5:
            comfort_values = [item['value'] for item in data['data'][0]['list']]
            interior_values = [item['value'] for item in data['data'][1]['list']]
            exterior_values = [item['value'] for item in data['data'][2]['list']]
            safety_values = [item['value'] for item in data['data'][3]['list']]
            entertainment_values = [item['value'] for item in data['data'][4]['list']]
            
            result_dict['Comfort'] = ', '.join(comfort_values)
            result_dict['Interior'] = ', '.join(interior_values)
            result_dict['Exterior'] = ', '.join(exterior_values)
            result_dict['Safety'] = ', '.join(safety_values)
            result_dict['Entertainment'] = ', '.join(entertainment_values)
            
    except Exception as e:
        # Handle potential errors during extraction
        pass

    return pd.Series(result_dict)

# Apply the extraction function to the 'new_car_feature' column for all rows
dd_53 = df_63['new_car_feature'].apply(extract_features)
# print(dd_53)

df_64 = pd.DataFrame(df_6['new_car_specs'])

data_list_64 = []

for i in df_64['new_car_specs']:
    overview_dict = literal_eval(i)
    
    data64 = {'Car_heading': overview_dict.get('heading', '')}

    # Iterate over the 'top' list
    for item in overview_dict.get('top', []):
        key = item.get('key', '')
        value = item.get('value', '')
        data64[key] = value  # Assuming you want to mark the presence of each feature with True

    # Iterate over 'data' list
    for section in overview_dict.get('data', []):
        for item in section.get('list', []):
            key = item.get('key', '')
            value = item.get('value', '')
            data64[key] = value  # Assuming you want to mark the presence of each feature with True

    data_list_64.append(data64)

# Convert the list of dictionaries to a DataFrame
dd_54 = pd.DataFrame(data_list_64)

# Display the DataFrame
# print(dd_54)

# Concate Datas

# new_car_details

dfs_1 = [dd_1, dd_11, dd_21, dd_31, dd_41, dd_51]
d_f_1 = pd.concat(dfs_1, ignore_index=True)
# print(d_f_1)

# new_car_overview

dfs_2 = [dd_2, dd_12, dd_22, dd_32, dd_42, dd_52]
d_f_2 = pd.concat(dfs_2, ignore_index=True)
# print(d_f_2)

# new_car_feature

dfs_3 = [dd_3, dd_13, dd_23, dd_33, dd_43, dd_53]
d_f_3 = pd.concat(dfs_3, ignore_index=True)
# print(d_f_3)

# new_car_specs

dfs_4 = [dd_4, dd_14, dd_24, dd_34, dd_44, dd_54]
d_f_4 = pd.concat(dfs_4, ignore_index=True)
# print(d_f_4)

# Overall Dataframe

result = pd.concat([d_f_1, d_f_2, d_f_3, d_f_4], axis=1)
# print(result)

current_year = 2024
result['car_age'] = current_year - result['Manufacture_Year']

rto_mapping = {
    'WB02': 'West Bengal - Kolkata',
    'WB24': 'West Bengal - Asansol',
    'WB06': 'West Bengal - Howrah',
    'WB18': 'West Bengal - Baharampur',
    'WB22': 'West Bengal - Jalpaiguri',
    'WB08': 'West Bengal - Siliguri',
    'WB20': 'West Bengal - Malda',
    'WB12': 'West Bengal - Kharagpur',
    'WB10': 'West Bengal - Burdwan',
    'WB96': 'West Bengal - RTO Barrackpore',
    'WB92': 'West Bengal - Alipore',
    'WB26': 'West Bengal - Haldia',
    'TN01': 'Tamil Nadu - Chennai Central',
    'UP33': 'Uttar Pradesh - Ghaziabad',
    'AR01': 'Arunachal Pradesh - Itanagar',
    'WB36': 'West Bengal - Krishnanagar',
    'WB42': 'West Bengal - Arambagh',
    'WB16': 'West Bengal - Raiganj',
    'WB40': 'West Bengal - Balurghat',
    'WB34': 'West Bengal - Cooch Behar',
    'WB74': 'West Bengal - Serampore',
    'WB98': 'West Bengal - RTO Diamond Harbour',
    'UP81': 'Uttar Pradesh - Noida',
    'OD33': 'Odisha - Cuttack',
    'WB44': 'West Bengal - Contai',
    'WB52': 'West Bengal - Rampurhat',
    'AS05': 'Assam - Tezpur',
    'Wb24': 'West Bengal - Raipur (Bankura)',
    'MP09': 'Madhya Pradesh - Rewa',
    'Wb10': 'West Bengal - Medinipur',
    'WB14': 'West Bengal - Berhampore',
    'WB32': 'West Bengal - Bankura',
    'WB50': 'West Bengal - Tamluk',
    'OD02': 'Odisha - Bhubaneswar',
    'WB94': 'West Bengal - Taki',
    'Wb14': 'West Bengal - Kalyani',
    'Wb26': 'West Bengal - RTO Ranaghat',
    'GJ05': 'Gujarat - Ahmedabad',
    'WB38': 'West Bengal - Bishnupur',
    'Wb08': 'West Bengal - Midnapore',
    'Wb06': 'West Bengal - RTO Burdwan',
    'HP47': 'Himachal Pradesh - Kangra',
    'KA03': 'Karnataka - Bangalore Central',
    'Wb02': 'West Bengal - Alipurduar',
    'Wb12': 'West Bengal - RTO Tamluk',
    'Wb52': 'West Bengal - RTO Alipurduar',
    'Wb16': 'West Bengal - RTO Durgapur',
    'Wb34': 'West Bengal - RTO Raghunathpur',
    'WB90': 'West Bengal - RTO Jhargram',
    'JH05': 'Jharkhand - Dhanbad',
    'UP16': 'Uttar Pradesh - Bareilly',
    'WB30': 'West Bengal - RTO Barasat',
    'MH04': 'Maharashtra - Mumbai West',
    'MH14': 'Maharashtra - Beed',
    'JH01': 'Jharkhand - Ranchi',
    'WW06': 'Unknown',
    'CG04': 'Chhattisgarh - Durg',
    'AN01': 'Andaman and Nicobar Islands - Port Blair',
    'KA04': 'Karnataka - Bangalore West',
    'WB68': 'West Bengal - RTO Salt Lake',
    'WB04': 'West Bengal - RTO Beldanga',
    'WB84': 'West Bengal - RTO Ghatal',
    'GJ15': 'Gujarat - Porbandar',
    'WB60': 'West Bengal - RTO Howrah',
    'HR26': 'Haryana - Gurgaon',
    'WB66': 'West Bengal - RTO Behala',
    'JH10': 'Jharkhand - Jamshedpur',
    'WB05': 'West Bengal - RTO Siliguri',
    'WB03': 'West Bengal - RTO Krishnagar',
    'BR01': 'Bihar - Patna',
    'KA41': 'Karnataka - Bangalore East',
    'DL2C': 'Delhi - New Delhi',
    'WB58': 'West Bengal - RTO Uluberia',
    'WB64': 'West Bengal - RTO Raniganj',
    'AS04': 'Assam - Golaghat',
    'WB72': 'West Bengal - RTO Tarakeswar',
    'PB10': 'Punjab - Amritsar',
    'RJ13': 'Rajasthan - Alwar',
    'RJ14': 'Rajasthan - Banswara',
    'RJ32': 'Rajasthan - Pratapgarh',
    'DL02': 'Delhi - New Delhi',
    'RJ45': 'Rajasthan - Dungarpur',
    'HP38': 'Himachal Pradesh - Mandi',
    'DL12': 'Delhi - New Delhi',
    'RJ19': 'Rajasthan - Dholpur',
    'RJ01': 'Rajasthan - Ajmer',
    'RJ27': 'Rajasthan - Jhunjhunu',
    'RJ02': 'Rajasthan - Bharatpur',
    'RJ37': 'Rajasthan - RTO Bikaner',
    'HR30': 'Haryana - Faridabad',
    '22BH': 'Unknown',
    'HR03': 'Haryana - Faridabad',
    'RJ05': 'Rajasthan - Baran',
    'RJ22': 'Rajasthan - Hanumangarh',
    'DL8C': 'Delhi - New Delhi',
    'RJ23': 'Rajasthan - Pali',
    'RJ18': 'Rajasthan - Chittorgarh',
    'RJ42': 'Rajasthan - RTO Churu',
    'RJ48': 'Rajasthan - RTO Dausa',
    'RJ10': 'Rajasthan - RTO Bharatpur',
    'RJ51': 'Rajasthan - RTO Ganganagar',
    'RJ40': 'Rajasthan - RTO Chittorgarh',
    'HE26': 'Haryana - Ambala',
    'RJ20': 'Rajasthan - RTO Kota',
    'RJ16': 'Rajasthan - Bundi',
    'RR23': 'Rajasthan - RTO Sirohi',
    'RJ07': 'Rajasthan - Bikaner',
    'RJ21': 'Rajasthan - Bhilwara',
    'UP14': 'Uttar Pradesh - Unnao',
    'MP17': 'Madhya Pradesh - Sehore',
    'UK06': 'Uttarakhand - Dehradun',
    'RJ29': 'Rajasthan - RTO Jalore',
    'RJ41': 'Rajasthan - RTO Jaipur South',
    'RJ36': 'Rajasthan - RTO Jhunjhunu',
    'TN07': 'Tamil Nadu - Chennai North West',
    'RJ11': 'Rajasthan - Bilara',
    'RJ44': 'Rajasthan - RTO Kishangarh',
    'RJ06': 'Rajasthan - RTO Barmer',
    'RJ26': 'Rajasthan - Bundi',
    'RH14': 'Unknown',
    'RJ25': 'Rajasthan - RTO Tonk',
    'PB22': 'Punjab - Patiala',
    'RJ31': 'Rajasthan - RTO Pali',
    'GJ10': 'Gujarat - Gandhinagar',
    'RJ34': 'Rajasthan - RTO Dungarpur',
    'RJ28': 'Rajasthan - RTO Jhalawar',
    'HR36': 'Haryana - Rewari',
    'MP04': 'Madhya Pradesh - Dhar',
    'RJ39': 'Rajasthan - RTO Udaipur',
    'DL3C': 'Delhi - New Delhi',
    'HR13': 'Haryana - Palwal',
    'MH02': 'Maharashtra - Mumbai',
    'RJ09': 'Rajasthan - RTO Ajmer',
    'HR51': 'Haryana - Ambala',
    'RJ04': 'Rajasthan - Banswara',
    'DL10': 'Delhi - New Delhi',
    'DL4C': 'Delhi - New Delhi',
    'KA53': 'Karnataka - Bangalore East',
    'RJ24': 'Rajasthan - RTO Chittorgarh',
    'RJ53': 'Rajasthan - RTO Chittorgarh',
    'PB65': 'Punjab - Tarn Taran',
    'RJ43': 'Rajasthan - RTO Jaipur North',
    'RJ12': 'Rajasthan - Barmer',
    'RJ47': 'Rajasthan - RTO Chittorgarh',
    'MP33': 'Madhya Pradesh - Ratlam',
    'DL9C': 'Delhi - New Delhi',
    'MH49': 'Maharashtra - Nashik',
    'UP32': 'Uttar Pradesh - Lucknow',
    'DL1C': 'Delhi - New Delhi',
    'HR27': 'Haryana - Jind',
    'RJ30': 'Rajasthan - RTO Bharatpur',
    'RJ52': 'Rajasthan - RTO Alwar',
    'UP85': 'Uttar Pradesh - Azamgarh',
    'GJ14': 'Gujarat - Mehsana',
    'R14C': 'Rajasthan - RTO Bikaner',
    'DL5C': 'Delhi - New Delhi',
    'RJ4G': 'Rajasthan - RTO Gangapur City',
    'UP72': 'Uttar Pradesh - Bijnor',
    'CH01': 'Chandigarh - Chandigarh',
    'HR43': 'Haryana - Jhajjar',
    'MP13': 'Madhya Pradesh - Khandwa',
    'BR3F': 'Bihar - Muzaffarpur',
    'RJ35': 'Rajasthan - RTO Baran',
    'MH18': 'Maharashtra - Solapur',
    'UP64': 'Uttar Pradesh - Saharanpur',
    'UP82': 'Uttar Pradesh - Kannauj',
    'TS04': 'Telangana - Hyderabad Central',
    'TS09': 'Telangana - Hyderabad North',
    'TS07': 'Telangana - Hyderabad South East',
    'TS08': 'Telangana - Hyderabad West',
    'TS32': 'Telangana - RTO Khairatabad',
    'AP10': 'Andhra Pradesh - Anantapur',
    'TS15': 'Telangana - RTO Bahadurpura',
    'TS22': 'Telangana - RTO Kukatpally',
    'TS13': 'Telangana - RTO Uppal',
    'AP39': 'Andhra Pradesh - Guntur',
    'TS11': 'Telangana - RTO Himayatnagar',
    'TS28': 'Telangana - RTO Kondapur',
    'TS30': 'Telangana - RTO Medchal',
    'TS10': 'Telangana - RTO Mehdipatnam',
    'TS27': 'Telangana - RTO Malakpet',
    'TS16': 'Telangana - RTO Vanasthalipuram',
    'TS17': 'Telangana - RTO L.B. Nagar',
    'TS05': 'Telangana - RTO Nampally',
    'TS23': 'Telangana - RTO Ranga Reddy',
    'TS06': 'Telangana - RTO Musheerabad',
    'AP11': 'Andhra Pradesh - Chittoor',
    'AP26': 'Andhra Pradesh - Nellore',
    'TS12': 'Telangana - RTO S.D. Road',
    'TS35': 'Telangana - RTO Shamshabad',
    'AP04': 'Andhra Pradesh - Kurnool',
    'TS33': 'Telangana - RTO Secunderabad',
    'TS03': 'Telangana - RTO Lakdikapul',
    'TS25': 'Telangana - RTO Mallepally',
    'AP13': 'Andhra Pradesh - Prakasam',
    'GJ06': 'Gujarat - Gandhinagar',
    'AP22': 'Andhra Pradesh - Guntakal',
    'AP28': 'Andhra Pradesh - Kadapa',
    'TS31': 'Telangana - RTO Nalgonda',
    'AP07': 'Andhra Pradesh - Eluru',
    'AP16': 'Andhra Pradesh - Nizamabad',
    'TS29': 'Telangana - RTO Rangareddy',
    'AP09': 'Andhra Pradesh - Anakapalle',
    'TS02': 'Telangana - RTO Hyderabad Central',
    'AP29': 'Andhra Pradesh - Kakinada',
    'TS19': 'Telangana - RTO Sircilla',
    'AP20': 'Andhra Pradesh - Vijayawada',
    'TS34': 'Telangana - RTO Wanaparthy',
    'AP36': 'Andhra Pradesh - Rajahmundry',
    'TS36': 'Telangana - RTO Vikarabad',
    'TS24': 'Telangana - RTO Nagarkurnool',
    'AP31': 'Andhra Pradesh - Ongole',
    'AP23': 'Andhra Pradesh - Visakhapatnam',
    'TS01': 'Telangana - RTO Adilabad',
    'AP21': 'Andhra Pradesh - Srikakulam',
    'AP05': 'Andhra Pradesh - Kakinada',
    'GA03': 'Goa - Panaji',
    'TA13': 'Tamil Nadu - Kumbakonam',
    'AP24': 'Andhra Pradesh - Vizianagaram',
    'AP12': 'Andhra Pradesh - Chirala',
    'AP27': 'Andhra Pradesh - Srikakulam',
    'TS21': 'Telangana - RTO Nizamabad',
    'TT07': 'Tiruchirappalli - Trichy',
    'KK04': 'Kerala - Kollam',
    'TA10': 'Tamil Nadu - Madurai South',
    'AP25': 'Andhra Pradesh - Rajahmundry',
    'TA07': 'Tamil Nadu - Erode',
    'AP02': 'Andhra Pradesh - Srikakulam',
    'PY05': 'Puducherry - Puducherry',
    'AP01': 'Andhra Pradesh - Srikakulam',
    'AA28': 'Arunachal Pradesh - Itanagar',
    'MH46': 'Maharashtra - Nagpur',
    'TN41': 'Tamil Nadu - Chennai West',
    'PB01': 'Punjab - Amritsar',
    'GJ04': 'Gujarat - Gandhinagar',
    'UP78': 'Uttar Pradesh - Varanasi',
    'TS26': 'Telangana - RTO Medak',
    'TT02': 'Tiruchirappalli - Trichy',
    'MH05': 'Maharashtra - Mumbai East',
    'TS18': 'Telangana - RTO Khammam',
    'TN39': 'Tamil Nadu - Chennai South East',
    'TD07': 'Telangana - RTO Adilabad',
    'AP37': 'Andhra Pradesh - Vijayawada',
    'DL01': 'Delhi - New Delhi',
    'DL9X': 'Delhi - New Delhi',
    'DL04': 'Delhi - New Delhi',
    'Up15': 'Uttar Pradesh - Aligarh',
    'DL03': 'Delhi - New Delhi',
    'UP21': 'Uttar Pradesh - Bareilly',
    'DL09': 'Delhi - New Delhi',
    'UK07': 'Uttarakhand - Dehradun',
    'UP80': 'Uttar Pradesh - Azamgarh',
    'HR72': 'Haryana - Ambala',
    'HR98': 'Haryana - Ambala',
    'DL26': 'Delhi - New Delhi',
    'PH65': 'Unknown',
    'Hr22': 'Haryana - Ambala',
    'UP15': 'Uttar Pradesh - Aligarh',
    'DD4L': 'Unknown',
    'DL14': 'Delhi - New Delhi',
    'GJ01': 'Gujarat - Ahmedabad',
    'DL6C': 'Delhi - New Delhi',
    'UP37': 'Uttar Pradesh - Ghaziabad',
    'DL7C': 'Delhi - New Delhi',
    'HR89': 'Haryana - Ambala',
    'HR29': 'Haryana - Ambala',
    'HR12': 'Haryana - Ambala',
    'UK14': 'Uttarakhand - Dehradun',
    'KA01': 'Karnataka - Bangalore Central',
    'HP01': 'Himachal Pradesh - Shimla',
    'UP01': 'Uttar Pradesh - Aligarh',
    'HR06': 'Haryana - Ambala',
    'HR01': 'Haryana - Ambala',
    'MH01': 'Maharashtra - Mumbai',
    'HR87': 'Haryana - Ambala',
    'DL07': 'Delhi - New Delhi',
    'DL13': 'Delhi - New Delhi',
    'DL05': 'Delhi - New Delhi',
    'UK18': 'Uttarakhand - Dehradun',
    'DL08': 'Delhi - New Delhi',
    'HP07': 'Himachal Pradesh - Kangra',
    'HR70': 'Haryana - Ambala',
    'UP10': 'Uttar Pradesh - Bareilly',
    'HR05': 'Haryana - Ambala',
    'UP12': 'Uttar Pradesh - Bareilly',
    'DL1Z': 'Delhi - New Delhi',
    'HR16': 'Haryana - Ambala',
    'DD8L': 'Unknown',
    'MP30': 'Madhya Pradesh - Dhar',
    'HP12': 'Himachal Pradesh - Kangra',
    'DL86': 'Delhi - New Delhi',
    'HP34': 'Himachal Pradesh - Solan',
    'HP52': 'Himachal Pradesh - Kangra',
    'UP13': 'Uttar Pradesh - Bareilly',
    'DL11': 'Delhi - New Delhi',
    'PB70': 'Punjab - Bathinda',
    'DL3F': 'Delhi - New Delhi',
    'HR90': 'Haryana - Ambala',
    'HR52': 'Haryana - Ambala',
    'HP17': 'Himachal Pradesh - Kangra',
    'HP62': 'Himachal Pradesh - Kangra',
    'UK17': 'Uttarakhand - Dehradun',
    'HR19': 'Haryana - Ambala',
    'PB08': 'Punjab - Batala',
    'HP05': 'Himachal Pradesh - Kangra',
    'HP93': 'Himachal Pradesh - Solan',
    'UP25': 'Uttar Pradesh - Ghaziabad',
    'AS01': 'Assam - Guwahati',
    'UP65': 'Uttar Pradesh - Rae Bareli',
    'DL1V': 'Delhi - New Delhi',
    'DL5P': 'Delhi - New Delhi',
    'Dl12': 'Delhi - New Delhi',
    'DL1U': 'Delhi - New Delhi',
    'HR08': 'Haryana - Ambala',
    'DL2B': 'Delhi - New Delhi',
    'DL9A': 'Delhi - New Delhi',
    'HP15': 'Himachal Pradesh - Kangra',
    'KA05': 'Karnataka - Bangalore South',
    'HR81': 'Haryana - Ambala',
    'UP53': 'Uttar Pradesh - Hapur',
    'MP07': 'Madhya Pradesh - Guna',
    'UK08': 'Uttarakhand - Dehradun',
    'HR10': 'Haryana - Ambala',
    'DL16': 'Delhi - New Delhi',
    'PB03': 'Punjab - Amritsar',
    'HR24': 'Haryana - Ambala',
    'HR78': 'Haryana - Ambala',
    'UK15': 'Uttarakhand - Dehradun',
    'HR35': 'Haryana - Ambala',
    'HR14': 'Haryana - Ambala',
    'HR42': 'Haryana - Ambala',
    'HP95': 'Himachal Pradesh - Kangra',
    'HR55': 'Haryana - Ambala',
    'UP11': 'Uttar Pradesh - Bareilly',
    'UP17': 'Uttar Pradesh - Bareilly',
    'DL8A': 'Delhi - New Delhi',
    'PB87': 'Punjab - Gurdaspur',
    'DD06': 'Unknown',
    'TN02': 'Tamil Nadu - Chennai Central',
    'TN04': 'Tamil Nadu - Chennai East',
    'TN22': 'Tamil Nadu - Salem',
    'TN11': 'Tamil Nadu - Chennai South',
    'TN06': 'Tamil Nadu - Chennai North',
    'TN09': 'Tamil Nadu - Trichy',
    'TN05': 'Tamil Nadu - Chennai South',
    'TN19': 'Tamil Nadu - Tiruvallur',
    'TN24': 'Tamil Nadu - Chennai West',
    'TN18': 'Tamil Nadu - Tirunelveli',
    'TN03': 'Tamil Nadu - Chennai North',
    'TN14': 'Tamil Nadu - Madurai North',
    'TN12': 'Tamil Nadu - Chennai East',
    'TN10': 'Tamil Nadu - Madurai South',
    'TN20': 'Tamil Nadu - Vellore',
    'TN23': 'Tamil Nadu - Krishnagiri',
    'TN08': 'Tamil Nadu - Chennai West',
    'TN16': 'Tamil Nadu - Karur',
    'TN15': 'Tamil Nadu - Namakkal',
    'TN13': 'Tamil Nadu - Madurai South',
    'TN01': 'Tamil Nadu - Chennai Central',
    'TN25': 'Tamil Nadu - Dharmapuri',
    'TN07': 'Tamil Nadu - Chennai North West',
    'TN17': 'Tamil Nadu - Tirupur',
    'TN21': 'Tamil Nadu - Vellore',
    'TN02': 'Tamil Nadu - Chennai North',
    'TN26': 'Tamil Nadu - Dindigul',
    'TN29': 'Tamil Nadu - Tuticorin',
    'TN28': 'Tamil Nadu - Nagercoil',
    'TN30': 'Tamil Nadu - Kancheepuram',
    'TN31': 'Tamil Nadu - Villupuram',
    'TN32': 'Tamil Nadu - Cuddalore',
    'TN34': 'Tamil Nadu - Pudukkottai',
    'TN33': 'Tamil Nadu - Thanjavur',
    'TN35': 'Tamil Nadu - Ramanathapuram',
    'TN36': 'Tamil Nadu - Sivaganga',
    'TN37': 'Tamil Nadu - Thoothukudi',
    'TN38': 'Tamil Nadu - Tenkasi',
    'TN39': 'Tamil Nadu - Chengalpattu',
    'TN40': 'Tamil Nadu - Ranipet',
    'TN41': 'Tamil Nadu - Tirupattur',
    'TN42': 'Tamil Nadu - Vaniyambadi',
    'TN43': 'Tamil Nadu - Viluppuram',
    'TN44': 'Tamil Nadu - Cuddalore',
    'TN45': 'Tamil Nadu - Nagapattinam',
    'TN46': 'Tamil Nadu - Tiruvarur',
    'TN47': 'Tamil Nadu - Perambalur',
    'TN48': 'Tamil Nadu - Ariyalur',
    'TN49': 'Tamil Nadu - Kallakurichi',
    'TN50': 'Tamil Nadu - Tiruvannamalai',
    'TN51': 'Tamil Nadu - Kanchipuram',
    'TN52': 'Tamil Nadu - Tirunelveli',
    'TN53': 'Tamil Nadu - Tenkasi',
    'TN54': 'Tamil Nadu - Tuticorin',
    'TN55': 'Tamil Nadu - Ramanathapuram',
    'TN56': 'Tamil Nadu - Sivaganga',
    'TN57': 'Tamil Nadu - Pudukkottai',
    'TN58': 'Tamil Nadu - Dindigul',
    'TN59': 'Tamil Nadu - Madurai',
    'TN60': 'Tamil Nadu - Theni',
    'TN61': 'Tamil Nadu - Virudhunagar',
    'TN62': 'Tamil Nadu - Ramanathapuram',
    'TN63': 'Tamil Nadu - Sivaganga',
    'TN64': 'Tamil Nadu - Madurai',
    'TN65': 'Tamil Nadu - Tirunelveli',
    'TN66': 'Tamil Nadu - Tenkasi',
    'TN67': 'Tamil Nadu - Thoothukudi',
    'TN68': 'Tamil Nadu - Kanyakumari',
    'TN69': 'Tamil Nadu - Theni',
    'TN70': 'Tamil Nadu - Dindigul',
    'TN71': 'Tamil Nadu - Virudhunagar',
    'TN72': 'Tamil Nadu - Ramanathapuram',
    'TN73': 'Tamil Nadu - Sivaganga',
    'TN74': 'Tamil Nadu - Madurai',
    'TN75': 'Tamil Nadu - Tirunelveli',
    'TN76': 'Tamil Nadu - Tenkasi',
    'TN77': 'Tamil Nadu - Thoothukudi',
    'TN78': 'Tamil Nadu - Kanyakumari',
    'TN79': 'Tamil Nadu - Theni',
    'TN80': 'Tamil Nadu - Dindigul',
    'TN81': 'Tamil Nadu - Virudhunagar',
    'TN82': 'Tamil Nadu - Ramanathapuram',
    'TN83': 'Tamil Nadu - Sivaganga',
    'TN84': 'Tamil Nadu - Madurai',
    'TN85': 'Tamil Nadu - Tirunelveli',
    'TN86': 'Tamil Nadu - Tenkasi',
    'TN87': 'Tamil Nadu - Thoothukudi',
    'TN88': 'Tamil Nadu - Kanyakumari',
    'TN89': 'Tamil Nadu - Theni',
    'TN90': 'Tamil Nadu - Dindigul',
    'TN91': 'Tamil Nadu - Virudhunagar',
    'TN92': 'Tamil Nadu - Ramanathapuram',
    'TN93': 'Tamil Nadu - Sivaganga',
    'TN94': 'Tamil Nadu - Madurai',
    'TN95': 'Tamil Nadu - Tirunelveli',
    'TN96': 'Tamil Nadu - Tenkasi',
    'TN97': 'Tamil Nadu - Thoothukudi',
    'TN98': 'Tamil Nadu - Kanyakumari',
    'TN99': 'Tamil Nadu - Theni',
}

# Assuming 'RTO' is a column with regional information
result['location'] = result['RTO'].map(rto_mapping)

# If there are NaN values after mapping, you can fill them with a default value
result['location'] = result['location'].fillna('Unknown')

# for ML Processing
selected_columns = ['Model', 'No_Of_Owners', 'car_age', 'Mileage', 'Fuel_Type', 'Kilomerters_driven', 'Features', 'location', 'Price']
new_df = result[selected_columns]
# print(new_df)

new_df.isnull().sum()

new_df.dtypes

le = LabelEncoder()
new_df['Fuel_Type'] = le.fit_transform(new_df['Fuel_Type'])

mapping_values = dict(zip(le.classes_, le.transform(le.classes_)))

# Converting string to float
def extract_numeric(mileage):
    if isinstance(mileage, str):
        try:
            return float(re.search(r'(\d+\.\d+|\d+)', mileage).group())
        except AttributeError:
            return mileage
    else:
        return mileage

# Apply the custom function to the 'Mileage' column
new_df['Mileage'] = new_df['Mileage'].apply(extract_numeric)

new_df['Kilomerters_driven'] = new_df['Kilomerters_driven'].astype(str).str.replace(',', '').astype(float)

def extract_numeric_price(price):
    if isinstance(price, str):
        try:
            # Remove currency symbols, commas, and other characters
            cleaned_price = re.sub(r'[^\d.]', '', price)
            return float(cleaned_price)
        except ValueError:
            return price
    else:
        return price

# Apply the custom function to the 'Price' column
new_df['Price'] = new_df['Price'].apply(extract_numeric_price)

# Handling Nan values
mean_mileage = new_df['Mileage'].mean()

# Fill missing values in 'Mileage' with the mean
new_df['Mileage'].fillna(mean_mileage, inplace=True)

new_df.isnull().sum()

# EDA Analysis:

# Assuming df is your DataFrame
df_corr = new_df[['No_Of_Owners', 'car_age', 'Mileage', 'Fuel_Type', 'Kilomerters_driven', 'Price']]

# Calculate the correlation matrix
corr_matrix = df_corr.corr()

# Plot the heatmap using Seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap')
plt.show()

# Univariate Analysis
sns.histplot(data=new_df, x='No_Of_Owners', bins=10)
plt.title('Distribution of No. of Owners')
plt.show()

sns.histplot(data=new_df, x='car_age', bins=20)
plt.title('Distribution of Car Age')
plt.show()

sns.histplot(data=new_df, x='Mileage', bins=20)
plt.title('Distribution of Mileage')
plt.show()

sns.histplot(data=new_df, x='Kilomerters_driven', bins=20)
plt.title('Distribution of Kilometers Driven')
plt.show()

sns.countplot(data=new_df, x='Fuel_Type')
plt.title('Distribution of Fuel Type')
plt.show()

sns.histplot(data=new_df, x='Price', bins=20)
plt.title('Distribution of Price')
plt.show()

# Bivariate Analysis
sns.scatterplot(data=new_df, x='No_Of_Owners', y='Price')
plt.title('Relationship between No. of Owners and Price')
plt.show()

sns.scatterplot(data=new_df, x='car_age', y='Price')
plt.title('Relationship between Car Age and Price')
plt.show()

sns.scatterplot(data=new_df, x='Mileage', y='Price')
plt.title('Relationship between Mileage and Price')
plt.show()

sns.scatterplot(data=new_df, x='Kilomerters_driven', y='Price')
plt.title('Relationship between Kilometers Driven and Price')
plt.show()

sns.boxplot(data=new_df, x='Fuel_Type', y='Price')
plt.title('Relationship between Fuel Type and Price')
plt.show()

# Categorical Variables Analysis
sns.countplot(data=new_df, x='Fuel_Type')
plt.title('Distribution of Fuel Type')
plt.show()

sns.countplot(data=new_df, x='location')
plt.title('Distribution of Location')
plt.show()

# Location Analysis
sns.countplot(data=new_df, x='location')
plt.title('Distribution of Cars Across Locations')
plt.xticks(rotation=45)
plt.show()

# Outlier Detection
# sns.boxplot(data=new_df, x='Price')
# plt.title('Boxplot of Car Prices')
# plt.show()

# Feature Engineering (Example: Creating a new feature 'Mileage per Year')
new_df['Mileage_Per_Year'] = new_df['Mileage'] / new_df['car_age']

# Price Distribution Across Different Categories
sns.boxplot(data=new_df, x='Fuel_Type', y='Price')
plt.title('Price Distribution Across Fuel Types')
plt.show()

sns.boxplot(data=new_df, x='No_Of_Owners', y='Price')
plt.title('Price Distribution Across No. of Owners')
plt.show()

# Multivariate Analysis
# sns.pairplot(new_df[['No_Of_Owners', 'car_age', 'Mileage', 'Kilomerters_driven', 'Price']])
# plt.suptitle('Multivariate Analysis', y=1.02)
# plt.show()

new_df = new_df.drop(columns=['Mileage_Per_Year'])

 #Assuming 'Features', 'Model', and 'location' are the categorical columns
categorical_columns = ['Features', 'Model', 'location']

# Label Encoding
label_encoder_mapping = {}
for column in categorical_columns:
    label_encoder = LabelEncoder()
    new_df[column + '_encoded'] = label_encoder.fit_transform(new_df[column])
    label_encoder_mapping[column] = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

# One-Hot Encoding
# If you want to keep the original columns, use the prefix parameter
new_df_encoded = pd.get_dummies(new_df, columns=categorical_columns, prefix=categorical_columns)

# label encoder mapping
for column, mapping in label_encoder_mapping.items():
    pass 

# To find the values of the encoding

# for column, mapping in label_encoder_mapping.items():
    # pprint(f"{column} Mapping:")
    # pprint(mapping)
    # pprint('\n') 

# Detect Outlier

plt.figure(figsize=(16,7))
sns.boxplot(data=new_df)
plt.show()

# Handle Outlier

new_df['Kilomerters_driven'] = np.log1p(new_df['Kilomerters_driven'])

new_df['Price'] = np.log1p(new_df['Price'])

# Feature Selection

new_df = new_df.drop(columns=['Model', 'location', 'Features'])

# Create csv file

zz=new_df.to_csv('cardekho.csv', index=False)


x = new_df.drop('Price',axis=1)
y = new_df['Price']

def model_regression(x,y,algorithm):
    for i in algorithm:
        xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2, random_state=42)
        model = i().fit(xtrain,ytrain)
        # predict for train and test accuracy # Predicts the target variable for both the training and testing sets using the trained model
        y_train_pred = model.predict(xtrain)
        y_test_pred  = model.predict(xtest)

       # R2 score
        training = r2_score(ytrain,y_train_pred)
        testing = r2_score(ytest,y_test_pred)
        data = {'Algorithm':i.__name__, 'Training R2 Score':training,'Testing R2 Score':testing}
        print(data)

model_regression(x,y,[DecisionTreeRegressor,ExtraTreesRegressor,RandomForestRegressor,XGBRegressor,KNeighborsRegressor,GradientBoostingRegressor,SVR,GaussianProcessRegressor])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Define the hyperparameter grid for ExtraTreesRegressor
param_grid = {
    'n_estimators': [50, 100, 200],    # controls the number of trees in the forest
    'max_depth': [None, 10, 20],       # determines the maximum depth of each tree in the forest
    'min_samples_split': [2, 5, 10],   # controls the minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],     # controls the minimum number of samples required to be at a leaf node
    'max_features': ['auto', 'sqrt', 'log2']  # determines the number of features to consider when looking for the best split
}

# Create ExtraTreesRegressor
et_model = ExtraTreesRegressor()

# Create GridSearchCV object   that performs hyperparameter tuning using cross-validation
grid_search = GridSearchCV(estimator=et_model, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1)

# Fit the grid search to the data
grid_search = grid_search.fit(x_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_

# Use the best hyperparameters to train the model
best_model = grid_search.best_estimator_

# Evaluate the performance of the best model on the test set
test_r2_score = best_model.score(x_test, y_test)

print("Best Hyperparameters:", best_params)
print("Test R2 Score:", test_r2_score)

# ExtraTreeRegressor

# Set a random seed for reproducibility
random_seed = 42

# Use the same random seed for both the splitting approaches
xtrain, xtest, ytrain, ytest = train_test_split(x.values, y, test_size=0.2, random_state=random_seed)

# Create and train the ExtraTreesRegressor model
model = ExtraTreesRegressor(random_state=random_seed).fit(xtrain, ytrain)

# Predict the target variable for the training set
y_pred_train = model.predict(xtrain)

# Predict the target variable for the test set
y_pred_test = model.predict(xtest)

# Calculate R-squared (coefficient of determination) score
r2_train = r2_score(ytrain, y_pred_train)
r2_test = r2_score(ytest, y_pred_test)

# Print R-squared scores
print("R2 Score - Training:", r2_train)
print("R2 Score - Testing:", r2_test)

# Calculate Mean Absolute Error (MAE) for training and testing sets
# MAE is a metric that measures the average absolute difference between the actual and predicted values. Lower MAE values indicate better model performance.

mae_train = mean_absolute_error(ytrain, y_pred_train)
mae_test = mean_absolute_error(ytest, y_pred_test)

# Calculate Mean Squared Error (MSE) for training and testing sets
# It gives higher weight to large errors compared to MAE. Lower MSE values indicate better model performance.

mse_train = mean_squared_error(ytrain, y_pred_train)
mse_test = mean_squared_error(ytest, y_pred_test)

# Print regression metrics
print("MAE - Training:", mae_train)
print("MAE - Testing:", mae_test)
print("MSE - Training:", mse_train)
print("MSE - Testing:", mse_test)

# Save model

with open('car_regression_model_3.pkl', 'wb') as f:
    pickle.dump(model, f)


with open('car_regression_model_3.pkl', 'rb') as f:
    model = pickle.load(f)

# Try the Model
    
sp = [3.0, 10.0, 19.160, 4.0,np.log1p(2.497845), 72.0, 241.0, 224.0]

sell = model.predict([sp])

print('The Price of the Car is ',np.exp(sell), 'Lakhs')

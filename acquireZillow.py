import env
import pandas as pd
import os
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

#Scaling Processers
import sklearn.preprocessing

from datetime import date

#Gets connection to Code Up database using env file
def get_connection(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'


#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------
# Get zillow.csv Data
def get_zillow_data():
    filename = "zillow.csv"

    if os.path.isfile(filename):
        zillow = pd.read_csv(filename)
    else:
        zillow = pd.read_sql('SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips FROM properties_2017 JOIN propertylandusetype using(propertylandusetypeid) WHERE propertylandusetypeid = 261', 
        get_connection('zillow'))
        zillow.to_csv(index = False)
    return zillow


#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------
# Remove outliers
def remove_outliers(df, k, col_list):
    ''' remove outliers from a list of columns in a dataframe 
        and return that dataframe
    '''

    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df

#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------


def prepare_zillow(df):
    ''' Prepare zillow data for exploration'''
    df = df.rename(columns = {'bedroomcnt':'bedrooms', 
                              'bathroomcnt':'bathrooms', 
                              'calculatedfinishedsquarefeet':'area',
                              'taxvaluedollarcnt':'tax_value', 
                              'yearbuilt':'year_built',})

    # removing outliers
    col_list = ['bedrooms', 'bathrooms', 'area', 'tax_value', 'taxamount']
    k = 1.5
    
    df = remove_outliers(df, k, col_list)

    
    # converting column datatypes
    df.fips = df.fips.astype(object)
    df.area = df.area.astype(int)
    df.year_built = df.year_built.astype(object)
    
    #df = yearbuilt_years(df)
    # train/validate/test split
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)
    
    # impute year built using mode
    imputer = SimpleImputer(strategy='median')

    imputer.fit(train[['year_built']])

    train[['year_built']] = imputer.transform(train[['year_built']])
    validate[['year_built']] = imputer.transform(validate[['year_built']])
    test[['year_built']] = imputer.transform(test[['year_built']])       
    
    train = yearbuilt_years(train)
    validate = yearbuilt_years(validate)
    test = yearbuilt_years(test)

    return train, validate, test 

#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------

def zillow_scaler(train, validate, test):

    # 1. create the object
    scaler_min_max = sklearn.preprocessing.MinMaxScaler()

    # 2. fit the object (learn the min and max value)
    scaler_min_max.fit(train[['bedrooms', 'taxamount']])

    # 3. use the object (use the min, max to do the transformation)
    scaled_bill = scaler_min_max.transform(train[['bedrooms', 'taxamount']])

    train[['bedrooms_scaled', 'taxamount_scaled']] = scaled_bill
    # Create them on the test and validate
    test[['bedrooms_scaled', 'taxamount_scaled']] = scaler_min_max.transform(test[['bedrooms', 'taxamount']])
    validate[['bedrooms_scaled', 'taxamount_scaled']] = scaler_min_max.transform(validate[['bedrooms', 'taxamount']])

    return train, validate, test


#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------

def yearbuilt_years(df):
    df.year_built =  df.year_built.astype(int)
    year = date.today().year
    df['age'] = year - df.year_built
    # dropping the 'yearbuilt' column now that i have the age
    df = df.drop(columns=['year_built'])
    
    return df

#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------

'''
def chart_zillow(df):

    plt.figure(figsize=(16, 3))
    # List of columns
    cols = ['bedrooms', 'bathrooms', 'area', 'tax_value', 'tax_amount', 'year_built','fips']
    # Note the enumerate code, which is functioning to make a counter for use in successive plots.
    for i, col in enumerate(cols):
        # i starts at 0, but plot nos should start at 1
        plot_number = i + 1
        # Create subplot.
        plt.subplot(1,9, plot_number)
        # Title with column name.
        plt.title(col)
        # Display histogram for column.
        df[col].hist(bins=10, edgecolor='black')
        # Hide gridlines.
        plt.grid(False)
        plt.tight_layout()
    plt.figure(figsize=(16, 3))

    # List of columns
    cols = ['bedrooms', 'bathrooms', 'area', 'tax_value', 'tax_amount', 'year_built','fips']
    # Note the enumerate code, which is functioning to make a counter for use in successive plots.
    for i, col in enumerate(cols):
        # i starts at 0, but plot nos should start at 1
        plot_number = i + 1
        # Create subplot.
        plt.subplot(1,9, plot_number)
        # Title with column name.
        plt.title(col)
        # Display histogram for column.
        quantile_scaled_zillow2[col].hist(bins=10, edgecolor='black')
        # Hide gridlines.
        plt.grid(False)
        plt.tight_layout()
    return df
'''

#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------

#Do both recieve and clean of zillow data
def wrangle_zillow():

    train, validate, test = prepare_zillow(get_zillow_data())

    return train, validate, test
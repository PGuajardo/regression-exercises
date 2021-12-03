import env
import pandas as pd
import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

#Gets connection to Code Up database using env file
def get_connection(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'


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
'''
# Wrangle the data for zillow
#def wrangle_data(df):
    # Replace any white spaces with a null
    # df = df.replace(r'^\s*$', np.nan, regex=True)

    # Drop all null values
    # df = df.dropna()

    # Change types to int dtypes except for tax amount which remains as a float
   # df = df.astype({'bedroomcnt': 'int64', 'bathroomcnt': 'float64', 'calculatedfinishedsquarefeet': 'int64'
                 ,'taxvaluedollarcnt': 'int64', 'taxamount' : 'float64'})

   #df = df.rename(columns = {'bedroomcnt':'bedrooms', 
                              'bathroomcnt':'bathrooms', 
                              'calculatedfinishedsquarefeet':'area',
                              'taxvaluedollarcnt':'tax_value', 
                              'yearbuilt':'year_built',})

    

  #  return df
'''



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
    
    # train/validate/test split
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)
    
    # impute year built using mode
    imputer = SimpleImputer(strategy='median')

    imputer.fit(train[['year_built']])

    train[['year_built']] = imputer.transform(train[['year_built']])
    validate[['year_built']] = imputer.transform(validate[['year_built']])
    test[['year_built']] = imputer.transform(test[['year_built']])       
    
    return train, validate, test 



#Do both recieve and clean of zillow data
def wrangle_zillow():

    train, validate, test = prepare_zillow(get_zillow_data())

    return train, validate, test
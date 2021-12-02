import env
import pandas as pd
import os
import numpy as np

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



# Wrangle the data for zillow
def wrangle_data(df):
    # Replace any white spaces with a null
    # df = df.replace(r'^\s*$', np.nan, regex=True)

    # Drop all null values
    df = df.dropna()

    # Change types to int dtypes except for tax amount which remains as a float
    df = df.astype({'bedroomcnt': 'int64', 'bathroomcnt': 'int64', 'calculatedfinishedsquarefeet': 'int64'
                 ,'taxvaluedollarcnt': 'int64', 'yearbuilt': 'int64', 'fips': 'int64', 'taxamount' : 'float64'})

    return df


#Do both recieve and clean of zillow data
def wrangle_zillow():

    df = get_zillow_data()
    df = wrangle_data(df)

    return df
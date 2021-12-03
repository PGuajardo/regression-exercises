import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import splitting and imputing functions
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# turn off pink boxes for demo
import warnings
warnings.filterwarnings("ignore")

# import our own acquire module
import acquireTelco

#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
# Clean the data
def clean_telco(df):
    '''
    This function will clean the data
    '''

     # Drops any duplicate values
    df = df.drop_duplicates()
  
   # Change Total_charges to a float64 type
    df['total_charges'] = df.total_charges.replace(' ', 0)
    df['total_charges'] = df.total_charges.astype('float64')

    dummy_df = pd.get_dummies(df[['gender', 'partner', 'dependents', 'phone_service', 'multiple_lines'
                             , 'online_security', 'online_backup', 'device_protection', 'tech_support'
                             , 'streaming_tv', 'streaming_movies', 'paperless_billing',
                             'churn', 'contract_type', 'internet_service_type', 'payment_type']],
                             dummy_na = False, drop_first = [True])

    df = pd.concat([df, dummy_df], axis = 1)
    
    # Drops columns that are already represented by other columns
    cols_to_drop = ['payment_type_id', 'internet_service_type_id', 'contract_type_id']
    df = df.drop(columns = cols_to_drop)
    
    return df

#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
#---------------------------------------------------------------------------


# Split the data function
def train_validate_test_split(df, target):
    '''
    This function takes in a dataframe, the name of the target variable
    (for stratification purposes)
    and splits the data into train, validate and test. 
    Test is 20% of the original dataset, validate is .3 * .8 = 24% of the 
    original dataset, and train is .7 * .80= 56% of the original dataset. 
    The function returns, in this order, train, validate and test dataframes. 
    '''
    train_validate, test = train_test_split(df, test_size=0.2,  
                                            stratify=df[target], random_state = 123)
    
    
    train, validate = train_test_split(train_validate, test_size=0.3, 
                                       stratify=train_validate[target], random_state = 123)
    return train, validate, test


#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
# Use both clean and train_validate_test_split
def prep_telco(df, target):
    
    df = clean_telco(df)
    
    df.rename(columns={'gender_Male': 'is_male', 'partner_Yes': 'has_partner', 'dependents_Yes': 'has_dependent',
                      'churn_Yes' : 'has_churn'}, inplace=True)
    
    ## ???
    train, validate, test = train_validate_test_split(df, target)
    
    return train, validate, test
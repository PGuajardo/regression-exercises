import env
import pandas as pd
import os

#Gets connection to Code Up database using env file
def get_connection(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'


# Get telco_churn .csv Data
def get_telco_data():
    filename = "telco_churn.csv"

    if os.path.isfile(filename):
        telco_churn = pd.read_csv(filename)
    else:
        telco_churn = pd.read_sql('SELECT * FROM customers JOIN contract_types using(contract_type_id) JOIN internet_service_types using(internet_service_type_id) JOIN payment_types using(payment_type_id)', 
        get_connection('telco_churn'))
        telco_churn.to_csv(index = False)
    return telco_churn
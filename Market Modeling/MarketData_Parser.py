#POWER GRID DATA PARSING

#import packages
import pandas as pd
from datetime import datetime

#import useful ML packages
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
import numpy as np
import keras.backend as kb
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
if not physical_devices:
    print("No GPU available.")

# Given the structure, we'll now proceed to read the entire CSV and parse the data as previously done
# but using the CSV reader for flexibility to handle varying data lengths
CAISO_file_path = r"C:\Users\liams\Documents\GitHub\ENERGY\Market Modeling\Data\20230306-20240305 CAISO Real-time Price.csv"

#PARSE MARKET DATA
def parse_market_data(file_path):
    # Read the entire CSV file
    df_full = pd.read_csv(file_path)

    # Specify the format for the datetime conversion
    date_format = "%m/%d/%Y %I:%M:%S %p"
    # Convert 'Date' column to datetime with specified format
    df_full['Date'] = pd.to_datetime(df_full['Date'], format=date_format)

    # Extract year, month, day, and time (in minutes)
    df_full['year'] = df_full['Date'].dt.year
    df_full['month'] = df_full['Date'].dt.month
    df_full['day'] = df_full['Date'].dt.day
    df_full['time'] = df_full['Date'].dt.hour * 60 + df_full['Date'].dt.minute

    # Convert month and day to day of year.
    #INVESTIGATE EFFECT OF LEAP DAY, MINUTE, SECOND
    df_full['day_of_year'] = df_full['Date'].dt.dayofyear

    # Splitting data by hub
    #Current Input Variables: Year, Month, day_of_year, Time,
    #MORE IMPORTANT VARIABLES TO CONSIDER: LOCATION, DAY OF WEEK, WEATHER/TEMP CONDITIONS, GENERATOR OUTAGES, COMMODITY PRICES
    th_np15 = df_full[df_full['hub'] == 'TH_NP15'][['year', 'month', 'day', 'day_of_year', 'time', 'price']]
    th_sp15 = df_full[df_full['hub'] == 'TH_SP15'][['year', 'month', 'day', 'day_of_year', 'time', 'price']]
    th_zp26 = df_full[df_full['hub'] == 'TH_ZP26'][['year', 'month', 'day', 'day_of_year', 'time', 'price']]

    return th_np15, th_sp15, th_zp26
th_np15, th_sp15, th_zp26 = parse_market_data(CAISO_file_path)

#DISPLAY SAMPLE
def print_sample_data(samplesz, th_np15, th_sp15, th_zp26):
    # Convert to arrays for a sample of the data to confirm successful conversion
    th_np15_sample_updated = th_np15.head(samplesz).values
    th_sp15_sample_updated = th_sp15.head(samplesz).values
    th_zp26_sample_updated = th_zp26.head(samplesz).values
    print(th_np15_sample_updated)
    print(th_sp15_sample_updated)
    print(th_zp26_sample_updated)
print_sample_data(3, th_np15, th_sp15, th_zp26)


#TAKE DATA AND IMPLEMENT PREDICTIVE MODEL. LSTM RECURENT NN MODEL PROBABLY BEST. SEE DOCUMENTATION ON KERAS

#EXAMPLE FROM 
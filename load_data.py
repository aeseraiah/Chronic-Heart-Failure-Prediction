import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def preprocess_music_data(file_path, test_size=0.2, random_state=42):
    # Load the dataset
    df = pd.read_csv(file_path+'subject-info.csv', sep=';')
    
    # # Handle missing values - example: drop rows with any NaN values
    # df = df.dropna()
    
    # # Encode categorical variables - example: Gender and Cause of death
    # df['Gender (male=1)'] = df['Gender (male=1)'].astype('category')
    # df['Cause of death'] = df['Cause of death'].astype('category')
    # df = pd.get_dummies(df, drop_first=True)
    
    # Separate features and labels
    label_column = 'Cause of death'

    df_cleaned = df.dropna(axis=1, how='all') # drops columns (axis=1) where every value is missing (how='all')
    # print(df_cleaned.columns.tolist())


    X = df_cleaned.drop(columns=[label_column])

    # Only the labels df
    labels = df_cleaned[label_column]


    # Specify the columns you want to one-hot encode
    columns_to_encode = [
        'Exit of the study', 
        'Cause of death', 
        'HF etiology - Diagnosis', 
        'Prior implantable device', 
        'Prior Revascularization', 
        'Syncope', 
        'Mitral valve insufficiency ',  # COLUMN NAME INCLUDES SPACE 
        'Mitral flow pattern', 
        'ECG rhythm ', # COLUMN NAME INCLUDES SPACE 
        'Intraventricular conduction disorder', 
        'Holter  rhythm ', # COLUMN NAME INCLUDES SPACE 
        'Ventricular Extrasystole', 
        'Ventricular Tachycardia', 
        'Paroxysmal supraventricular tachyarrhythmia', 
        'Bradycardia'
    ]

    # One-hot encode the specified columns
    df_encoded = pd.get_dummies(df_cleaned, columns=columns_to_encode, drop_first=True)

    # Display the first few rows of the encoded DataFrame
    print(df_encoded.head())

    print(df_encoded.columns.tolist())




    class_counts = labels.value_counts()
    # print(class_counts)
    # classes = np.unique(labels)
    # print(classes)
    return X, labels

    # # Scale the features to the range [0, 1]
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)

    # # Split the dataset into training and testing sets
    # X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)

    # return X_train, X_test, y_train, y_test

file_path = './data/'
# X_train, X_test, y_train, y_test = preprocess_music_data(file_path)
X, y = preprocess_music_data(file_path)


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from lime import lime_tabular

def preprocess_music_data(file_path, test_size=0.2, random_state=42):
    # Load the dataset
    data_df = pd.read_csv(file_path+'subject-info.csv', sep=';')
    data_codes_df = pd.read_csv(file_path+'subject-info_codes.csv', sep=';', encoding='ISO-8859-1')

    print(data_df)
    print(data_codes_df['Codes'])

    # # Handle missing values - example: drop rows with any NaN values
    # df = df.dropna()
    
    # # Encode categorical variables - example: Gender and Cause of death
    # df['Gender (male=1)'] = df['Gender (male=1)'].astype('category')
    # df['Cause of death'] = df['Cause of death'].astype('category')
    # df = pd.get_dummies(df, drop_first=True)
    
    df_cleaned = data_df.dropna(axis=1, how='all') # drops columns (axis=1) where every value is missing (how='all')
    # print(df_cleaned.columns.tolist())


    # Separate features and labels
    label_column = 'Cause of death'

    # Only the labels df
    labels = df_cleaned[label_column]

    # cleaned_csv = df_cleaned.to_csv(file_path+'cleaned_data.csv')
    X = df_cleaned.drop(columns=[label_column])
    columns_to_drop = ['Patient ID', 'Exit of the study', 'Follow-up period from enrollment (days)']
    X_dropped = X.drop(columns=columns_to_drop)

    print(f"X_dropped:{X_dropped}")
    columns_with_commas = ['Albumin (g/L)', 'Body Mass Index (Kg/m2)', 'Total Cholesterol (mmol/L)', 'Glucose (mmol/L)', 'HDL (mmol/L)', 'Potassium (mEq/L)']
    X_dropped[columns_with_commas].replace(",", ".", regex=True)
    print(X_dropped)

    # print(len(X_dropped))

    # exlude subject code 1 from cleaned_data set. This code corresponds to non-cardiac deaths (Total Deaths = 266 - 61 non-cardiac deaths = 205 cardiac deaths)
    non_zero_counts_per_row = (labels != 0).sum(axis=0)

    non_cardiac_death = (labels == 1).sum(axis=0)
            
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
    # X_dropped_dummies = pd.get_dummies(X_dropped, columns=columns_to_encode, drop_first=True)

    # Display the first few rows of the encoded DataFrame
    # print(df_encoded.values)

    # print(df_encoded.columns.tolist())




    class_counts = labels.value_counts()
    # print(class_counts)

    classes = np.unique(labels)
    print(classes)

    # Scale the features to the range [0, 1]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_dropped)

    feature_names = X_scaled.columns.tolist()
    label_name = labels.columns.tolist()
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, labels, test_size=test_size, random_state=random_state)

    # Train a random forest classifier
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Create a LIME explainer
    explainer = lime_tabular.LimeTabularExplainer(
        X_train.values,
        feature_names=feature_names,
        class_names=label_name,
        mode='classification'
    )

    # Explain a prediction
    i = 0  # Index of the instance to explain
    exp = explainer.explain_instance(X_test.values[i], clf.predict_proba, num_features=4)

    # Visualize the explanation
    exp.show_in_notebook(show_table=True)

    return X, labels

file_path = './data/'
# X_train, X_test, y_train, y_test = preprocess_music_data(file_path)
X, y = preprocess_music_data(file_path)


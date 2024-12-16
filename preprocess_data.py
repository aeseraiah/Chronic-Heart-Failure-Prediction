from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import numpy as np


def cleaning(df):

    # Exclude subjects who were catgeorized as 'Lost to follow-up':
    df = df.loc[df['Exit of the study'] != 1]
    df.to_csv('after_dropping_subs_df.csv', index=False)

    # Drop 'Exit of the study' column since 'Cause of death' is the target variable:
    df = df.drop('Exit of the study', axis=1)

    # Following columns are not direct risk indicators that can be measured before follow-up period:
    df = df.drop('Follow-up period from enrollment (days)', axis=1)
    df = df.drop('days_4years', axis=1)

    # Convert 'Holster Onset' column to seconds since it's original format is in 'HH:MM:SS':
    val_arr = []
    for val in df['Holter onset (hh:mm:ss)']:
        if pd.isna(val):
            val_arr.append(val)
        else:
            hours = (int(val.strip("'").split(":")[0]))
            minutes = (int(val.strip("'").split(":")[1]))
            seconds = (int(val.strip("'").split(":")[2]))

            val_arr.append((hours*360) + (minutes*60) + seconds)
  
    df['Holter onset (hh:mm:ss)'] = val_arr

    df = df.replace(",", ".", regex=True)

    nans_removed_df = df.dropna(axis=1, how='all') 

    # Drop Patient ID column as it's not a feature
    # Drop 'Number of ventricular premature contractions per hour' due to nonsensical values
    # Drop 'cigarettes /year' due to unrealistic, inflated values
    # Drop 'Hig-resolution ECG available' and 'Holter available' since they are not features
    columns_to_be_dropped = ['Patient ID', 'Number of ventricular premature contractions per hour', 'cigarettes /year', 'Hig-resolution ECG available', 'Holter available']

    df_columns_removed = nans_removed_df.drop(columns_to_be_dropped, axis=1)
    
    # Replace age >89 with age 90: 
    df_columns_removed['Age'] = df_columns_removed['Age'].replace('>89', 90)

    return df_columns_removed


def columns_to_one_hot(df):

    # Specify the columns you want to one-hot encode
    columns_to_encode = [
        'Calcium channel blocker (yes=1)',
        'SCD_4years SinusRhythm',
        'Q-waves (necrosis, yes=1)',
        'QRS > 120 ms ',
        'HF_4years SinusRhythm',
        'Gender (male=1)',
        'NYHA class',
        'Diabetes (yes=1)',
        'History of dyslipemia (yes=1)',
        'Peripheral vascular disease (yes=1)',
        'History of hypertension (yes=1)',
        'Prior Myocardial Infarction (yes=1)',
        'Signs of pulmonary venous hypertension (yes=1)',
        'Right ventricle contractility (altered=1)',
        'Left ventricular hypertrophy (yes=1)',
        'Non-sustained ventricular tachycardia (CH>10)',
        'Diabetes medication (yes=1)',
        'Amiodarone (yes=1)',
        'Angiotensin-II receptor blocker (yes=1)',
        'Anticoagulants/antitrombotics  (yes=1)',
        'Betablockers (yes=1)',
        'Digoxin (yes=1)',
        'Loop diuretics (yes=1)',
        'Spironolactone (yes=1)',
        'Statins (yes=1)',
        'Hidralazina (yes=1)',
        'ACE inhibitor (yes=1)',
        'Nitrovasodilator (yes=1)',
        'HF etiology - Diagnosis', 
        'Prior implantable device', 
        'Prior Revascularization', 
        'Syncope', 
        'Mitral valve insufficiency ',
        'Mitral flow pattern', 
        'ECG rhythm ',
        'Intraventricular conduction disorder', 
        'Holter  rhythm ',
        'Ventricular Extrasystole', 
        'Ventricular Tachycardia', 
        'Paroxysmal supraventricular tachyarrhythmia', 
        'Bradycardia' 
    ]

    binary_categorical_columns = []
    one_hot_encoded_columns = []

    # Process binary categorical features
    for col in columns_to_encode:
        # Check if the column has binary values (0 or 1)
        if df[col].isin([0, 1]).all():
            # Create two separate columns: (True) and (False)
            true_col = f'{col} (True)'
            false_col = f'{col} (False)'
            df[true_col] = (df[col] == 1).astype(int)
            df[false_col] = (df[col] == 0).astype(int)
            
            # Drop the original binary column
            df.drop(columns=[col], inplace=True)
            
            # Track new binary columns
            binary_categorical_columns.extend([true_col, false_col])
        else:
            # Keep non-binary categorical columns for one-hot encoding
            one_hot_encoded_columns.append(col)

    # Perform one-hot encoding for remaining categorical columns
    df_after_one_hot = pd.get_dummies(df, columns=one_hot_encoded_columns, drop_first=True)
    df_after_one_hot.to_csv('df_after_one_hot_before_booling.csv', index=False)

    # Capture new one-hot-encoded column names
    one_hot_columns = [col for col in df_after_one_hot.columns if col not in df.columns]
    all_categorical_columns = binary_categorical_columns + one_hot_columns

    # Convert boolean columns to integers
    for col in df_after_one_hot.select_dtypes(include=['bool']).columns:
        df_after_one_hot[col] = df_after_one_hot[col].astype(int)

    # Separate continuous and categorical features
    continuous_features = df_after_one_hot.drop(columns=all_categorical_columns)
    # continuous_features = df.drop(columns=columns_to_encode, errors='ignore')
    categorical_features_df = df_after_one_hot[all_categorical_columns]
    print(f'Number of categorical features: {len(categorical_features_df.columns)}')
    print(f'Number of continuous features: {len(continuous_features.columns)}')

    return df_after_one_hot, continuous_features, categorical_features_df


def standard_scalar(continuous_features_df, categorical_columns_df):

    continuous_features_df = continuous_features_df.drop('Cause of death', axis=1)
    scaler = StandardScaler()
    df_scaled_continuous = scaler.fit_transform(continuous_features_df)

    df_scaled_continuous = pd.DataFrame(
        df_scaled_continuous, 
        columns=continuous_features_df.columns, 
        index=continuous_features_df.index
    )

    # Check indices and reset if needed
    if not continuous_features_df.index.equals(categorical_columns_df.index):
        print("Index mismatch detected. Resetting indices.")
        continuous_features_df = continuous_features_df.reset_index(drop=True)
        categorical_columns_df = categorical_columns_df.reset_index(drop=True)
        df_scaled_continuous.index = continuous_features_df.index

    # Combine the scaled continuous features and categorical features
    combined_df = pd.concat([df_scaled_continuous, categorical_columns_df], axis=1)

    print(f'Number of continuous features: {len(continuous_features_df.columns)}')
    print(f'Number of categorical features: {len(categorical_columns_df.columns)}')
    print(f'Total number of features in combined DataFrame: {len(combined_df.columns)}')

    return combined_df


def impute_missing_values(continuous_features_df, categorical_columns_df):

    imputer = KNNImputer()

    continuous_features_imputed = imputer.fit_transform(continuous_features_df)
    # Total missing after KNN imputation, should be 0
    print('Missing: %d' % sum(np.isnan(continuous_features_imputed).flatten()))
    continuous_features_imputed_df = pd.DataFrame(continuous_features_imputed, columns=continuous_features_df.columns)
    
    categorical_features_imputed = imputer.fit_transform(categorical_columns_df)
    # Total missing after KNN imputation, should be 0
    print('Missing: %d' % sum(np.isnan(categorical_features_imputed).flatten()))
    categorical_features_imputed_df = pd.DataFrame(categorical_features_imputed, columns=categorical_columns_df.columns)

    return continuous_features_imputed_df, categorical_features_imputed_df


def load_labels(file_path):
    data = pd.read_csv(file_path)

    print(data.columns.tolist())

    label_column = 'Cause of death'
    labels = data[label_column]

    for index, label in enumerate(labels):
        if label == 7.0:
            labels[index] = 6.0
            

    encoder = LabelEncoder()
    encoder.fit(labels)
    # [0, 1, 2, 3] == [survivor, non-cardiac death, SCD, Pump-Failure]
    encoded_labels = encoder.transform(labels)
    one_hot_labels = to_categorical(encoded_labels, num_classes=len(np.unique(encoded_labels)))
    
    return one_hot_labels
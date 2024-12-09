from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import pandas as pd

def cleaning(df):
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

    df['Number of ventricular premature contractions per hour'] = df['Number of ventricular premature contractions per hour'].replace(",", ".", regex=True)
    
    df = df.replace(",", ".", regex=True)

    nans_removed_df = df.dropna(axis=1, how='all') 

    # Drop Patient ID column since row values are strings (KNN needs numerical values)
    dropped_patient_id = nans_removed_df.drop('Patient ID', axis=1)

    # Remove columns with nonsensical or inflated values: 
    removed_inflated_df = dropped_patient_id.drop('cigarettes /year', axis=1) # values are too high to be realistic

    # Remove range of ages above 89
    for index, val in enumerate(removed_inflated_df['Age']):
        try:
            age = val
            if age == '>89':
                removed_inflated_df.drop(index, inplace=True)
            else:
                age = int(val)
        except ValueError:
            print(f"ValueError: {index, val}")
            pass

    return removed_inflated_df

def columns_to_one_hot(df):

    # Specify the columns you want to one-hot encode
    columns_to_encode = [
        'Exit of the study', 
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
    X_dropped_dummies = pd.get_dummies(df, columns=columns_to_encode, drop_first=True)

    for col in X_dropped_dummies.select_dtypes(include=['bool']).columns:
        X_dropped_dummies[col] = X_dropped_dummies[col].astype(int)

    return X_dropped_dummies


def standard_scalar(df):

    continuous_features = [
        'Exit of the study', 
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

    continuous_features_df = df[continuous_features]
    print(f"Continuous Features: {continuous_features_df.columns}")
    non_continuous_features_df = df.drop(columns=continuous_features)

    scaler = StandardScaler()
    df_scaled_continuous = scaler.fit_transform(continuous_features_df)

    df_scaled_continuous = pd.DataFrame(df_scaled_continuous, columns=continuous_features)
    df_final = pd.concat([df_scaled_continuous, non_continuous_features_df.reset_index(drop=True)], axis=1)

    return df_final


def impute_missing_values(df):
    imputer = KNNImputer()
    imputer.fit(df)
    Xtrans = imputer.transform(df)

    return Xtrans
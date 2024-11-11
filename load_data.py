import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from lime import lime_tabular
from sklearn.metrics import accuracy_score
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
# import autosklearn.classification


def preprocessing_missing_values(data_df):
    # Drop columns with ALL NaN values for rows:
    data_without_all_nan_values = data_df.dropna(axis=1, how='all') # EXLCUDES UNNAMED FEATURES
    return 


def drop_data(data_df):
    # Drop columns with ALL NaN values for rows (excludes 'unnamed' features):
    nans_removed_df = data_df.dropna(axis=1, how='all') 

    # Drop Patient ID column since row values are strings (KNN needs numerical values)
    dropped_patient_id = nans_removed_df.drop('Patient ID', axis=1)

    # Remove columns with nonsensical or inflated values: 
    removed_inflated_df = dropped_patient_id.drop('cigarettes /year', axis=1) # values are too high to be realistic

    return removed_inflated_df


def standard_scalar(data_df):
    # Scale the features to the range [0, 1]
    scaler = StandardScaler()
    data_df_scaled = scaler.fit_transform(data_df)
    return data_df_scaled


def simple_neural_net(xtrain):
    input_d = xtrain.shape[1]
    num_classes = 5

    model = Sequential([
        Dense(100, activation='relu', input_dim=input_d), 
        Dense(50, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model


def k_fold_cross_validation(features, labels):

    print(labels.shape)
    # WHY ARE THERE 5 LABELS RATHER THAN 3?
    print(np.unique(labels))

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    accuracy_scores = []

    for train_index, test_index in kf.split(features):
        X_train, X_test = features.iloc[train_index], features.iloc[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        model = simple_neural_net(X_train)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        history = model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        accuracy_scores.append(accuracy)

    average_accuracy = np.mean(accuracy_scores)

    return average_accuracy, history


def plot_metrics(model_history):
    plt.plot(model_history['accuracy'])
    plt.plot(model_history['val_accuracy'])
    plt.title('Training and Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')


def auto_sklearn(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, random_state=1)
    automl = autosklearn.classification.AutoSklearnClassifier()
    automl.fit(X_train, y_train)
    y_hat = automl.predict(X_test)
    print("Accuracy score", accuracy_score(y_test, y_hat))


def feature_importance(X_train, X_test, feature_names, label_name, model):
    # Create a LIME explainer
    explainer = lime_tabular.LimeTabularExplainer(
        X_train.values,
        feature_names=feature_names,
        class_names=label_name,
        mode='classification'
    )

    # Explain a prediction
    i = 0  # Index of the instance to explain
    exp = explainer.explain_instance(X_test.values[i], model.predict_proba)

    # Visualize the explanation
    exp.show_in_notebook(show_table=True)

    # Print the explanation as a list of feature contributions
    for feature, importance in exp.as_list():
        print(f"{feature}: {importance}")


def preprocess_data(file_path, test_size=0.2, random_state=42):

    # Load the dataset
    data_df = pd.read_csv(file_path+'subject-info.csv', sep=';')
    data_codes_df = pd.read_csv(file_path+'subject-info_codes.csv', sep=';', encoding='ISO-8859-1')

    # Convert 'Holster Onset' column to seconds since it's original format is in 'HH:MM:SS':
    val_arr = []
    for val in data_df['Holter onset (hh:mm:ss)']:
        if pd.isna(val):
            val_arr.append(val)
        else:
            hours = (int(val.strip("'").split(":")[0]))
            minutes = (int(val.strip("'").split(":")[1]))
            seconds = (int(val.strip("'").split(":")[2]))

            val_arr.append((hours*360) + (minutes*60) + seconds)
  
    data_df['Holter onset (hh:mm:ss)'] = val_arr

    # Example value to be changed: 1,458,2000 --> where it should be 1.458
    data_df['Number of ventricular premature contractions per hour'] = data_df['Number of ventricular premature contractions per hour'].replace(",", ".", regex=True)
    
    data_df = data_df.replace(",", ".", regex=True)

    # Remove irrelevant or erroneous data:
    data_df_dropped = drop_data(data_df)

    # Replace range of ages above 89 with 90:
    # val_arr_bmi = []
    val_arr_age = []
    for index, val in enumerate(zip(data_df_dropped['Age'], data_df_dropped['Body Mass Index (Kg/m2)'])):
        try:
            age = val[0]
            if age == '>89':
                age = int(90)
                print(f"Subject number: {index}, {val}")
                # age = int(age.strip(">"))
            else:
                age = int(val[0])

            bmi = float(val[1])
            val_arr_age.append(age)
            # val_arr_bmi.append(bmi)
        except ValueError:
            print(f"TYPE ERROR: {index, val}")
            pass

    # data_df_dropped['Body Mass Index (Kg/m2)'] = val_arr_bmi
    data_df_dropped['Age'] = val_arr_age

    data_df_dropped.to_csv('before_imputation.csv')

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
    X_dropped_dummies = pd.get_dummies(data_df_dropped, columns=columns_to_encode, drop_first=True)

    for col in X_dropped_dummies.select_dtypes(include=['bool', 'uint8']).columns:
        X_dropped_dummies[col] = X_dropped_dummies[col].astype(int)

    X_dropped_dummies.to_csv('after_dummy_creation.csv', index=False)

    imputer = KNNImputer()
    imputer.fit(X_dropped_dummies)
    Xtrans = imputer.transform(X_dropped_dummies)
    # print total missing
    print('Missing: %d' % sum(np.isnan(Xtrans).flatten()))

    final_cleaned_df = pd.DataFrame(Xtrans, columns=X_dropped_dummies.columns)
    final_cleaned_df.to_csv('after_imputation.csv', index=False)

    label_column = 'Cause of death'
    labels = final_cleaned_df[label_column]

    # Encoding labels is not neccesary because there are already in a numerical format 
    # Assuming y is your target label column
    # label_encoder = LabelEncoder()
    # labels_encoded = label_encoder.fit_transform(labels) 

    features = final_cleaned_df.drop('Cause of death', axis=1)

    # 3. Define columns to be imputed and then apply imputation method (mean, mode, or median)
    # 4. Verify that no rows have all NaN values. If we implemnet imputation, then the row/subject with all missing values will have all of their data generated, which is less than ideal

    # # exlude subject code 1 from cleaned_data set. This code corresponds to non-cardiac deaths (Total Deaths = 266 - 61 non-cardiac deaths = 205 cardiac deaths)
    # non_zero_counts_per_row = (labels != 0).sum(axis=0)
    # non_cardiac_death = (labels == 1).sum(axis=0)

    # from [0. 1. 3. 6. 7.] to [0 1 2 3 4]

    encoder = LabelEncoder()
    encoder.fit(labels)
    encoded_labels = encoder.transform(labels)
    print(np.unique(encoded_labels))

    one_hot_labels = to_categorical(labels, num_classes=len(np.unique(labels)))
    print(one_hot_labels)

    clf = RandomForestClassifier(random_state=42)
    average_testing_accuracy, history = k_fold_cross_validation(features, one_hot_labels)
    print(f"Average Test Accuracy: {average_testing_accuracy}")

    # feature_names = features.columns.tolist()
    # label_name = labels.tolist()

    # plot_metrics(model_history=history)


data_file_path = './data/'
preprocess_data(data_file_path)


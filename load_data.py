import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, Conv1D, Flatten, BatchNormalization, MaxPooling1D
import numpy as np
from sklearn.utils import class_weight
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import pandas as pd
import re
import glob
import tensorflow as tf
from sklearn.metrics import roc_auc_score
import os
from sklearn.ensemble import RandomForestClassifier
# from lime import lime_tabular
from sklearn.metrics import accuracy_score
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
# import autosklearn.classification
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import mne


def preprocessing_missing_values(data_df):
    # Drop columns with ALL NaN values for rows:
    data_without_all_nan_values = data_df.dropna(axis=1, how='all') # EXLCUDES UNNAMED FEATURES
    return 


def pca(X, num_components, plot_bool):

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=num_components)  # Reduce to 2 components
    X_pca = pca.fit_transform(X_scaled)

    if plot_bool == True:
        plt.figure(figsize=(8, 6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], color='blue', edgecolors='k', alpha=0.6)
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.title('PCA Result (2D Projection)')
        plt.show()

    return pca.explained_variance_ratio_, pca.explained_variance_, pca.components_, X_pca

def drop_subject(data_df):
    # Remove range of ages above 89
    for index, val in enumerate(data_df['Age']):
        try:
            age = val
            if age == '>89':
                data_df.drop(index, inplace=True)
            else:
                age = int(val)
        except ValueError:
            print(f"ValueError: {index, val}")
            pass

    return data_df


def drop_data(data_df):
    # Drop columns with ALL NaN values for rows (excludes 'unnamed' features):
    nans_removed_df = data_df.dropna(axis=1, how='all') 

    # Drop Patient ID column since row values are strings (KNN needs numerical values)
    dropped_patient_id = nans_removed_df.drop('Patient ID', axis=1)

    # Remove columns with nonsensical or inflated values: 
    removed_inflated_df = dropped_patient_id.drop('cigarettes /year', axis=1) # values are too high to be realistic

    final_sub_df = drop_subject(removed_inflated_df)
    return final_sub_df


def standard_scalar(data_df):
    # Scale the features to the range [0, 1]
    scaler = StandardScaler()
    data_df_scaled = scaler.fit_transform(data_df)
    return data_df_scaled


def simple_neural_net(xtrain):
    input_d = xtrain.shape[1]
    print(f'Input dimension: {input_d}')
    num_classes = 4

    model = Sequential([
        # Dense(512, activation='relu', input_dim=input_d), 
        Dense(128, activation='relu', input_dim=input_d), 
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model

def cnn(X_train):
    num_classes = 4
    model = Sequential()
    # model.add(Conv1D(64,  kernel_size=3, activation='relu', kernel_initializer='he_uniform',padding = 'same', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Conv1D(128, 3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    # model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.20))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes,activation = 'softmax'))

    # model.add(Conv1D(64,  kernel_size=3, activation='relu', kernel_initializer='he_uniform',padding = 'same', input_shape=(X_train.shape[1], X_train.shape[2]))) 
    # model.add(BatchNormalization(axis = -1))
    # model.add(MaxPooling1D(pool_size=2))
    # model.add(Dropout(0.20))
    # model.add(Conv1D(64,  kernel_size=3, activation='relu', kernel_initializer='he_uniform',padding = 'same'))
    # model.add(BatchNormalization(axis=-1))
    # model.add(MaxPooling1D(pool_size=2))
    # model.add(Dropout(0.20))
    # model.add(Flatten())
    # model.add(Dropout(0.20))
    # model.add(Dense(32, activation = 'relu'))
    # model.add(Dense(16, activation = 'relu'))
    # model.add(Dense(num_classes,activation = 'softmax'))

    return model


def k_fold_cross_validation(features, labels, num_epochs):

    features_df = pd.DataFrame(features)

    kf = KFold(n_splits=5, shuffle=True)
    # kf = KFold(n_splits=5, shuffle=True, random_state=42)

    cnn_AUC_scores = []
    simple_nn_AUC_scores = []
    ensemble_AUC_scores = []
    
    for fold_num, (train_index, test_index) in enumerate(kf.split(features), 1):
        X_train, X_test = features_df.iloc[train_index], features_df.iloc[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        # print(f"Testing Subjects: {test_index}")

        X_train_arr = np.array(X_train)
        X_test_arr = np.array(X_test)
        # reshape array for expected cnn input
        X_train_cnn = X_train_arr.reshape((X_train_arr.shape[0], X_train_arr.shape[1], 1))  # Reshaping to (samples, timesteps, channels) OR (subjects, features, channels)
        X_test_cnn = X_test_arr.reshape((X_test_arr.shape[0], X_test_arr.shape[1], 1))  # Reshaping to (samples, timesteps, channels)
        
        cnn_model = cnn(X_train_cnn)
        simple_nn_model = simple_neural_net(X_train_arr)

        cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=tf.keras.metrics.AUC())
        simple_nn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=tf.keras.metrics.AUC())

        early_stopping = EarlyStopping(
            monitor='val_loss',       # Can also monitor 'val_accuracy', 'loss', etc.
            patience=2,               # Number of epochs to wait for improvement
            restore_best_weights=True, # Restore the weights of the best model (i.e., the one with the lowest val_loss)
            verbose=1                  # Prints information about stopping
        )

        print('TRAINING CNN MODEL:')
        print(f'X_train_cnn: {X_train_cnn.shape}')
        print(f'X_test_cnn: {X_test_cnn.shape}')
        print(f'y_train: {y_train.shape}')
        cnn_history = cnn_model.fit(X_train_cnn, y_train, epochs=num_epochs, validation_split=0.2, callbacks=[early_stopping])
        print('TRAINING SIMPLE NN MODEL:')
        nn_history = simple_nn_model.fit(X_train_arr, y_train, epochs=num_epochs, validation_split=0.2, callbacks=[early_stopping])

        cnn_y_pred = cnn_model.predict(X_test_cnn)
        nn_y_pred = simple_nn_model.predict(X_test)
        cnn_y_pred_train = cnn_model.predict(X_train_cnn)
        
        # print(f'Training labels for subjects 1-10: {y_train[:10]}\n')
        # print(f'Testing labels for subjects 1-10: {y_test[:10]}\n')
        # print(f'Probabilities for first CNN test prediction (subjects 1-10): {cnn_y_pred[:10]}\n') # Corresponds to probabilities for first prediction
        print(f'Probabilities for first CNN train prediction (subjects 1-10): {cnn_y_pred_train[:10]}\n') # Corresponds to probabilities for first prediction
        print(f'Labels for training data (subjects 1-10): {y_train}')

        y_train_ground_truth_df = pd.DataFrame(y_train)
        y_train_ground_truth_df.to_csv(f'y_train_ground_truth_df_{fold_num}.csv')

        y_train_pred_df = pd.DataFrame(cnn_y_pred_train)
        y_train_pred_df.to_csv(f'y_train_pred_df_{fold_num}.csv')

        y_train_ground_truth_counts = y_train_ground_truth_df.sum(axis=0)
        print(f'The ground truth totals for each column: {y_train_ground_truth_counts}')
        # print(f'X_test_cnn (data) for subjects 1-10: {X_test_cnn[:10]}\n')
        # print(f'Ground truth for subjects 1-10: {y_test[:10]}\n')
        # # print(f'Shape of X_test_cnn: {X_test_cnn.shape}') 
        # print(f'Probabilities for first NN prediction (subject 1): {nn_y_pred[0]}') # Corresponds to probabilities for first prediction

        ensemble_y_pred = (cnn_y_pred + nn_y_pred) / 2
        # print(f'Shape of ensemble preds: {ensemble_y_pred.shape}')
        # print(f'Probabilities for first NN prediction: {ensemble_y_pred[0]}')

        cnn_new_y_pred = []
        simple_nn_new_y_pred = []
        ensemble_new_y_pred = []

        for cnn_subject_prob, simple_nn_subject_prob, ensemble_subject_prob in zip(cnn_y_pred, nn_y_pred, ensemble_y_pred):
            # Convert each probability to 1 if it's >= 0.5, otherwise 0
            cnn_binary_subject_pred = [1 if subject_probability >= 0.5 else 0 for subject_probability in cnn_subject_prob]

            cnn_new_y_pred.append(cnn_binary_subject_pred)

            simple_nn_binary_subject_pred = [1 if subject_probability >= 0.5 else 0 for subject_probability in simple_nn_subject_prob]
            simple_nn_new_y_pred.append(simple_nn_binary_subject_pred)

            ensemble_binary_subject_pred = [1 if subject_probability >= 0.5 else 0 for subject_probability in ensemble_subject_prob]
            ensemble_new_y_pred.append(ensemble_binary_subject_pred)

        predicted_probabilities = np.array(cnn_y_pred)

        # Create a one-hot encoded array
        one_hot_predictions = np.zeros_like(predicted_probabilities)

        # For each sample, find the index of the max probability and set it to 1
        for i, prob in enumerate(predicted_probabilities):
            max_class = np.argmax(prob)  # Find index of the max probability
            one_hot_predictions[i, max_class] = 1  # Set that index to 1

        # Print the one-hot encoded predictions
        print("One-hot Encoded Predictions:")
        print(one_hot_predictions)

        y_train_pred_df = pd.DataFrame(cnn_new_y_pred)
        y_pred_counts = y_train_pred_df.sum(axis=0)
        print(f'The prediction for training totals for each column: {y_pred_counts}')

        # cnn_AUC = roc_auc_score(y_test, cnn_y_pred, average='macro', multi_class='ovo')
        # print(f"CNN Test AUC: {cnn_AUC}")

        # simple_nn_AUC = roc_auc_score(y_test, nn_y_pred, average='macro', multi_class='ovo')
        # print(f"Simple NN Test AUC: {simple_nn_AUC}")

        # ensemble_AUC = roc_auc_score(y_test, ensemble_y_pred, average='macro', multi_class='ovo')
        # print(f"Ensemble Test AUC: {ensemble_AUC}")

        cnn_auc_metric = tf.keras.metrics.AUC()
        cnn_auc_metric.update_state(y_test, cnn_y_pred)
        cnn_auc_score = cnn_auc_metric.result().numpy()
        print(f"CNN Test AUC: {cnn_auc_score}")

        simple_nn_auc_metric = tf.keras.metrics.AUC()
        simple_nn_auc_metric.update_state(y_test, nn_y_pred)
        simple_nn_auc_score = simple_nn_auc_metric.result().numpy()
        print(f"Simple NN Test AUC: {simple_nn_auc_score}")

        ensemble_auc_metric = tf.keras.metrics.AUC()
        ensemble_auc_metric.update_state(y_test, ensemble_y_pred)
        ensemble_auc_score = ensemble_auc_metric.result().numpy()
        print("ensemble_auc_score:", ensemble_auc_score)

        test_index_arr = None
    
        cnn_AUC_scores.append(cnn_auc_score)
        simple_nn_AUC_scores.append(simple_nn_auc_score)
        ensemble_AUC_scores.append(ensemble_auc_score)

    cnn_AUC = np.mean(cnn_AUC_scores)
    simple_nn_AUC = np.mean(simple_nn_AUC_scores)
    ensemble_AUC = np.mean(ensemble_AUC_scores)

    return cnn_AUC, simple_nn_AUC, ensemble_AUC, cnn_history, nn_history, test_index_arr


def plot_metrics(model_history, metrics_plot_name, model_name):
    plt.plot(model_history.history['AUC'], label='Training AUC')
    plt.plot(model_history.history['val_AUC'], label='Validation AUC')
    plt.title(f'{model_name} AUC')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.legend()
    plt.savefig(f'{metrics_plot_name}.png')
    plt.show()


def auto_sklearn(features, labels, num_epochs):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, random_state=42)
    automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=30)
    automl.fit(X_train, y_train)
    y_pred = automl.predict(X_test)
    y_pred_rounded = np.round(y_pred)
    test_accuracy = accuracy_score(y_test, y_pred_rounded)

    return test_accuracy, automl


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


def load_labels(file_path):
    data = pd.read_csv(file_path+'subject-info.csv', sep=';')

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

        
def preprocess_tabular_data(file_path, test_size=0.2, random_state=42):

    # Load the dataset
    data_df = pd.read_csv(file_path+'subject-info.csv', sep=';')
    data_codes_df = pd.read_csv(file_path+'subject-info_codes.csv', sep=';', encoding='ISO-8859-1')
    one_hot_labels = load_labels(file_path)
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

    for col in X_dropped_dummies.select_dtypes(include=['bool']).columns:
        X_dropped_dummies[col] = X_dropped_dummies[col].astype(int)

    X_dropped_dummies.to_csv('after_dummy_creation.csv', index=False)

    imputer = KNNImputer()
    imputer.fit(X_dropped_dummies)
    Xtrans = imputer.transform(X_dropped_dummies)
    # print total missing
    print('Missing: %d' % sum(np.isnan(Xtrans).flatten()))

    final_cleaned_df = pd.DataFrame(Xtrans, columns=X_dropped_dummies.columns)
    final_cleaned_df.to_csv('after_imputation.csv', index=False)


    features = final_cleaned_df.drop('Cause of death', axis=1)


    # clf = RandomForestClassifier(random_state=42)
    # nn_model = simple_neural_net(features,labels)

    # auto_sklearn_accuracy, auto_sklearn_history = auto_sklearn(features, one_hot_labels, num_epochs=2)
    # print(f'Autosklearn Accuracy: {auto_sklearn_accuracy}')

    n_comps = None
    # explained_variance_ratio, eigenvals, pcs, pca_features = pca(features, num_components=n_comps, plot_bool=False)
 
    # print(f"PCA Features shape: {pca_features.shape}")
    # print("Explained Variance Ratio:", explained_variance_ratio)  # Variance explained by each component
    # print(f'Explained Variance Ratio Sum: {sum(explained_variance_ratio)}')

    nn_plot_name=f'simple_nn_metrics_{n_comps}'
    cnn_plot_name=f'cnn_metrics_{n_comps}'

    num_epochs = 10
    cnn_average_test_AUC, simple_nn_average_test_AUC, ensemble_average_test_AUC, cnn_history, simple_nn_history, test_index_arr = k_fold_cross_validation(features, one_hot_labels, num_epochs=num_epochs)

    print(f"Average CNN Test AUC: {cnn_average_test_AUC}")
    print(f"Average Simple NN Test AUC: {simple_nn_average_test_AUC}")
    print(f"Average Ensemble Test AUC: {ensemble_average_test_AUC}")

    # ADD VARIANCE FOR TOTAL
    # print(f'Explained Variance Ratio Sum: {sum(explained_variance_ratio)}')

    plot_metrics(model_history=cnn_history, metrics_plot_name=cnn_plot_name, model_name='CNN')
    plot_metrics(model_history=simple_nn_history, metrics_plot_name=nn_plot_name, model_name='Simple NN')


data_file_path = './data/'
# preprocess_tabular_data(data_file_path)
preprocess_tabular_data(data_file_path)



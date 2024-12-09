from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
# import autosklearn.classification
from lime import lime_tabular
import tensorflow as tf
import pandas as pd
import numpy as np


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

    return feature, importance


def auto_sklearn(features, labels, num_epochs):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, random_state=42)
    automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=30)
    automl.fit(X_train, y_train)
    y_pred = automl.predict(X_test)
    y_pred_rounded = np.round(y_pred)
    test_accuracy = accuracy_score(y_test, y_pred_rounded)

    return test_accuracy, automl


def pca(X, num_components, plot_bool):

    X_scaled = standard_scalar(X)

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


def plot_metrics(model_history, metrics_plot_name, model_name):
    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(model_history.history['loss'], label='Training Loss')
    plt.plot(model_history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{metrics_plot_name}_loss.png')
    plt.show()

    # Plot AUC
    plt.figure(figsize=(10, 5))
    auc_key = list(model_history.history.keys())[0]
    plt.plot(model_history.history[auc_key], label='Training AUC')
    plt.plot(model_history.history['val_' + auc_key], label='Validation AUC')
    plt.title(f'{model_name} AUC')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.legend()
    plt.savefig(f'{metrics_plot_name}_auc.png')
    plt.show()


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from lime import lime_tabular
import tensorflow as tf
import pandas as pd
import numpy as np



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


def pca(X, num_components, plot_bool):

    pca = PCA(n_components=num_components)
    X_pca = pca.fit_transform(X)

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


import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import models
from sklearn.model_selection import StratifiedKFold
from collections import Counter
import utils

def train_k_fold(features, labels, patience=3, num_splits=5, num_epochs=10, perform_shap='False'):

    feature_names = features.columns.tolist()

    labels_df = pd.DataFrame(labels)
    label_names = labels_df.columns.tolist()   

    # Initialize StratifiedKFold for classification tasks
    skf = StratifiedKFold(n_splits=num_splits, shuffle=True)

    AUC_scores = []
    feature_counts = Counter()

    for fold_num, (train_index, test_index) in enumerate(skf.split(features, np.argmax(labels, axis=1)), 1):
        # Split data into training and testing for this fold
        X_train, X_test = features.iloc[train_index], features.iloc[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        # feature_names = X_train.columns.tolist()
        X_train_arr = np.array(X_train)
        X_test_arr = np.array(X_test)

        # if CNN model:
        # Reshaping to (samples, timesteps, channels) OR (subjects, features, channels)
        # X_train_arr = X_train_arr.reshape((X_train_arr.shape[0], X_train_arr.shape[1], 1))
        # X_test_arr = X_test_arr.reshape((X_test_arr.shape[0], X_test_arr.shape[1], 1))
        # model = models.cnn(X_train_arr)

        model = models.deep_neural_net(X_train_arr)

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[tf.keras.metrics.AUC(multi_label=True, num_labels=4)])

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )

        print('TRAINING MODEL:')
        history = model.fit(X_train_arr, y_train, epochs=num_epochs, validation_split=0.2, callbacks=[early_stopping])

        y_pred = model.predict(X_test_arr)
        
        auc_metric = tf.keras.metrics.AUC(multi_label=True, num_labels=4)
        auc_metric.update_state(y_test, y_pred)
        auc_score = auc_metric.result().numpy()
        print(f"Test AUC: {auc_score}")
    
        AUC_scores.append(auc_score)

        important_features = utils.feature_importance(X_train_arr, X_test_arr, feature_names, label_names, model)
        
        # Update the counter with the features from this fold
        feature_counts.update(important_features)

    auc = np.mean(AUC_scores)

    sorted_feature_counts = feature_counts.most_common()

    return auc, history, sorted_feature_counts
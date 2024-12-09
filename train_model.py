import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping
import models

def train_k_fold(features, labels, num_epochs=10):
    features_df = pd.DataFrame(features)

    kf = KFold(n_splits=5, shuffle=True)

    cnn_AUC_scores = []
    
    for fold_num, (train_index, test_index) in enumerate(kf.split(features), 1):
        X_train, X_test = features_df.iloc[train_index], features_df.iloc[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        X_train_arr = np.array(X_train)
        X_test_arr = np.array(X_test)

        # Reshaping to (samples, timesteps, channels) OR (subjects, features, channels)
        X_train_cnn = X_train_arr.reshape((X_train_arr.shape[0], X_train_arr.shape[1], 1))
        X_test_cnn = X_test_arr.reshape((X_test_arr.shape[0], X_test_arr.shape[1], 1))
        
        cnn_model = models.cnn(X_train_cnn)

        cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[tf.keras.metrics.AUC()])

        early_stopping = EarlyStopping(
            monitor='val_loss',       # Can also monitor 'val_accuracy', 'loss', etc.
            patience=3,               # Number of epochs to wait for improvement
            restore_best_weights=True, # Restore the weights of the best model (i.e., the one with the lowest val_loss)
            verbose=1                  # Prints information about stopping
        )

        print('TRAINING CNN MODEL:')
        cnn_history = cnn_model.fit(X_train_cnn, y_train, epochs=num_epochs, validation_split=0.2, callbacks=[early_stopping])

        cnn_y_pred = cnn_model.predict(X_test_cnn)
        cnn_y_pred_train = cnn_model.predict(X_train_cnn)
        
        # print(f'Probabilities for first CNN train prediction (subjects 1-10): {cnn_y_pred_train[:10]}\n') # Corresponds to probabilities for first prediction
        # print(f'Labels for training data (subjects 1-10): {y_train}')

        y_train_ground_truth_df = pd.DataFrame(y_train)
        # y_train_ground_truth_df.to_csv(f'y_train_ground_truth_df_{fold_num}.csv')

        y_train_pred_df = pd.DataFrame(cnn_y_pred_train)
        # y_train_pred_df.to_csv(f'y_train_pred_df_{fold_num}.csv')

        y_train_ground_truth_counts = y_train_ground_truth_df.sum(axis=0)
        # print(f'The ground truth totals for each column: {y_train_ground_truth_counts}')

        cnn_new_y_pred = []

        for cnn_subject_prob in cnn_y_pred:
            # Convert each probability to 1 if it's >= 0.5, otherwise 0
            cnn_binary_subject_pred = [1 if subject_probability >= 0.5 else 0 for subject_probability in cnn_subject_prob]

            cnn_new_y_pred.append(cnn_binary_subject_pred)

        y_train_pred_df = pd.DataFrame(cnn_new_y_pred)
        y_pred_counts = y_train_pred_df.sum(axis=0)
        # print(f'The prediction for training totals for each column: {y_pred_counts}')

        predicted_probabilities = np.array(cnn_y_pred)
        # Create a one-hot encoded array
        one_hot_predictions = np.zeros_like(predicted_probabilities)

       # Print the one-hot encoded predictions
        # print(f"One-hot Encoded Predictions: {one_hot_predictions}")

        # For each sample, find the index of the max probability and set it to 1
        for i, prob in enumerate(predicted_probabilities):
            max_class = np.argmax(prob)  # Find index of the max probability
            one_hot_predictions[i, max_class] = 1  # Set that index to 1

        cnn_auc_metric = tf.keras.metrics.AUC()
        cnn_auc_metric.update_state(y_test, cnn_y_pred)
        cnn_auc_score = cnn_auc_metric.result().numpy()
        print(f"CNN Test AUC: {cnn_auc_score}")
    
        cnn_AUC_scores.append(cnn_auc_score)

    cnn_AUC = np.mean(cnn_AUC_scores)

    return cnn_AUC, cnn_history

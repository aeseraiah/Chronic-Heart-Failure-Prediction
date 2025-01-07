import numpy as np
import pandas as pd
import numpy as np
import train_model
import preprocess_data
import re
import utils
import tensorflow as tf
        
def main(file_path):

    df = pd.read_csv(file_path+'subject-info.csv', sep=';')
    print(f'Num of cols in original df: {len(df.columns)}, Num of subjects in original df: {len(df)}')
    # df.to_csv('original_df.csv', index=False)

    cleaned_df = preprocess_data.cleaning(df)
    print(f'Num of cols after cleaning: {len(cleaned_df.columns)}, Num of subjects after cleaning: {len(cleaned_df)}')
    cleaned_df.to_csv('cleaned_df. csv', index=False)

    df_after_one_hot, continuous_features, categorical_features = preprocess_data.columns_to_one_hot(cleaned_df)
    print(f'Num of cols after one hot encoding: {len(df_after_one_hot.columns)}, Num of subjects after one hot encoding: {len(df_after_one_hot)}')
    df_after_one_hot.to_csv('df_after_one_hot.csv', index=False)

    continuous_features_imputed_df, categorical_features_imputed_df = preprocess_data.impute_missing_values(continuous_features, categorical_features)
    continuous_features_imputed_df.to_csv('continuous_features_imputed_df.csv', index=False)
    categorical_features_imputed_df.to_csv('categorical_features_imputed_df.csv', index=False)

    one_hot_labels = preprocess_data.load_labels(file_path='continuous_features_imputed_df.csv')

    scaled_df = preprocess_data.standard_scalar(continuous_features_imputed_df, categorical_features_imputed_df)
    preprocessed_df = scaled_df
    
    preprocessed_df.to_csv('preprocessed_data.csv', index=False)
    features = preprocessed_df

    num_epochs = 100
    num_splits = 20
    patience = 3

    features = pd.DataFrame(features)

    average_test_AUC, history, sorted_feature_counts = train_model.train_k_fold(features, one_hot_labels, patience=patience, num_splits=num_splits, num_epochs=num_epochs, perform_shap='False')
    print(f"Average Test AUC: {average_test_AUC}")

    num_top_features = 10
    top_features_count = sorted_feature_counts[:num_top_features]

    features_list = []
    for feature, count in top_features_count:
        features_list.append(feature)
        print(f"{feature}: {count}")

    top_features_cleaned = [re.sub(r'\s*<=\s*[-+]?\d*\.?\d+', '', feature) for feature in features_list]
    print(top_features_cleaned)

    top_features = features[top_features_cleaned]

    num_splits = 5
    average_test_AUC, history, sorted_feature_counts = train_model.train_k_fold(top_features, one_hot_labels, patience=patience, num_splits=num_splits, num_epochs=num_epochs, perform_shap='False')
    print(f"Average Test AUC: {average_test_AUC}")

    utils.plot_metrics(model_history=history, metrics_plot_name=f'metrics', model_name='deep_neural_net')

if __name__ == '__main__':
    main(file_path='./data/')


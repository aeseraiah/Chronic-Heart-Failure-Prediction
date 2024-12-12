import numpy as np
import pandas as pd
import numpy as np
import train_model
import preprocess_data
import utils
        
def main(file_path):

    df = pd.read_csv(file_path+'subject-info.csv', sep=';')
    print(f'Num of cols in original df: {len(df.columns)}, Num of subjects in original df: {len(df)}')
    # df.to_csv('original_df.csv', index=False)

    cleaned_df = preprocess_data.cleaning(df)
    print(f'Num of cols after cleaning: {len(cleaned_df.columns)}, Num of subjects after cleaning: {len(cleaned_df)}')
    cleaned_df.to_csv('cleaned_df.csv', index=False)

    df_after_one_hot, continuous_features, categorical_features = preprocess_data.columns_to_one_hot(cleaned_df)
    print(f'Num of cols after one hot encoding: {len(df_after_one_hot.columns)}, Num of subjects after one hot encoding: {len(df_after_one_hot)}')
    df_after_one_hot.to_csv('df_after_one_hot.csv', index=False)

    continuous_features_imputed_df, categorical_features_imputed_df = preprocess_data.impute_missing_values(continuous_features, categorical_features)
    continuous_features_imputed_df.to_csv('continuous_features_imputed_df.csv', index=False)
    categorical_features_imputed_df.to_csv('categorical_features_imputed_df.csv', index=False)

    scaled_df = preprocess_data.standard_scalar(continuous_features_imputed_df, categorical_features_imputed_df)
    preprocessed_df = scaled_df
    
    preprocessed_df.to_csv('preprocessed_data.csv', index=False)

    features = preprocessed_df.drop('Cause of death', axis=1)
    one_hot_labels = preprocess_data.load_labels(file_path)

    n_comps = 5

    explained_variance_ratio, eigenvals, pcs, pca_features = utils.pca(features, num_components=n_comps, plot_bool=False)
    print(pca_features)
    print(f"PCA Features shape: {pca_features.shape}")
    print("Explained Variance Ratio:", explained_variance_ratio)  # Variance explained by each component
    print(f'Explained Variance Ratio Sum: {sum(explained_variance_ratio)}')

    num_epochs = 10
    cnn_average_test_AUC, cnn_history = train_model.train_k_fold(pca_features, one_hot_labels, num_epochs=num_epochs)

    print(f"Average CNN Test AUC: {cnn_average_test_AUC}")
    utils.plot_metrics(model_history=cnn_history, metrics_plot_name=f'cnn_metrics_{n_comps}_components', model_name='CNN')

if __name__ == '__main__':
    main(file_path='./data/')



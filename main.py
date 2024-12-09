import numpy as np
import pandas as pd
import numpy as np
import train_model
import preprocess_data
import utils
        
def main(file_path):

    df = pd.read_csv(file_path+'subject-info.csv', sep=';')

    cleaned_df = preprocess_data.cleaning(df)
    df_after_one_hot = preprocess_data.columns_to_one_hot(cleaned_df)
    imputed_df = preprocess_data.impute_missing_values(df_after_one_hot)
    

    # Total missing after KNN imputation, should be 0
    print('Missing: %d' % sum(np.isnan(imputed_df).flatten()))

    preprocessed_df = pd.DataFrame(imputed_df, columns=df_after_one_hot.columns)
    preprocessed_df.to_csv('preprocessed_data.csv', index=False)

    features = preprocessed_df.drop('Cause of death', axis=1)
    one_hot_labels = utils.load_labels(file_path)

    n_comps = None
    # explained_variance_ratio, eigenvals, pcs, pca_features = utils.pca(features, num_components=n_comps, plot_bool=False)
 
    # print(f"PCA Features shape: {pca_features.shape}")
    # print("Explained Variance Ratio:", explained_variance_ratio)  # Variance explained by each component
    # print(f'Explained Variance Ratio Sum: {sum(explained_variance_ratio)}')

    num_epochs = 2
    cnn_average_test_AUC, cnn_history = train_model.train_k_fold(features, one_hot_labels, num_epochs=num_epochs)

    print(f"Average CNN Test AUC: {cnn_average_test_AUC}")
    utils.plot_metrics(model_history=cnn_history, metrics_plot_name=f'cnn_metrics_{n_comps}_components', model_name='CNN')

if __name__ == '__main__':
    main(file_path='./data/')



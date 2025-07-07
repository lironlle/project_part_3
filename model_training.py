import pandas as pd
import pickle
from assets_data_prep import prepare_data
from sklearn.linear_model import ElasticNet

if __name__ == "__main__":
    df = pd.read_csv('train_data.csv')
    df_prepared = prepare_data(df, dataset_type='train')

    X = df_prepared.drop(columns='price').values
    y = df_prepared['price'].values

    print("DEBUG: X.shape =", X.shape)
    print("DEBUG: y.shape =", y.shape)

    model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
    model.fit(X, y)

    with open('trained_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("ElasticNet model saved!")

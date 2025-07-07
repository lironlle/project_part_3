import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

def prepare_data(df: pd.DataFrame, dataset_type='train') -> pd.DataFrame:
    df = df.copy()

    # סינון חריגים רק אם יש price
    if 'price' in df.columns:
        df = df[df['price'].notna()]
        lower = df['price'].quantile(0.01)
        df = df[(df['price'] >= lower) & (df['price'] <= 17000)]

    # המרת מרחקים (נניח שיש עמודה בשם distance_from_center)
    if 'distance_from_center' in df.columns:
        df.loc[df['distance_from_center'] > 100, 'distance_from_center'] /= 1000

    # טיוב קומות (מכניס אם לא קיימים)
    df['floor'] = df.get('floor', np.nan)
    df['total_floors'] = df.get('total_floors', np.nan)

    # ניסיון להשלים שכונה לפי כתובת (כרגע לא ממומש)
    if 'neighborhood' in df.columns and 'address' in df.columns:
        for i, row in df[df['neighborhood'].isna() & df['address'].notna()].iterrows():
            pass

    # הסרת עמודות לא רלוונטיות (אבל שים לב מה אתה מוחק!)
    columns_to_drop = [
        'description', 'address', 'num_of_images',
        'days_to_enter', 'num_of_payments',
        'garden_area', 'building_tax', 'handicap', 'has_bars'
    ]
    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    # טיפול בערכים חסרים
    if dataset_type == 'train':
        df.dropna(inplace=True)
        if df.empty:
            raise ValueError("⚠️ DataFrame ריק לאחר ניקוי - אין שורות לאימון.")
    else:
        df.fillna(0, inplace=True)

    # Target Encoding לקטגוריות
    categorical_cols = ['property_type', 'neighborhood']
    if dataset_type == 'train':
        category_means = {}
        for col in categorical_cols:
            means = df.groupby(col)['price'].mean()
            category_means[col] = means.to_dict()
            df[col + '_encoded'] = df[col].map(category_means[col])
        with open("category_means.pkl", "wb") as f:
            pickle.dump(category_means, f)
    else:
        with open("category_means.pkl", "rb") as f:
            category_means = pickle.load(f)
        for col in categorical_cols:
            df[col + '_encoded'] = df[col].map(category_means[col]).fillna(
                np.mean(list(category_means[col].values()))
            )

    df.drop(columns=categorical_cols, inplace=True, errors='ignore')

    # נרמול העמודות
    if dataset_type == 'train':
        feature_cols = df.drop(columns='price').columns
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[feature_cols])
        with open("train_columns.pkl", "wb") as f:
            pickle.dump(feature_cols, f)
        with open("scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        df_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=df.index)
        df_scaled["price"] = df["price"].values
    else:
        with open("train_columns.pkl", "rb") as f:
            feature_cols = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0

        df = df[feature_cols]
        if df.empty:
            raise ValueError("⚠️ DataFrame ריק לאחר השלמת עמודות - אין שורות לחיזוי.")

        X_scaled = scaler.transform(df)
        df_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=df.index)

    print(f"✅ prepare_data סיים. צורת הקלט: {df_scaled.shape}")
    return df_scaled

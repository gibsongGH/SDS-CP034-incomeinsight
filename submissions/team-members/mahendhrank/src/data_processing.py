import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def load_data(filepath='data/adult.csv'):
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.replace('.', '-', regex=False)
    df = df.drop(columns=["fnlwgt"])
    # Replace '?' with NaN
    df.replace('?', np.nan, inplace=True)
    # Drop rows with missing values for simplicity
    #df.dropna(inplace=True)
    return handle_missingValues(df)

def handle_missingValues(df):
    df_drop = df.dropna()
    # Original row count
    n_original = len(df)
    # cleaned after Drop rows with any missing values
    n_clean = len(df_drop)
    # Calculate retention
    retained_pct = n_clean / n_original * 100
    loss_pct = (n_original - n_clean) / n_original * 100 
    if  loss_pct > 5 :
        #print(f"Data Loss is {loss_pct:.2f}%. Since > 5%, dropping the rows is NOT recommended, instead use Imputer") 
        return simple_imputer(df)
    else:
        return df_drop

def simple_imputer(df):
    # As identiifed all the ? columns are categorical columns, we can replace them with the mode of the column
    missed_categorical_columns = ['workclass', 'occupation', 'native-country']
    categorical_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    categorical_imputer.fit(df[missed_categorical_columns])
    org_df = df.copy() #for count purpose
    df[missed_categorical_columns] = categorical_imputer.transform(df[missed_categorical_columns])
    print(df.isna().sum())
    return df

def encode_features(df):
    # Encode categorical features
    categorical_cols = df.select_dtypes(include='object').columns
    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le
    return df, le_dict

def scale_features(df, numeric_cols):
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df, scaler

import pickle

def save_preprocessors(le_dict, scaler, feature_names, path="src/preprocessors.pkl"):
    """Save label encoders, scaler, and feature order to a pickle file."""
    with open(path, "wb") as f:
        pickle.dump({
            "label_encoders": le_dict,
            "scaler": scaler,
            "features": feature_names
        }, f)
    print(f"✅ Preprocessors saved at {path}")


if __name__ == "__main__":
    df = load_data('data/adult.csv')
    df, le_dict = encode_features(df)
    numeric_cols = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    df, scaler = scale_features(df, numeric_cols)
    print(df.head())

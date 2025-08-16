
import pandas as pd
from joblib import dump
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder, 
    StandardScaler,
    MinMaxScaler,
    FunctionTransformer, 
    OrdinalEncoder, 
    PolynomialFeatures
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin 

# ------------------------
# Custom Transformer
# ------------------------
class EducationBinner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # X will be a 2D array or DataFrame, get the first column
        X = pd.Series(X.ravel())
        return X.apply(self._bin_education).to_frame()
    
    def _bin_education(self, edu_num):
        if edu_num <= 4:
            return 'PreHighSchool'
        elif edu_num <= 7:
            return 'IncompleteHS'
        elif edu_num <= 10:
            return 'HighSchool'
        elif edu_num <= 12:
            return 'Associate'
        elif edu_num == 13:
            return 'Bachelors'
        else:
            return 'Advanced'
        
# ------------------------
# Binary Mapping Function
# ------------------------
def map_binary_cols(df):
    df = df.copy()
    df['sex'] = df['sex'].map({'Male': 1, 'Female': 0})
    df['income'] = df['income'].map({'>50K': 1, '<=50K': 0})
    df['married_together'] = df['marital_status'].map({
        'Married-AF-spouse': 1,
        'Married-civ-spouse': 1
    }).fillna(0)
    df['has_capital_gain'] = (df['capital_gain'] > 0).astype(int)
    return df

# ------------------------
# Preprocessor Builder
# ------------------------
def build_preprocessor():
    # Define column groups
    # cat_interact = ['marital_status', 'occupation', 'workclass']
    numeric_interact = ['hours_per_week', 'education_num', 'age', 'capital_gain']
    onehot_cols = ['native.country', 'race']
    edu_num_col = ['education_num']
    normal_cols = ['age', 'hours_per_week', 'education_num']
    skewed_col = ['capital_gain']
    bin_flags = ['income', 'sex', 'marital_status']

    # Pipelines
    #cat_interact_pipeline = Pipeline([
    #    ('onehot', OneHotEncoder(drop='first')),
    #    ('interact', PolynomialFeatures(interaction_only=True, include_bias=False))
    #])

    numeric_interact_pipeline = Pipeline([
        ('interact', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False))
    ])

    edu_pipeline = Pipeline([
        ('bin', EducationBinner()),
        ('ord', OrdinalEncoder(categories=[[
            'PreHighSchool', 'IncompleteHS', 'HighSchool', 
            'Associate', 'Bachelors', 'Advanced'
        ]]))
    ])

    onehot_pipeline = OneHotEncoder(drop='first')

    normal_pipeline = StandardScaler()
    skewed_pipeline = MinMaxScaler()

    # ColumnTransformer
    preprocessor = ColumnTransformer(transformers=[
        ('binary_flags', FunctionTransformer(map_binary_cols, validate=False), bin_flags),
        #('interact', cat_interact_pipeline, cat_interact),
        ('edu_bin_ord', edu_pipeline, edu_num_col),
        ('onehot', onehot_pipeline, onehot_cols),
        ('normal', normal_pipeline, normal_cols),
        ('skewed', skewed_pipeline, skewed_col),
        ('interact_num', numeric_interact_pipeline, numeric_interact)
    ], remainder='passthrough')  # To keep binary columns like sex, has_capital_gain, etc.

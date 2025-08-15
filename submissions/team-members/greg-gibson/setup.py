from setuptools import setup, find_packages

setup(
    name="my_pipeline",          # package name (whatever you like)
    version="0.1.0",              # version string
    packages=find_packages(),     # auto-detects 'pipeline' package
    install_requires=[
        "pandas",
        "scikit-learn",
        "xgboost",
        "mlflow",
        "joblib"
    ],
    python_requires=">=3.8",
)
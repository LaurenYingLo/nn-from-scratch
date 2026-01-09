import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def make_dataset(n_samples=2000, noise=0.25, random_state=42):
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    # make_moons: generates a 2D binary classification dataset with a “two moons” shape.
    # This dataset is non-linearly separable, making it ideal to demonstrate the advantage of neural networks.
    X = X.astype(np.float64)#X: shape (n_samples, 2) – 2D coordinates (x1, x2)
    y = y.astype(np.int64)#y: shape (n_samples,) – binary labels {0,1}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )
    #Splits data into train/test sets
    #test_size=0.25: 25% test data
    #stratify=y: stratified sampling, preserving class balance (a professional detail) 分層抽樣，確保訓練集與測試集的 0/1 比例一致
    #random_state: reproducible split

    scaler = StandardScaler()#standardizes features to zero mean and unit variance, improving training stability and speed.
    X_train = scaler.fit_transform(X_train)
    #fit_transform(X_train): compute statistics from training data only
    X_test = scaler.transform(X_test)
    #transform(X_test): apply the same statistics to test data

    return X_train, X_test, y_train, y_test

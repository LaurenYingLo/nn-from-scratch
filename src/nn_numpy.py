import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_grad(a):
    return a * (1.0 - a)
#a=sigmoid(z),da/dz=a(1-a)

class TwoLayerNN:
    """
    2-layer NN: x -> (W1,b1) -> sigmoid -> (W2,b2) -> sigmoid -> y_hat
    binary classification with BCE loss
    """
    def __init__(self, in_dim, hidden_dim=16, seed=42):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.5, size=(in_dim, hidden_dim))
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = rng.normal(0, 0.5, size=(hidden_dim, 1))
        self.b2 = np.zeros((1, 1))

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = sigmoid(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = sigmoid(z2)
        cache = (X, z1, a1, z2, a2)
        return a2, cache

    def bce_loss(self, y_hat, y): #Binary Cross Entropy
        y = y.reshape(-1, 1)
        eps = 1e-9
        return -np.mean(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))

    def backward(self, cache, y):
        X, z1, a1, z2, a2 = cache
        y = y.reshape(-1, 1)
        N = X.shape[0]

        dZ2 = (a2 - y) / N
        dW2 = a1.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * sigmoid_grad(a1)
        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        return (dW1, db1, dW2, db2)

    def step(self, grads, lr=0.1):
        dW1, db1, dW2, db2 = grads
        self.W1 -= lr * dW1 #learning rate
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

    def predict_proba(self, X):
        y_hat, _ = self.forward(X)
        return y_hat

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int).reshape(-1)
    # ŷ >= 0.5 → 1
    # ŷ < 0.5 → 0

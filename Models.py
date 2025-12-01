import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


class RandomForestModel:
    def __init__(self,
                 n_estimators=500,
                 max_depth=15,
                 random_state=42,
                 n_jobs=-1):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=n_jobs
        )
        self.trained = False

    # Train the model on original target values
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        self.trained = True

    # Predict on original scale
    def predict(self, X):
        if not self.trained:
            raise ValueError("Model must be trained first")
        return self.model.predict(X)

    # Evaluate on original scale
    def evaluate(self, X_test, y_test, return_preds=False):
        if not self.trained:
            raise ValueError("Model must be trained first")

        y_pred = self.model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        if return_preds:
            return mse, r2, y_test, y_pred
        return mse, r2

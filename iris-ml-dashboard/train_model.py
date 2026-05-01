import sklearn
print("sklearn version:", sklearn.__version__)

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
import warnings

# Suppress convergence warnings (you should still be aware of them)
warnings.filterwarnings("ignore")

# 1. Load data
X, y = load_iris(return_X_y=True)

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Models
models = {
    "logistic_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=500, random_state=42))
    ]),
    "svm": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(probability=True))
    ]),
    "random_forest": RandomForestClassifier(
        random_state=42, n_estimators=100
    ),
    "neural_network": Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPClassifier(
            hidden_layer_sizes=(100,),
            max_iter=1000,
            random_state=42
        ))
    ])
}

# 4. Directory
os.makedirs("saved_models", exist_ok=True)

# 5. Train, evaluate, save
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    
    results[name] = accuracy
    print(f"{name:20}: {accuracy:.4f}")
    
    joblib.dump(model, os.path.join("saved_models", f"{name}.pkl"))

print("\nSaved models:", os.listdir("saved_models"))
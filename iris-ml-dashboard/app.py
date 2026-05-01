import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import pandas as pd

# Load dataset
iris = load_iris()
feature_names = iris.feature_names
target_names = iris.target_names

# Load scaler
scaler = joblib.load("scaler.pkl")

# Load models
models = {
    "Logistic Regression": joblib.load("Logistic Regression.pkl"),
    "SVM": joblib.load("SVM.pkl"),
    "Random Forest": joblib.load("Random Forest.pkl"),
    "Neural Network": joblib.load("Neural Network.pkl")
}

# Title
st.title("🌸 Iris Flower Classification Dashboard")

# Sidebar
st.sidebar.header("Input Features")

sepal_length = st.sidebar.slider("Sepal Length", 4.0, 8.0, 5.1)
sepal_width = st.sidebar.slider("Sepal Width", 2.0, 4.5, 3.5)
petal_length = st.sidebar.slider("Petal Length", 1.0, 7.0, 1.4)
petal_width = st.sidebar.slider("Petal Width", 0.1, 2.5, 0.2)

input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
input_scaled = scaler.transform(input_data)

# Model selection
model_choice = st.selectbox("Select Model", list(models.keys()))
model = models[model_choice]

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]

    st.subheader("Prediction Result")
    st.success(f"Predicted Species: {target_names[prediction]}")

    st.subheader("Prediction Probabilities")
    for i, prob in enumerate(probabilities):
        st.write(f"{target_names[i]}: {prob:.2f}")

# Visualization
st.subheader("📊 Feature Visualization")

df = pd.DataFrame(iris.data, columns=feature_names)
df['species'] = iris.target

fig, ax = plt.subplots()
sns.scatterplot(
    x=df[feature_names[2]], 
    y=df[feature_names[3]], 
    hue=df['species'], 
    palette='viridis', 
    ax=ax
)

# Plot user input point
ax.scatter(petal_length, petal_width, color='red', s=100, label='Your Input')
ax.legend()

st.pyplot(fig)

# Model comparison (hardcoded from your results)
st.subheader("📈 Model Comparison")

comparison_data = pd.DataFrame({
    "Model": ["Logistic Regression", "SVM", "Random Forest", "Neural Network"],
    "Accuracy": [1.0, 0.97, 1.0, 0.98]
})

st.bar_chart(comparison_data.set_index("Model"))

# Feature Importance (Random Forest only)
if model_choice == "Random Forest":
    st.subheader("🌟 Feature Importance")
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    st.bar_chart(importance_df.set_index("Feature"))
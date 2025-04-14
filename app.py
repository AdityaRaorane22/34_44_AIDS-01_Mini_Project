import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Load model
with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🌍 Ozone Layer Detection - AI Classifier")
st.markdown("Enter your environmental data (T85, T70, TT, KI, SLP) and predict ozone damage levels using a trained XGBoost model.")

# Input fields for parameters
T85 = st.number_input("T85", min_value=-1000000.0, max_value=1000000.0, step=0.1, value=0.0)
T70 = st.number_input("T70", min_value=-1000000.0, max_value=1000000.0, step=0.1, value=0.0)
TT = st.number_input("TT", min_value=-1000000.0, max_value=1000000.0, step=0.1, value=0.0)
KI = st.number_input("KI", min_value=-1000000.0, max_value=1000000.0, step=0.1, value=0.0)
SLP = st.number_input("SLP", min_value=-1000000.0, max_value=1000000.0, step=0.1, value=0.0)

# Button to trigger prediction
if st.button("Predict"):
    # Create a DataFrame from the input values
    input_data = pd.DataFrame([[T85, T70, TT, KI, SLP]], columns=['T85', 'T70', 'TT', 'KI', 'SLP'])

    # Prediction
    prediction = model.predict(input_data)
    pred_proba = model.predict_proba(input_data)

    st.subheader("🧪 Prediction Result")
    st.write(f"Predicted Result: {prediction[0]}")
    st.write(f"Probability of Class 1: {pred_proba[0][1]:.2f}")

    # Visualizations: Plotting the probabilities
    fig, ax = plt.subplots()
    ax.bar([0, 1], pred_proba[0], color=['red', 'green'])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Class 0', 'Class 1'])
    ax.set_ylabel('Probability')
    ax.set_title('Prediction Probabilities')
    st.pyplot(fig)

    # Optional: Provide classification report and confusion matrix for model analysis
    # Assuming you have actual labels to compare (in practice, you might use a real dataset)
    # Here I will just assume it's a demo scenario:
    y_true = np.array([1])  # Example true label for testing
    cm = confusion_matrix(y_true, prediction)
    report = classification_report(y_true, prediction, output_dict=True)

    st.subheader("📊 Confusion Matrix")
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap="YlGnBu", ax=ax_cm)
    st.pyplot(fig_cm)

    st.subheader("📋 Classification Report")
    st.write(pd.DataFrame(report).transpose())

    st.success("✅ Prediction complete!")

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.title("Simple Linear Regression - Salary Prediction")
st.header("Step 1: Load and Display Data")

# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file is not None:
    # Read CSV
    data = pd.read_csv(uploaded_file)
    st.dataframe(data)

    st.header("Step 2: Visualize Salary vs Experience")
    fig, ax = plt.subplots()
    ax.scatter(data["YearsExperience"], data["Salary"], color="blue")
    ax.set_xlabel("Years of Experience")
    ax.set_ylabel("Salary")
    ax.set_title("Scatter Plot")
    st.pyplot(fig)

    st.header("Step 3: Train Linear Regression Model")

    x = data[["YearsExperience"]]
    y = data["Salary"]

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Create and train the model
    model = LinearRegression()
    model.fit(x_train, y_train)
    st.success("Model Trained Successfully!")

    st.header("Step 4: Visualize Regression Line")
    
    fig2, ax2 = plt.subplots()
    ax2.scatter(x, y, color='blue', label='Actual Data')
    ax2.plot(x, model.predict(x), color='red', label='Regression Line')
    ax2.set_xlabel("Years of Experience")
    ax2.set_ylabel("Salary")
    ax2.set_title("Regression Line vs Actual Data")
    ax2.legend()
    st.pyplot(fig2)

    st.header("Step 5: Predict Salary")

    # ✅ Corrected number_input usage
    experience = st.number_input("Enter years of experience", min_value=0.0, step=0.1)

    if experience:
        input_df = pd.DataFrame([[experience]], columns=["YearsExperience"])
        predicted_salary = model.predict(input_df)[0]
        st.success(f"Predicted Salary for {experience} years of experience is ₹{predicted_salary:,.2f}")

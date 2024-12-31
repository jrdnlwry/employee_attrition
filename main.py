import streamlit as st
import pandas as pd
import numpy as np
import joblib
from ISLP import confusion_table
from sklearn.metrics import accuracy_score

def main():
    st.title("Machine Learning App with Streamlit")
    st.write("This app will automatically load a CSV file from the specified location, show a sample row, run a pre-trained model, and display the confusion matrix and accuracy score.")
    st.write("Please note: the response column as indicated by AttrValue is removed prior to any predictions being made.")

    # 1. Load the CSV from a known location
    csv_path = "test_set.csv" 
    st.write(f"Loading CSV file from: {csv_path}")

    # 2. Read the CSV into a DataFrame
    try:
        df = pd.read_csv(csv_path)
        df = df.loc[:, df.columns != 'Unnamed: 0']
    except FileNotFoundError:
        st.error(f"Could not find the file '{csv_path}'. Please check the path.")
        return
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return

    st.subheader("DataFrame Preview")
    st.write(df.head())

    # 3. Let the user pick a row index to see sample data
    st.subheader("Select a row index to see sample data")
    sample_idx = st.number_input(
        "Row index", 
        min_value=0, 
        max_value=len(df) - 1, 
        value=0, 
        step=1
    )

    # 4. Display the chosen sample
    st.write("Sample row data:")
    st.write(df.iloc[sample_idx])

    # 5. Load the pre-trained model from a .joblib file
    model_path = "svmEmployee.joblib"  # <-- Replace with the correct path/filename for your model
    try:
        model = joblib.load(model_path)
        st.success(f"Model loaded successfully from {model_path}")
    except FileNotFoundError:
        st.error(f"Could not find the file '{model_path}'. Please check the path.")
        return
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return

    # ------------------------------------------------------------------------
    # Uncomment and modify the lines below to perform predictions and
    # evaluate your model on the loaded CSV. Be sure to adjust 'target_column'
    # to match the actual name of your target in the CSV.


    XtestDF = df.iloc[:,:-1]
    YtestDF = df.iloc[:,-1]

    # if "target_column" not in df.columns:
    #     st.error("Please ensure your CSV has a column named 'target_column' for the target.")
    #     return
    
    # X = df.drop(columns=["target_column"])
    # y = df["target_column"]

    predictions = model.predict(XtestDF)

    cm = confusion_table(predictions, YtestDF)
    # accuracyPred = np.sum(predictTest == YtestDF) / len(predictTest)
    acc = np.sum(predictions == YtestDF) / len(predictions)

    st.subheader("Confusion Matrix")
    st.write(cm)

    st.subheader("Accuracy Score")
    st.write(f"{acc:.2f}")
    # ------------------------------------------------------------------------

    st.subheader("Summary")
    st.write("Out of the spreadsheet of 441 employees the model correctly predicts that 376 employees have not quit.")
    st.write("Whereas the model correctly predicts that out of 441 employees 65 have quit.")

    # 4. Filter rows where prediction == 0 (employees who will quit)
    df_quit = df[df["AttrValue"] == 0]

    # 5. Count how many times each performance score appears among those who will quit
    quit_score_counts = df_quit["PerformanceRating"].value_counts()

    st.subheader("Number of Employees Predicted to Quit by Performance Score")
    # 6. Streamlit bar chart
    st.bar_chart(quit_score_counts)


if __name__ == "__main__":
    main()
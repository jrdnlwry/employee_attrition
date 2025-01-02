import streamlit as st
import pandas as pd
import numpy as np
import joblib
from ISLP import confusion_table
from sklearn.metrics import accuracy_score

def main():
    st.title("Employee Retention Predictor: Identifying Likelihood of Attrition Using SVM")
    st.write("""This model was built using a Support Vector Machine model.
             This is a type of machine learning model used to make decisions or predictions by identifying patterns in data.
             You can think of it as a very smart line-drawing tool that helps classify things into different groups.""")
    
    st.write("Example scenario:")

    st.write("""You’re sorting apples and oranges on a table. 
             The goal is to separate them with a straight line so all the apples are on one side, and all the oranges are on the other.
            The SVM does something similar, but instead of sorting fruits, it separates groups based on their characteristics.""")

    # 1. Load the CSV from a known location
    csv_path = "test_set.csv" 
    # st.write(f"Loading CSV file from: {csv_path}")

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

    st.subheader("Data used for the prediction")
    st.write("""Below is a preview of the data used to build the prediction model.
                The data set consist of the predictors used along with the actual value the model is trying to predict called AttrValue.""")
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
    st.write("""This is like a scorecard that helps us see how well a prediction model is performing.
                Imagine you’re a teacher grading a quiz where the answers are either "yes" or "no", 
                and you want to compare what your students guessed (predictions) to the correct answers (reality).""")
    
    st.write("""In our case, when we look at the 0, 0 columns we can get a sense of how frequently the model accurately predicts an employee will not quit.
                Conversely, 1, 1 column and row tells us how frequently the model predicts an employee will quit.""")
    st.write(cm)

    st.subheader("Accuracy Score")
    st.write(f"{acc:.2f}")
    # ------------------------------------------------------------------------
    st.subheader("Generate Random Employee Data")


    # Initialize session state
    if "rand_df" not in st.session_state:
        st.session_state.rand_df = None

    if st.button("Generate Random Data"):
        randData = {
            "Age": np.random.randint(min(XtestDF["Age"]), max(XtestDF["Age"]), size=1)[0],
            "DistanceFromHome": np.random.randint(min(XtestDF["DistanceFromHome"]), max(XtestDF["DistanceFromHome"]), size=1)[0],
            "Education": np.random.randint(min(XtestDF["Education"]), max(XtestDF["Education"]), size=1)[0],
            "EmployeeCount": 1,
            "JobLevel": np.random.randint(min(XtestDF["JobLevel"]), max(XtestDF["JobLevel"]), size=1)[0],
            "MonthlyIncome": np.random.randint(min(XtestDF["MonthlyIncome"]), max(XtestDF["MonthlyIncome"]), size=1)[0],
            "NumCompaniesWorked": np.random.randint(min(XtestDF["NumCompaniesWorked"]), max(XtestDF["NumCompaniesWorked"]), size=1)[0],
            "PercentSalaryHike": np.random.randint(min(XtestDF["PercentSalaryHike"]), max(XtestDF["PercentSalaryHike"]), size=1)[0],
            "StandardHours": 8,
            "StockOptionLevel": np.random.randint(min(XtestDF["StockOptionLevel"]), max(XtestDF["StockOptionLevel"]), size=1)[0],
            "TotalWorkingYears": np.random.randint(min(XtestDF["TotalWorkingYears"]), max(XtestDF["TotalWorkingYears"]), size=1)[0],
            "TrainingTimesLastYear": np.random.randint(min(XtestDF["TrainingTimesLastYear"]), max(XtestDF["TrainingTimesLastYear"]), size=1)[0],
            "YearsAtCompany": np.random.randint(min(XtestDF["YearsAtCompany"]), max(XtestDF["YearsAtCompany"]), size=1)[0],
            "YearsSinceLastPromotion": np.random.randint(min(XtestDF["YearsSinceLastPromotion"]), max(XtestDF["YearsSinceLastPromotion"]), size=1)[0],
            "YearsWithCurrManager": np.random.randint(min(XtestDF["YearsWithCurrManager"]), max(XtestDF["YearsWithCurrManager"]), size=1)[0],
            "EnvironmentSatisfaction": np.random.randint(min(XtestDF["EnvironmentSatisfaction"]), max(XtestDF["EnvironmentSatisfaction"]), size=1)[0],
            "JobSatisfaction": np.random.randint(min(XtestDF["JobSatisfaction"]), max(XtestDF["JobSatisfaction"]), size=1)[0],
            "WorkLifeBalance": np.random.randint(min(XtestDF["WorkLifeBalance"]), max(XtestDF["WorkLifeBalance"]), size=1)[0],
            "JobInvolvement": np.random.randint(min(XtestDF["JobInvolvement"]), max(XtestDF["JobInvolvement"]), size=1)[0],
            "PerformanceRating": np.random.randint(min(XtestDF["PerformanceRating"]), max(XtestDF["PerformanceRating"]), size=1)[0],
            "BusinessTravel_Non-Travel": np.random.randint(min(XtestDF["BusinessTravel_Non-Travel"]), max(XtestDF["BusinessTravel_Non-Travel"]), size=1)[0],
            "BusinessTravel_Travel_Frequently": np.random.randint(min(XtestDF["BusinessTravel_Travel_Frequently"]), max(XtestDF["BusinessTravel_Travel_Frequently"]), size=1)[0],
            "BusinessTravel_Travel_Rarely": np.random.randint(min(XtestDF["BusinessTravel_Travel_Rarely"]), max(XtestDF["BusinessTravel_Travel_Rarely"]), size=1)[0],
            "Gender_Female": np.random.randint(min(XtestDF["Gender_Female"]), max(XtestDF["Gender_Female"]), size=1)[0],
            "Gender_Male": np.random.randint(min(XtestDF["Gender_Male"]), max(XtestDF["Gender_Male"]), size=1)[0],
            "Department_Human Resources": np.random.randint(min(XtestDF["Department_Human Resources"]), max(XtestDF["Department_Human Resources"]), size=1)[0],
            "Department_Research & Development": np.random.randint(
                min(XtestDF["Department_Research & Development"]), max(XtestDF["Department_Research & Development"]), size=1)[0],
            "Department_Sales": np.random.randint(min(XtestDF["Department_Sales"]), max(XtestDF["Department_Sales"]), size=1)[0],
        }

        st.session_state.rand_df = pd.DataFrame([randData])


    # Display DataFrame if it exists
    if st.session_state.rand_df is not None:
        st.write("Generated Random Data:")
        st.dataframe(st.session_state.rand_df)


    # Predict with Random Data
    if st.session_state.rand_df is not None and st.button("Make prediction"):
        try:
            mOutput = model.predict(st.session_state.rand_df)
            if mOutput[0] == 1:
                st.success("Prediction: Employee will quit")
            else:
                st.success("Prediction: Employee will not quit")
        except Exception as e:
            st.error(f"Error in prediction: {e}")



    st.subheader("Summary")
    st.write("""Employee attrition is a significant challenge for organizations worldwide, directly impacting operational efficiency, productivity, and financial stability. High attrition rates can lead to increased costs associated with recruitment, training, and onboarding of new employees while causing disruptions in team dynamics and overall organizational morale.""")
    st.write("""Predicting employee attrition has become a critical area of focus, leveraging machine learning techniques to understand the factors contributing to employee turnover. By identifying employees who are at risk of leaving, organizations can proactively implement retention strategies, optimize workforce planning, and maintain a competitive edge.""")
    st.write("""This project utilizes the **Employee Attrition Dataset**, which provides detailed information on various factors influencing employee turnover, such as demographic information, job roles, work-life balance, performance ratings, and compensation details.""")
    st.write("""The goal of this project is to develop a predictive machine learning model to identify employees who are most likely to leave the organization. Insights generated from the model will provide actionable recommendations for HR departments to improve employee engagement, refine policies, and allocate resources effectively to retain top talent.""")

    # 4. Filter rows where prediction == 0 (employees who will quit)
    df_quit = df[df["AttrValue"] == 0]

    # 5. Count how many times each performance score appears among those who will quit
    quit_score_counts = df_quit["PerformanceRating"].value_counts()




if __name__ == "__main__":
    main()
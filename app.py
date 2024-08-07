import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.experimental import enable_iterative_imputer  # Add this import
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle
import io

# 1. Greet the user
st.title("Welcome to the Machine Learning Application")
st.write("This application allows you to build and evaluate machine learning models using your own data or example datasets.")

# 2. Ask the user for data input method
data_choice = st.sidebar.radio("Would you like to upload your own data or use an example dataset?", ('Upload Data', 'Use Example Data'))

# 3. If user selects to upload data
if data_choice == 'Upload Data':
    uploaded_file = st.sidebar.file_uploader("Upload your dataset (csv, xlsx, tsv)", type=['csv', 'xlsx', 'tsv'])
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.tsv'):
            df = pd.read_csv(uploaded_file, sep='\t')
else:
    # 4. If user wants to use example data
    dataset_name = st.sidebar.selectbox("Select a dataset", ['titanic', 'tips', 'iris'])
    df = sns.load_dataset(dataset_name)

# 5. Print basic data information
if df is not None:
    st.write("### Data Information")
    st.write("#### Data Head")
    st.write(df.head())
    st.write("#### Data Shape")
    st.write(df.shape)
    st.write("#### Data Description")
    st.write(df.describe())
    st.write("#### Data Info")
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    st.text(info_str)
    st.write("#### Column Names")
    st.write(df.columns)

    # 6. Ask for features and target
    feature_cols = st.multiselect("Select features", df.columns)
    target_col = st.selectbox("Select target", df.columns)

    if feature_cols and target_col:
        X = df[feature_cols]
        y = df[target_col]

        # 7. Identify problem type
        if pd.api.types.is_numeric_dtype(y):
            problem_type = "regression" if y.nunique() > 10 else "classification"
        else:
            problem_type = "classification"
        st.write(f"This is a {problem_type} problem.")

        # 8. Pre-process the data
        st.write("### Data Pre-processing")

        # Encode categorical variables
        encoders = {}
        for col in X.columns:
            if pd.api.types.is_categorical_dtype(X[col]) or X[col].dtype == object:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                encoders[col] = le

        # Handle missing values
        imputer = IterativeImputer()
        X_imputed = imputer.fit_transform(X)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)
        
        # 9. Train-test split
        test_size = st.slider("Select train-test split size", 0.1, 0.5, 0.2)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)
        
        # 10. Model selection
        st.write("### Model Selection")
        if problem_type == "regression":
            model_choice = st.sidebar.selectbox("Select a model", ['Linear Regression', 'Decision Tree Regressor', 'Random Forest Regressor', 'Support Vector Regressor'])
        else:
            model_choice = st.sidebar.selectbox("Select a model", ['Logistic Regression', 'Decision Tree Classifier', 'Random Forest Classifier', 'Support Vector Classifier'])
        
        # 11. Train the model
        if problem_type == "regression":
            if model_choice == 'Linear Regression':
                model = LinearRegression()
            elif model_choice == 'Decision Tree Regressor':
                model = DecisionTreeRegressor()
            elif model_choice == 'Random Forest Regressor':
                model = RandomForestRegressor()
            elif model_choice == 'Support Vector Regressor':
                model = SVR()
        else:
            if model_choice == 'Logistic Regression':
                model = LogisticRegression(max_iter=200)
            elif model_choice == 'Decision Tree Classifier':
                model = DecisionTreeClassifier()
            elif model_choice == 'Random Forest Classifier':
                model = RandomForestClassifier()
            elif model_choice == 'Support Vector Classifier':
                model = SVC(probability=True)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # 12. Evaluate the model
        st.write("### Model Evaluation")
        if problem_type == "regression":
            mse = mean_squared_error(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            st.write(f"Mean Squared Error: {mse}")
            st.write(f"Root Mean Squared Error: {rmse}")
            st.write(f"Mean Absolute Error: {mae}")
            st.write(f"R2 Score: {r2}")
        else:
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            cm = confusion_matrix(y_test, y_pred)
            st.write(f"Accuracy: {accuracy}")
            st.write(f"Precision: {precision}")
            st.write(f"Recall: {recall}")
            st.write(f"F1 Score: {f1}")
            st.write(f"Confusion Matrix: \n {cm}")
        
        # 14. Highlight best model (Note: Simplified as only one model is used)
        st.write("### Best Model")
        st.write(f"The selected model is: {model_choice}")
        
        # 15. Option to download the model
        if st.button("Download Model"):
            with open('model.pkl', 'wb') as f:
                pickle.dump(model, f)
            st.write("Model downloaded as model.pkl")
        
        # 16. Option to make predictions
        if st.button("Make Predictions"):
            st.write("Upload data to make predictions")
            pred_file = st.file_uploader("Upload your dataset for predictions (csv, xlsx, tsv)", type=['csv', 'xlsx', 'tsv'])
            if pred_file is not None:
                if pred_file.name.endswith('.csv'):
                    pred_df = pd.read_csv(pred_file)
                elif pred_file.name.endswith('.xlsx'):
                    pred_df = pd.read_excel(pred_file)
                elif pred_file.name.endswith('.tsv'):
                    pred_df = pd.read_csv(pred_file, sep='\t')

                # Encode categorical variables for prediction data
                for col in encoders:
                    le = encoders[col]
                    pred_df[col] = le.transform(pred_df[col])
                
                # Pre-process the prediction data
                pred_df_imputed = imputer.transform(pred_df)
                pred_df_scaled = scaler.transform(pred_df_imputed)
                
                # Make predictions
                predictions = model.predict(pred_df_scaled)
                st.write("Predictions: ")
                st.write(predictions)

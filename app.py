import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from imblearn.combine import SMOTETomek
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_predict, KFold, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import plotly.express as px
from streamlit_option_menu import option_menu
import joblib
import os

st.title("Water Quality Monitoring System")
st.subheader("Analyze water quality and predict potability using machine learning.")

page = option_menu(
    menu_title= None,
    options =  ['Dataset Overview','Preprocessing','Prediction'],
    orientation='horizontal'
)

# File Upload Section
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your water quality CSV file:", type=["csv"])


if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data_cl = data.dropna()
    output = data_cl['Potability']
    corrs = data_cl.corrwith(output)[:-1]
    data_cl_ = data_cl.copy()
    best_features = list(corrs.sort_values(ascending=False)[1:].index[:5])

    mm = MinMaxScaler()
    for i in best_features:
        data_cl_[i + ' mm'] = mm.fit_transform(data_cl_[[i]])

    mm_trans = [i + ' mm' for i in best_features]

    smt = SMOTETomek(random_state=42)
    X_smt, y_smt = smt.fit_resample(data_cl_[mm_trans], output)



    if page == 'Dataset Overview':
        st.header("Dataset Preview")
        n = st.number_input(label='Enter the first n columns to display', value=5, min_value=1,max_value=len(data))
        st.write(f'#### First {n} columns of the dataset')
        st.table(data.head(n))

        st.write('#### The properties of each column')
        st.write(f"Shape of the DataFrame : **{data.shape}**")
        st.write("##### Columns and Data Types:")
        st.write(data.dtypes)
        st.write("##### Memory Usage:")
        st.write(data.memory_usage(deep=True))

        st.write('#### The distribution of values of each column')
        st.write(data.describe())

    if page == 'Preprocessing' :
        st.subheader("Missing Values Summary")
        st.write(data.isna().sum())
    
        st.write("### Feature Correlations with Potability")
        st.bar_chart(corrs)

        st.write("### Selected Features for Modeling")
        st.table(best_features)

        st.write("### Feature Distributions")
        graphs = st.selectbox(label="Select the feature for which you want to observe the distribution", options=['Hardness' , 'Turbidity' ,'Chloramines','ph','Trihalomethanes'])
        st.write(f"#### {graphs} Distribution")
    
        bins = st.slider("Select Number of Bins:", min_value=5, max_value=len(data_cl[graphs]), value=100)
        fig = px.histogram(data_cl[graphs], nbins=bins, title= f"Distribution of data of {graphs}", labels={"value": graphs})
        fig.update_layout(xaxis_title=graphs, yaxis_title="Frequency",width = 800,height = 525)
        st.plotly_chart(fig)

        st.write('***')

        st.write('### Initial Potability Distribution')
        st.bar_chart(output.value_counts())

        st.write("### Resampled Potability Distribution")
        st.bar_chart(y_smt.value_counts())

    if page == 'Prediction':
        with st.spinner(text='Training the data'):
            model_name = 'model.pkl' 
            Xtrain, Xtest, ytrain, ytest = train_test_split(X_smt, y_smt, test_size=0.3, random_state=42)
            
            if os.path.exists(model_name):
                grid_rfc = joblib.load(model_name)

            else:
                params = {
                    'criterion': ['gini', 'entropy', 'log_loss'],
                    'max_features': ['sqrt', 'log2'],
                    'n_estimators': range(50, 250, 50),
                }

                kf = KFold(shuffle=True, n_splits=5)
                grid_rfc = GridSearchCV(rfc, params, cv=kf, scoring='accuracy')
                grid_rfc.fit(Xtrain, ytrain)

                joblib.dump(grid_rfc, model_name)   

           
            predicts = grid_rfc.predict(Xtest)
            cm = confusion_matrix(ytest, predicts)
            accuracy = accuracy_score(ytest, predicts)

            st.write("### Model Performance")
            st.write(f"##### Accuracy of the best Random Forest model: ***{accuracy:.2f}***")

            st.write("#### Confusion Matrix")
            fig = px.imshow(cm, color_continuous_scale="viridis", text_auto=True)
            st.plotly_chart(fig)

        st.header("Predicting Water Quality")
        new_vals = {}
        for feature in data.drop(columns=['Potability']):
            val = st.number_input(f"Enter the value for {feature}", value=float(data[feature].mean()))
            new_vals[feature] = val

        if st.button("Predict Potability"):
            new_vals_list = [new_vals[feature] for feature in best_features]

            scaled_new_vals = []
            for i in new_vals_list:
                transformed_value = mm.transform(np.array([[i]]))[0][0]
                scaled_new_vals.append(transformed_value)
            prediction = grid_rfc.predict([scaled_new_vals])

            st.write('***')
            st.subheader('Prediction : ')
            st.write(f"#### For the given set of values the predicted category is => ***{'Potable' if prediction[0] == 1 else 'Not Potable'}***")
else:
    st.write("### Please upload a dataset to start.")
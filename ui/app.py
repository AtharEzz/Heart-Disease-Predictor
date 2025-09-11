import streamlit as st 
import pandas as pd
import joblib
import plotly.graph_objects as go

from preprocess import preprocess_df, preprocess_single_input




st.title("Heart Disease Application")
st.subheader("Enter patient data: ")

age = st.number_input("Age", min_value=10, max_value=100, value=56)

sex= st.selectbox("Sex", options=[0,1], format_func= lambda x:"Female" if x== 0 else "Male")

cp= st.selectbox("Chest Pain Type (cp)", 
    options=[1,2,3,4], 
    format_func= lambda x:{
        1:"Typical Angina", 
        2:"Atypical Angina", 
        3:"Non-anginal Pain", 
        4:"Asymptomatic"}[x])
        
trestbps = st.number_input("Resting BP", min_value=80, max_value=200, value=130)

chol = st.number_input("Cholesterol", min_value=100, max_value=600, value=241)

fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl ",options= [0, 1], format_func= lambda x: "False" if x==0  else "True")

restecg = st.selectbox("Resting ECG ", options=[0, 1, 2], format_func= lambda x:{
        0: "Normal",
        1:"Having ST- T wave abnormality",
        2:"showing probable or definite left ventricular hypertrophy"
        }[x])

thalach=st.number_input("Max Heart Rate (thalach)", min_value=60, max_value=250, value=150)

exang= st.selectbox("Exercise Induced Angina (exang)", 
       options=[0,1], 
       format_func= lambda x: "No" if x== 0 else "Yes")
       
oldpeak= st.number_input("ST Depression (oldpeak)", min_value= 0.0, max_value=10.0, value=0.8, step=0.1) 

slope= st.selectbox("ST Segment Slope (slope)", 
        options=[1,2,3], 
        format_func= lambda x:{
            1: "Upsloping", 
            2: "Flat", 
            3: "Downsloping"}[x])
            
ca = st.selectbox("Number of Major Vessels Colored (ca)",
     options=[0,1,2,3], 
     format_func= lambda x:f"{x} vessel(s)")
     
     
thal = st.selectbox("Thalassemia Test Result (thal)", 
        options=[3,6,7], 
        format_func= lambda x:{
            3:"Unknown (test not performed)",
            6:"Fixed Defect",
            7:"Reversible Defect"}[x])
            



input_raw = pd.DataFrame([{
    'age': age,
    'sex': sex,
    'cp': cp,
    'trestbps': trestbps,
    'chol': chol,
    'fbs': fbs,
    'restecg': restecg,
    'thalach': thalach,
    'exang': exang,
    'oldpeak': oldpeak,
    'slope': slope,
    'ca': ca,
    'thal': thal,
    'target': 0  
}])



# X_train, X_test, y_train, y_test, scaler, final_features = preprocess_df(input_raw)
model = joblib.load('../models/final_model.pkl')
scaler = joblib.load('../models/scaler.pkl')           
final_features = joblib.load('../models/final_features.pkl')  


X_final = preprocess_single_input(input_raw, scaler, final_features)

# --- MAKE PREDICTION ---
if st.button("Predict"):
    prediction = model.predict(X_final)
    probability = model.predict_proba(X_final)[0][1]
    
    try:
        
        heart_df = pd.read_csv('../data/heart_disease.csv')  # Adjust path if needed

        compare_features= ['age', 'thalach', 'chol', 'trestbps', 'oldpeak']
        
        mean_healthy = heart_df[heart_df['target'] == 0][compare_features].mean()
        mean_disease = heart_df[heart_df['target'] == 1][compare_features].mean()

        user_vals = pd.Series({
            'age': age,
            'thalach': thalach,
            'chol': chol,
            'trestbps': trestbps,
            'oldpeak': oldpeak
        })

    except FileNotFoundError:
        st.warning("Original dataset not found — comparison chart disabled.")
        mean_healthy = None
        
    
    if mean_healthy is not None:
        
        fig = go.Figure()

        
        fig.add_trace(go.Bar(
            x=compare_features,
            y=mean_healthy.values,
            name='Healthy Patients (Avg)',
            marker_color='lightgreen',
            hovertemplate='<b>%{x}</b><br>Avg: %{y:.1f}<extra></extra>'
        ))

        
        fig.add_trace(go.Bar(
            x=compare_features,
            y=mean_disease.values,
            name='Heart Disease Patients (Avg)',
            marker_color='salmon',
            hovertemplate='<b>%{x}</b><br>Avg: %{y:.1f}<extra></extra>'
        ))

        fig.add_trace(go.Bar(
            x=compare_features,
            y=user_vals.values,
            name='Your Value',
            marker_color='steelblue',
            hovertemplate='<b>%{x}</b><br>Your Value: %{y:.1f}<extra></extra>'
        ))

        # Customize layout
        fig.update_layout(
            title="Your Values vs. Patient Averages",
            xaxis_title="Clinical Feature",
            yaxis_title="Value",
            barmode='group',  
            template='plotly_white',
            hovermode='x unified',  
            legend_title="Legend"
        )

        st.plotly_chart(fig, use_container_width=True)


    if prediction == 1:
        st.error(f"High risk of heart disease with being  {probability:.2%}  confident having it")
    else:
        st.success(f"Low risk of heart disease with being   {1 - probability:.2%}  confident not having it")
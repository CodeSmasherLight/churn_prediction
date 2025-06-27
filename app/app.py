import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

@st.cache_resource
def load_model():
    with open('C:/churn_prediction/notebooks/models/tuned_churn_model_xgb_smote.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

st.title('Telecom Customer Churn Prediction')
st.write('This app predicts whether a customer will churn based on their characteristics.')

page = st.sidebar.selectbox('Page Navigation', ['Prediction', 'Model Insights'])

if page == 'Prediction':
    st.header('Customer Information')
    
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox('Gender', ['Male', 'Female'])
        senior_citizen = st.selectbox('Senior Citizen', ['No', 'Yes'])
        partner = st.selectbox('Partner', ['No', 'Yes'])
        dependents = st.selectbox('Dependents', ['No', 'Yes'])
        
        tenure = st.slider('Tenure (months)', 0, 72, 12)
        phone_service = st.selectbox('Phone Service', ['No', 'Yes'])
        multiple_lines = st.selectbox('Multiple Lines', ['No', 'Yes', 'No phone service'])
        internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
        
    with col2:
        online_security = st.selectbox('Online Security', ['No', 'Yes', 'No internet service'])
        online_backup = st.selectbox('Online Backup', ['No', 'Yes', 'No internet service'])
        device_protection = st.selectbox('Device Protection', ['No', 'Yes', 'No internet service'])
        tech_support = st.selectbox('Tech Support', ['No', 'Yes', 'No internet service'])
        streaming_tv = st.selectbox('Streaming TV', ['No', 'Yes', 'No internet service'])
        streaming_movies = st.selectbox('Streaming Movies', ['No', 'Yes', 'No internet service'])
        
        contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
        paperless_billing = st.selectbox('Paperless Billing', ['No', 'Yes'])
        payment_method = st.selectbox('Payment Method', 
                                      ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
        monthly_charges = st.slider('Monthly Charges ($)', 0, 150, 50)
        total_charges = st.slider('Total Charges ($)', 0, 10000, tenure * monthly_charges)
    
    input_data = {
        'gender': gender,
        'SeniorCitizen': 1 if senior_citizen == 'Yes' else 0,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    
    input_df = pd.DataFrame([input_data])
    
    input_df['CustomerLifetimeValue'] = input_df['tenure'] * input_df['MonthlyCharges']
    input_df['AvgMonthlyCharges'] = input_df['TotalCharges'] / input_df['tenure']
    input_df['AvgMonthlyCharges'].replace([np.inf, -np.inf], np.nan, inplace=True)
    input_df['AvgMonthlyCharges'].fillna(input_df['MonthlyCharges'], inplace=True)
    
    service_columns = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                     'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    input_df['ServiceCount'] = 0
    for column in service_columns:
        input_df['ServiceCount'] += np.where(~input_df[column].isin(['No', 'No internet service', 'No phone service']), 1, 0)
    
    input_df['HasFamily'] = np.where((input_df['Partner'] == 'Yes') | (input_df['Dependents'] == 'Yes'), 1, 0)
    input_df['IsPaperlessBilling'] = np.where(input_df['PaperlessBilling'] == 'Yes', 1, 0)
    input_df['IsAutomaticPayment'] = np.where(input_df['PaymentMethod'].isin(['Bank transfer (automatic)', 'Credit card (automatic)']), 1, 0)
    
    contract_mapping = {'Month-to-month': 1, 'One year': 12, 'Two year': 24}
    input_df['ContractDuration'] = input_df['Contract'].map(contract_mapping)
    
    input_df['ChargePerTenure'] = input_df['TotalCharges'] / input_df['tenure']
    input_df['ChargePerTenure'].replace([np.inf, -np.inf], np.nan, inplace=True)
    input_df['ChargePerTenure'].fillna(input_df['MonthlyCharges'], inplace=True)

    input_df['charge_ratio'] = input_df['MonthlyCharges'] / input_df['TotalCharges']
    input_df['charge_ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)
    input_df['charge_ratio'].fillna(0, inplace=True)

    def get_tenure_group(tenure):
        if tenure <= 12:
            return '0-1 year'
        elif tenure <= 24:
            return '1-2 years'
        elif tenure <= 48:
            return '2-4 years'
        elif tenure <= 60:
            return '4-5 years'
        else:
            return '5+ years'

    input_df['tenure_group'] = input_df['tenure'].apply(get_tenure_group)

    if st.button('Predict Churn'):
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)
        
        st.subheader('Prediction Result')
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            churn_probability = prediction_proba[0][1] * 100
            st.metric("Churn Probability", f"{churn_probability:.1f}%")
          
            if prediction[0] == 1:
                st.error("⚠️ High risk of churn detected!")
            else:
                st.success("✅ Customer likely to stay")
        
        with col2:
            
            fig, ax = plt.subplots(figsize=(4, 3))
            
            cmap = plt.cm.RdYlGn_r
            colors = cmap(np.linspace(0, 1, 100))
            
            ax.pie([prediction_proba[0][1], 1-prediction_proba[0][1]], 
                   startangle=90, counterclock=False,
                   colors=[cmap(prediction_proba[0][1]), 'white'],
                   wedgeprops=dict(width=0.3, edgecolor='w'))
            
            circle = plt.Circle((0,0), 0.7, color='white')
            ax.add_artist(circle)
            
            ax.axis('equal')
            ax.set_title("Churn Risk Meter", fontsize=10)
            st.pyplot(fig)
        
        if prediction[0] == 1:
            st.subheader("Retention Recommendations")
            
            if hasattr(model[-1], 'feature_importances_'):
                feature_names = model[:-1].get_feature_names_out()
                importances = model[-1].feature_importances_
                
                indices = np.argsort(importances)[::-1]
                
                if 'Contract' in input_data and input_data['Contract'] == 'Month-to-month':
                    st.info("📋 Offer longer-term contract with special incentives")
                
                if input_data['TechSupport'] == 'No':
                    st.info("🛠️ Offer complementary tech support for a limited time")
                
                if 'InternetService' in input_data and input_data['InternetService'] == 'Fiber optic':
                    st.info("💻 Check for service issues with fiber connection")
                
                st.info("💰 Offer a loyalty discount based on tenure")
            else:
                st.info("📱 Offer a device upgrade or discount")
                st.info("💰 Provide promotional pricing on their current services")
                st.info("📋 Suggest a contract extension with additional benefits")

elif page == 'Model Insights':
    st.header('Model Performance Insights')
    
    st.subheader('Confusion Matrix')
    img = plt.imread('C:/churn_prediction/notebooks/notebooks/figures/confusion_matrix.png')
    st.image(img, caption='Confusion Matrix')
    
    st.subheader('ROC Curve')
    img = plt.imread('C:/churn_prediction/notebooks/notebooks/figures/roc_curve.png')
    st.image(img, caption='ROC Curve')
    
    try:
        st.subheader('Feature Importance')
        img = plt.imread('C:/churn_prediction/notebooks/notebooks/figures/feature_importances.png')
        st.image(img, caption='Feature Importance')
    except:
        st.write("Feature importance plot not available for this model.")

    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'],
        'Value': [0.8048, 0.6052, 0.7560, 0.6722, 0.8745]
    })    
    
    st.subheader('Model Performance Metrics')
    st.table(metrics_df)
    
    st.subheader('Key Insights')
    st.write("""
    - Contract type is the most important predictor of churn
    - Customers with month-to-month contracts are 3x more likely to churn
    - Tenure is strongly negatively correlated with churn
    - Customers without tech support or online security are more likely to churn
    - Higher monthly charges correlate with increased churn probability
    """)

metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'],
        'Value': [0.8048, 0.6052, 0.7560, 0.6722, 0.8745]
})

accuracy = metrics_df.loc[metrics_df['Metric'] == 'Accuracy', 'Value'].values[0]
roc_auc = metrics_df.loc[metrics_df['Metric'] == 'ROC AUC', 'Value'].values[0]

if __name__ == '__main__':
    st.sidebar.markdown(f"""
### About
This app predicts customer churn for a telecom company using machine learning.

### How to use
1. Navigate to the Prediction page  
2. Enter customer information  
3. Click 'Predict Churn' to see results  

### Model
XGBoost classifier with **{accuracy:.2%}** accuracy and **{roc_auc:.2f}** ROC AUC score.
""")
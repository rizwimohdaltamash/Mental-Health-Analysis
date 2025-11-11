import joblib
import pandas as pd
import numpy as np
import openai  # Ensure you install openai library
import streamlit as st

# Load the trained model
class MentalHealthPredictor:
    def __init__(self, model_path='models/mental_health_model.pkl'):
        self.pipeline = joblib.load(model_path)
        self.class_names = {
            0: 'No Diagnosis',
            1: 'Depression Only',
            2: 'Anxiety Only',
            3: 'Comorbid Depression & Anxiety'
        }
    
    def predict(self, input_data):
        required_features = self.pipeline[:-1].get_feature_names_out()
        df = pd.DataFrame(columns=required_features)

        for col in ['phq_score', 'gad_score', 'age', 'gender']:
            if col in input_data:
                df[col] = [input_data[col]]

        df_imputed = self.pipeline[:-1].transform(df)
        prediction = self.pipeline.named_steps['model'].predict(df_imputed)[0]
        probabilities = self.pipeline.named_steps['model'].predict_proba(df_imputed)[0]
        severity_index = input_data.get('phq_score', 0) + input_data.get('gad_score', 0)

        return {
            'diagnosis': self.class_names[prediction],
            'probabilities': dict(zip(self.class_names.values(), probabilities)),
            'severity_index': severity_index,
            'recommendations': self._generate_recommendations(prediction, severity_index)
        }

    def _generate_recommendations(self, prediction, severity_index):
        severity = 'Mild' if severity_index < 15 else 'Moderate' if severity_index < 25 else 'Severe'
        recommendations = {
            'Mild': ["Consider weekly mood tracking", "Practice mindfulness exercises"],
            'Moderate': ["Schedule clinical evaluation", "Begin cognitive behavioral therapy (CBT) exercises"],
            'Severe': ["Immediate professional consultation recommended", "Contact crisis hotline if experiencing suicidal thoughts"]
        }
        return recommendations[severity]

# Function to get explanation from LLM


def get_llm_explanation(diagnosis, severity_index):
    client = openai.OpenAI(api_key="")  # Add your API key here

    prompt = f"""
    The user has been diagnosed with {diagnosis} with a severity index of {severity_index}/48.
    Provide a brief, compassionate explanation of this condition and suggest coping mechanisms.
    """
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a mental health assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    
    return response.choices[0].message.content


# Streamlit UI
st.title("Mental Health Prediction Tool")
st.write("Enter your details below to receive a mental health assessment.")

phq_score = st.number_input("PHQ-9 Score (0-27)", min_value=0, max_value=27, step=1)
gad_score = st.number_input("GAD-7 Score (0-21)", min_value=0, max_value=21, step=1)
age = st.number_input("Age", min_value=1, max_value=100, step=1)
gender = st.radio("Gender", options={0: "Female", 1: "Male"}, format_func=lambda x: {0: "Female", 1: "Male"}[x])

if st.button("Predict"):
    predictor = MentalHealthPredictor()
    result = predictor.predict({'phq_score': phq_score, 'gad_score': gad_score, 'age': age, 'gender': gender})

    st.subheader("Prediction Results")
    st.write(f"**Diagnosis:** {result['diagnosis']}")
    st.write(f"**Severity Index:** {result['severity_index']}/48")

    # Display recommendations
    st.subheader("Recommendations")
    for rec in result['recommendations']:
        st.write(f"- {rec}")

    # Get LLM-based explanation
    explanation = get_llm_explanation(result['diagnosis'], result['severity_index'])
    st.subheader("AI Explanation & Coping Strategies")
    st.write(explanation)

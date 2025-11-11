import joblib
import pandas as pd
import numpy as np

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
        """Make prediction with severity assessment"""
        required_features = self.pipeline[:-1].get_feature_names_out()
        df = pd.DataFrame(columns=required_features)

        # Fill provided values
        for col in ['phq_score', 'gad_score', 'age', 'gender']:
            if col in input_data:
                df[col] = [input_data[col]]

        # Impute missing values using the pipeline
        df_imputed = self.pipeline[:-1].transform(df)

        # Make prediction
        prediction = self.pipeline.named_steps['model'].predict(df_imputed)[0]
        probabilities = self.pipeline.named_steps['model'].predict_proba(df_imputed)[0]

        # Calculate severity index
        severity_index = input_data.get('phq_score', 0) + input_data.get('gad_score', 0)

        return {
            'diagnosis': self.class_names[prediction],
            'probabilities': dict(zip(self.class_names.values(), probabilities)),
            'severity_index': severity_index,
            'recommendations': self._generate_recommendations(prediction, severity_index)
        }
    
    def _generate_recommendations(self, prediction, severity_index):
        """Generate simple recommendations based on severity index"""
        severity = 'Mild' if severity_index < 15 else \
                  'Moderate' if severity_index < 25 else 'Severe'
        
        recommendations = {
            'No Diagnosis': ["Maintain a healthy lifestyle", "Stay physically active", "Engage in social activities"],
            'Depression Only': ["Practice mindfulness", "Seek professional help if needed", "Engage in hobbies"],
            'Anxiety Only': ["Try deep breathing exercises", "Limit caffeine and sugar intake", "Consider therapy"],
            'Comorbid Depression & Anxiety': ["Follow a structured routine", "Engage in regular physical exercise", "Seek counseling"]
        }
        
        return recommendations[self.class_names[prediction]]

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Mental Health Prediction Tool')
    parser.add_argument('--phq', type=int, required=True, help='PHQ-9 score (0-27)')
    parser.add_argument('--gad', type=int, required=True, help='GAD-7 score (0-21)')
    parser.add_argument('--age', type=int, required=True, help='Age')
    parser.add_argument('--gender', type=int, required=True, help='Gender (0=Female, 1=Male)')
    
    args = parser.parse_args()
    
    predictor = MentalHealthPredictor()
    result = predictor.predict({
        'phq_score': args.phq,
        'gad_score': args.gad,
        'age': args.age,
        'gender': args.gender
    })
    
    print("\nPrediction Results:")
    print(f"Diagnosis: {result['diagnosis']}")
    print(f"Severity Index: {result['severity_index']}/48")
    print("\nRecommendations:")
    for rec in result['recommendations']:
        print(f"- {rec}")

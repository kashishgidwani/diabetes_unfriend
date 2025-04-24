import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('diabetes_assistant.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class PredictiveRecommendation:
    def __init__(self):
        """Initialize the predictive recommendation system."""
        self.model = None
        self.features = ['age', 'bmi', 'blood_pressure', 'glucose', 'insulin', 'skin_thickness', 'pregnancies', 'diabetes_pedigree', 'heart_rate', 'cholesterol']
        self.load_model()

    def load_model(self):
        """Load and train the predictive model."""
        try:
            # Load the dataset
            dataset_path = 'diabetes_health_data_5000_specific.csv'
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
            
            df = pd.read_csv(dataset_path)
            
            # Prepare data
            df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
            
            # Train model
            X = df[self.features]
            y = df['diabetes_type'].astype('category').cat.codes
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X, y)
            self.diabetes_types = df['diabetes_type'].astype('category').cat.categories
            logger.info("Predictive model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading predictive model: {str(e)}")
            self.model = None

    def get_recommendation(self, user_data):
        """Generate personalized recommendations based on user data."""
        try:
            if self.model is None:
                raise ValueError("Model not loaded")
            
            features = pd.DataFrame([user_data], columns=self.features)
            prediction = self.model.predict(features)
            diabetes_type = self.diabetes_types[prediction[0]]
            
            recommendations = {
                'diet_plan': self.get_diet_plan(diabetes_type, user_data['age']),
                'exercise_plan': self.get_exercise_plan(diabetes_type, user_data['age']),
                'lifestyle_changes': self.get_lifestyle_recommendations(user_data)
            }
            return recommendations
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return None

    def get_diet_plan(self, diabetes_type, age):
        """Generate personalized diet plan."""
        if diabetes_type.lower() == 'type 1':
            return {
                'meals': [
                    'Breakfast: High-fiber cereal with low-fat milk, fresh fruits',
                    'Lunch: Grilled chicken salad with olive oil dressing',
                    'Dinner: Baked fish with quinoa and steamed vegetables'
                ],
                'snacks': ['Apple with almond butter', 'Greek yogurt with berries'],
                'guidelines': [
                    'Count carbohydrates carefully',
                    'Eat at regular times',
                    'Monitor portion sizes',
                    'Choose low glycemic index foods'
                ]
            }
        else:
            return {
                'meals': [
                    'Breakfast: Oatmeal with nuts and cinnamon',
                    'Lunch: Turkey and avocado sandwich on whole grain bread',
                    'Dinner: Lean protein with brown rice and vegetables'
                ],
                'snacks': ['Hummus with vegetables', 'Mixed nuts'],
                'guidelines': [
                    'Limit refined carbohydrates',
                    'Include lean proteins',
                    'Add healthy fats',
                    'Eat plenty of fiber'
                ]
            }

    def get_exercise_plan(self, diabetes_type, age):
        """Generate personalized exercise plan."""
        base_plan = {
            'cardio': ['Walking', 'Swimming', 'Cycling'],
            'strength': ['Resistance bands', 'Body weight exercises'],
            'flexibility': ['Gentle stretching', 'Yoga'],
            'duration': '30 minutes per session',
            'frequency': '5 days per week'
        }
        
        if age > 60:
            base_plan['cardio'] = ['Walking', 'Water aerobics']
            base_plan['duration'] = '20 minutes per session'
        
        return base_plan

    def get_lifestyle_recommendations(self, user_data):
        """Generate personalized lifestyle recommendations."""
        recommendations = [
            'Monitor blood sugar regularly',
            'Keep a consistent meal schedule',
            'Stay hydrated',
            'Get adequate sleep'
        ]
        
        if user_data.get('bloodpressureissue'):
            recommendations.append('Monitor blood pressure daily')
        if user_data.get('heartproblem'):
            recommendations.append('Regular cardiac check-ups')
        if user_data.get('kidneyissue'):
            recommendations.append('Limit sodium intake')
            
        return recommendations
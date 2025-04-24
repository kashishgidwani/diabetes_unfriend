import streamlit as st

# Set page config first and only once - must be the first Streamlit command
st.set_page_config(
    page_title="Diabetes Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

import os
import logging
from model_manager import model_manager, ModelManager
from db_utils import db_manager
from PIL import Image
import io
import whisper
import tempfile
import time
from fallback_responses import generate_fallback_response
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from datetime import datetime
import cv2
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import google.generativeai as genai
from dotenv import load_dotenv
from pymongo import MongoClient
from typing import Dict, List, Optional, Tuple
import json
import base64
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
import torch
import torchaudio
from pydub import AudioSegment
import subprocess
from speech_to_text import SpeechToText
import joblib

# Import custom components
from pdf_report_generator import PDFReportGenerator
from Predictive_recommendation import PredictiveRecommendation

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

# Load environment variables
load_dotenv()

class ImageAnalyzer:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError("Google API key not found. Please set GOOGLE_API_KEY environment variable.")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        logger.info("ImageAnalyzer initialized successfully")

    def analyze_image(self, image_data: bytes, prompt: str = None) -> Dict:
        try:
            image = Image.open(io.BytesIO(image_data))
            default_prompt = """
            Analyze this medical image and provide a VERY CONCISE response with ONLY:
            1. Test Type & Result (one line)
            2. Key Finding (one line)
            3. Quick Recommendation (one line)
            Keep total response under 100 words.
            """
            prompt = prompt or default_prompt
            response = self.model.generate_content([prompt, image])
            return {
                'description': response.text,
                'timestamp': datetime.now().isoformat(),
                'analysis_type': 'medical_report'
            }
        except Exception as e:
            logger.error(f"Error analyzing image: {str(e)}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

# Initialize MongoDB connection
def get_database():
    """Initialize MongoDB connection."""
    try:
        uri = os.getenv('MONGODB_URI')
        if not uri:
            raise ValueError("MONGODB_URI not found in environment variables")
        
        # Parse the database name correctly from the URI
        db_name = uri.split('/')[-1].split('?')[0]  # Remove query parameters
        if len(db_name) > 38:
            raise ValueError("Database name is too long. Max length is 38 bytes.")
        
        client = MongoClient(uri)
        db = client[db_name]
        logger.info("MongoDB connection established successfully")
        return db
    except Exception as e:
        logger.error(f"Error connecting to MongoDB: {str(e)}")
        raise

# Initialize database
try:
    db = get_database()
    logger.info("Database connection established successfully")
except Exception as e:
    logger.error(f"Error connecting to database: {str(e)}")
    db = None

# Initialize components
pdf_generator = PDFReportGenerator()
image_analyzer = ImageAnalyzer()
predictive_recommendation = PredictiveRecommendation()

# Force CPU usage for whisper
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['USE_MPS_DEVICE'] = ''

# Check for ffmpeg installation
def check_ffmpeg():
    """Check if ffmpeg is installed and use local binary if available."""
    try:
        # Try to use local ffmpeg first
        local_ffmpeg = os.path.join(os.path.dirname(__file__), 'bin', 'ffmpeg')
        if os.path.exists(local_ffmpeg):
            # Make it executable
            os.chmod(local_ffmpeg, 0o755)
            # Test if it works
            subprocess.run([local_ffmpeg, '-version'], capture_output=True, check=True)
            logger.info("Using local ffmpeg binary")
            return local_ffmpeg
    except Exception as e:
        logger.warning(f"Local ffmpeg not available: {str(e)}")
    
    try:
        # Try system ffmpeg
        subprocess.run(['/usr/bin/ffmpeg', '-version'], capture_output=True, check=True)
        logger.info("Using system ffmpeg")
        return '/usr/bin/ffmpeg'
    except Exception as e:
        logger.warning(f"System ffmpeg not available: {str(e)}")
    
    try:
        # Try PATH ffmpeg
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        logger.info("Using PATH ffmpeg")
        return 'ffmpeg'
    except Exception as e:
        logger.warning(f"PATH ffmpeg not available: {str(e)}")
    
    logger.warning("No ffmpeg installation found. Audio processing may be limited.")
    return None

# Check ffmpeg installation at startup
check_ffmpeg()

def check_api_rate_limit():
    """Check if enough time has passed since the last API call"""
    current_time = time.time()
    if current_time - st.session_state.last_api_call < 60:  # 60 seconds cooldown
        remaining_time = int(60 - (current_time - st.session_state.last_api_call))
        st.warning(f"Please wait {remaining_time} seconds before making another request.")
        return False
    st.session_state.last_api_call = current_time
    return True

def show_login_form():
    """Display login/registration interface with tabs"""
    st.title("🏥 Diabeticanfriend")
    
    # Create tabs for login and registration
    login_tab, register_tab = st.tabs(["Login", "Register"])
    
    with login_tab:
        st.subheader("Login to Your Account")
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
            
            if submit:
                if not email or not password:
                    st.error("Please enter both email and password")
                else:
                    try:
                        user = db_manager.authenticate_user(email, password)
                        if user:
                            st.session_state.user = user
                            st.session_state.is_authenticated = True
                            st.session_state.is_logged_in = True
                            
                            # Load user's data from MongoDB
                            try:
                                user_data = db.users.find_one({"_id": user["_id"]})
                                if user_data:
                                    st.session_state.user_data = user_data
                                    st.session_state.health_assessment = user_data.get("health_assessment")
                                    st.session_state.image_analyses = user_data.get("image_analyses", [])
                                    st.session_state.reports = user_data.get("reports", [])
                                    
                                    # Update last login time in MongoDB
                                    db.users.update_one(
                                        {"_id": user["_id"]},
                                        {"$set": {"last_login": datetime.now()}}
                                    )
                            except Exception as e:
                                logger.error(f"Error loading user data: {str(e)}")
                            
                            st.success("Login successful!")
                            st.experimental_rerun()
                        else:
                            st.error("Invalid credentials. Please check your email and password.")
                    except Exception as e:
                        logger.error(f"Error during login: {str(e)}")
                        st.error("An error occurred during login. Please try again.")
    
    with register_tab:
        st.subheader("Create New Account")
        with st.form("registration_form"):
            name = st.text_input("Name")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            
            diabetes_type = st.radio(
                "Diabetes Type",
                ["Type 1", "Type 2", "Type 3", "No diabetes (curious)"]
            )
            
            age = st.number_input("Age", min_value=0, max_value=120)
            
            gender = st.radio(
                "Gender",
                ["Male", "Female", "Prefer not to say"]
            )
            
            medical_history = st.text_area("Medical History (optional)")
            
            submitted = st.form_submit_button("Register")
            
            if submitted:
                if password != confirm_password:
                    st.error("Passwords do not match")
                    return
                    
                success, message = db_manager.register_user(
                    name=name,
                    email=email,
                    password=password,
                    diabetes_type=diabetes_type,
                    age=age,
                    gender=gender,
                    medical_history=medical_history
                )
                
                if success:
                    st.success(message)
                    st.info("Please login with your credentials")
                else:
                    st.error(message)

class PredictiveRecommendation:
    def __init__(self):
        self.model = None
        self.label_encoders = None
        self.scaler = None
        self.load_model()

    def load_model(self):
        try:
            # Check if model files exist
            model_path = 'diabetes_model.joblib'
            encoders_path = 'label_encoders.joblib'
            scaler_path = 'feature_scaler.joblib'
            
            if not all(os.path.exists(path) for path in [model_path, encoders_path, scaler_path]):
                logger.error("Model files not found. Please ensure the following files are present:")
                logger.error(f"- {model_path}")
                logger.error(f"- {encoders_path}")
                logger.error(f"- {scaler_path}")
                return
            
            # Load the trained model, label encoders, and scaler
            self.model = joblib.load(model_path)
            self.label_encoders = joblib.load(encoders_path)
            self.scaler = joblib.load(scaler_path)
            logger.info("Predictive model loaded successfully")
            
            # Verify model is working
            test_data = pd.DataFrame([{
                'Age': 30,
                'Gender': self.label_encoders['Gender'].transform(['Male'])[0],
                'BloodPressureIssue': 0,
                'KidneyIssue': 0,
                'HeartProblem': 0,
                'JointPainIssue': 0,
                'AgeGroup': 0,
                'HealthScore': 0
            }])
            test_features = self.scaler.transform(test_data)
            test_prediction = self.model.predict(test_features)
            logger.info(f"Model test prediction successful: {test_prediction[0]}")
            
        except Exception as e:
            logger.error(f"Error loading predictive model: {str(e)}")
            self.model = None
            self.label_encoders = None
            self.scaler = None

    def get_recommendation(self, user_data):
        try:
            if self.model is None:
                return self.get_default_recommendations()

            # Prepare user data for prediction
            features = pd.DataFrame([{
                'Age': user_data.get('age', 30),
                'Gender': self.label_encoders['Gender'].transform([user_data.get('gender', 'Male')])[0],
                'BloodPressureIssue': 1 if user_data.get('blood_pressure', False) else 0,
                'KidneyIssue': 1 if user_data.get('kidney_issue', False) else 0,
                'HeartProblem': 1 if user_data.get('heart_problem', False) else 0,
                'JointPainIssue': 1 if user_data.get('joint_pain', False) else 0,
                'AgeGroup': pd.cut([user_data.get('age', 30)], bins=[0, 30, 50, 70, 100], labels=[0, 1, 2, 3])[0],
                'HealthScore': sum([
                    1 if user_data.get('blood_pressure', False) else 0,
                    1 if user_data.get('kidney_issue', False) else 0,
                    1 if user_data.get('heart_problem', False) else 0,
                    1 if user_data.get('joint_pain', False) else 0
                ])
            }])

            # Scale features
            features_scaled = self.scaler.transform(features)

            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            diabetes_type = self.label_encoders['DiabetesType'].inverse_transform([prediction])[0]

            # Get recommendations based on prediction
            recommendations = {
                'diet_plan': self.get_diet_plan(diabetes_type, user_data.get('age', 30)),
                'exercise_plan': self.get_exercise_plan(diabetes_type, user_data.get('age', 30)),
                'lifestyle_changes': self.get_lifestyle_recommendations(user_data)
            }

            # Store recommendations in user's medical record
            if st.session_state.is_authenticated and st.session_state.user:
                try:
                    medical_record = {
                        'user_id': str(st.session_state.user["_id"]),
                        'recommendations': recommendations,
                        'timestamp': datetime.now(),
                        'diabetes_type': diabetes_type,
                        'user_data': user_data,
                        'prediction_confidence': self.model.predict_proba(features_scaled)[0].max()
                    }
                    db.medical_records.insert_one(medical_record)
                    logger.info("Medical record saved successfully")
                except Exception as e:
                    logger.error(f"Error saving medical record: {str(e)}")

            return recommendations

        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return self.get_default_recommendations()

    def get_default_recommendations(self):
        return {
            'diet_plan': {
                'meals': ['Consult with your healthcare provider for personalized diet recommendations'],
                'snacks': ['Healthy snacks as recommended by your doctor'],
                'guidelines': ['Follow your healthcare provider\'s advice']
            },
            'exercise_plan': {
                'cardio': ['Walking', 'Swimming'],
                'strength': ['Light resistance training'],
                'flexibility': ['Gentle stretching'],
                'duration': '30 minutes per session',
                'frequency': '5 days per week'
            },
            'lifestyle_changes': [
                'Monitor blood sugar regularly',
                'Maintain a healthy diet',
                'Stay physically active',
                'Get regular check-ups'
            ]
        }

    def get_diet_plan(self, diabetes_type, age):
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
        recommendations = [
            'Monitor blood sugar regularly',
            'Keep a consistent meal schedule',
            'Stay hydrated',
            'Get adequate sleep'
        ]
        
        if user_data.get('blood_pressure'):
            recommendations.append('Monitor blood pressure daily')
        if user_data.get('heart_problem'):
            recommendations.append('Regular cardiac check-ups')
        if user_data.get('kidney_issue'):
            recommendations.append('Limit sodium intake')
            
        return recommendations

class PDFReportGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.custom_style = ParagraphStyle(
            'CustomStyle',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=20
        )

    def create_report(self, user_data, recommendations, image_analysis=None):
        try:
            filename = f"diabetes_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            doc = SimpleDocTemplate(filename, pagesize=letter)
            story = []

            # Title
            title_style = self.styles['Heading1']
            story.append(Paragraph("Diabetes Management Report", title_style))
            story.append(Spacer(1, 20))

            # User Information
            story.append(Paragraph("Patient Information", self.styles['Heading2']))
            user_info = [
                f"Age: {user_data['age']}",
                f"Blood Pressure Issues: {'Yes' if user_data['blood_pressure'] else 'No'}",
                f"Heart Problems: {'Yes' if user_data['heart_problem'] else 'No'}",
                f"Kidney Issues: {'Yes' if user_data['kidney_issue'] else 'No'}"
            ]
            for info in user_info:
                story.append(Paragraph(info, self.custom_style))
            story.append(Spacer(1, 20))

            # Recommendations
            story.append(Paragraph("Personalized Recommendations", self.styles['Heading2']))
            
            # Diet Plan
            story.append(Paragraph("Diet Plan", self.styles['Heading3']))
            for meal in recommendations['diet_plan']['meals']:
                story.append(Paragraph(f"• {meal}", self.custom_style))
            
            # Exercise Plan
            story.append(Paragraph("Exercise Plan", self.styles['Heading3']))
            exercise_plan = recommendations['exercise_plan']
            story.append(Paragraph(f"Frequency: {exercise_plan['frequency']}", self.custom_style))
            story.append(Paragraph(f"Duration: {exercise_plan['duration']}", self.custom_style))
            story.append(Paragraph("Recommended Activities:", self.custom_style))
            for activity_type, activities in exercise_plan.items():
                if isinstance(activities, list):
                    for activity in activities:
                        story.append(Paragraph(f"• {activity}", self.custom_style))

            # Lifestyle Recommendations
            story.append(Paragraph("Lifestyle Recommendations", self.styles['Heading3']))
            for rec in recommendations['lifestyle_changes']:
                story.append(Paragraph(f"• {rec}", self.custom_style))

            # Image Analysis (if available)
            if image_analysis:
                story.append(Paragraph("Medical Image Analysis", self.styles['Heading2']))
                story.append(Paragraph("Image Quality Assessment:", self.styles['Heading3']))
                for metric, value in image_analysis['clarity'].items():
                    story.append(Paragraph(f"• {metric}: {value}", self.custom_style))
                
                story.append(Paragraph("Findings:", self.styles['Heading3']))
                findings = image_analysis['findings']
                story.append(Paragraph(f"• Confidence: {findings['confidence']:.2%}", self.custom_style))
                story.append(Paragraph(f"• Primary Finding: {findings['primary_finding']}", self.custom_style))
                story.append(Paragraph(f"• Suggestion: {findings['suggestion']}", self.custom_style))

            # Build PDF
            doc.build(story)
            logger.info(f"PDF report generated successfully: {filename}")
            return filename
        except Exception as e:
            logger.error(f"Error generating PDF report: {str(e)}")
            return None

class HealthAssessment:
    """Handles health assessment and recommendations."""
    
    def __init__(self):
        """Initialize the health assessment system."""
        self.model_manager = ModelManager()
    
    def assess_health(self, age: int, weight: float, height: float, 
                     activity_level: str, diet_type: str) -> Dict:
        """Generate health assessment and recommendations."""
        try:
            prompt = f"""
            Based on the following health information:
            - Age: {age} years
            - Weight: {weight} kg
            - Height: {height} cm
            - Activity Level: {activity_level}
            - Diet Type: {diet_type}
            
            Please provide:
            1. A health assessment
            2. Personalized diet recommendations
            3. Exercise recommendations
            4. Lifestyle suggestions
            """
            
            response = self.model_manager.generate_content(prompt)
            return {
                "assessment": response,
                "raw_response": response
            }
        except Exception as e:
            logger.error(f"Error in health assessment: {str(e)}")
            return {"error": str(e)}

class SpeechToText:
    """Handles speech-to-text conversion using Whisper."""
    
    def __init__(self, model_size: str = "base"):
        """
        Initialize the speech-to-text converter.
        
        Args:
            model_size (str): Size of the Whisper model to use (tiny, base, small, medium, large)
        """
        self.model_size = model_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Force CPU usage to avoid FP16 warning
        self.device = "cpu"
        self.model = whisper.load_model(model_size, device=self.device)
        self.ffmpeg_path = check_ffmpeg()
        logger.info(f"SpeechToText initialized with model size: {model_size} on {self.device}")
    
    def preprocess_audio(self, audio_data: bytes) -> str:
        """
        Preprocess audio data for transcription.
        
        Args:
            audio_data (bytes): Raw audio data
            
        Returns:
            str: Path to the preprocessed audio file
        """
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_path = temp_file.name

            # Convert audio to the right format using ffmpeg if available
            if self.ffmpeg_path:
                try:
                    output_path = temp_path + ".converted.wav"
                    subprocess.run([
                        self.ffmpeg_path,
                        '-i', temp_path,
                        '-ar', '16000',
                        '-ac', '1',
                        '-y',
                        output_path
                    ], capture_output=True, check=True)
                    os.unlink(temp_path)
                    return output_path
                except Exception as e:
                    logger.warning(f"Could not preprocess audio with ffmpeg: {str(e)}")
                    return temp_path
            else:
                # Try pydub as fallback
                try:
                    audio = AudioSegment.from_file(temp_path)
                    audio = audio.set_frame_rate(16000)
                    audio = audio.set_channels(1)
                    audio.export(temp_path, format="wav")
                except Exception as e:
                    logger.warning(f"Could not preprocess audio with pydub: {str(e)}")
                    # If preprocessing fails, try to use the file as is
                    pass

            return temp_path
            
        except Exception as e:
            logger.error(f"Error preprocessing audio: {str(e)}")
            raise
    
    def transcribe(self, audio_data: bytes) -> str:
        """
        Transcribe audio data to text.
        
        Args:
            audio_data (bytes): Raw audio data
            
        Returns:
            str: Transcribed text
        """
        try:
            temp_path = self.preprocess_audio(audio_data)
            result = self.model.transcribe(temp_path)
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            return result["text"]
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            raise
    
    def process_audio_file(self, file_path: str) -> str:
        """
        Process an audio file and transcribe it.
        
        Args:
            file_path (str): Path to the audio file
            
        Returns:
            str: Transcribed text
        """
        try:
            with open(file_path, "rb") as f:
                audio_data = f.read()
            return self.transcribe(audio_data)
            
        except Exception as e:
            logger.error(f"Error processing audio file: {str(e)}")
            raise

# Updated MongoDB schema for crucial medical data
class MedicalRecord:
    def __init__(self):
        self.schema = {
            "medical_bills": [{
                "date": datetime,
                "amount": float,
                "description": str,
                "insurance_coverage": float,
                "remaining_balance": float
            }],
            "current_medications": [{
                "name": str,
                "dosage": str,
                "frequency": str,
                "prescribed_date": datetime,
                "notes": str,
                "prescribed_by": str,
                "end_date": datetime,
                "status": str  # active, completed, discontinued
            }],
            "vitals_history": [{
                "date": datetime,
                "blood_glucose": float,
                "blood_pressure": str,
                "heart_rate": int,
                "weight": float,
                "hba1c": float,
                "notes": str
            }],
            "image_analyses": [{
                "date": datetime,
                "image_type": str,
                "analysis_result": dict,
                "image_data": str,  # Base64 encoded image
                "doctor_notes": str,
                "follow_up_date": datetime,
                "status": str  # pending, reviewed, action_required
            }],
            "current_condition": {
                "diabetes_type": str,
                "last_updated": datetime,
                "complications": list,
                "allergies": list,
                "recent_symptoms": list
            },
            "chat_history": [{
                "timestamp": datetime,
                "message": str,
                "type": str,  # user, assistant
                "extracted_info": {
                    "medications": list,
                    "symptoms": list,
                    "vitals": dict
                }
            }]
        }

def initialize_session_state():
    """Initialize session state with crucial medical data structure"""
    # Authentication states
    if 'is_authenticated' not in st.session_state:
        st.session_state.is_authenticated = False
    if 'is_logged_in' not in st.session_state:
        st.session_state.is_logged_in = False
    if 'user' not in st.session_state:
        st.session_state.user = None
    
    # Medical record states
    if 'medical_record' not in st.session_state:
        st.session_state.medical_record = {
            'medical_bills': [],
            'current_medications': [],
            'vitals_history': [],
            'image_analyses': [],
            'current_condition': {},
            'chat_history': []
        }
    
    # Load user data if authenticated
    if st.session_state.is_authenticated and st.session_state.user:
        try:
            user_data = db.users.find_one({"_id": st.session_state.user["_id"]})
            if user_data:
                st.session_state.medical_record = user_data.get("medical_record", st.session_state.medical_record)
                st.session_state.health_assessment = user_data.get("health_assessment")
                st.session_state.image_analyses = user_data.get("image_analyses", [])
                st.session_state.reports = user_data.get("reports", [])
        except Exception as e:
            logger.error(f"Error loading user data: {str(e)}")
    
    # API and component states
    if 'last_api_call' not in st.session_state:
        st.session_state.last_api_call = 0
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Initialize components with error handling
    if 'speech_to_text' not in st.session_state:
        try:
            st.session_state.speech_to_text = SpeechToText()
        except Exception as e:
            logger.error(f"Error initializing speech-to-text: {str(e)}")
            st.session_state.speech_to_text = None
    
    if 'image_analyzer' not in st.session_state:
        try:
            st.session_state.image_analyzer = ImageAnalyzer()
        except Exception as e:
            logger.error(f"Error initializing image analyzer: {str(e)}")
            st.session_state.image_analyzer = None
    
    if 'predictive_recommendation' not in st.session_state:
        try:
            st.session_state.predictive_recommendation = PredictiveRecommendation()
        except Exception as e:
            logger.error(f"Error initializing predictive recommendation: {str(e)}")
            st.session_state.predictive_recommendation = None
    
    # Additional states
    if 'health_assessment' not in st.session_state:
        st.session_state.health_assessment = None
    if 'image_analyses' not in st.session_state:
        st.session_state.image_analyses = []
    if 'reports' not in st.session_state:
        st.session_state.reports = []
    if 'speech_inputs' not in st.session_state:
        st.session_state.speech_inputs = []

def process_user_input(user_input: str) -> str:
    """Process user input and generate response."""
    try:
        if not user_input.strip():
            return "Please provide a valid input."
        
        # Add user message to chat history
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Extract medical information from chat
        extracted_info = extract_medical_info_from_chat([{"role": "user", "content": user_input}])
        
        # Update medical record with extracted information
        if extracted_info['prescriptions']:
            for prescription in extracted_info['prescriptions']:
                st.session_state.medical_record['current_medications'].append({
                    "name": prescription['text'],
                    "prescribed_date": datetime.now(),
                    "status": "active"
                })
        
        if extracted_info['vitals']:
            for vital in extracted_info['vitals']:
                st.session_state.medical_record['vitals_history'].append({
                    "date": datetime.now(),
                    "notes": vital['text']
                })
        
        # Save updated medical record
        save_medical_record()
        
        # Generate response using the model
        try:
            # Check if Gemini API key is available
            api_key = os.getenv('GOOGLE_API_KEY')
            if not api_key:
                raise ValueError("Google API key not found. Please set GOOGLE_API_KEY environment variable.")
            
            # Configure Gemini with the correct model
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            # Generate response
            response = model.generate_content(user_input)
            if not response.text:
                raise ValueError("No response generated from the model")
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response.text})
            
            # Save conversation to database
            try:
                if st.session_state.is_authenticated and st.session_state.user:
                    conversation_data = {
                        "user_id": str(st.session_state.user["_id"]),
                        "messages": [
                            {"role": "user", "content": user_input},
                            {"role": "assistant", "content": response.text}
                        ],
                        "timestamp": datetime.now(),
                        "extracted_info": extracted_info
                    }
                    db.conversations.insert_one(conversation_data)
                    logger.info("Conversation saved to database successfully")
            except Exception as e:
                logger.error(f"Error saving conversation: {str(e)}")
            
            return response.text
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            fallback_response = "I apologize, but I'm having trouble generating a response right now. Please try again later."
            st.session_state.messages.append({"role": "assistant", "content": fallback_response})
            return fallback_response
            
    except Exception as e:
        logger.error(f"Error processing user input: {str(e)}")
        return "I apologize, but I encountered an error. Please try again or rephrase your question."

def show_medical_history():
    """Display medical history including prescriptions and image analyses"""
    st.title("Medical History")
    
    if not st.session_state.is_authenticated:
        st.warning("Please log in to view your medical history")
        return
    
    # Create tabs for different sections
    meds_tab, vitals_tab, images_tab, chat_tab = st.tabs([
        "Medications", "Vitals", "Image Analyses", "Chat History"
    ])
    
    with meds_tab:
        st.subheader("Current Medications")
        if st.session_state.medical_record.get('current_medications'):
            for med in st.session_state.medical_record['current_medications']:
                with st.expander(f"{med['name']} - {med.get('prescribed_date', datetime.now()).strftime('%Y-%m-%d')}"):
                    st.write(f"Dosage: {med.get('dosage', 'Not specified')}")
                    st.write(f"Frequency: {med.get('frequency', 'Not specified')}")
                    st.write(f"Status: {med.get('status', 'active')}")
                    if med.get('notes'):
                        st.write(f"Notes: {med['notes']}")
        else:
            st.info("No medications recorded")
    
    with vitals_tab:
        st.subheader("Vitals History")
        if st.session_state.medical_record.get('vitals_history'):
            for vital in st.session_state.medical_record['vitals_history']:
                with st.expander(f"Vitals from {vital.get('date', datetime.now()).strftime('%Y-%m-%d %H:%M')}"):
                    st.write(f"Notes: {vital.get('notes', 'No notes')}")
        else:
            st.info("No vitals recorded")
    
    with images_tab:
        st.subheader("Image Analyses")
        if st.session_state.medical_record.get('image_analyses'):
            for analysis in st.session_state.medical_record['image_analyses']:
                with st.expander(f"Analysis from {analysis.get('date', datetime.now()).strftime('%Y-%m-%d %H:%M')}"):
                    if analysis.get('image_data'):
                        try:
                            image_bytes = base64.b64decode(analysis['image_data'])
                            st.image(image_bytes, width=300)
                        except Exception as e:
                            logger.error(f"Error displaying image: {str(e)}")
                            st.warning("Could not display image")
                    
                    st.write("Analysis Results:")
                    if analysis.get('analysis_result'):
                        st.write(analysis['analysis_result'])
                    else:
                        st.write("No analysis results available")
                    
                    if analysis.get('doctor_notes'):
                        st.write(f"Doctor's Notes: {analysis['doctor_notes']}")
                    
                    if analysis.get('follow_up_date'):
                        st.write(f"Follow-up Date: {analysis['follow_up_date'].strftime('%Y-%m-%d')}")
                    
                    if analysis.get('status'):
                        st.write(f"Status: {analysis['status']}")
        else:
            st.info("No image analyses recorded")
    
    with chat_tab:
        st.subheader("Chat History")
        if st.session_state.medical_record.get('chat_history'):
            for chat in st.session_state.medical_record['chat_history']:
                with st.expander(f"Chat from {chat.get('timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M')}"):
                    st.write(f"{chat['type'].title()}: {chat['message']}")
                    if chat.get('extracted_info'):
                        st.write("Extracted Information:")
                        if chat['extracted_info'].get('medications'):
                            st.write("Medications:", chat['extracted_info']['medications'])
                        if chat['extracted_info'].get('symptoms'):
                            st.write("Symptoms:", chat['extracted_info']['symptoms'])
                        if chat['extracted_info'].get('vitals'):
                            st.write("Vitals:", chat['extracted_info']['vitals'])
        else:
            st.info("No chat history recorded")

def show_main_interface():
    """Show the main interface with navigation."""
    try:
        # Sidebar navigation
        st.sidebar.title("Navigation")
        page = st.sidebar.radio(
            "Go to",
            ["Chat", "Image Analysis", "Health Assessment", "Medical History", "Reports", "Speech Input"]
        )
        
        # Main content area
        st.title("Diabetes Assistant")
        
        if page == "Chat":
            show_chat_interface()
        elif page == "Image Analysis":
            show_image_analysis()
        elif page == "Health Assessment":
            show_health_assessment()
        elif page == "Medical History":
            show_medical_history()
        elif page == "Reports":
            show_reports()
        elif page == "Speech Input":
            show_speech_input()
            
    except Exception as e:
        logger.error(f"Error in main interface: {str(e)}")
        st.error("An error occurred while loading the interface. Please try again.")

def show_chat_interface():
    """Show the chat interface."""
    try:
        st.title("Chat with Diabetes Assistant")
        
        # Display chat messages
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.write("You: " + message["content"])
            else:
                st.write("Assistant: " + message["content"])
        
        # Chat input
        user_input = st.text_input("What would you like to know about diabetes?", key="chat_input")
        send_button = st.button("Send")
        
        if send_button and user_input:
            with st.spinner("Thinking..."):
                response = process_user_input(user_input)
                st.write("Assistant: " + response)
                
    except Exception as e:
        logger.error(f"Error in chat interface: {str(e)}")
        st.error("An error occurred while processing your message. Please try again.")

def show_speech_input():
    """Show the speech input interface."""
    try:
        st.title("Speech Input")
        st.write("Upload an audio file or use the microphone to ask questions about diabetes.")
        
        # File uploader for audio files
        audio_file = st.file_uploader(
            "Upload an audio file",
            type=["wav", "mp3", "m4a", "ogg"],
            help="Supported formats: WAV, MP3, M4A, OGG"
        )
        
        if audio_file:
            # Display audio player
            st.audio(audio_file)
            
            # Process button
            if st.button("Process Audio"):
                with st.spinner("Processing audio..."):
                    try:
                        # Save the uploaded file to a temporary location
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                            tmp_file.write(audio_file.getvalue())
                            tmp_path = tmp_file.name
                        
                        try:
                            # Process audio
                            transcription = st.session_state.speech_to_text.process_audio_file(tmp_path)
                            
                            if not transcription:
                                raise ValueError("No transcription generated")
                                
                            # Generate response using the same process as text input
                            response = process_user_input(transcription)
                            
                            # Save to database
                            if st.session_state.is_authenticated and st.session_state.user:
                                speech_data = {
                                    "user_id": str(st.session_state.user["_id"]),
                                    "transcription": transcription,
                                    "response": response,
                                    "timestamp": datetime.now()
                                }
                                db.speech_inputs.insert_one(speech_data)
                                logger.info("Speech input saved to database successfully")
                            
                            st.write("**Transcription:**")
                            st.write(transcription)
                            st.write("**Response:**")
                            st.write(response)
                            
                            # Add to chat history
                            if 'messages' not in st.session_state:
                                st.session_state.messages = []
                            st.session_state.messages.append({
                                "role": "user",
                                "content": transcription
                            })
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": response
                            })
                            
                        finally:
                            # Clean up temporary file
                            os.unlink(tmp_path)
                            
                    except Exception as e:
                        logger.error(f"Error processing audio: {str(e)}")
                        st.error("An error occurred while processing the audio. Please try again.")
                        
    except Exception as e:
        logger.error(f"Error in speech input interface: {str(e)}")
        st.error("An error occurred while loading the speech input interface. Please try again.")

def show_reports():
    """Display medical reports with crucial information"""
    st.subheader("📋 Medical Reports")
    
    if not st.session_state.is_authenticated:
        st.warning("Please log in to view your medical reports.")
        return

    # Add download PDF button at the top
    if st.button("📥 Download Complete Medical Report"):
        try:
            report_data = {
                'medications': st.session_state.medical_record.get('current_medications', []),
                'vitals': st.session_state.medical_record.get('vitals_history', []),
                'bills': st.session_state.medical_record.get('medical_bills', []),
                'images': st.session_state.medical_record.get('image_analyses', [])
            }
            pdf_file = generate_pdf_report(
                st.session_state.user, 
                report_data
            )
            if pdf_file:
                with open(pdf_file, "rb") as f:
                    pdf_bytes = f.read()
                st.download_button(
                    label="Click to Download PDF",
                    data=pdf_bytes,
                    file_name=f"medical_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf"
                )
                # Clean up the temporary file
                os.remove(pdf_file)
        except Exception as e:
            st.error(f"Error generating PDF: {str(e)}")
    
    # Display current medications
    st.subheader("Current Medications")
    if st.session_state.medical_record.get('current_medications'):
        meds_df = pd.DataFrame(st.session_state.medical_record['current_medications'])
        st.dataframe(meds_df)
    else:
        st.info("No current medications recorded.")
    
    # Display latest vitals
    st.subheader("Latest Vitals")
    if st.session_state.medical_record.get('vitals_history'):
        latest_vitals = st.session_state.medical_record['vitals_history'][-1]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Blood Glucose", f"{latest_vitals['blood_glucose']} mg/dL")
        with col2:
            st.metric("Blood Pressure", latest_vitals['blood_pressure'])
        with col3:
            st.metric("Heart Rate", f"{latest_vitals['heart_rate']} bpm")
    
    # Display image analysis history with individual PDF downloads
    st.subheader("Image Analysis History")
    if st.session_state.medical_record.get('image_analyses'):
        for idx, analysis in enumerate(st.session_state.medical_record['image_analyses']):
            with st.expander(f"Analysis {idx + 1} - {analysis['timestamp'].strftime('%Y-%m-%d %H:%M')}"):
                col1, col2 = st.columns([2, 1])
                with col1:
                    if analysis.get('image_data'):
                        image_bytes = base64.b64decode(analysis['image_data'])
                        st.image(image_bytes, width=300)
                with col2:
                    st.write("Analysis:")
                    st.write(analysis['analysis']['description'])
                    
                    # Add individual report download button
                    if st.button(f"📄 Download This Report", key=f"download_report_{idx}"):
                        pdf_file = generate_single_analysis_report(analysis)
                        if pdf_file:
                            with open(pdf_file, "rb") as f:
                                pdf_bytes = f.read()
                            st.download_button(
                                label="Download PDF",
                                data=pdf_bytes,
                                file_name=f"analysis_report_{analysis['timestamp'].strftime('%Y%m%d_%H%M')}.pdf",
                                mime="application/pdf",
                                key=f"download_button_{idx}"
                            )
                            os.remove(pdf_file)
    else:
        st.info("No image analyses recorded.")

def show_health_assessment():
    """Show the health assessment interface."""
    try:
        st.title("Health Assessment")
        
        if not st.session_state.is_authenticated:
            st.warning("Please log in to complete a health assessment")
            return
        
        # Load latest assessment from MongoDB
        try:
            latest_assessment = db.health_assessments.find_one(
                {"user_id": str(st.session_state.user["_id"])},
                sort=[("timestamp", -1)]
            )
            
            if latest_assessment:
                st.subheader("Current Assessment Results")
                
                # BMI Information
                st.write(f"BMI: {latest_assessment.get('bmi', 'Not available'):.1f} ({latest_assessment.get('bmi_category', 'Not available')})")
                
                # Risk Factors
                st.write("Risk Factors:")
                for factor, value in latest_assessment.get('risk_factors', {}).items():
                    if value:
                        st.write(f"- {factor.replace('_', ' ').title()}")
                
                # Lifestyle Factors
                st.write("Lifestyle Factors:")
                st.write(f"- Activity Level: {latest_assessment.get('lifestyle_factors', {}).get('activity_level', 'Not available')}")
                st.write(f"- Diet Type: {latest_assessment.get('lifestyle_factors', {}).get('diet_type', 'Not available')}")
                
                # Display current recommendations
                if latest_assessment.get('recommendations'):
                    st.subheader("Current Recommendations")
                    recommendations = latest_assessment['recommendations']
                    
                    st.write("Diet Plan:")
                    for meal in recommendations['diet_plan']['meals']:
                        st.write(f"- {meal}")
                    
                    st.write("\nExercise Plan:")
                    exercise_plan = recommendations['exercise_plan']
                    st.write(f"Frequency: {exercise_plan['frequency']}")
                    st.write(f"Duration: {exercise_plan['duration']}")
                    st.write("Recommended Activities:")
                    for activity_type, activities in exercise_plan.items():
                        if isinstance(activities, list):
                            for activity in activities:
                                st.write(f"- {activity}")
                    
                    st.write("\nLifestyle Recommendations:")
                    for rec in recommendations['lifestyle_changes']:
                        st.write(f"- {rec}")
                
                st.write("---")
        except Exception as e:
            logger.error(f"Error loading latest assessment: {str(e)}")
        
        with st.form("health_assessment_form"):
            st.subheader("Basic Information")
            age = st.number_input("Age", min_value=0, max_value=120, value=st.session_state.user.get('age', 30))
            weight = st.number_input("Weight (kg)", min_value=0.0, max_value=500.0, value=st.session_state.user.get('weight', 70.0))
            height = st.number_input("Height (cm)", min_value=0.0, max_value=300.0, value=st.session_state.user.get('height', 170.0))
            
            st.subheader("Health Conditions")
            blood_pressure = st.checkbox("Do you have blood pressure issues?", value=st.session_state.user.get('blood_pressure', False))
            heart_problems = st.checkbox("Do you have any heart problems?", value=st.session_state.user.get('heart_problem', False))
            kidney_issues = st.checkbox("Do you have any kidney issues?", value=st.session_state.user.get('kidney_issue', False))
            joint_pain = st.checkbox("Do you have any joint pain issues?", value=st.session_state.user.get('joint_pain', False))
            
            st.subheader("Lifestyle")
            activity_level = st.select_slider(
                "Activity Level",
                options=["Sedentary", "Lightly Active", "Moderately Active", "Very Active", "Extremely Active"],
                value=st.session_state.user.get('activity_level', "Moderately Active")
            )
            
            diet_type = st.select_slider(
                "Diet Type",
                options=["Regular", "Low Carb", "Low Sugar", "Mediterranean", "Vegan", "Keto"],
                value=st.session_state.user.get('diet_type', "Regular")
            )
            
            submit = st.form_submit_button("Submit Assessment")
            
            if submit:
                user_data = {
                    'age': age,
                    'weight': weight,
                    'height': height,
                    'blood_pressure': blood_pressure,
                    'heart_problem': heart_problems,
                    'kidney_issue': kidney_issues,
                    'joint_pain': joint_pain,
                    'activity_level': activity_level,
                    'diet_type': diet_type
                }
                
                # Calculate BMI
                bmi = weight / ((height/100) ** 2)
                bmi_category = "Underweight" if bmi < 18.5 else "Normal" if bmi < 25 else "Overweight" if bmi < 30 else "Obese"
                
                # Get recommendations
                recommendations = st.session_state.predictive_recommendation.get_recommendation(user_data)
                
                if recommendations:
                    # Create health assessment record
                    health_assessment = {
                        'user_id': str(st.session_state.user["_id"]),
                        'user_data': user_data,
                        'bmi': bmi,
                        'bmi_category': bmi_category,
                        'recommendations': recommendations,
                        'timestamp': datetime.now(),
                        'risk_factors': {
                            'blood_pressure': blood_pressure,
                            'heart_problem': heart_problems,
                            'kidney_issue': kidney_issues,
                            'joint_pain': joint_pain
                        },
                        'lifestyle_factors': {
                            'activity_level': activity_level,
                            'diet_type': diet_type
                        }
                    }
                    
                    # Save to database
                    try:
                        # Insert new assessment
                        db.health_assessments.insert_one(health_assessment)
                        logger.info("Health assessment saved to database successfully")
                        
                        # Update user's medical record
                        st.session_state.medical_record['current_condition'] = {
                            'diabetes_type': recommendations.get('diabetes_type', 'Unknown'),
                            'last_updated': datetime.now(),
                            'complications': [
                                'High Blood Pressure' if blood_pressure else None,
                                'Heart Problems' if heart_problems else None,
                                'Kidney Issues' if kidney_issues else None,
                                'Joint Pain' if joint_pain else None
                            ],
                            'bmi': bmi,
                            'bmi_category': bmi_category,
                            'activity_level': activity_level,
                            'diet_type': diet_type
                        }
                        st.session_state.medical_record['recommendations'] = recommendations
                        save_medical_record()
                        
                    except Exception as e:
                        logger.error(f"Error saving health assessment: {str(e)}")
                    
                    st.success("Assessment submitted successfully!")
                    st.experimental_rerun()
                    
    except Exception as e:
        logger.error(f"Error in health assessment interface: {str(e)}")
        st.error("An error occurred while processing your health assessment. Please try again.")

def show_image_analysis():
    """Show the image analysis interface."""
    try:
        st.title("Image Analysis")
        
        if not st.session_state.is_authenticated:
            st.warning("Please log in to use image analysis")
            return
        
        uploaded_file = st.file_uploader("Upload a medical image", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_file:
            image_bytes = uploaded_file.getvalue()
            st.image(image_bytes, caption="Uploaded Image", use_container_width=True)
            
            if st.button("Analyze Image"):
                with st.spinner("Analyzing image..."):
                    try:
                        # Process and store the image
                        result = process_image_input(uploaded_file)
                        
                        if result:
                            st.subheader("Analysis Results")
                            st.write(result)
                            
                            # Show success message
                            st.success("Image analysis saved successfully!")
                            
                            # Force a rerun to update the display
                            st.experimental_rerun()
                    except Exception as e:
                        logger.error(f"Error analyzing image: {str(e)}")
                        st.error("An error occurred while analyzing the image. Please try again.")
                        
    except Exception as e:
        logger.error(f"Error in image analysis interface: {str(e)}")
        st.error("An error occurred while loading the image analysis interface. Please try again.")

def extract_medical_info_from_chat(messages):
    """Extract medical information from chat history."""
    medical_info = {
        'prescriptions': [],
        'vitals': [],
        'symptoms': []
    }
    
    try:
        for msg in messages:
            if msg['role'] == 'user':
                content = msg['content'].lower()
                
                # Extract prescriptions
                if any(keyword in content for keyword in ['prescribed', 'medication', 'medicine', 'drug', 'pill']):
                    medical_info['prescriptions'].append({
                        'text': msg['content'],
                        'timestamp': msg.get('timestamp', datetime.now())
                    })
                
                # Extract vitals
                if any(keyword in content for keyword in ['blood sugar', 'glucose', 'blood pressure', 'heart rate', 'weight']):
                    medical_info['vitals'].append({
                        'text': msg['content'],
                        'timestamp': msg.get('timestamp', datetime.now())
                    })
                
                # Extract symptoms
                if any(keyword in content for keyword in ['symptom', 'feeling', 'pain', 'ache', 'discomfort']):
                    medical_info['symptoms'].append({
                        'text': msg['content'],
                        'timestamp': msg.get('timestamp', datetime.now())
                    })
        
        return medical_info
    except Exception as e:
        logger.error(f"Error extracting medical info from chat: {str(e)}")
        return medical_info

def generate_pdf_report(user_data, recommendations, image_analysis=None):
    """Generate a comprehensive PDF report including all user interactions."""
    try:
        # Create a PDF document
        filename = f"diabetes_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        doc = SimpleDocTemplate(filename, pagesize=letter)
        story = []
        
        # Add title
        title_style = ParagraphStyle(
            'Title',
            parent=getSampleStyleSheet()['Heading1'],
            fontSize=24,
            spaceAfter=30
        )
        story.append(Paragraph("Diabetes Health Assessment Report", title_style))
        
        # Add timestamp
        story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                             getSampleStyleSheet()['Normal']))
        story.append(Spacer(1, 20))
        
        # Add patient information
        story.append(Paragraph("Patient Information", getSampleStyleSheet()['Heading2']))
        if st.session_state.user:
            user_info = [
                f"Name: {st.session_state.user.get('name', 'Not provided')}",
                f"Email: {st.session_state.user.get('email', 'Not provided')}",
                f"Age: {st.session_state.user.get('age', 'Not provided')}",
                f"Gender: {st.session_state.user.get('gender', 'Not provided')}",
                f"Diabetes Type: {st.session_state.user.get('diabetes_type', 'Not provided')}",
                f"Medical History: {st.session_state.user.get('medical_history', 'Not provided')}"
            ]
            for info in user_info:
                story.append(Paragraph(info, getSampleStyleSheet()['Normal']))
        story.append(Spacer(1, 20))
        
        # Load latest assessment from MongoDB
        try:
            latest_assessment = db.health_assessments.find_one(
                {"user_id": str(st.session_state.user["_id"])},
                sort=[("timestamp", -1)]
            )
            
            if latest_assessment:
                # Add health assessment results
                story.append(Paragraph("Health Assessment Results", getSampleStyleSheet()['Heading2']))
                
                # BMI Information
                story.append(Paragraph(f"BMI: {latest_assessment.get('bmi', 'Not available'):.1f} ({latest_assessment.get('bmi_category', 'Not available')})", 
                                     getSampleStyleSheet()['Normal']))
                
                # Risk Factors
                story.append(Paragraph("Risk Factors:", getSampleStyleSheet()['Normal']))
                for factor, value in latest_assessment.get('risk_factors', {}).items():
                    if value:
                        story.append(Paragraph(f"• {factor.replace('_', ' ').title()}", 
                                             getSampleStyleSheet()['Normal']))
                
                # Lifestyle Factors
                story.append(Paragraph("Lifestyle Factors:", getSampleStyleSheet()['Normal']))
                story.append(Paragraph(f"• Activity Level: {latest_assessment.get('lifestyle_factors', {}).get('activity_level', 'Not available')}", 
                                     getSampleStyleSheet()['Normal']))
                story.append(Paragraph(f"• Diet Type: {latest_assessment.get('lifestyle_factors', {}).get('diet_type', 'Not available')}", 
                                     getSampleStyleSheet()['Normal']))
                
                story.append(Spacer(1, 20))
                
                # Add recommendations
                if latest_assessment.get('recommendations'):
                    story.append(Paragraph("Personalized Recommendations", getSampleStyleSheet()['Heading2']))
                    recommendations = latest_assessment['recommendations']
                    
                    # Diet Plan
                    story.append(Paragraph("Diet Plan", getSampleStyleSheet()['Heading3']))
                    for meal in recommendations['diet_plan']['meals']:
                        story.append(Paragraph(f"• {meal}", getSampleStyleSheet()['Normal']))
                    
                    # Exercise Plan
                    story.append(Paragraph("Exercise Plan", getSampleStyleSheet()['Heading3']))
                    exercise_plan = recommendations['exercise_plan']
                    story.append(Paragraph(f"Frequency: {exercise_plan['frequency']}", getSampleStyleSheet()['Normal']))
                    story.append(Paragraph(f"Duration: {exercise_plan['duration']}", getSampleStyleSheet()['Normal']))
                    story.append(Paragraph("Recommended Activities:", getSampleStyleSheet()['Normal']))
                    for activity_type, activities in exercise_plan.items():
                        if isinstance(activities, list):
                            for activity in activities:
                                story.append(Paragraph(f"• {activity}", getSampleStyleSheet()['Normal']))
                    
                    # Lifestyle Recommendations
                    story.append(Paragraph("Lifestyle Recommendations", getSampleStyleSheet()['Heading3']))
                    for rec in recommendations['lifestyle_changes']:
                        story.append(Paragraph(f"• {rec}", getSampleStyleSheet()['Normal']))
                    
                    story.append(Spacer(1, 20))
        except Exception as e:
            logger.error(f"Error loading latest assessment for report: {str(e)}")
        
        # Add current medications
        story.append(Paragraph("Current Medications", getSampleStyleSheet()['Heading2']))
        if st.session_state.medical_record.get('current_medications'):
            for med in st.session_state.medical_record['current_medications']:
                story.append(Paragraph(f"• {med['name']}", getSampleStyleSheet()['Normal']))
                story.append(Paragraph(f"  Dosage: {med.get('dosage', 'Not specified')}", 
                                     getSampleStyleSheet()['Normal']))
                story.append(Paragraph(f"  Frequency: {med.get('frequency', 'Not specified')}", 
                                     getSampleStyleSheet()['Normal']))
                story.append(Paragraph(f"  Status: {med.get('status', 'active')}", 
                                     getSampleStyleSheet()['Normal']))
                if med.get('notes'):
                    story.append(Paragraph(f"  Notes: {med['notes']}", 
                                         getSampleStyleSheet()['Normal']))
                story.append(Spacer(1, 5))
        else:
            story.append(Paragraph("No medications recorded", getSampleStyleSheet()['Normal']))
        story.append(Spacer(1, 20))
        
        # Add latest vitals
        story.append(Paragraph("Latest Vitals", getSampleStyleSheet()['Heading2']))
        if st.session_state.medical_record.get('vitals_history'):
            latest_vitals = st.session_state.medical_record['vitals_history'][-1]
            story.append(Paragraph(f"Date: {latest_vitals.get('date', datetime.now()).strftime('%Y-%m-%d %H:%M')}", 
                                 getSampleStyleSheet()['Normal']))
            story.append(Paragraph(f"Notes: {latest_vitals.get('notes', 'No notes')}", 
                                 getSampleStyleSheet()['Normal']))
        else:
            story.append(Paragraph("No vitals recorded", getSampleStyleSheet()['Normal']))
        story.append(Spacer(1, 20))
        
        # Add recent image analyses
        story.append(Paragraph("Recent Medical Image Analyses", getSampleStyleSheet()['Heading2']))
        if st.session_state.medical_record.get('image_analyses'):
            recent_analyses = sorted(
                st.session_state.medical_record['image_analyses'],
                key=lambda x: x.get('date', datetime.now()),
                reverse=True
            )[:5]  # Only last 5 analyses
            
            for analysis in recent_analyses:
                story.append(Paragraph(f"Analysis from {analysis.get('date', datetime.now()).strftime('%Y-%m-%d %H:%M')}", 
                                     getSampleStyleSheet()['Heading3']))
                if analysis.get('analysis_result'):
                    story.append(Paragraph(analysis['analysis_result'].get('description', 'No description available'), 
                                         getSampleStyleSheet()['Normal']))
                if analysis.get('doctor_notes'):
                    story.append(Paragraph(f"Doctor's Notes: {analysis['doctor_notes']}", 
                                         getSampleStyleSheet()['Normal']))
                if analysis.get('status'):
                    story.append(Paragraph(f"Status: {analysis['status']}", 
                                         getSampleStyleSheet()['Normal']))
                story.append(Spacer(1, 10))
        else:
            story.append(Paragraph("No recent image analyses available", getSampleStyleSheet()['Normal']))
        
        # Build PDF
        doc.build(story)
        return filename
    except Exception as e:
        logger.error(f"Error generating PDF report: {str(e)}")
        return None

def process_speech_input(audio_file):
    """Process speech input and save to database."""
    try:
        if not st.session_state.speech_to_text:
            raise ValueError("Speech-to-text converter not initialized")
            
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_file.getvalue())
            tmp_path = tmp_file.name
            
        try:
            # Process audio
            transcription = st.session_state.speech_to_text.process_audio_file(tmp_path)
            
            if not transcription:
                raise ValueError("No transcription generated")
                
            # Generate response using the same process as text input
            response = process_user_input(transcription)
            
            # Save to database
            if st.session_state.is_authenticated and st.session_state.user:
                speech_data = {
                    "user_id": str(st.session_state.user["_id"]),
                    "transcription": transcription,
                    "response": response,
                    "timestamp": datetime.now()
                }
                db.speech_inputs.insert_one(speech_data)
                logger.info("Speech input saved to database successfully")
            
            return {
                "transcription": transcription,
                "response": response
            }
            
        finally:
            # Clean up temporary file
            os.unlink(tmp_path)
            
    except Exception as e:
        logger.error(f"Error processing speech input: {str(e)}")
        return {
            "error": str(e),
            "transcription": None,
            "response": None
        }

def process_image_input(image_file, text_input=""):
    """Process image input and save to database."""
    try:
        if not check_api_rate_limit():
            return "Please wait a moment before making another request."
        
        # Read image file
        image_bytes = image_file.getvalue()
        
        # Analyze image
        analysis = st.session_state.image_analyzer.analyze_image(image_bytes)
        
        if "error" in analysis:
            return f"Error analyzing image: {analysis['error']}"
        
        # Generate response
        response = f"Image Analysis Results:\n{analysis['description']}"
        
        # Save to database
        if st.session_state.is_authenticated and st.session_state.user:
            try:
                # Create image analysis record with consistent field names
                image_analysis_record = {
                    "user_id": str(st.session_state.user["_id"]),
                    "image_data": base64.b64encode(image_bytes).decode('utf-8'),
                    "analysis_result": {
                        "description": analysis['description'],
                        "timestamp": datetime.now().isoformat(),
                        "analysis_type": analysis.get('analysis_type', 'medical_report')
                    },
                    "date": datetime.now(),
                    "image_type": image_file.type,
                    "status": "pending",
                    "doctor_notes": "",
                    "follow_up_date": None
                }
                
                # Save to image_analyses collection
                db.image_analyses.insert_one(image_analysis_record)
                logger.info("Image analysis saved to image_analyses collection")
                
                # Update user's medical record
                db.users.update_one(
                    {"_id": st.session_state.user["_id"]},
                    {"$push": {"medical_record.image_analyses": image_analysis_record}}
                )
                logger.info("Image analysis saved to user's medical record")
                
                # Update session state
                if 'medical_record' not in st.session_state:
                    st.session_state.medical_record = {
                        'medical_bills': [],
                        'current_medications': [],
                        'vitals_history': [],
                        'image_analyses': [],
                        'current_condition': {},
                        'chat_history': []
                    }
                
                if 'image_analyses' not in st.session_state.medical_record:
                    st.session_state.medical_record['image_analyses'] = []
                
                st.session_state.medical_record['image_analyses'].append(image_analysis_record)
                logger.info("Image analysis saved to session state")
                
            except Exception as e:
                logger.error(f"Error saving image analysis to database: {str(e)}")
                return "Error saving image analysis. Please try again."
        
        return response
                
    except Exception as e:
        logger.error(f"Error processing image input: {str(e)}")
        return "I apologize, but I encountered an error while processing your image. Please try again."

def generate_single_analysis_report(analysis_data):
    """Generate a PDF report for a single image analysis."""
    try:
        filename = f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        doc = SimpleDocTemplate(filename, pagesize=letter)
        story = []

        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=getSampleStyleSheet()['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1  # Center alignment
        )
        story.append(Paragraph("Medical Image Analysis Report", title_style))
        story.append(Spacer(1, 20))

        # Date and Time
        story.append(Paragraph(
            f"Analysis Date: {analysis_data['timestamp'].strftime('%Y-%m-%d %H:%M')}",
            getSampleStyleSheet()['Normal']
        ))
        story.append(Spacer(1, 20))

        # Add image if available
        if analysis_data.get('image_data'):
            image_data = base64.b64decode(analysis_data['image_data'])
            img_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            img_temp.write(image_data)
            img_temp.close()
            
            img = RLImage(img_temp.name, width=400, height=300)
            story.append(img)
            story.append(Spacer(1, 20))
            
            # Clean up temporary image file
            os.unlink(img_temp.name)

        # Analysis Results
        story.append(Paragraph("Analysis Results", getSampleStyleSheet()['Heading2']))
        story.append(Spacer(1, 10))
        
        # Format the analysis description
        description = analysis_data['analysis']['description']  # Use original field name
        for line in description.split('\n'):
            if line.strip():
                story.append(Paragraph(line.strip(), getSampleStyleSheet()['Normal']))
                story.append(Spacer(1, 5))

        # Build PDF
        doc.build(story)
        return filename
    except Exception as e:
        logger.error(f"Error generating single analysis report: {str(e)}")
        return None

def save_medical_record():
    """Save the current medical record to MongoDB"""
    if st.session_state.is_authenticated and st.session_state.user:
        try:
            db.users.update_one(
                {"_id": st.session_state.user["_id"]},
                {"$set": {
                    "medical_record": st.session_state.medical_record,
                    "last_updated": datetime.now()
                }}
            )
            logger.info("Medical record saved successfully")
        except Exception as e:
            logger.error(f"Error saving medical record: {str(e)}")

def main():
    """Main function to run the Streamlit app."""
    try:
        # Initialize session state
        initialize_session_state()
        
        # Show appropriate interface based on login status
        if not st.session_state.is_logged_in:
            show_login_form()
        else:
            show_main_interface()
            
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        st.error("An error occurred. Please try refreshing the page or contact support.")

if __name__ == "__main__":
    main()
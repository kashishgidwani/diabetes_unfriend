# model_manager.py
import logging
import os
import time
from fallback_responses import generate_fallback_response
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_manager.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class ModelManager:
    _instance = None
    _initialized = False
    _model = None
    last_api_call = 0
    rate_limit_delay = 0.5  # Reduced delay to 0.5 seconds between API calls

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._initialize()

    def _initialize(self):
        """Initialize Gemini model"""
        try:
            # Load environment variables
            load_dotenv()
            
            # Get API key from environment
            api_key = os.getenv('GOOGLE_API_KEY')
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment variables")
            
            # Configure Gemini
            genai.configure(api_key=api_key)
            
            # Set default generation config
            generation_config = {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "max_output_tokens": 1024,
            }
            
            # Initialize the model with Gemini 1.5 Flash
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }
            
            # List available models
            available_models = []
            for m in genai.list_models():
                logger.info(f"Available model: {m.name}")
                available_models.append(m.name)
            
            # Try to use Gemini 1.5 Flash model
            model_name = "models/gemini-1.5-flash"
            if model_name not in available_models:
                logger.warning(f"Model {model_name} not found, trying fallback models")
                # Try fallback models
                fallback_models = [
                    "models/gemini-1.5-flash-latest",
                    "models/gemini-1.5-flash-001",
                    "models/gemini-1.5-flash-002",
                    "models/gemini-1.5-pro"  # Last resort fallback
                ]
                for fallback in fallback_models:
                    if fallback in available_models:
                        model_name = fallback
                        logger.info(f"Using fallback model: {model_name}")
                        break
                else:
                    logger.error("No suitable Gemini models found")
                    raise ValueError("No suitable Gemini models found")
            
            self._model = genai.GenerativeModel(
                model_name=model_name,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            logger.info(f"Model manager initialized successfully with {model_name}")
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Error initializing Gemini model: {str(e)}")
            self._initialized = False

    def _check_rate_limit(self):
        """Check and enforce rate limiting."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_api_call
        
        if time_since_last_request < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last_request
            logger.info(f"Rate limit: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self.last_api_call = time.time()

    def generate_content(self, prompt):
        """Generate content using Gemini model"""
        self._check_rate_limit()
        
        try:
            response = self._model.generate_content(prompt)
            
            if not response or not response.text:
                logger.warning("Empty or invalid response received")
                return "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
            
            return response.text
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error generating content: {error_msg}")
            
            if "429" in error_msg:  # Rate limit error
                # Try to switch to a different model if available
                if self._model.model_name != "models/gemini-1.5-flash":
                    try:
                        self._model = genai.GenerativeModel("models/gemini-1.5-flash")
                        logger.info("Switched to Gemini 1.5 Flash model")
                        return self.generate_content(prompt)  # Retry with new model
                    except:
                        pass
                return "I'm receiving too many requests right now. Please wait a moment and try again."
            elif "quota" in error_msg.lower():
                return "I've reached my quota limit. Please try again later."
            else:
                return "I encountered an error while processing your request. Please try again."

    def generate_vision_response(self, image_bytes, prompt):
        """Generate response for image analysis using Gemini Vision"""
        try:
            # Check rate limit
            self._check_rate_limit()
            
            # If model is not initialized, use fallback
            if not self._initialized:
                logger.info("Using fallback response generator")
                return generate_fallback_response(prompt)
            
            # Initialize Gemini Pro Vision model
            vision_model = genai.GenerativeModel('models/gemini-1.5-pro-vision-latest')
            
            # Create image part for the model
            image_part = {"mime_type": "image/jpeg", "data": image_bytes}
            
            # Generate response
            response = vision_model.generate_content(
                [prompt, image_part],
                generation_config={
                    "max_output_tokens": 1024,
                    "temperature": 0.4,
                    "top_p": 0.8,
                    "top_k": 40
                }
            )
            
            return response.text if response.text else generate_fallback_response(prompt)
            
        except Exception as e:
            logger.error(f"Error generating vision response: {str(e)}")
            return generate_fallback_response(prompt)

# Create a global instance
model_manager = ModelManager()
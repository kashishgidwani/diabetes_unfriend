from pymongo import MongoClient
from dotenv import load_dotenv
import os
from datetime import datetime
import bcrypt
import logging
from typing import Dict, List, Optional, Tuple
import re
from bson import ObjectId

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('db_utils.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def init_collections(db):
    """Initialize MongoDB collections with proper schemas and indexes."""
    # Patients collection
    if 'patients' not in db.list_collection_names():
        patients = db.create_collection('patients')
        patients.create_index('patient_id', unique=True)
        patients.create_index('name')

    # Images collection
    if 'images' not in db.list_collection_names():
        images = db.create_collection('images')
        images.create_index('patient_id')
        images.create_index('upload_date')

    # Interactions collection
    if 'interactions' not in db.list_collection_names():
        interactions = db.create_collection('interactions')
        interactions.create_index([('patient_id', 1), ('timestamp', -1)])

    # Recommendations collection
    if 'recommendations' not in db.list_collection_names():
        recommendations = db.create_collection('recommendations')
        recommendations.create_index([('patient_id', 1), ('date', -1)])

def get_database():
    """
    Get the MongoDB database instance using the connection URI from environment variables.

    Returns:
        pymongo.database.Database: A MongoDB database instance.

    Raises:
        ValueError: If MONGODB_URI environment variable is not set.
        ConnectionError: If connection to MongoDB fails.
        Exception: For other unexpected errors during database connection.

    Example:
        >>> db = get_database()
        >>> db.list_collection_names()  # List all collections in the database
    """
    try:
        # Get MongoDB URI from environment variables
        mongo_uri = os.getenv('MONGODB_URI')
        if not mongo_uri:
            logger.error("MONGODB_URI environment variable not found")
            raise ValueError("MONGODB_URI environment variable is not set")
        
        try:
            # Create a connection using MongoClient
            client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
            # Test the connection
            client.server_info()
        except Exception as conn_err:
            logger.error(f"Failed to connect to MongoDB: {str(conn_err)}")
            raise ConnectionError(f"Could not connect to MongoDB: {str(conn_err)}")
        
        # Get the database name from the connection string
        db_name = mongo_uri.split('/')[-1].split('?')[0]
        
        # Get database
        db = client[db_name]
        logger.info("Successfully connected to MongoDB database")
        
        return db
    except Exception as e:
        logger.error(f"Unexpected error while connecting to database: {str(e)}")
        raise Exception(f'Database connection error: {str(e)}')

def test_connection():
    """Test the MongoDB connection and initialize collections."""
    try:
        db = get_database()
        # Try to execute a simple command to test the connection
        db.command('ping')
        # Initialize collections
        init_collections(db)
        print('Successfully connected to MongoDB and initialized collections!')
        return True
    except Exception as e:
        print(f'Failed to connect to MongoDB: {str(e)}')
        return False

def add_patient(db, name, medical_history=None):
    """Add a new patient to the database."""
    patient_data = {
        'patient_id': str(uuid.uuid4()),
        'name': name,
        'medical_history': medical_history or [],
        'registration_date': datetime.utcnow(),
        'last_visit': datetime.utcnow()
    }
    db.patients.insert_one(patient_data)
    return patient_data['patient_id']

def add_interaction(db, patient_id, message, response):
    """Log a chat interaction."""
    interaction = {
        'patient_id': patient_id,
        'timestamp': datetime.utcnow(),
        'message': message,
        'response': response
    }
    db.interactions.insert_one(interaction)

def add_image(db, patient_id, image_name, extracted_text):
    """Store image information."""
    image_data = {
        'patient_id': patient_id,
        'image_name': image_name,
        'extracted_text': extracted_text,
        'upload_date': datetime.utcnow()
    }
    db.images.insert_one(image_data)

def add_recommendation(db, patient_id, recommendation_type, details):
    """Add a medical recommendation."""
    recommendation = {
        'patient_id': patient_id,
        'type': recommendation_type,
        'details': details,
        'date': datetime.utcnow()
    }
    db.recommendations.insert_one(recommendation)

class DatabaseManager:
    _instance = None
    _initialized = False
    _client = None
    _db = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._initialize()

    def _initialize(self):
        """Initialize MongoDB connection"""
        try:
            self._client = MongoClient(os.getenv('MONGODB_URI'))
            self._db = self._client.get_database()
            logger.info("MongoDB connection established successfully")
            self._initialized = True
        except Exception as e:
            logger.error(f"Error connecting to MongoDB: {str(e)}")
            self._initialized = False

    def get_database(self):
        """Get MongoDB database instance"""
        if not self._initialized:
            self._initialize()
        return self._db

    def register_user(self, name, email, password, diabetes_type, age, gender, medical_history=""):
        """Register a new user"""
        try:
            # Check if user already exists
            if self._db.users.find_one({"email": email}):
                return False, "User with this email already exists"
            
            # Create new user
            user = {
                "name": name,
                "email": email,
                "password": password,  # In production, this should be hashed
                "diabetes_type": diabetes_type,
                "age": age,
                "gender": gender,
                "medical_history": medical_history,
                "created_at": datetime.now()
            }
            
            result = self._db.users.insert_one(user)
            return True, "User registered successfully"
        except Exception as e:
            logger.error(f"Error registering user: {str(e)}")
            return False, f"Error registering user: {str(e)}"

    def authenticate_user(self, email, password):
        """Authenticate a user"""
        try:
            user = self._db.users.find_one({"email": email, "password": password})
            return user
        except Exception as e:
            logger.error(f"Error authenticating user: {str(e)}")
            return None

    def save_conversation(self, user_id, message, response):
        """Save a conversation to the database"""
        try:
            conversation = {
                "user_id": user_id,
                "message": message,
                "response": response,
                "timestamp": datetime.now()
            }
            
            result = self._db.conversations.insert_one(conversation)
            logger.info(f"Conversation saved successfully with ID: {result.inserted_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving conversation: {str(e)}")
            return False

    def get_user_conversations(self, user_id, limit=10):
        """Get recent conversations for a user"""
        try:
            conversations = self._db.conversations.find(
                {"user_id": user_id}
            ).sort("timestamp", -1).limit(limit)
            
            return list(conversations)
        except Exception as e:
            logger.error(f"Error retrieving conversations: {str(e)}")
            return []

    def get_conversation_keywords(self, user_id):
        """Get common keywords from user conversations"""
        try:
            # This is a simple implementation - in production, you might want to use NLP
            conversations = self._db.conversations.find({"user_id": user_id})
            
            # Extract keywords (simple implementation)
            keywords = {}
            for conv in conversations:
                words = conv["message"].lower().split()
                for word in words:
                    if len(word) > 3:  # Only consider words longer than 3 characters
                        keywords[word] = keywords.get(word, 0) + 1
            
            # Convert to list of dictionaries
            keyword_list = [{"keyword": k, "frequency": v} for k, v in keywords.items()]
            
            # Sort by frequency
            keyword_list.sort(key=lambda x: x["frequency"], reverse=True)
            
            return keyword_list[:10]  # Return top 10 keywords
        except Exception as e:
            logger.error(f"Error extracting keywords: {str(e)}")
            return []

    def list_all_users(self):
        """List all users (for debugging)"""
        try:
            return list(self._db.users.find({}, {"password": 0}))
        except Exception as e:
            logger.error(f"Error listing users: {str(e)}")
            return []

# Create a global instance
db_manager = DatabaseManager()

# Example usage:
if __name__ == "__main__":
    try:
        db = DatabaseManager()
        print("Successfully connected to MongoDB")
    except Exception as e:
        print(f"Error connecting to MongoDB: {str(e)}")
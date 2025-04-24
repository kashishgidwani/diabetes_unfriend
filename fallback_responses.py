import re
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fallback_responses.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Common diabetes-related terms and their responses
DIABETES_TERMS = {
    "blood sugar": {
        "normal_range": "70-100 mg/dL (fasting), 80-130 mg/dL (before meals), <180 mg/dL (2 hours after meals)",
        "tips": [
            "Monitor your blood sugar regularly as recommended by your healthcare provider",
            "Keep a log of your readings to track patterns",
            "Learn to recognize signs of high and low blood sugar",
            "Always carry fast-acting carbohydrates for low blood sugar emergencies"
        ]
    },
    "insulin": {
        "types": ["Rapid-acting", "Short-acting", "Intermediate-acting", "Long-acting"],
        "tips": [
            "Store insulin properly (refrigerated, not frozen)",
            "Rotate injection sites to prevent lipodystrophy",
            "Check expiration dates regularly",
            "Never share insulin pens or needles"
        ]
    },
    "diet": {
        "recommendations": [
            "Focus on whole grains, lean proteins, and healthy fats",
            "Limit refined carbohydrates and sugary foods",
            "Eat regular, balanced meals",
            "Stay hydrated with water",
            "Consider working with a registered dietitian"
        ]
    },
    "exercise": {
        "benefits": [
            "Helps lower blood sugar levels",
            "Improves insulin sensitivity",
            "Reduces cardiovascular risk",
            "Helps with weight management"
        ],
        "tips": [
            "Check blood sugar before and after exercise",
            "Stay hydrated during physical activity",
            "Carry fast-acting carbohydrates during exercise",
            "Start slowly and gradually increase intensity"
        ]
    },
    "complications": {
        "types": [
            "Diabetic retinopathy (eye problems)",
            "Diabetic neuropathy (nerve damage)",
            "Diabetic nephropathy (kidney problems)",
            "Cardiovascular disease",
            "Foot problems"
        ],
        "prevention": [
            "Maintain good blood sugar control",
            "Get regular check-ups",
            "Manage blood pressure and cholesterol",
            "Take care of your feet daily"
        ]
    }
}

def extract_key_terms(text):
    """Extract key diabetes-related terms from the text"""
    text = text.lower()
    found_terms = []
    
    for term in DIABETES_TERMS.keys():
        if term in text:
            found_terms.append(term)
    
    # Look for numbers that might be blood sugar readings
    numbers = re.findall(r'\d+', text)
    blood_sugar_readings = [num for num in numbers if 40 <= int(num) <= 400]  # Reasonable blood sugar range
    
    return found_terms, blood_sugar_readings

def generate_fallback_response(prompt):
    """Generate a helpful response when the main model is unavailable"""
    try:
        # Extract key terms and potential blood sugar readings
        found_terms, blood_sugar_readings = extract_key_terms(prompt)
        
        # Start building the response
        response = "I understand you're asking about diabetes. Here's what I can tell you:\n\n"
        
        # Add information about found terms
        if found_terms:
            response += "Based on your query, here's relevant information:\n\n"
            for term in found_terms:
                term_info = DIABETES_TERMS[term]
                response += f"About {term}:\n"
                
                if "normal_range" in term_info:
                    response += f"Normal range: {term_info['normal_range']}\n"
                
                if "tips" in term_info:
                    response += "Important tips:\n"
                    for tip in term_info["tips"]:
                        response += f"- {tip}\n"
                
                if "recommendations" in term_info:
                    response += "Recommendations:\n"
                    for rec in term_info["recommendations"]:
                        response += f"- {rec}\n"
                
                if "benefits" in term_info:
                    response += "Benefits:\n"
                    for benefit in term_info["benefits"]:
                        response += f"- {benefit}\n"
                
                response += "\n"
        
        # Add information about blood sugar readings if found
        if blood_sugar_readings:
            response += "I noticed some blood sugar readings in your message. Here's what you should know:\n"
            response += "- Keep a log of your readings\n"
            response += "- Share these numbers with your healthcare provider\n"
            response += "- Note any patterns or unusual readings\n\n"
        
        # Add general tips if no specific terms were found
        if not found_terms and not blood_sugar_readings:
            response += "Here are some general tips for diabetes management:\n"
            response += "- Monitor your blood sugar regularly\n"
            response += "- Follow a balanced diet\n"
            response += "- Stay physically active\n"
            response += "- Take medications as prescribed\n"
            response += "- Schedule regular check-ups with your healthcare provider\n\n"
        
        response += "For more detailed information, please try asking your question again in a few moments."
        return response
        
    except Exception as e:
        logger.error(f"Error generating fallback response: {str(e)}")
        return "I apologize, but I'm having trouble processing your request. Please try again in a few moments." 
# Diabetes Assistant

A comprehensive web application for diabetes management and health monitoring.

## Features

- Health Assessment
- Medical Image Analysis
- Personalized Recommendations
- Medical History Tracking
- Report Generation
- Speech Input Support
- Chat Interface

## Prerequisites

- Python 3.10
- MongoDB
- FFmpeg (for audio processing)
- Required Python packages (see requirements.txt)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/kashishgidwani/diabetes_unfriend.git
cd diabetes_unfriend
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your configuration:
```
MONGODB_URI=your_mongodb_uri
GOOGLE_API_KEY=your_google_api_key
```

## Usage

1. Start the application:
```bash
streamlit run diabetes_assistant.py
```

2. Open your browser and navigate to `http://localhost:8501`

## Deployment

### Deploying to Streamlit Cloud

1. Create a Streamlit Cloud account at https://streamlit.io/cloud

2. Connect your GitHub repository to Streamlit Cloud

3. Configure your environment variables in Streamlit Cloud:
   - MONGODB_URI
   - GOOGLE_API_KEY

4. Deploy your app

### Deploying to Heroku

1. Create a Heroku account and install the Heroku CLI

2. Create a new Heroku app:
```bash
heroku create your-app-name
```

3. Set environment variables:
```bash
heroku config:set MONGODB_URI=your_mongodb_uri
heroku config:set GOOGLE_API_KEY=your_google_api_key
```

4. Deploy to Heroku:
```bash
git push heroku main
```

### Deploying to AWS

1. Create an AWS account and set up AWS CLI

2. Create an EC2 instance

3. Install required dependencies on the EC2 instance

4. Set up environment variables

5. Run the application using:
```bash
streamlit run diabetes_assistant.py
```

## Project Structure

- `diabetes_assistant.py`: Main application file
- `model_manager.py`: Model management and predictions
- `db_utils.py`: Database utilities
- `pdf_report_generator.py`: PDF report generation
- `speech_to_text.py`: Speech-to-text conversion
- `requirements.txt`: Project dependencies

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Streamlit for the web framework
- MongoDB for database
- Google's Gemini API for image analysis
- Whisper for speech-to-text conversion

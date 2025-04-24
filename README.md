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

- Python 3.8+
- MongoDB
- FFmpeg (for audio processing)
- Required Python packages (see requirements.txt)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/kashishgidwani9/diabetes-assistant.git
cd diabetes-assistant
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
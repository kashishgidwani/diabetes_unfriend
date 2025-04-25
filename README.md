# Diabetes Assistant

A comprehensive AI-powered diabetes management and analysis application that helps users track, analyze, and understand their diabetes-related health data.

## 🌟 Features

- **Voice & Text Input**: Interact with the assistant through voice commands or text input
- **Image Analysis**: Upload and analyze diabetes-related images (e.g., food, wounds, glucose readings)
- **Health Data Tracking**: Monitor blood glucose, blood pressure, and heart rate
- **Medication Management**: Track current medications and dosages
- **Medical Report Generation**: Generate comprehensive PDF reports of your health data
- **AI-Powered Insights**: Get personalized insights and recommendations
- **Secure Authentication**: Protected user data with secure login system

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- FFmpeg (for audio processing)
- MongoDB (for data storage)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/kashishgidwani/diabetes_unfriend.git
   cd diabetes_unfriend
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv myenv
   source myenv/bin/activate  # On Windows: myenv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   Create a `.env` file in the root directory with:
   ```
   MONGODB_URI=your_mongodb_uri
   GOOGLE_API_KEY=your_google_api_key
   ```

5. Run the application:
   ```bash
   streamlit run diabetes_assistant.py
   ```

## 📊 Features in Detail

### Voice & Text Input
- Natural language processing for understanding user queries
- Voice command support for hands-free operation
- Multi-language support

### Image Analysis
- Upload and analyze diabetes-related images
- Get AI-powered insights about food, wounds, or glucose readings
- Store and track image analysis history

### Health Data Tracking
- Record and monitor vital signs
- Track blood glucose levels
- Monitor blood pressure and heart rate
- View historical trends

### Medication Management
- Track current medications
- Set reminders for medication
- View medication history
- Get medication-related insights

### Medical Reports
- Generate comprehensive PDF reports
- Include all health data and analyses
- Download and share reports
- Track progress over time

## 🔒 Security

- Secure user authentication
- Encrypted data storage
- Protected API keys
- Regular security updates

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for providing the AI models
- Streamlit for the web framework
- MongoDB for database support
- All contributors and users of the application

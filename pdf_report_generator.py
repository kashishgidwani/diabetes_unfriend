from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PDFReportGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.styles.add(ParagraphStyle(
            name='MedicalTitle',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=20,
            textColor=colors.darkblue
        ))
        self.styles.add(ParagraphStyle(
            name='MedicalSubtitle',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=15,
            textColor=colors.darkblue
        ))
        self.styles.add(ParagraphStyle(
            name='MedicalText',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=10
        ))

    def create_report(self, user_data, assessment_data, image_analysis=None, conversation_history=None):
        try:
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"medical_report_{timestamp}.pdf"
            
            # Create document
            doc = SimpleDocTemplate(
                filename,
                pagesize=letter,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            
            # Create story (content)
            story = []
            
            # Add title
            story.append(Paragraph("Medical Assessment Report", self.styles['MedicalTitle']))
            story.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y')}", self.styles['MedicalText']))
            story.append(Spacer(1, 20))
            
            # Patient Information
            story.append(Paragraph("Patient Information", self.styles['MedicalSubtitle']))
            patient_data = [
                ["Name:", user_data.get('name', 'Not provided')],
                ["Age:", str(user_data.get('age', 'Not provided'))],
                ["Gender:", user_data.get('gender', 'Not provided')],
                ["Contact:", user_data.get('contact', 'Not provided')],
                ["Email:", user_data.get('email', 'Not provided')]
            ]
            patient_table = Table(patient_data, colWidths=[1.5*inch, 3*inch])
            patient_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(patient_table)
            story.append(Spacer(1, 20))
            
            # Health Metrics
            story.append(Paragraph("Health Assessment", self.styles['MedicalSubtitle']))
            health_data = [
                ["Blood Pressure:", assessment_data.get('blood_pressure', 'Not provided')],
                ["Blood Sugar:", assessment_data.get('blood_sugar', 'Not provided')],
                ["BMI:", assessment_data.get('bmi', 'Not provided')],
                ["Cholesterol:", assessment_data.get('cholesterol', 'Not provided')],
                ["Predicted Diabetes Type:", assessment_data.get('predicted_type', 'Not provided')]
            ]
            health_table = Table(health_data, colWidths=[1.5*inch, 3*inch])
            health_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(health_table)
            story.append(Spacer(1, 20))
            
            # Treatment Plan
            story.append(Paragraph("Personalized Treatment Plan", self.styles['MedicalSubtitle']))
            
            # Dietary Recommendations
            story.append(Paragraph("Dietary Recommendations:", self.styles['MedicalText']))
            diet_plan = assessment_data.get('diet_plan', [])
            for item in diet_plan:
                story.append(Paragraph(f"• {item}", self.styles['MedicalText']))
            story.append(Spacer(1, 10))
            
            # Exercise Plan
            story.append(Paragraph("Exercise Plan:", self.styles['MedicalText']))
            exercise_plan = assessment_data.get('exercise_plan', [])
            for item in exercise_plan:
                story.append(Paragraph(f"• {item}", self.styles['MedicalText']))
            story.append(Spacer(1, 10))
            
            # Lifestyle Modifications
            story.append(Paragraph("Lifestyle Modifications:", self.styles['MedicalText']))
            lifestyle_plan = assessment_data.get('lifestyle_plan', [])
            for item in lifestyle_plan:
                story.append(Paragraph(f"• {item}", self.styles['MedicalText']))
            story.append(Spacer(1, 20))
            
            # Image Analysis (if available)
            if image_analysis:
                story.append(Paragraph("Medical Image Analysis", self.styles['MedicalSubtitle']))
                story.append(Paragraph(image_analysis, self.styles['MedicalText']))
                story.append(Spacer(1, 20))
            
            # Conversation History (if available)
            if conversation_history:
                story.append(Paragraph("Consultation Notes", self.styles['MedicalSubtitle']))
                for message in conversation_history:
                    role = "Patient" if message['role'] == 'user' else "Doctor"
                    story.append(Paragraph(f"{role}: {message['content']}", self.styles['MedicalText']))
                story.append(Spacer(1, 20))
            
            # Medical Disclaimer
            story.append(Paragraph("Medical Disclaimer", self.styles['MedicalSubtitle']))
            disclaimer = """
            This report is generated based on the information provided and should not be considered as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
            """
            story.append(Paragraph(disclaimer, self.styles['MedicalText']))
            
            # Build PDF
            doc.build(story)
            logger.info(f"PDF report generated successfully: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error generating PDF report: {str(e)}")
            raise 
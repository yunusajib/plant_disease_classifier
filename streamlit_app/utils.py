# streamlit_app/utils.py
from datetime import datetime

plant_care_tips = {
    "Corn_(maize)___healthy": {
        "short_term": [
            "Continue regular watering schedule",
            "Monitor for any changes in leaf color",
            "Maintain good air circulation"
        ],
        "long_term": [
            "Regular soil testing",
            "Crop rotation planning",
            "Preventive pest management"
        ]
    },
    "Tomato___Late_blight": {
        "short_term": [
            "Remove infected leaves immediately",
            "Apply appropriate fungicide",
            "Improve air circulation around plants"
        ],
        "long_term": [
            "Use resistant varieties next season",
            "Improve soil drainage",
            "Practice crop rotation"
        ]
    }
}

def format_prediction_history(history_item):
    return {
        'timestamp': history_item['timestamp'],
        'prediction': history_item['prediction'].replace('_', ' '),
        'confidence': f"{history_item['confidence']*100:.2f}%"
    }
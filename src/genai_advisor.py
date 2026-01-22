def generate_health_advice(prediction):
    if prediction == 1:
        return (
            "High risk of diabetes detected.\n"
            "• Reduce sugar intake\n"
            "• Exercise daily\n"
            "• Monitor blood glucose\n"
            "• Consult a doctor"
        )
    else:
        return (
            "Low risk of diabetes.\n"
            "• Maintain healthy diet\n"
            "• Stay physically active\n"
            "• Regular health checkups"
        )

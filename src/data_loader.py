import google.generativeai as genai

# Replace with your actual API key
# For production, it's best to use environment variables
API_KEY = "YOUR_API_KEY"
genai.configure(api_key=API_KEY)

def get_gemini_prediction(prompt):
    """
    Sends a prompt to the Gemini model and returns the text response.
    """
    model = genai.GenerativeModel('gemini-1.0-pro')
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Example usage in main.py
if __name__ == "__main__":
    prompt = "Predict the winners and final scores for the following EPL matches: Man City vs Liverpool, Arsenal vs Chelsea."
    prediction_text = get_gemini_prediction(prompt)

    if prediction_text:
        print("\nGemini's Prediction:")
        print(prediction_text)


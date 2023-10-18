import openai
import speech_recognition as sr
import pyttsx3

# Set your OpenAI API key
openai.api_key = "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"



# Initialize the TTS engine
tts_engine = pyttsx3.init()
tts_engine = None

def initialize_tts():
    global tts_engine
    if tts_engine is None:
        tts_engine = pyttsx3.init()

# Initialize the recognizer outside of the speech recognition function
recognizer = sr.Recognizer()

def recognize_speech():
    with sr.Microphone() as source:
        print("Listening for your input...")
        audio = recognizer.listen(source)

    try:
        user_input = recognizer.recognize_google(audio)
        return user_input
    except sr.UnknownValueError:
        print("Sorry, I could not understand your speech.")
        return ""
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
        return ""

def chat_with_gpt(user_input):
    if not user_input:
        return "Sorry, I didn't understand your input."

    response = openai.Completion.create(
        model="text-davinci-003",  
        prompt=f"You: {user_input}",
        max_tokens=50  # Adjust this based on your desired response length
    )
    return response.choices[0].text.strip()

def text_to_speech(response):
    initialize_tts()  # Ensure TTS engine is initialized before use
    tts_engine.say(response)
    tts_engine.runAndWait()

def process_text_prompt(user_prompt):
    response = chat_with_gpt(user_prompt)
    text_to_speech(response)
    return response

def process_voice_command():
    user_input = recognize_speech()
    response = chat_with_gpt(user_input)
    text_to_speech(response)
    return response


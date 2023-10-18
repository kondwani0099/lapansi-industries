import openai
import speech_recognition as sr
import pyttsx3

# Set your OpenAI API key
openai.api_key = "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"

def recognize_speech():
    recognizer = sr.Recognizer()

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
        engine="davinci",
        prompt=f"You: {user_input}",
        max_tokens=50  # Adjust this based on your desired response length
    )
    return response.choices[0].text.strip()

def text_to_speech(response):
    engine = pyttsx3.init()
    engine.say(response)
    engine.runAndWait()

while True:
    user_input = recognize_speech()
    if user_input.lower() == "quit":
        break
    response = chat_with_gpt(user_input)
    print(f"AI: {response}")
    text_to_speech(response)

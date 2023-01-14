import speech_recognition as sr
import serial
import pyttsx3

engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)

newVoiceRate = 145
engine.setProperty('rate', newVoiceRate)
def talk(text):
    engine.say(text)
    engine.runAndWait()
ser = serial.Serial('COM5', 115200, timeout=0)
r = sr.Recognizer()

while True:
    with sr.Microphone() as mic:
        try:
            print("Silence please, calibrating...")
            r.adjust_for_ambient_noise(mic, duration=2)
            print("calibrated, speak now...")
            talk('calibrated, speak now...')
            audio = r.listen(mic)
            text = r.recognize_google(audio)
            text = text.lower()
            print("You said "+text+"\n")
            ser.write(str.encode(text))
            talk("You said "+text+"\n")
            
            
        except sr.UnknownValueError:
            print("Could not understand audio")
            talk("Could not understand audio")
            
        except sr.RequestError as e:
            print("Request error; {0}".format(e))
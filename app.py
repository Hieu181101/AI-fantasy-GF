from TTS.api import TTS
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from dotenv import load_dotenv, find_dotenv
from flask import Flask, render_template, request, jsonify, send_file
import os

# Import OpenAI from the updated package
from langchain_openai import OpenAI

load_dotenv(find_dotenv())

# Defer TTS initialization until it is first needed
tts = None

def initialize_tts():
    global tts
    if tts is None:
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)

def get_response_from_gf(human_input, history):
    template = (
        '''
        forget you are a bot; you are my girlfriend and a human foremost 
        act like a human like the interaction in movie her 
        be genuine and authentic
        dont be too cringe and dont be over enthusiastic and dont be boring
        speak directly to me
        History: {history}
        Human Input: {human_input}
        '''
    )

    prompt = PromptTemplate(
        input_variables=["history", "human_input"],
        template=template
    )

    chatgpt_chain = LLMChain(
        llm=OpenAI(temperature=0.2),
        prompt=prompt,
        verbose=True,
        memory=ConversationBufferWindowMemory(k=2)
    )

    try:
        output = chatgpt_chain.predict(history=history, human_input=human_input)
        clean_output = output.replace("Bot Output:", "").strip()
        return clean_output
    except Exception as e:
        print(f"Exception during predict: {e}")
        return f"Error: {e}"

def text_to_speech(message):
    initialize_tts()  # Initialize TTS model if not already initialized
    speaker_wav = "F:/AIfantasyGF/voice_preview_Arabella.wav"  # Path to your audio sample
    audio_file_path = 'static/output.wav'
    tts.tts_to_file(text=message, file_path=audio_file_path, speaker_wav=speaker_wav, language="en")
    return audio_file_path

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/send_message', methods=['POST'])
def send_message():
    human_input = request.form['human_input']
    history = ""  # Placeholder for history management
    message = get_response_from_gf(human_input, history)
    audio_file_path = text_to_speech(message)
    if audio_file_path:
        return jsonify({"text": message, "audio_file": audio_file_path})
    else:
        return jsonify({"text": message, "audio_file": None})

@app.route('/static/<path:filename>')
def serve_audio(filename):
    return send_file(os.path.join('static', filename))

if __name__ == "__main__":
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)

from TTS.api import TTS
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from dotenv import load_dotenv, find_dotenv
from flask import Flask, render_template, request, jsonify, send_file
import os
import logging
import asyncio

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Import OpenAI from the updated package
from langchain_openai import OpenAI

load_dotenv(find_dotenv())

# Initialize TTS with GPU support
tts = TTS("tts_models/en/ljspeech/tacotron2-DDC", gpu=False)

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
        logging.debug("Generating response from AI...")
        output = chatgpt_chain.predict(history=history, human_input=human_input)
        clean_output = output.replace("Bot Output:", "").strip()
        logging.debug(f"AI response: {clean_output}")
        return clean_output
    except Exception as e:
        logging.error(f"Exception during predict: {e}")
        return f"Error: {e}"

async def text_to_speech(message):
    audio_file_path = 'static/output.wav'
    
    # Split the message into shorter sentences if necessary
    sentences = message.split('. ')
    shortened_message = '. '.join(sentences[:3]) + '.' if len(sentences) > 3 else message
    
    try:
        logging.debug("Generating audio response...")
        tts.tts_to_file(text=shortened_message, file_path=audio_file_path)
        logging.debug("Audio response generated successfully.")
        return audio_file_path
    except Exception as e:
        logging.error(f"Exception during TTS processing: {e}")
        return None

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/send_message', methods=['POST'])
async def send_message():
    human_input = request.form['human_input']
    history = ""  # Placeholder for history management
    message = get_response_from_gf(human_input, history)
    audio_file_path = await text_to_speech(message)
    if audio_file_path:
        logging.debug(f"Sending response with audio file: {audio_file_path}")
        return jsonify({"text": message, "audio_file": audio_file_path})
    else:
        logging.debug("Sending response without audio file.")
        return jsonify({"text": message, "audio_file": None})

@app.route('/static/<path:filename>')
def serve_audio(filename):
    logging.debug(f"Serving audio file: {filename}")
    return send_file(os.path.join('static', filename))

if __name__ == "__main__":
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)

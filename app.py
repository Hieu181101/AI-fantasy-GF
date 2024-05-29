from TTS.api import TTS
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from dotenv import load_dotenv, find_dotenv
from quart import Quart, render_template, request, jsonify, send_file
import os
import logging
import uuid
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Import OpenAI from the updated package
from langchain_openai import OpenAI

load_dotenv(find_dotenv())

# Initialize TTS with CPU support
tts = TTS("tts_models/en/ljspeech/tacotron2-DDC", gpu=False)  

# Global variable to track the last audio file
last_audio_file = None

# Initialize the diffusion model
pipe = StableDiffusionPipeline.from_pretrained(
    'hakurei/waifu-diffusion',
    torch_dtype=torch.float32
).to('cuda')

def get_response_from_gf(human_input, history):
    template = (
        '''
        forget you are a bot; you are my girlfriend and a human foremost 
        act like a human like the interaction in movie her 
        be genuine and authentic
        dont be too cringe and dont be over enthusiastic and dont be boring
        speak directly to me
        your response should not be over 160 characters 
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
        clean_output = output.replace("Bot Output:", "").strip() #we bassically delete any occurence of bot output from chat gpt response 
        logging.debug(f"AI response: {clean_output}")
        return clean_output
    except Exception as e:
        logging.error(f"Exception during predict: {e}")
        return f"Error: {e}"

async def text_to_speech(message):
    global last_audio_file
    unique_filename = f'output_{uuid.uuid4().hex}.wav'
    audio_file_path = os.path.join('static', unique_filename)
    
    # Split the message into shorter sentences if necessary
    sentences = message.split('. ')
    shortened_message = '. '.join(sentences[:3]) + '.' if len(sentences) > 3 else message
    
    try:
        logging.debug(f"Generating audio response for text: {shortened_message}")
        tts.tts_to_file(text=shortened_message, file_path=audio_file_path)
        logging.debug("Audio response generated successfully.")

        # Delete the old audio file if it exists
        if last_audio_file and os.path.exists(last_audio_file):
            os.remove(last_audio_file)
            logging.debug(f"Deleted old audio file: {last_audio_file}")

        # Update the last audio file variable
        last_audio_file = audio_file_path

        return unique_filename
    except Exception as e:
        logging.error(f"Exception during TTS processing: {e}")
        return None

#generate our gf appearence 
gf_promt = "1girl, aqua eyes, baseball cap, blonde hair, closed mouth, earrings, green background, hat, hoop earrings, jewelry, looking at viewer, shirt, short hair, simple background, solo, upper body, yellow shirt"
async def generate_image(gf_promt):
    unique_image_filename = f'image_{uuid.uuid4().hex}.png'
    image_file_path = os.path.join('static', unique_image_filename)
    with autocast("cuda"):
        image = pipe(gf_promt, guidance_scale=6)["sample"][0]
    image.save(image_file_path)
    return unique_image_filename


#using Quart to hand server request
app = Quart(__name__)

@app.route("/")
async def home():
    return await render_template("index.html")

@app.route('/send_message', methods=['POST'])
async def send_message():
    human_input = (await request.form)['human_input']
    history = ""  # Placeholder for history management
    message = get_response_from_gf(human_input, history)
    audio_file_name = await text_to_speech(message)
    image_file_name = await generate_image(human_input)
    
    response = {"text": message, "audio_file": audio_file_name, "image_file": image_file_name}
    
    logging.debug(f"Sending response: {response}")
    return jsonify(response)

@app.route('/static/<path:filename>')
async def serve_static(filename):
    file_path = os.path.join('static', filename)
    logging.debug(f"Serving file: {file_path}")
    return await send_file(file_path)

if __name__ == "__main__":
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)

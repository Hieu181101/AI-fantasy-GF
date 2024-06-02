# Install required packages
!pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
!pip install TTS
!pip install diffusers
!pip install openai
!pip install gradio
!pip install accelerate
!pip install SpeechRecognition pydub

import os
import logging
import uuid
import torch
from torch.cuda.amp import autocast  # Import autocast for mixed precision
from TTS.api import TTS
from diffusers import StableDiffusionPipeline
import openai
import gradio as gr
import time
from PIL import Image
import speech_recognition as sr

# Setup logging
logging.basicConfig(level=logging.DEBUG)
# Set the OpenAI API key directly
openai.api_key = "sk-proj-4MMN5HZhyKasV74nyIEBT3BlbkFJLiKM2QwLHdyCUkmvxu2m"

os.environ['OPENAI_API_KEY'] = 'sk-proj-4MMN5HZhyKasV74nyIEBT3BlbkFJLiKM2QwLHdyCUkmvxu2m'
# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
assert openai.api_key is not None, "OpenAI API key not found. Please set it in the environment variable."

# Check GPU availability
assert torch.cuda.is_available(), "CUDA is not available on this machine."

# Initialize TTS with CPU support to save GPU memory
tts = TTS("tts_models/en/ljspeech/tacotron2-DDC", gpu=True)

# Clear GPU cache to avoid memory issues
torch.cuda.empty_cache()

# Define the gf_prompt variable
gf_prompt = "1girl, aqua eyes, baseball cap, blonde hair, closed mouth, earrings, green background, hat, hoop earrings, jewelry, looking at viewer, shirt, short hair, simple background, solo, upper body, yellow shirt"

# Function to generate the initial image
def generate_initial_image(gf_prompt):
    pipe = StableDiffusionPipeline.from_pretrained(
        'hakurei/waifu-diffusion',
        torch_dtype=torch.float16  # Use float16 for reduced memory usage
    ).to('cuda')
    try:
        with autocast(enabled=True):
            result = pipe(gf_prompt, guidance_scale=6)
            if "images" in result:
                image = result.images[0]
                image_file_path = "./my_gf.png"
                image.save(image_file_path)
                logging.debug(f"Image saved at: {image_file_path}")
                return image_file_path
            else:
                logging.error("Expected key 'images' not found in the output of the pipeline")
                return None
    except Exception as e:
        logging.error(f"Exception during initial image generation: {e}")
        return None


initial_image_file_path = generate_initial_image(gf_prompt)
logging.debug(f"Initial image file path: {initial_image_file_path}")

def get_response_from_gf(human_input, history):
    start_time = time.time()
    prompt = (
        f"Forget you are a bot; you are my girlfriend and a human foremost. "
        f"Act like a human like the interaction in the movie 'Her'. "
        f"Be genuine and authentic. "
        f"Don't be too cringe and don't be over enthusiastic and don't be boring. "
        f"Speak directly to me. "
        f"Your response should not be over 160 characters. "
        f"History: {history} "
        f"Human Input: {human_input}"
    )

    try:
        logging.debug("Generating response from AI...")
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        output = response['choices'][0]['message']['content'].strip()
        end_time = time.time()
        logging.debug(f"AI response: {output}")
        logging.debug(f"Response generation time: {end_time - start_time} seconds")
        return output
    except Exception as e:
        logging.error(f"Exception during predict: {e}")
        return f"Error: {e}"

def text_to_speech(message):
    start_time = time.time()
    unique_filename = f'output_{uuid.uuid4().hex}.wav'
    audio_file_path = os.path.join('/content', unique_filename)

    try:
        logging.debug(f"Generating audio response for text: {message}")
        tts.tts_to_file(text=message, file_path=audio_file_path)
        end_time = time.time()
        logging.debug("Audio response generated successfully.")
        logging.debug(f"TTS generation time: {end_time - start_time} seconds")
        return audio_file_path
    except Exception as e:
        logging.error(f"Exception during TTS processing: {e}")
        return None

def speech_to_text(audio_file):
    recognizer = sr.Recognizer()
    try:
        # Ensure audio_file is a file path
        if isinstance(audio_file, str):
            with sr.AudioFile(audio_file) as source:
                audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
            logging.debug(f"Recognized text: {text}")
            return text
        else:
            raise ValueError("Given audio file must be a filename string or a file-like object")
    except sr.UnknownValueError:
        logging.error("Google Speech Recognition could not understand audio")
        return "Sorry, I did not get that."
    except sr.RequestError as e:
        logging.error(f"Could not request results from Google Speech Recognition service; {e}")
        return "Sorry, I am having trouble understanding you right now."
    except ValueError as ve:
        logging.error(f"Value error: {ve}")
        return "Error: Invalid audio file input."


def chat_with_gf(human_input):
    history = ""  # Placeholder for history management
    
    # Get AI response
    response_start = time.time()
    response = get_response_from_gf(human_input, history)
    response_end = time.time()
    logging.debug(f"Total response generation time: {response_end - response_start} seconds")
    
    # Generate TTS audio
    tts_start = time.time()
    audio_file_path = text_to_speech(response)
    tts_end = time.time()
    logging.debug(f"Total TTS generation time: {tts_end - tts_start} seconds")
    
    return response, audio_file_path, initial_image_file_path

def voice_chat_with_gf(audio_file):
    # Convert speech to text
    human_input = speech_to_text(audio_file)
    return chat_with_gf(human_input)

# Define the Gradio interface
interface = gr.Interface(
    fn=voice_chat_with_gf,
    inputs=gr.Audio(type="filepath"),
    outputs=[
        gr.Textbox(label="Response from AI"),
        gr.Audio(label="Generated Audio"),
        gr.Image(label="Generated Image")
    ],
    title="Fantasy GF Chatbot with Voice Input"
)

# Launch the interface
interface.launch(share=True, debug = True)

!pip install numpy==1.23.5
!pip install pandas==1.23.5
!pip install openai==0.28.0
!pip install flask tts diffusers nest_asyncio pyngrok langchain python-dotenv
!pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1+cu116
!pip install accelerate
!pip install gradio

import os
import logging
import uuid
import torch
from TTS.api import TTS
from diffusers import StableDiffusionPipeline
import openai
from dotenv import load_dotenv, find_dotenv
import gradio as gr
import time

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Set the OpenAI API key directly
openai.api_key = "sk-proj-4MMN5HZhyKasV74nyIEBT3BlbkFJLiKM2QwLHdyCUkmvxu2m"

os.environ['OPENAI_API_KEY'] = 'sk-proj-4MMN5HZhyKasV74nyIEBT3BlbkFJLiKM2QwLHdyCUkmvxu2m'



openai.api_key = os.getenv("OPENAI_API_KEY")
assert openai.api_key is not None, "OpenAI API key not found. Please set it in the environment variable."

# Check GPU availability
assert torch.cuda.is_available(), "CUDA is not available on this machine."

# Initialize TTS with GPU support
tts = TTS("tts_models/en/ljspeech/tacotron2-DDC", gpu=True)

# Initialize the diffusion model to use GPU with low memory usage
pipe = StableDiffusionPipeline.from_pretrained(
    'hakurei/waifu-diffusion',
    torch_dtype=torch.float16,  # Use float16 for faster computation
    low_cpu_mem_usage=True
).to('cuda')  # Use 'cuda' for GPU

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
        # Split the message into shorter sentences if necessary
        sentences = message.split('. ')
        shortened_message = '. '.join(sentences[:3]) + '.' if len(sentences) > 3 else message

        logging.debug(f"Generating audio response for text: {shortened_message}")
        tts.tts_to_file(text=shortened_message, file_path=audio_file_path)
        end_time = time.time()
        logging.debug("Audio response generated successfully.")
        logging.debug(f"TTS generation time: {end_time - start_time} seconds")
        return audio_file_path
    except Exception as e:
        logging.error(f"Exception during TTS processing: {e}")
        return None

def generate_image(gf_prompt):
    start_time = time.time()
    unique_image_filename = f'image_{uuid.uuid4().hex}.png'
    image_file_path = os.path.join('/content', unique_image_filename)

    try:
        image = pipe(gf_prompt, guidance_scale=6)["sample"][0]
        image.save(image_file_path)
        end_time = time.time()
        logging.debug(f"Image generation time: {end_time - start_time} seconds")
        return image_file_path
    except Exception as e:
        logging.error(f"Exception during image generation: {e}")
        return None

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

    # Generate Image
    image_start = time.time()
    gf_prompt = "1girl, aqua eyes, baseball cap, blonde hair, closed mouth, earrings, green background, hat, hoop earrings, jewelry, looking at viewer, shirt, short hair, simple background, solo, upper body, yellow shirt"
    image_file_path = generate_image(gf_prompt)
    image_end = time.time()
    logging.debug(f"Total image generation time: {image_end - image_start} seconds")

    return response, audio_file_path, image_file_path

# Define the Gradio interface
interface = gr.Interface(
    fn=chat_with_gf,
    inputs="text",
    outputs=[
        gr.Textbox(label="Response from AI"),
        gr.Audio(label="Generated Audio"),
        gr.Image(label="Generated Image")
    ],
    title="Fantasy GF Chatbot"
)

# Launch the interface
interface.launch(share=True)

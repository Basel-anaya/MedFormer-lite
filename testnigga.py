from io import BytesIO
import torch
from transformers import BitsAndBytesConfig, pipeline
import locale
import re
import nltk
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from PIL import Image
import requests
import whisper
import gradio as gr
import time
import warnings
import os
from gtts import gTTS

# Configure quantization for better performance
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

# Load the LLaVA model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "weizhiwang/LLaVA-Llama-3-8B"
tokenizer, model, image_processor, context_len = load_pretrained_model(model_name, None, model_name, False, True, device=device)

# Create an image-to-text pipeline with the LLAVA model
pipe = pipeline(model=model_name, task="text-generation")

# Set the path to the input image
image_path = "data/test.jpg"
image = Image.open(image_path)

# Ensure locale is set correctly for Whisper
locale.setlocale(locale.LC_ALL, '')

# Set the maximum number of new tokens for generation
max_new_tokens = 200

# Define the prompt instructions for the user
prompt_instructions = """
Describe the image using as much detail as possible. Is it a painting or a photograph? 
What colors are predominant? What is the image about?
"""

# Prepare inputs for the model
text = '<image>' + '\n' + "Describe the image in detail. Are there any notable objects or people? What colors stand out?"
conv = conv_templates["llama_3"].copy()
conv.append_message(conv.roles[0], text)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()
input_ids = tokenizer_image_token(prompt, tokenizer, -200, return_tensors='pt').unsqueeze(0).to(device)

# Prepare image input
image_url = "https://example.com/image.jpg"  # Replace with your image URL
response = requests.get(image_url)
image = Image.open(BytesIO(response.content)).convert('RGB')
image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().to(device)

# Autoregressively generate text
with torch.no_grad():
    output_ids = model.generate(
        input_ids,
        images=image_tensor,
        do_sample=False,
        max_new_tokens=512,
        use_cache=True
    )

# Decode the generated output
generated_text = tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
print("Generated Response:")
print(generated_text)

# Extract and print the generated response
generated_response = generated_text
print("Generated Response:")
print(generated_response)

# Tokenize the response into sentences and print each sentence
for sentence in nltk.sent_tokenize(generated_response):
    print(sentence)

# Ignore certain warnings
warnings.filterwarnings("ignore")

import numpy as np

# Check if CUDA is available and set the device accordingly
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using torch {torch.__version__} ({DEVICE})")

# Load the Whisper model for speech recognition
model = whisper.load_model("medium", device=DEVICE)
print(f"Model is {'multilingual' if model.is_multilingual else 'English-only'} and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters.")

# Regular expression to extract the assistant's response from the generated text
response_pattern = r'ASSISTANT:\s*(.*)'

# Function to convert image and text input into text output
def img2txt(input_text, input_image):
    # Load the image
    image = Image.open(input_image)

    # Set prompt instructions based on input type
    if isinstance(input_text, tuple):
        prompt_instructions = """
        Describe the image using as much detail as possible. Is it a painting or a photograph? 
        What colors are predominant? What is the image about?
        """
    else:
        prompt_instructions = """
        Act as an expert in imagery descriptive analysis, using as much detail as possible from the image, respond to the following prompt:
        """ + input_text

    # Construct the prompt
    prompt = "USER: <image>\n" + prompt_instructions + "\nASSISTANT:"

    # Generate a response using the image-to-text pipeline
    outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": max_new_tokens})

    # Extract the assistant's response
    match = re.search(response_pattern, outputs[0]["generated_text"])
    if match:
        reply = match.group(1)
    else:
        reply = "No response found."

    return reply

# Function to transcribe audio input using Whisper
def transcribe(audio):
    # Check if audio input is empty
    if audio is None or audio == '':
        return ('', '', None)

    # Load and preprocess audio
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)

    # Perform language detection
    _, probs = model.detect_language(audio)

    # Decode the audio and get the transcribed text
    options = whisper.DecodingOptions()
    result = whisper.decode(model, audio, options)
    result_text = result.text

    return result_text

# Function to convert text to speech using gTTS
def text_to_speech(text, file_path):
    language = 'en'
    audioobj = gTTS(text=text, lang=language, slow=False)
    audioobj.save(file_path)
    return file_path

# Function to process audio and image inputs
def process_inputs(audio_path, image_path):
    # Transcribe the audio input
    speech_to_text_output = transcribe(audio_path)

    # Process the image input using img2txt function
    if image_path:
        chatgpt_output = img2txt(speech_to_text_output, image_path)
    else:
        chatgpt_output = "No image provided."

    # Convert the chatgpt output to speech
    processed_audio_path = text_to_speech(chatgpt_output, "Temp3.mp3")

    return speech_to_text_output, chatgpt_output, processed_audio_path

# Create a Gradio interface for the multimodal RAG system
iface = gr.Interface(
    fn=process_inputs,
    inputs=[
        gr.Audio(sources=["microphone"], type="filepath"),
        gr.Image(type="filepath")
    ],
    outputs=[
        gr.Textbox(label="Speech to Text"),
        gr.Textbox(label="ChatGPT Output"),
        gr.Audio("Temp.mp3")
    ],
    title="Multimodal RAG System",
    description="Upload an image and interact via voice input and audio response."
)

# Launch the Gradio interface
iface.launch(debug=True)

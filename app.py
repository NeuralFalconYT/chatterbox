# my useless funtions
import random
import numpy as np
import torch
import gradio as gr
import gc
import subprocess

def get_max_gpu_memory():
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory
        return round(total_memory / (1024 ** 3), 2)  # Convert to GB
    else:
        print("CUDA is not available.")
        return None

def is_gpu_memory_over_limit(safety_margin_gb=0.8):
    """
    Returns True if GPU memory usage exceeds (total - safety_margin).
    """
    max_memory_gb = get_max_gpu_memory()
    if max_memory_gb is None:
        return False  # Can't check memory if no GPU

    limit_gb = max_memory_gb - safety_margin_gb

    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        memory_used_mb_list = result.stdout.strip().splitlines()
        for i, memory_used_mb in enumerate(memory_used_mb_list):
            memory_used_gb = int(memory_used_mb) / 1024.0
            if memory_used_gb > limit_gb:
                print(f"Maximum GPU memory available: {max_memory_gb} GB")
                print(f"Memory limit set: {limit_gb} GB")
                print("⚠️ GPU memory usage has exceeded the safe threshold. Clearing memory and reloading the model...")
                return True
        # print("✅ GPU memory is within safe limits.")
        return False
    except Exception as e:
        print(f"Failed to check GPU memory: {e}")
        return False



from sentence_splitter import SentenceSplitter
import re
import uuid
import numpy as np
import wave
from pydub import AudioSegment
from pydub.silence import split_on_silence
import os


def word_split(text, char_limit=300):
    words = text.split()
    chunks = []
    current_chunk = ""

    for word in words:
        if len(current_chunk) + len(word) + (1 if current_chunk else 0) <= char_limit:
            current_chunk += (" " if current_chunk else "") + word
        else:
            chunks.append(current_chunk)
            current_chunk = word

    if current_chunk:
        chunks.append(current_chunk)

    return chunks
def split_into_chunks(text, max_char_limit=300):
    if len(text)>=300:
      print("⚠️ The text is too long. Breaking it into smaller pieces so the voice generation works correctly.")

      splitter = SentenceSplitter(language='en')
      raw_sentences = splitter.split(text)

      # Flattened list of sentence-level word chunks
      sentence_chunks = []
      for sen in raw_sentences:
          sentence_chunks.extend(word_split(sen, char_limit=max_char_limit))

      chunks = []
      temp_str = ""

      for sentence in sentence_chunks:
          if len(temp_str) + len(sentence) + (1 if temp_str else 0) <= max_char_limit:
              temp_str += (" " if temp_str else "") + sentence
          else:
              chunks.append(temp_str)
              temp_str = sentence

      if temp_str:
          chunks.append(temp_str)

      return chunks
    else:
      return [text]


def clean_text(text):
    # Define replacement rules
    replacements = {
        "–": " ",  # Replace en-dash with space
        "—": " ",  #
        "-": " ",  # Replace hyphen with space
        "**": " ", # Replace double asterisks with space
        "*": " ",  # Replace single asterisk with space
        "#": " ",  # Replace hash with space
    }

    # Apply replacements
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Remove emojis using regex (covering wide range of Unicode characters)
    emoji_pattern = re.compile(
        r'[\U0001F600-\U0001F64F]|'  # Emoticons
        r'[\U0001F300-\U0001F5FF]|'  # Miscellaneous symbols and pictographs
        r'[\U0001F680-\U0001F6FF]|'  # Transport and map symbols
        r'[\U0001F700-\U0001F77F]|'  # Alchemical symbols
        r'[\U0001F780-\U0001F7FF]|'  # Geometric shapes extended
        r'[\U0001F800-\U0001F8FF]|'  # Supplemental arrows-C
        r'[\U0001F900-\U0001F9FF]|'  # Supplemental symbols and pictographs
        r'[\U0001FA00-\U0001FA6F]|'  # Chess symbols
        r'[\U0001FA70-\U0001FAFF]|'  # Symbols and pictographs extended-A
        r'[\U00002702-\U000027B0]|'  # Dingbats
        r'[\U0001F1E0-\U0001F1FF]'   # Flags (iOS)
        r'', flags=re.UNICODE)

    text = emoji_pattern.sub(r'', text)

    # Remove multiple spaces and extra line breaks
    text = re.sub(r'\s+', ' ', text).strip()

    return text




def tts_file_name(text, language="en"):
    global temp_audio_dir
    # Clean and process the text
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only letters and spaces
    text = text.lower().strip().replace(" ", "_")

    # Ensure the text is not empty
    if not text:
        text = "audio"

    # Truncate to first 20 characters for filename
    truncated_text = text[:20]

    # Sanitize and format the language tag
    language = re.sub(r'\s+', '_', language.strip().lower()) if language else "unknown"

    # Generate random suffix
    random_string = uuid.uuid4().hex[:8].upper()

    # Construct the filename
    file_name = f"{temp_audio_dir}/{truncated_text}_{language}_{random_string}.wav"
    return file_name








def remove_silence_function(file_path,minimum_silence=50):
    # Extract file name and format from the provided path
    output_path = file_path.replace(".wav", "_no_silence.wav")
    audio_format = "wav"
    # Reading and splitting the audio file into chunks
    sound = AudioSegment.from_file(file_path, format=audio_format)
    audio_chunks = split_on_silence(sound,
                                    min_silence_len=100,
                                    silence_thresh=-45,
                                    keep_silence=minimum_silence)
    # Putting the file back together
    combined = AudioSegment.empty()
    for chunk in audio_chunks:
        combined += chunk
    combined.export(output_path, format=audio_format)
    return output_path

#chatterbox code
from src.chatterbox.tts import ChatterboxTTS


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

chatterbox_model=None
def load_model():
    global chatterbox_model
    if chatterbox_model is None:
      del chatterbox_model
      gc.collect()
      torch.cuda.empty_cache()
    chatterbox_model = ChatterboxTTS.from_pretrained(DEVICE)
    return chatterbox_model

# for the first time
chatterbox_model=load_model()

import tempfile
import shutil
import os
import numpy as np
import soundfile as sf
from tqdm.auto import tqdm

def generate(chatterbox_model, text, audio_prompt_path, exaggeration, temperature, seed_num, cfgw):
    if is_gpu_memory_over_limit():
        chatterbox_model = load_model()

    # Removed seed setting here to avoid conflicts
    # if seed_num != 0:
    #         set_seed(int(seed_num))

    wav = chatterbox_model.generate(
        text,
        audio_prompt_path=audio_prompt_path,
        exaggeration=exaggeration,
        temperature=temperature,
        cfg_weight=cfgw,
    )
    sr = chatterbox_model.sr
    audio = wav.squeeze(0).numpy()
    return sr, audio

def generate_and_save_all(text, audio_prompt_path, exaggeration, temperature, seed_num, cfgw):
    global chatterbox_model
    text = clean_text(text)
    chunks = split_into_chunks(text, max_char_limit=300)

    temp_dir = tempfile.mkdtemp(prefix="audio_chunks_")
    temp_files = []

    try:
        for idx, chunk in tqdm(enumerate(chunks), total=len(chunks), desc="Generating audio"):
            try:
                # Set seed once per chunk for reproducible variation
                if seed_num != 0:
                    set_seed(int(seed_num) + idx)

                sr, audio = generate(
                    chatterbox_model,
                    chunk,
                    audio_prompt_path,
                    exaggeration,
                    temperature,
                    seed_num=0,  # Pass 0 to skip seed setting inside generate
                    cfgw=cfgw
                )

                chunk_path = os.path.join(temp_dir, f"chunk_{idx:03}.wav")
                sf.write(chunk_path, audio, sr)
                temp_files.append(chunk_path)

            except Exception as e:
                print(f"⚠️ [Chunk {idx}] Generation failed: {e}")
                print(f"Text: {chunk}")
                print(f"Length: {len(chunk)}")
                continue  # Skip failed chunk

        # Merge all valid chunks
        final_audio = []
        for file_path in temp_files:
            try:
                data, _ = sf.read(file_path)
                final_audio.append(data)
            except Exception as e:
                print(f"💀 [Merging] Failed to read chunk: {file_path} ({e})")

        if final_audio:
            final_audio = np.concatenate(final_audio)
            final_path = tts_file_name(text, language="en")
            sf.write(final_path, final_audio, sr)
        else:
            raise RuntimeError("All audio chunk generations failed.")

    finally:
        shutil.rmtree(temp_dir)

    return final_path


import gradio as gr



def ui_wrapper(text, ref_wav, exaggeration, cfg_weight, seed_num, temp, remove_silence):
    cloned_voice_path = generate_and_save_all(
        text=text,
        audio_prompt_path=ref_wav,
        exaggeration=exaggeration,
        temperature=temp,
        seed_num=seed_num,
        cfgw=cfg_weight
    )
    params_info = f"""Reference Voice Path: {os.path.basename(ref_wav)}
Exaggeration: {exaggeration}
CFG/Pace: {cfg_weight}
Random seed: {seed_num}
Temperature: {temp}
Remove Silence: {remove_silence}"""

    if remove_silence:
        final_audio = remove_silence_function(cloned_voice_path)
        return final_audio, final_audio, params_info
    else:
        return cloned_voice_path, cloned_voice_path, params_info

def ui():
    with gr.Blocks() as demo:
        gr.Markdown(
            """
            # Chatterbox TTS: No character length limitation, until you run out of memory.
            Generate high-quality speech from text with reference audio styling.
            """
        )
        with gr.Row():
            with gr.Column():
                text = gr.Textbox(
                    value="Now let's make my mum's favourite. So three mars bars into the pan. Then we add the tuna and just stir for a bit, just let the chocolate and fish infuse. A sprinkle of olive oil and some tomato ketchup. Now smell that. Oh boy this is going to be incredible.",
                    label="Text to synthesize",
                    max_lines=5
                )

                ref_wav = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label="Reference Audio File (Optional)",
                    value="https://storage.googleapis.com/chatterbox-demo-samples/prompts/female_shadowheart4.flac"
                )
                exaggeration = gr.Slider(0.25, 2, step=0.05, label="Exaggeration (Neutral = 0.5, extreme values can be unstable)", value=0.5)
                cfg_weight = gr.Slider(0.0, 1.0, step=0.05, label="CFG/Pace", value=0.5)
                remove_silence = gr.Checkbox(value=False, label='✂️ Remove Silence From TTS')
                with gr.Accordion("More options", open=False):
                    seed_num = gr.Number(value=0, label="Random seed (0 for random)")
                    temp = gr.Slider(0.05, 5.0, step=0.05, label="Temperature", value=0.8)

                run_btn = gr.Button("Generate", variant="primary")

            with gr.Column():
                audio_output = gr.Audio(label="Output Audio")
                audio_file = gr.File(label='📥 Download Audio')
                with gr.Accordion("Used Parameters (copy for future use)", open=False):
                  params_display = gr.Textbox(label="Copy It", interactive=True, lines=7)

        inputs = [text, ref_wav, exaggeration, cfg_weight, seed_num, temp, remove_silence]
        outputs = [audio_output, audio_file, params_display]

        text.submit(ui_wrapper, inputs=inputs, outputs=outputs)
        run_btn.click(ui_wrapper, inputs=inputs, outputs=outputs)

    return demo

# demo = ui()
# demo.queue().launch(share=True, debug=True)

import click
@click.command()
@click.option("--debug", is_flag=True, default=False, help="Enable debug mode.")
@click.option("--share", is_flag=True, default=False, help="Enable sharing of the interface.")
def main(debug, share):
    demo= ui()
    demo.queue().launch(debug=debug, share=share)
    # demo.queue().launch(debug=debug, share=share,server_port=9000)
    #Run on local network
    # laptop_ip="192.168.0.30"
    # port=8080
    # demo.queue().launch(debug=debug, share=share,server_name=laptop_ip,server_port=port)

# Initialize default pipeline
temp_audio_dir="./cloned_voices"
os.makedirs(temp_audio_dir, exist_ok=True)
if __name__ == "__main__":
    main()  

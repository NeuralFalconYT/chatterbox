#@title Podcast

"""
Chatterbox Podcast
"""

import gc
import json
import os
import random
import re
import shutil
import subprocess
import tempfile
import uuid
from typing import List

import gradio as gr
import numpy as np
import torch
from pydub import AudioSegment
from pydub.silence import split_on_silence
from sentence_splitter import SentenceSplitter
import soundfile as sf
from tqdm.auto import tqdm

# --- Chatterbox Model Import ---
from src.chatterbox.tts import ChatterboxTTS

# --- Configuration Constants ---
BASE_DIR = os.getcwd()
VOICES_DIR = os.path.join(BASE_DIR, "voices")
EXAMPLES_DIR = os.path.join(BASE_DIR, "text_examples")
OUTPUT_DIR = os.path.join(BASE_DIR, "podcast_audio")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Setup Directories ---
os.makedirs(VOICES_DIR, exist_ok=True)
os.makedirs(EXAMPLES_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# region Chatterbox Core Logic
def set_seed(seed: int):
    """Sets the seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def get_max_gpu_memory():
    """Returns total GPU memory in GB if available."""
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory
        return round(total_memory / (1024 ** 3), 2)
    return None

def is_gpu_memory_over_limit(safety_margin_gb=1.0):
    """Checks if GPU memory usage has exceeded a safe threshold."""
    max_memory_gb = get_max_gpu_memory()
    if not max_memory_gb: return False

    limit_gb = max_memory_gb - safety_margin_gb
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
            stdout=subprocess.PIPE, text=True, check=True
        )
        memory_used_gb = int(result.stdout.strip()) / 1024.0
        if memory_used_gb > limit_gb:
            print(f"⚠️ GPU memory usage ({memory_used_gb:.2f}GB) exceeded safe threshold ({limit_gb:.2f}GB).")
            return True
    except Exception as e:
        print(f"Failed to check GPU memory: {e}")
    return False

def load_model():
    """Loads or reloads the ChatterboxTTS model onto the correct device."""
    global chatterbox_model
    if 'chatterbox_model' in globals() and chatterbox_model is not None:
        del chatterbox_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"🧠 Loading ChatterboxTTS model onto {DEVICE.upper()}...")
    chatterbox_model = ChatterboxTTS.from_pretrained(DEVICE)
    print("✅ Model loaded successfully.")
    return chatterbox_model

def split_into_chunks(text, max_char_limit=300):
    """Splits long text into smaller chunks for stable generation."""
    if len(text) <= max_char_limit:
        return [text]
    print("⚠️ Text is long, splitting into manageable chunks.")
    splitter = SentenceSplitter(language='en')
    sentences = splitter.split(text)
    
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 > max_char_limit:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence
        else:
            current_chunk += (" " if current_chunk else "") + sentence
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def clean_text(text):
    """Cleans text by removing special characters and emoji."""
    text = re.sub(r"[–—*#]", " ", text)
    emoji_pattern = re.compile("[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002702-\U000027B0]+")
    text = emoji_pattern.sub(r'', text)
    return re.sub(r'\s+', ' ', text).strip()

def generate(chatterbox_model, text, audio_prompt_path, exaggeration, temperature, seed_num, cfgw):
    """Generates a single audio chunk."""
    if seed_num != 0: set_seed(int(seed_num))
    wav = chatterbox_model.generate(text, audio_prompt_path, exaggeration, temperature, cfgw)
    return chatterbox_model.sr, wav.squeeze(0).cpu().numpy()

def generate_and_save_chunk(text, audio_prompt_path, exaggeration, temperature, seed_num, cfgw, remove_silence):
    """Handles chunking and generates a temporary audio file for a given text line."""
    global chatterbox_model
    text = clean_text(text)
    chunks = split_into_chunks(text, max_char_limit=300)
    
    final_audio_data = []
    sr = None

    for idx, chunk in enumerate(chunks):
        if is_gpu_memory_over_limit(): chatterbox_model = load_model()
        
        current_seed = (seed_num + idx) if seed_num != 0 else 0
        try:
            sr, audio = generate(chatterbox_model, chunk, audio_prompt_path, exaggeration, temperature, current_seed, cfgw)
            final_audio_data.append(audio)
        except Exception as e:
            print(f"⚠️ Generation failed for chunk: '{chunk[:50]}...'. Error: {e}")
            continue

    if not final_audio_data:
        raise RuntimeError("All audio chunk generations failed for the line.")
    
    final_audio = np.concatenate(final_audio_data)
    
    temp_path = tempfile.mktemp(suffix=".wav")
    sf.write(temp_path, final_audio, sr)

    if remove_silence:
        sound = AudioSegment.from_file(temp_path, format="wav")
        audio_chunks = split_on_silence(sound, min_silence_len=100, silence_thresh=-45, keep_silence=50)
        combined = sum(audio_chunks, AudioSegment.empty())
        combined.export(temp_path, format="wav")
        
    return temp_path
# endregion

def generate_file_name(text):
    """Generates a clean, unique base filename from the first line of a script."""
    cleaned = re.sub(r"^\s*speaker\s*\d+\s*:\s*", "", text, flags=re.IGNORECASE)
    short = cleaned[:30].strip()
    short = re.sub(r'[^a-zA-Z0-9\s]', '', short).lower().strip().replace(" ", "_")
    if not short: short = "podcast_output"
    return f"{short}_{uuid.uuid4().hex[:6]}"

class ChatterboxDemo:
    def __init__(self):
        self.is_generating = False
        self.stop_generation = False
        self.setup_voice_presets()
        self.load_example_scripts()

    def setup_voice_presets(self):
        self.available_voices = {}
        for f in sorted(os.listdir(VOICES_DIR)):
            if f.lower().endswith(('.wav', '.mp3', '.flac')):
                name = os.path.splitext(f)[0]
                self.available_voices[name] = os.path.join(VOICES_DIR, f)
        print(f"Found {len(self.available_voices)} voice files in {VOICES_DIR}")

    def load_example_scripts(self):
        self.example_scripts = []
        if not os.path.exists(EXAMPLES_DIR): return
        for f in sorted(os.listdir(EXAMPLES_DIR)):
            if f.lower().endswith('.txt'):
                try:
                    with open(os.path.join(EXAMPLES_DIR, f), 'r', encoding='utf-8') as file:
                        script = file.read().strip()
                    if script:
                        num_speakers = self._get_num_speakers_from_script(script)
                        self.example_scripts.append([num_speakers, script])
                except Exception as e:
                    print(f"Error loading example script {f}: {e}")

    def _get_num_speakers_from_script(self, script: str) -> int:
        speakers = set(re.findall(r'^Speaker\s+(\d+)\s*:', script, re.MULTILINE | re.IGNORECASE))
        return max(int(s) for s in speakers) if speakers else 1

    def stop_audio_generation(self):
        if self.is_generating:
            self.stop_generation = True
            print("🛑 Audio generation stop requested.")

    def generate_podcast(self, num_speakers, script,
                         speaker_1, speaker_2, speaker_3, speaker_4,
                         exaggeration, temperature, cfgw, seed, remove_silence,
                         progress=gr.Progress()):
        try:
            self.is_generating, self.stop_generation = True, False

            if not script.strip(): raise gr.Error("Error: Please provide a script.")
            selected_speakers = [speaker_1, speaker_2, speaker_3, speaker_4][:num_speakers]
            if any(not sp or sp not in self.available_voices for sp in selected_speakers):
                raise gr.Error("Error: Please select a valid voice for each speaker.")
            
            lines = [ln.strip() for ln in script.strip().split('\n') if ln.strip()]
            formatted_lines = [f"Speaker {(i % num_speakers) + 1}: {line}" if not re.match(r'Speaker\s*\d+:', line, re.IGNORECASE) else line for i, line in enumerate(lines)]
            if not formatted_lines: raise gr.Error("Error: Script is empty after formatting.")

            base_name = generate_file_name(formatted_lines[0])
            final_audio_path = os.path.join(OUTPUT_DIR, f"{base_name}.wav")
            final_json_path = os.path.join(OUTPUT_DIR, f"{base_name}.json")
            
            timestamps, current_time = {}, 0.0
            
            with sf.SoundFile(final_audio_path, 'w', samplerate=24000, channels=1, subtype='PCM_16') as final_file:
                for i, line in enumerate(formatted_lines):
                    if self.stop_generation:
                        print("\n🛑 Generation stopped by user.")
                        break
                    
                    progress(i / len(formatted_lines), desc=f"Line {i+1}/{len(formatted_lines)}")
                    
                    match = re.match(r'Speaker\s*(\d+):\s*(.*)', line, re.IGNORECASE)
                    if not match: continue
                    
                    speaker_idx, text_content = int(match.group(1)) - 1, match.group(2).strip()
                    if not (0 <= speaker_idx < num_speakers): continue
                    
                    voice_ref_path = self.available_voices[selected_speakers[speaker_idx]]
                    
                    temp_clip_path = generate_and_save_chunk(
                        text_content, voice_ref_path, exaggeration, temperature, int(seed), cfgw, remove_silence
                    )
                    
                    data, _ = sf.read(temp_clip_path)
                    final_file.write(data)
                    
                    duration = len(data) / final_file.samplerate
                    timestamps[str(i + 1)] = {"text": text_content, "speaker_id": speaker_idx + 1, "start": round(current_time, 3), "end": round(current_time + duration, 3)}
                    current_time += duration
                    os.remove(temp_clip_path)

            with open(final_json_path, "w") as f: json.dump(timestamps, f, indent=2)

            self.is_generating = False
            return final_audio_path, final_audio_path, final_json_path, gr.update(visible=True), gr.update(visible=False)

        except Exception as e:
            self.is_generating = False
            import traceback
            traceback.print_exc()
            # Errors will now appear as a pop-up in the UI and in the console
            raise gr.Error(str(e))

def create_demo_interface(demo_instance: ChatterboxDemo):
    """Creates and returns the Gradio web interface."""
    with gr.Blocks(title="Chatterbox Podcast Generator") as interface:
        gr.HTML("""
        <div style="text-align: center; margin: 20px auto; max-width: 800px;">
            <h1 style="font-size: 2.5em; margin-bottom: 5px;">🎙️ Chatterbox Podcast Generator</h1>
            <p style="font-size: 1.2em; color: #555;">Generate AI Podcasts with Custom Voices inspired from VibeVoice</p>
        </div>""")

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### 🎛️ Podcast Settings")
                    num_speakers = gr.Slider(1, 4, 2, step=1, label="Number of Speakers")
                    
                    gr.Markdown("### 🎭 Speaker Voice Selection")
                    speaker_selections = [gr.Dropdown(choices=list(demo_instance.available_voices.keys()), label=f"Speaker {i+1}", visible=(i < 2)) for i in range(4)]
                    remove_silence = gr.Checkbox(label="Trim Silence from Segments", value=False)
                    with gr.Accordion("🎤 Upload Custom Voices", open=False):
                        upload_audio = gr.File(label="Upload Voice Samples (.wav, .mp3)", file_count="multiple", file_types=["audio"])
                        process_upload_btn = gr.Button("Add Uploaded Voices")

                    with gr.Accordion("⚙️ Advanced Generation Settings", open=False):
                        exaggeration = gr.Slider(0, 2, 0.5, step=0.1, label="Exaggeration")
                        temperature = gr.Slider(0, 1, 0.5, step=0.05, label="Temperature")
                        cfgw = gr.Slider(0, 2, 0.5, step=0.1, label="CFG Weight")
                        seed = gr.Number(0, label="Seed (0 for random)", precision=0)
                        

            with gr.Column(scale=2):
                with gr.Group():
                    gr.Markdown("### 📝 Script Input")
                    script_input = gr.Textbox(label="Conversation Script", placeholder="Enter script here...", lines=10)
                    with gr.Row():
                        generate_btn = gr.Button("🚀 Generate Podcast", variant="primary", scale=2)
                        stop_btn = gr.Button("🛑 Stop", variant="stop", visible=False, scale=1)

                    gr.Markdown("### 🎵 **Generated Output**")
                    audio_output = gr.Audio(label="Podcast Audio")
                    
                    with gr.Accordion("📦 Download Files", open=False):
                      download_audio = gr.File(label="Download Audio (.wav)")
                      download_json = gr.File(label="Download Timestamps (.json)")

        gr.Examples(examples=demo_instance.example_scripts, inputs=[num_speakers, script_input], label="Example Scripts")

        def process_and_refresh_voices(uploaded_files):
            if uploaded_files:
                for f in uploaded_files: shutil.copy(f.name, os.path.join(VOICES_DIR, os.path.basename(f.name)))
                demo_instance.setup_voice_presets()
            return [gr.update(choices=list(demo_instance.available_voices.keys())) for _ in speaker_selections] + [gr.update(value=None)]

        num_speakers.change(lambda n: [gr.update(visible=i < n) for i in range(4)], num_speakers, speaker_selections)
        process_upload_btn.click(process_and_refresh_voices, upload_audio, speaker_selections + [upload_audio])
        
        gen_inputs = [num_speakers, script_input] + speaker_selections + [exaggeration, temperature, cfgw, seed, remove_silence]
        # REMOVED: status_display is no longer an output
        gen_outputs = [audio_output, download_audio, download_json, generate_btn, stop_btn]
        
        gen_event = generate_btn.click(
            lambda: (gr.update(visible=False), gr.update(visible=True)), 
            outputs=[generate_btn, stop_btn] # REMOVED: status_display from outputs
        ).then(
            fn=demo_instance.generate_podcast, inputs=gen_inputs, outputs=gen_outputs
        )
        
        def handle_stop_generation():
            demo_instance.stop_audio_generation()
            # REMOVED: status_display update from return
            return gr.update(visible=True), gr.update(visible=False)

        stop_btn.click(
            fn=handle_stop_generation,
            outputs=[generate_btn, stop_btn], # REMOVED: status_display from outputs
            cancels=[gen_event]
        )

    return interface
def build_conversation_prompt(topic, *speaker_names):
    """
    Generates the final prompt. It takes the topic and a variable number of speaker names.
    """
    names = [name for name in speaker_names if name and name.strip()]

    # Error checking
    if not topic or not topic.strip():
        return "Error: Please provide a topic."
    if not names:
        return "Error: Please provide at least one speaker name."

    num_speakers = len(names)
    speaker_mapping_str = "Speaker mapping (for context only, DO NOT use these names as labels):\n"
    for i, name in enumerate(names):
        speaker_mapping_str += f"- Speaker {i+1} = {name}\n"
    
    speaker_labels = [f"\"Speaker {i+1}:\"" for i in range(num_speakers)]
    
    introductions_str = ""
    for i, name in enumerate(names):
        introductions_str += f"  - Speaker {i+1} introduces themselves by saying: \"I’m {name}...\"\n"
        
    example_str = "STRICT Example (follow this format exactly):\n"
    example_str += f"Speaker 1: Hi everyone, I’m {names[0]}, and I’m excited to be here today.\n"
    if num_speakers > 1:
        for i in range(1, num_speakers):
            example_str += f"Speaker {i+1}: And I’m {names[i]}. Thanks for joining us.\n"
    example_str += "Speaker 1: So, let’s dive into our topic...\n"
    
    prompt = f"""
You are a professional podcast scriptwriter. 
Write a natural, engaging conversation between {num_speakers} speakers on the topic: "{topic}".

{speaker_mapping_str}
Formatting Rules:
- You MUST always format dialogue with {', '.join(speaker_labels)} ONLY. 
- Never replace the labels with real names. The labels stay exactly as they are.
- At the beginning:
{introductions_str}
- During the conversation, they may occasionally mention each other's names ({', '.join(names)}) naturally in the dialogue, but the labels must remain unchanged.
- Do not add narration, descriptions, or any extra formatting.

{example_str}
"""
    return prompt

def update_speaker_name_visibility(num_speakers):
    """
    Shows or hides the speaker name textboxes based on the slider value.
    """
    num = int(num_speakers)
    updates = []
    for i in range(4):
        if i < num:
            updates.append(gr.update(visible=True))
        else:
            updates.append(gr.update(visible=False, value=""))
    
    return tuple(updates) 

def ui2():

    with gr.Blocks(title="Prompt Builder") as demo:
        gr.HTML("""
        <div style="text-align: center; margin: 20px auto; max-width: 800px;">
            <h1 style="font-size: 2.5em; margin-bottom: 5px;">🎙️ Sample Podcast Prompt Generator</h1>
            <p style="font-size: 1.2em; color: #555;">Paste the prompt into any LLM, and customize the propmt if you want.</p>
        </div>""")
        
        with gr.Row():
            with gr.Column(scale=1):
                topic = gr.Textbox(label="Topic", placeholder="e.g., The Future of Artificial Intelligence")
                
                num_speakers = gr.Slider(
                    minimum=1, 
                    maximum=4, 
                    value=2, 
                    step=1, 
                    label="Number of Speakers"
                )
                
                with gr.Group():
                    speaker_textboxes = [
                        gr.Textbox(label=f"Speaker {i+1} Name", visible=(i < 2), placeholder=f"e.g., Speaker {i+1}")
                        for i in range(4)
                    ]
                
                gen_btn = gr.Button("Generate Prompt", variant="primary")


                gr.Examples(
                    examples=[
                        ["The Ethics of Gene Editing", 2, "Dr. Evelyn Reed", "Dr. Ben Carter", "", ""],
                        ["Exploring the Deep Sea", 3, "Maria", "Leo", "Samira", ""],
                        ["The Future of Space Tourism", 4, "Alex", "Zara", "Kenji", "Isla"]
                    ],
                    # The inputs list must match the order of items in the examples list
                    inputs=[topic, num_speakers] + speaker_textboxes,
                    label="Quick Examples"
                )

            with gr.Column(scale=2):
                output_prompt = gr.Textbox(label="Generated Prompt", lines=25, interactive=False, show_copy_button=True)

        
        num_speakers.change(
            fn=update_speaker_name_visibility, 
            inputs=num_speakers, 
            outputs=speaker_textboxes
        )
        
        gen_btn.click(
            fn=build_conversation_prompt, 
            inputs=[topic] + speaker_textboxes, 
            outputs=[output_prompt]
        )

    return demo

import click
@click.command()
@click.option("--debug", is_flag=True, default=False, help="Enable debug mode.")
@click.option("--share", is_flag=True, default=False, help="Enable sharing of the interface.")
def main(debug, share):
# def main(debug=True, share=True):
    """Main function to load the model and launch the Gradio interface."""
    global chatterbox_model
    chatterbox_model = load_model()

    demo_instance = ChatterboxDemo()
    
    demo1 = create_demo_interface(demo_instance)
    demo2= ui2()
    custom_css = """
      .gradio-container {
          font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
      }"""

    demo = gr.TabbedInterface([demo1, demo2],["Chatterbox Podcast","Generate Sample Podcast Script"],title="",theme=gr.themes.Soft(),css=custom_css)

    print("🚀 Launching Gradio Demo...")
    demo.queue().launch(debug=debug, share=share)

if __name__ == "__main__":
    main()

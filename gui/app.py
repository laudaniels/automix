"""Simple GUI for generating a song transition: two drop areas for the input
songs, a button, and a preview player for the result.

Run from the repo root:
    python gui/app.py
Then open the local URL it prints (usually http://127.0.0.1:7860).
"""
import os
import sys
import tempfile

import gradio as gr
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformer_models.bert import config
from validation.generate_transition import generate_transition, load_models

OUTPUT_DIR = os.path.join(tempfile.gettempdir(), "automix-gui-outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device(config.DEVICE)
print(f"Loading Encodec and BERT model on {DEVICE} (this can take a moment)...")
ENCODEC_MODEL, BERT_MODEL = load_models(config.CHECKPOINT_PATH, DEVICE)
print("Models loaded.")


def on_generate(song_a_path, song_b_path):
    if not song_a_path or not song_b_path:
        return None, "Drop a song in both boxes first."
    try:
        out_path = tempfile.mktemp(suffix=".wav", dir=OUTPUT_DIR)
        generate_transition(song_a_path, song_b_path, out_path, ENCODEC_MODEL, BERT_MODEL, DEVICE)
        return out_path, "Done."
    except ValueError as e:
        return None, f"Error: {e}"


with gr.Blocks(title="Automix") as demo:
    gr.Markdown(
        "# Automix\n"
        "Drop two songs below, then generate a bridge between them using the trained BERT model. "
        "This isn't a general song mixer -- it reuses the model's actual trained task "
        "(masked-token infilling) to fill in a short gap between the two clips."
    )
    with gr.Row():
        song_a = gr.Audio(label="Song A (plays first)", type="filepath")
        song_b = gr.Audio(label="Song B (plays second)", type="filepath")

    generate_btn = gr.Button("Generate transition", variant="primary")
    status = gr.Markdown("")
    output_audio = gr.Audio(label="Result", type="filepath")

    generate_btn.click(on_generate, inputs=[song_a, song_b], outputs=[output_audio, status])

if __name__ == "__main__":
    demo.launch()

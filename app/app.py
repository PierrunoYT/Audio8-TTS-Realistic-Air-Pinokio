import os
from pathlib import Path

import gradio as gr
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoProcessor

ROOT = Path(__file__).resolve().parent
MODEL_ID = "Edge0/Audio8-TTS-Realistic-Air"
# The official playground Space hosts the two reference clips from the model's listening room.
SPACE_ID = "Edge0/Audio8-TTS-Realistic-Air"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

NO_REFERENCE = "none"
CUSTOM = "custom"
PRESETS = {
    "example10": (
        "Example 10 · English",
        "examples/sample-10-reference.wav",
        "What a season! Amy figured out what I was up to, and now she's gonna use my DNA to make a multiverse waypoint.",
    ),
    "example18": (
        "Example 18 · 中文",
        "examples/sample-18-reference.wav",
        "又顺着床沿摸到了台灯的开关，按了下去，屋子里仍然漆黑一片。",
    ),
}
VOICE_CHOICES = [(label, key) for key, (label, _, _) in PRESETS.items()] + [
    ("Custom recording", CUSTOM),
    ("No reference (random voice)", NO_REFERENCE),
]
TEXT_EXAMPLES = [
    "Some days, the smallest moments are the ones we remember. A familiar voice, a shared laugh, a little good news.",
    "有时候，最让人难忘的，恰恰是那些平凡的小事。一个熟悉的声音，一次开心的交谈，还有一个等了很久的好消息。",
    "窓を開けると、朝の風がふわりと入ってきました。今日は何か、いいことがありそうです。",
    "A veces, los momentos más sencillos son los que recordamos: una voz familiar, una sonrisa y una buena noticia.",
]

print(f"Loading {MODEL_ID} on {DEVICE} ({DTYPE})...", flush=True)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True, dtype=DTYPE).eval().to(DEVICE)
model.load_codec()
PRESET_PATHS = {
    key: hf_hub_download(SPACE_ID, filename, repo_type="space", local_dir=ROOT)
    for key, (_, filename, _) in PRESETS.items()
}


def choose_voice(voice):
    if voice in PRESETS:
        return PRESET_PATHS[voice], PRESETS[voice][2], gr.update(visible=True)
    if voice == NO_REFERENCE:
        return gr.update(), gr.update(), gr.update(visible=False)
    # Custom: keep whatever the user uploaded or recorded.
    return gr.update(), gr.update(), gr.update(visible=True)


def mark_custom():
    return CUSTOM, ""


def synthesize(text, voice, ref_audio, ref_text, temperature, top_p, top_k, max_new_tokens, seed):
    text = (text or "").strip()
    if not text:
        raise gr.Error("Enter some text to speak.")
    kwargs = {"text": [text]}
    if voice != NO_REFERENCE:
        ref_text = (ref_text or "").strip()
        if not ref_audio:
            raise gr.Error("Pick an example voice, or upload/record a 3-15 s reference clip.")
        if not ref_text:
            raise gr.Error("Enter the exact transcript of the reference clip.")
        kwargs.update(reference_audio=[ref_audio], reference_text=[ref_text])
    try:
        inputs = processor(**kwargs, return_tensors="pt")
    except (ValueError, RuntimeError) as exc:
        raise gr.Error(f"Could not prepare the inputs: {exc}") from None
    inputs = {name: value.to(DEVICE) for name, value in inputs.items()}
    if seed is not None and seed >= 0:
        torch.manual_seed(int(seed))
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            do_sample=True,
            return_dict_in_generate=True,
        )
        waveforms, lengths = model.decode_audio(output.codes)
    audio = waveforms[0, : int(lengths[0])].float().cpu().numpy()
    return model.config.codec_sample_rate, audio


with gr.Blocks(title="Audio8-TTS-Realistic-Air") as demo:
    gr.Markdown(
        "# Audio8-TTS-Realistic-Air\n"
        "0.6B text-to-speech with zero-shot voice cloning in 23 languages. "
        f"Running on **{DEVICE}**. Clone only voices you have the right to use."
    )
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(label="Text", lines=6, value=TEXT_EXAMPLES[0])
            gr.Examples([[t] for t in TEXT_EXAMPLES], inputs=text, label="Example texts")
            voice = gr.Dropdown(VOICE_CHOICES, value="example10", label="Voice")
            with gr.Group() as reference_box:
                ref_audio = gr.Audio(
                    value=PRESET_PATHS["example10"],
                    label="Reference clip (clean speech, 3-15 s, one speaker)",
                    type="filepath",
                    sources=["upload", "microphone"],
                )
                ref_text = gr.Textbox(
                    value=PRESETS["example10"][2],
                    label="Exact transcript of the reference clip",
                    lines=2,
                )
            with gr.Accordion("Advanced", open=False):
                temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.05, label="Temperature")
                top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.01, label="Top-p")
                top_k = gr.Slider(1, 200, value=50, step=1, label="Top-k")
                max_new_tokens = gr.Slider(
                    64, 2048, value=1024, step=64,
                    label="Max audio frames (~21.5 per second; capped by the 2048-position context)",
                )
                seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
            button = gr.Button("Generate", variant="primary")
        with gr.Column():
            output = gr.Audio(label="Output (44.1 kHz)", type="numpy")

    voice.change(choose_voice, voice, [ref_audio, ref_text, reference_box], api_name=False)
    ref_audio.upload(mark_custom, outputs=[voice, ref_text], api_name=False)
    ref_audio.stop_recording(mark_custom, outputs=[voice, ref_text], api_name=False)
    button.click(
        synthesize,
        [text, voice, ref_audio, ref_text, temperature, top_p, top_k, max_new_tokens, seed],
        output,
        api_name="generate",
    )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        allowed_paths=[str(ROOT / "examples")],
    )

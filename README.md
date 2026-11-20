# Audio8-TTS-Realistic-Air (Pinokio launcher)

1-click launcher for [Audio8-TTS-Realistic-Air](https://huggingface.co/Edge0/Audio8-TTS-Realistic-Air) by Edge0. It is a 0.6B text-to-speech model with zero-shot voice cloning from a short reference clip and its transcript. It supports 23 languages: Arabic, Cantonese, Chinese, Dutch, English, Finnish, French, German, Hindi, Indonesian, Italian, Japanese, Korean, Malay, Norwegian, Polish, Portuguese, Russian, Spanish, Swedish, Thai, Turkish and Vietnamese. Output is 44.1 kHz.

Model and code are **Apache-2.0**. Clone only voices you have the right to use.

## Usage

1. **Install**: creates `app/env`, installs Transformers, Gradio and the right PyTorch build for your GPU.
2. **Start**: launches the Gradio UI. The weights (~2.6 GB: `model.safetensors` and `codec.pth`) download from Hugging Face on first start. The app runs on CUDA (bf16) when it's available and falls back to CPU (fp32) otherwise. CPU works but is slow.
3. In the UI:
   - **Voice → Example 10 / Example 18**: built-in reference clips from the official playground.
   - **Voice → Custom recording**: upload or record a clean 3-15 s clip of one speaker, then type its **exact** transcript.
   - **Voice → No reference**: the model picks a random voice.
   - Enter text and click **Generate**. There's no language selector because the model detects the language from the text.

**Advanced** options: temperature (0.8), top-p (0.95), top-k (50), max audio frames (1024 ≈ 47 s; the model holds 2048 text and audio positions in total), and seed (-1 = random). Short passages work best. Long text may get cut off when it hits the frame limit.

## API

The UI exposes a `/generate` endpoint. Arguments go in this order: `text, voice, ref_audio, ref_text, temperature, top_p, top_k, max_new_tokens, seed`. `voice` is `"example10"`, `"example18"`, `"custom"` or `"none"`. With `"none"`, `ref_audio` and `ref_text` are ignored. Replace the URL with the one shown in Pinokio.

### Python

```python
from gradio_client import Client, handle_file

client = Client("http://127.0.0.1:7860/")
result = client.predict(
    "Hello from Audio8!", "custom",
    handle_file("reference.wav"), "Exact transcript of reference.wav.",
    0.8, 0.95, 50, 1024, -1,
    api_name="/generate",
)
print(result)  # path to the generated wav
```

### JavaScript

```javascript
import { Client, handle_file } from "@gradio/client";

const client = await Client.connect("http://127.0.0.1:7860/");
const result = await client.predict("/generate", [
  "Hello from Audio8!", "none", null, "",
  0.8, 0.95, 50, 1024, -1,
]);
console.log(result.data);
```

### Curl

```bash
curl -X POST http://127.0.0.1:7860/gradio_api/call/generate \
  -H "Content-Type: application/json" \
  -d '{"data": ["Hello from Audio8!", "none", null, "", 0.8, 0.95, 50, 1024, -1]}'
# returns {"event_id": "..."}; then:
curl -N http://127.0.0.1:7860/gradio_api/call/generate/<event_id>
```

### Direct Python (inside `app/env`)

```python
import soundfile as sf
import torch
from transformers import AutoModel, AutoProcessor

model_id = "Edge0/Audio8-TTS-Realistic-Air"
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModel.from_pretrained(model_id, trust_remote_code=True, dtype=torch.bfloat16).eval().to("cuda")

inputs = processor(
    text=["Hello from Audio8!"],
    reference_audio=["reference.wav"],
    reference_text=["Exact transcript of reference.wav."],
    return_tensors="pt",
)
inputs = {k: v.to("cuda") for k, v in inputs.items()}
with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=1024, temperature=0.8, top_p=0.95, top_k=50,
                         do_sample=True, return_dict_in_generate=True)
    wavs, lengths = model.decode_audio(out.codes)
sf.write("out.wav", wavs[0, : int(lengths[0])].float().cpu().numpy(), model.config.codec_sample_rate)
```

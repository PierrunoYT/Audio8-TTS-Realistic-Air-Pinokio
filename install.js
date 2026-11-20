module.exports = {
  requires: {
    bundle: "ai"
  },
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "python -c \"from pathlib import Path; Path('env/.installed').unlink(missing_ok=True)\"",
          "uv pip install -r requirements.txt",
        ],
      },
    },
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
          path: "app",
        },
      },
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: "python -c \"import torch, torchaudio, soundfile, gradio; from transformers import AutoModel, AutoProcessor; from pathlib import Path; Path('env/.installed').touch(); print('Audio8-TTS dependencies verified. CUDA:', torch.cuda.is_available())\"",
      },
    },
    {
      when: "{{exists('app/env/.installed')}}",
      method: "input",
      params: {
        title: "Install Complete",
        description: "Audio8-TTS-Realistic-Air installed successfully. Click Start to launch the app (~2.6 GB of weights download on first start).",
      },
    },
  ],
}

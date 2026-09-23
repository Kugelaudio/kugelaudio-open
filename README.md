<h1 align="center">🎙️ KugelAudio</h1>

<p align="center">
  <strong>Open-source text-to-speech for European languages</strong><br>
  Powered by an AR + Diffusion architecture
</p>

<p align="center">
  <a href="https://huggingface.co/kugelaudio/kugelaudio-0-open"><img src="https://img.shields.io/badge/🤗-Hugging_Face_Model-blue" alt="HuggingFace Model"></a>
  <a href="https://github.com/Kugelaudio/kugelaudio-open"><img src="https://img.shields.io/badge/GitHub-Repository-black" alt="GitHub Repository"></a>
</p>

<p align="center">
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="https://huggingface.co/kugelaudio"><img src="https://img.shields.io/badge/🤗-Models-yellow" alt="HuggingFace"></a>
  <a href="https://kugelaudio.com"><img src="https://img.shields.io/badge/☁️-Hosted_API-blue" alt="Hosted API"></a>
</p>

<table align="center" style="border-collapse: collapse; border: none;">
  <tr style="border: none;">
    <td style="border: none; padding: 0 20px;">
      <a href="https://kugelaudio.com">
        <img src="https://www.kugelaudio.com/logos/Logo%20Short.svg" alt="KugelAudio" style="height: 60px; width: auto;">
      </a>
    </td>
    <td style="border: none; padding: 0 20px;">
      <a href="https://hpi.de/ki-servicezentrum/">
        <img src="https://docs.sc.hpi.de/attachments/aisc/aisc-logo.png" alt="KI-Servicezentrum Berlin-Brandenburg" style="height: 60px; width: auto;">
      </a>
    </td>
    <td style="border: none; padding: 0 20px;">
      <a href="https://www.bmftr.bund.de">
        <img src="https://hpi.de/fileadmin/_processed_/a/3/csm_BMFTR_de_Web_RGB_gef_durch_cd1f5345bd.jpg" alt="Gefördert durch BMFTR" style="height: 60px; width: auto;">
      </a>
    </td>
  </tr>
</table>

---

## Motivation

**Open-source text-to-speech models for European languages are significantly lagging behind.** While English TTS has seen remarkable progress, speakers of German, French, Spanish, Polish, and dozens of other European languages have been underserved by the open-source community.

KugelAudio aims to change this. Building on the excellent foundation laid by the [VibeVoice team at Microsoft](https://github.com/microsoft/VibeVoice), we've trained a model specifically focused on European language coverage, using approximately **200,000 hours** of highly pre-processed and enhanced speech data from the [YODAS2 dataset](https://huggingface.co/datasets/espnet/yodas).

## 🏆 Benchmark Results: Outperforming ElevenLabs

**KugelAudio achieves state-of-the-art performance**, beating industry leaders including ElevenLabs in rigorous human preference testing. This breakthrough demonstrates that open-source models can now rival - and surpass - the best commercial TTS systems.

### Human Preference Benchmark (A/B Testing)

We conducted extensive A/B testing with **339 human evaluations** to compare KugelAudio against leading TTS models. Participants listened to a reference voice sample, then compared outputs from two models and selected which sounded more human and closer to the original voice.

### German Language Evaluation

The evaluation specifically focused on **German language samples** with diverse emotional expressions and speaking styles:

* **Neutral Speech**: Standard conversational tones
* **Shouting**: High-intensity, elevated volume speech
* **Singing**: Melodic and rhythmic speech patterns
* **Drunken Voice**: Slurred and irregular speech characteristics

These diverse test cases demonstrate the model's capability to handle a wide range of speaking styles beyond standard narration.

### OpenSkill Ranking Results

| Rank | Model | Score | Record | Win Rate |
|------|-------|-------|--------|----------|
| 🥇 1 | **KugelAudio** | **26** | 71W / 20L / 23T | **78.0%** |
| 🥈 2 | ElevenLabs Multi v2 | 25 | 56W / 34L / 22T | 62.2% |
| 🥉 3 | ElevenLabs v3 | 21 | 64W / 34L / 16T | 65.3% |
| 4 | Cartesia | 21 | 55W / 38L / 19T | 59.1% |
| 5 | VibeVoice | 10 | 30W / 74L / 8T | 28.8% |
| 6 | CosyVoice v3 | 9 | 15W / 91L / 8T | 14.2% |

_Based on 339 evaluations using Bayesian skill-rating system (OpenSkill)_

## Audio Samples

Listen to KugelAudio's diverse voice capabilities across different speaking styles and languages:

### German Voice Samples

| Sample | Description | Audio Player |
|--------|-------------|--------------|
| **Whispering** | Soft whispering voice | <audio controls><source src="https://huggingface.co/kugelaudio/kugelaudio-0-open/resolve/main/samples/258_Lukas_der_Flüsterer.wav" type="audio/wav"></audio> |
| **Female Narrator** | Professional female reader voice | <audio controls><source src="https://huggingface.co/kugelaudio/kugelaudio-0-open/resolve/main/samples/266_Petra_die_Vorleserin.wav" type="audio/wav"></audio> |
| **Angry Voice** | Irritated and frustrated speech | <audio controls><source src="https://huggingface.co/kugelaudio/kugelaudio-0-open/resolve/main/samples/261_Sauerer_Felix.wav" type="audio/wav"></audio> |
| **Radio Announcer** | Professional radio broadcast voice | <audio controls><source src="https://huggingface.co/kugelaudio/kugelaudio-0-open/resolve/main/samples/277_Radio_Lars.wav" type="audio/wav"></audio> |

*All samples are generated using pre-encoded voice embeddings.*

### Training Details

- **Base Model**: [Microsoft VibeVoice](https://github.com/microsoft/VibeVoice)
- **Training Data**: ~200,000 hours from [YODAS2](https://huggingface.co/datasets/espnet/yodas)
- **Hardware**: 8x NVIDIA H100 GPUs
- **Training Duration**: 5 days

### Supported Languages

KugelAudio supports **24 major European languages** with varying levels of quality based on dataset representation:

| Language | Code | Flag | Language | Code | Flag | Language | Code | Flag |
|----------|------|------|----------|------|------|----------|------|------|
| English | en | 🇺🇸 | German | de | 🇩🇪 | French | fr | 🇫🇷 |
| Spanish | es | 🇪🇸 | Italian | it | 🇮🇹 | Portuguese | pt | 🇵🇹 |
| Dutch | nl | 🇳🇱 | Polish | pl | 🇵🇱 | Russian | ru | 🇷🇺 |
| Ukrainian | uk | 🇺🇦 | Czech | cs | 🇨🇿 | Romanian | ro | 🇷🇴 |
| Hungarian | hu | 🇭🇺 | Swedish | sv | 🇸🇪 | Danish | da | 🇩🇰 |
| Finnish | fi | 🇫🇮 | Norwegian | no | 🇳🇴 | Greek | el | 🇬🇷 |
| Bulgarian | bg | 🇧🇬 | Slovak | sk | 🇸🇰 | Croatian | hr | 🇭🇷 |
| Serbian | sr | 🇷🇸 | Turkish | tr | 🇹🇷 | | | |

> **📊 Language Coverage Disclaimer**: Quality varies significantly by language. Spanish, French, English, and German have the strongest representation in our training data (~200,000 hours from YODAS2). Other languages may have reduced quality, prosody, or vocabulary coverage depending on their availability in the training dataset.

## 📖 Start Here

Get started with KugelAudio quickly using our documentation:

| | |
|---|---|
| 📥 [**Installation**](#installation) | Set up KugelAudio on your machine |
| 🎯 [**Quick Start**](#quick-start) | Generate your first speech in minutes |
| 🎭 [**Voices**](#voices) | Use pre-encoded voices for different speakers |
| ☁️ [**Hosted API**](#hosted-api) | Use our cloud API for zero-setup inference |
| 🔒 [**Watermarking**](#audio-watermarking) | Verify AI-generated audio |
| 📦 [**Models**](#models) | Available model variants and benchmarks |

---

## Features

- 🏆 **State-of-the-Art Performance**: Outperforms ElevenLabs and other leading TTS models in human evaluations
- 🌍 **European Language Focus**: Trained specifically for 24 major European languages
- **High-Quality TTS**: State-of-the-art speech synthesis using AR + Diffusion
- 🎭 **Pre-encoded Voices**: Select from a set of pre-encoded speaker voices
- **Audio Watermarking**: All generated audio is watermarked using [Facebook's AudioSeal](https://huggingface.co/facebook/audioseal)
- 🎭 **Emotional Range**: Supports various speaking styles including shouting, singing, and expressive speech
- **Web Interface**: Easy-to-use Gradio UI for non-technical users
- **HuggingFace Integration**: Seamless loading from HuggingFace Hub

## Quick Start

### Installation

#### Prerequisites

- Python 3.10 or higher
- PyTorch 2.0 or higher
- CUDA (recommended for GPU acceleration)
- [uv](https://docs.astral.sh/uv/) (recommended package manager)

#### Install uv

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip
pip install uv
```

#### Installation

```bash
# Clone the repository
git clone https://github.com/Kugelaudio/kugelaudio-open.git
cd kugelaudio-open

# Run directly with uv (recommended - handles all dependencies automatically)
uv run python start.py
```

That's it! The `uv run` command will automatically create a virtual environment and install all dependencies.

### Launch Web Interface

```bash
# Quick start with uv (recommended)
uv run python start.py

# With a public share link
uv run python start.py ui --share

# Custom host and port
uv run python start.py ui --host 0.0.0.0 --port 8080
```

Then open http://127.0.0.1:7860 in your browser.

### Command Line Usage

```bash
# Generate speech from text
uv run python start.py generate "Hello, this is KugelAudio!" -o hello.wav

# With a specific pre-encoded voice
uv run python start.py generate "Hallo, das ist eine klare Stimme!" --voice clear -o clear.wav

# Using the default model for higher quality
uv run python start.py generate "Premium quality speech" --model kugelaudio/kugelaudio-0-open -o premium.wav

# Check if audio contains watermark
uv run python start.py verify audio.wav
```

### Python API

```python
from kugelaudio_open import (
    KugelAudioForConditionalGenerationInference,
    KugelAudioProcessor,
)
import torch

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = KugelAudioForConditionalGenerationInference.from_pretrained(
    "kugelaudio/kugelaudio-0-open",
    torch_dtype=torch.bfloat16,
).to(device)
model.eval()

# Strip encoder weights to save VRAM (only decoders needed for inference)
model.model.strip_encoders()

processor = KugelAudioProcessor.from_pretrained("kugelaudio/kugelaudio-0-open")

# See available voices
print(processor.get_available_voices())  # ["default", "clear", "english_female", "english_male"]

# Generate speech (watermark is automatically applied)
inputs = processor(text="Hello world!", voice="default", return_tensors="pt")
inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

with torch.no_grad():
    outputs = model.generate(**inputs, cfg_scale=3.0)

# Save audio
processor.save_audio(outputs.speech_outputs[0], "output.wav")
```

### Voices

KugelAudio provides pre-encoded voices that can be selected by name. The voices are stored as `.pt` files in the model repository and are automatically downloaded from HuggingFace when needed.

| Voice | Language | Description |
|-------|----------|-------------|
| `default` | German | Calm female narrator |
| `clear` | German | Clear, young female conversational voice |
| `english_female` | English | Friendly female teacher (British English) |
| `english_male` | English | Conversational male voice (British English) |

Voices work across languages but sound most natural, and are most reliable, in their native language.

```python
# List available voices
voices = processor.get_available_voices()
print(voices)  # ["default", "clear", "english_female", "english_male"]

# Generate with a specific voice
inputs = processor(text="Hello world!", voice="english_male", return_tensors="pt")
inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

with torch.no_grad():
    outputs = model.generate(**inputs, cfg_scale=3.0)

processor.save_audio(outputs.speech_outputs[0], "english_male_output.wav")
```

> **Note:** Voice cloning from raw audio is not supported in this open-source release. Only the pre-encoded voices listed in `voices/voices.json` are available.

## Hosted API

Don't want to run your own infrastructure? Use our **hosted API at [kugelaudio.com](https://kugelaudio.com)**:

- ⚡ **Ultra-low latency**: Inference as fast as **39ms**, end-to-end latency **<100ms** including network
- 🌍 **Global edge deployment**: Low latency worldwide
- 🔧 **Zero setup**: No GPU required, just API calls
- 📈 **Auto-scaling**: Handle any traffic volume

### Python SDK

```bash
uv pip install kugelaudio
```

```python
from kugelaudio import KugelAudio

# Initialize the client
client = KugelAudio(api_key="your_api_key")

# Generate speech
audio = client.tts.generate(
    text="Hello from KugelAudio!",
    model="kugel-1-turbo",
)

# Save to file
audio.save("output.wav")
print(f"Generated {audio.duration_seconds:.2f}s in {audio.generation_ms:.0f}ms")
```


📚 [Full SDK Documentation →](https://docs.kugelaudio.com/sdks/python)

## Audio Watermarking

All audio generated by KugelAudio contains an imperceptible watermark using [Facebook's AudioSeal](https://huggingface.co/facebook/audioseal) technology. This helps identify AI-generated content and prevent misuse.

### Verify Watermark

```python
from kugelaudio_open.watermark import AudioWatermark

watermark = AudioWatermark()

# Check if audio is watermarked
result = watermark.detect(audio, sample_rate=24000)

print(f"Detected: {result.detected}")
print(f"Confidence: {result.confidence:.1%}")
```

### Watermark Features

- **Imperceptible**: No audible difference in audio quality
- **Robust**: Survives compression, resampling, and editing
- **Fast Detection**: Real-time capable detection
- **Sample-Level**: 1/16k second resolution

## Models

| Model | Parameters | Quality | RTF | Speed | VRAM |
|-------|------------|---------|-----|-------|------|
| [kugelaudio-0-open](https://huggingface.co/kugelaudio/kugelaudio-0-open) | 7B | Best | 1.00 | 1.0x realtime | ~19GB |

*RTF = Real-Time Factor (generation time / audio duration). Lower is faster.*
## Architecture

KugelAudio uses a hybrid AR + Diffusion architecture:

1. **Text Encoder**: Qwen2-based language model encodes input text
2. **TTS Backbone**: Upper transformer layers generate speech representations
3. **Diffusion Head**: Predicts speech latents using denoising diffusion
4. **Acoustic Decoder**: Converts latents to audio waveforms

## Configuration

### Environment Variables

```bash
# Use specific GPU
export CUDA_VISIBLE_DEVICES=0

# Enable TF32 for faster computation on Ampere GPUs
export TORCH_ALLOW_TF32=1
```

### Advanced Generation Parameters

```python
outputs = model.generate(
    **inputs,
    cfg_scale=3.0,                  # Guidance scale (1.0-10.0)
    max_new_tokens=4096,            # Maximum generation length
)
```

## Responsible Use

This technology is intended for legitimate purposes:

✅ **Appropriate Uses:**
- Accessibility (TTS for visually impaired)
- Content creation (podcasts, videos, audiobooks)
- Voice assistants and chatbots
- Language learning applications
- Creative projects with consent

❌ **Prohibited Uses:**
- Creating deepfakes or misleading content
- Impersonating individuals without consent
- Fraud or deception
- Any illegal activities

All generated audio is watermarked to enable detection of AI-generated content.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

This model would not have been possible without the contributions of many individuals and organizations:

- **[Microsoft VibeVoice Team](https://github.com/microsoft/VibeVoice)**: For the excellent foundation architecture that this model builds upon
- **[YODAS2 Dataset](https://huggingface.co/datasets/espnet/yodas)**: For providing the large-scale multilingual speech data
- **[Qwen Team](https://huggingface.co/Qwen)**: For the powerful language model backbone
- **[Facebook AudioSeal](https://huggingface.co/facebook/audioseal)**: For the audio watermarking technology
- **[HuggingFace](https://huggingface.co)**: For model hosting and the transformers library

### Special Thanks

- **Carlos Menke**: For his invaluable efforts in gathering the first datasets and extensive work benchmarking the model
- **AI Service Center Berlin-Brandenburg (KI-Servicezentrum)**: For providing the GPU resources (8x H100) that made training this model possible





## Authors

**Kajo Kratzenstein**  
📧 [kajo@kugelaudio.com](mailto:kajo@kugelaudio.com)  
🌐 [kugelaudio.com](https://kugelaudio.com)

**Carlos Menke**

## Citation

```bibtex
@software{kugelaudio2026,
  title = {KugelAudio: Open-Source Text-to-Speech for European Languages},
  author = {Kratzenstein, Kajo and Menke, Carlos},
  year = {2026},
  institution = {Hasso-Plattner-Institut},
  url = {https://github.com/kugelaudio/kugelaudio}
}
```


---

<p align="center">
  <strong>Funding Notice</strong>
</p>

<p align="center">
  Das zugrunde liegende Vorhaben wurde mit Mitteln des Bundesministeriums für Forschung, Technologie und Raumfahrt unter dem Förderkennzeichen »KI-Servicezentrum Berlin-Brandenburg« 16IS22092 gefördert. Die Verantwortung für den Inhalt dieser Seite liegt bei der Autorin/beim Autor.
</p>

<p align="center">
  <em>This project was funded by the German Federal Ministry of Research, Technology and Space under the funding code "AI Service Center Berlin-Brandenburg" 16IS22092. The responsibility for the content of this publication lies with the author.</em>
</p>

---

#!/usr/bin/env python3
"""Command-line interface for KugelAudio."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="KugelAudio - Open-source text-to-speech",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch web interface
  kugelaudio ui
  
  # Launch with public share link
  kugelaudio ui --share
  
  # Generate speech from command line
  kugelaudio generate "Hello world!" -o output.wav
  
  # Generate with a specific voice
  kugelaudio generate "Hello world!" --voice default -o output.wav
  
  # Encode your own voice from a reference recording
  kugelaudio encode-voice reference.wav -o my_voice.pt
  kugelaudio generate "Hello world!" --voice my_voice.pt -o output.wav
  
  # Check watermark in audio file
  kugelaudio verify audio.wav
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # UI command
    ui_parser = subparsers.add_parser("ui", help="Launch Gradio web interface")
    ui_parser.add_argument("--share", action="store_true", help="Create public share link")
    ui_parser.add_argument("--host", default="127.0.0.1", help="Server hostname")
    ui_parser.add_argument("--port", type=int, default=7860, help="Server port")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate speech from text")
    gen_parser.add_argument("text", help="Text to synthesize")
    gen_parser.add_argument("-o", "--output", default="output.wav", help="Output file path")
    gen_parser.add_argument(
        "-v",
        "--voice",
        help="Pre-encoded voice: a name from the voices.json registry or a path to a .pt file",
    )
    gen_parser.add_argument("--model", default="kugelaudio/kugelaudio-0-open", help="Model ID")
    gen_parser.add_argument("--cfg-scale", type=float, default=3.0, help="Guidance scale")

    # Encode voice command
    encode_parser = subparsers.add_parser(
        "encode-voice", help="Encode a reference recording into a pre-encoded voice (.pt)"
    )
    encode_parser.add_argument(
        "audio", help="Reference audio file (a few seconds of clean speech)"
    )
    encode_parser.add_argument("-o", "--output", default="voice.pt", help="Output .pt file path")
    encode_parser.add_argument("--model", default="kugelaudio/kugelaudio-0-open", help="Model ID")

    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Check watermark in audio")
    verify_parser.add_argument("audio", help="Audio file to check")

    args = parser.parse_args()

    if args.command == "ui":
        from kugelaudio_open.ui import launch_app

        launch_app(
            share=args.share,
            server_name=args.host,
            server_port=args.port,
        )

    elif args.command == "generate":
        import torch

        from kugelaudio_open.models import KugelAudioForConditionalGenerationInference
        from kugelaudio_open.processors import KugelAudioProcessor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        print(f"Loading model {args.model}...")
        model = KugelAudioForConditionalGenerationInference.from_pretrained(
            args.model, torch_dtype=dtype
        ).to(device)
        model.eval()

        processor = KugelAudioProcessor.from_pretrained(args.model)

        # Strip encoders to save VRAM (only decoders needed for inference)
        model.model.strip_encoders()

        # Process inputs with optional pre-encoded voice
        inputs = processor(text=args.text, voice=args.voice, return_tensors="pt")
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        print("Generating speech...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                cfg_scale=args.cfg_scale,
                max_new_tokens=4096,
            )

        # Audio is already watermarked by the model's generate method
        audio = outputs.speech_outputs[0]

        # Save
        processor.save_audio(audio, args.output)
        print(f"Audio saved to {args.output}")

    elif args.command == "encode-voice":
        import os

        import torch

        from kugelaudio_open.models import KugelAudioForConditionalGenerationInference
        from kugelaudio_open.processors import KugelAudioProcessor

        if not os.path.exists(args.audio):
            print(f"Error: audio file not found: {args.audio}")
            sys.exit(1)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        print(f"Loading model {args.model}...")
        model = KugelAudioForConditionalGenerationInference.from_pretrained(
            args.model, torch_dtype=dtype
        ).to(device)
        model.eval()
        # Note: encoders are NOT stripped here; the acoustic encoder is
        # exactly what we need to turn audio into a voice embedding.

        processor = KugelAudioProcessor.from_pretrained(args.model)

        print(f"Encoding {args.audio}...")
        audio = processor.audio_processor(args.audio, return_tensors="pt")["audio"]
        audio = audio.to(device=device, dtype=dtype)
        with torch.no_grad():
            encoded = model.acoustic_tokenizer.encode(audio)

        # Store as float32 so the voice file is independent of the dtype/device
        # used for encoding; the model casts it at generation time.
        acoustic_mean = encoded.mean.to(torch.float32).cpu()
        torch.save({"acoustic_mean": acoustic_mean}, args.output)
        print(f"Saved voice ({acoustic_mean.shape[1]} frames) to {args.output}")
        print(f'Use it with: kugelaudio generate "Hello world!" --voice {args.output}')

    elif args.command == "verify":
        import numpy as np
        import soundfile as sf

        from kugelaudio_open.watermark import AudioWatermark

        audio, sr = sf.read(args.audio)

        watermark = AudioWatermark()
        result = watermark.detect(audio, sample_rate=sr)

        if result.detected:
            print(f"✅ Watermark DETECTED (confidence: {result.confidence:.1%})")
            print("This audio was generated by KugelAudio.")
        else:
            print(f"❌ No watermark detected (confidence: {result.confidence:.1%})")
            print("This audio does not appear to be generated by KugelAudio.")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

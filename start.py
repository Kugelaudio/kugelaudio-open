#!/usr/bin/env python3
"""Quick start script for KugelAudio.

This script provides easy access to all KugelAudio functionality:
- Launch the web UI
- Generate speech from command line
- Verify watermarks

Usage:
    # Launch web UI (default)
    python start.py
    python start.py ui
    python start.py ui --share  # With public link
    
    # Generate speech
    python start.py generate "Hello world!" -o output.wav
    python start.py generate "Hallo Welt!" --voice clear -o clear.wav
    
    # Verify watermark
    python start.py verify audio.wav
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="KugelAudio - Open-source text-to-speech with pre-encoded voices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch web interface (default action)
  python start.py
  python start.py ui
  python start.py ui --share
  python start.py ui --host 0.0.0.0 --port 8080
  
  # Generate speech from command line
  python start.py generate "Hello world!" -o output.wav
  python start.py generate "Hallo Welt!" --voice clear -o clear.wav
  python start.py generate "Premium quality" --model kugelaudio/kugelaudio-0-open -o premium.wav
  
  # Verify watermark in audio
  python start.py verify audio.wav
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # UI command (default)
    ui_parser = subparsers.add_parser("ui", help="Launch Gradio web interface")
    ui_parser.add_argument("--share", action="store_true", help="Create public share link")
    ui_parser.add_argument("--host", default="127.0.0.1", help="Server hostname (use 0.0.0.0 for network access)")
    ui_parser.add_argument("--port", type=int, default=7860, help="Server port")
    ui_parser.add_argument("--model", default="kugelaudio/kugelaudio-0-open", help="Default model to load")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate speech from text")
    gen_parser.add_argument("text", help="Text to synthesize")
    gen_parser.add_argument("-o", "--output", default="output.wav", help="Output file path")
    gen_parser.add_argument("-v", "--voice", default="default", help="Pre-encoded voice name (from voices.json registry)")
    gen_parser.add_argument("--model", default="kugelaudio/kugelaudio-0-open", help="Model ID")
    gen_parser.add_argument("--cfg-scale", type=float, default=3.0, help="Guidance scale (1.0-10.0)")
    gen_parser.add_argument("--max-tokens", type=int, default=4096, help="Maximum generation tokens")

    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Check watermark in audio")
    verify_parser.add_argument("audio", help="Audio file to check")

    args = parser.parse_args()

    # Default to UI if no command specified
    if args.command is None:
        args.command = "ui"
        args.share = False
        args.host = "127.0.0.1"
        args.port = 7860
        args.model = "kugelaudio/kugelaudio-0-open"

    if args.command == "ui":
        print("🎙️ Starting KugelAudio Web Interface...")
        print(f"   Host: {args.host}")
        print(f"   Port: {args.port}")
        print(f"   Share: {args.share}")
        print()
        
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

        print(f"🎙️ KugelAudio Speech Generation")
        print(f"   Model: {args.model}")
        print(f"   Device: {device}")
        print(f"   Text: {args.text[:50]}..." if len(args.text) > 50 else f"   Text: {args.text}")
        print(f"   Voice: {args.voice}")
        print()

        print("Loading model...")
        model = KugelAudioForConditionalGenerationInference.from_pretrained(
            args.model, torch_dtype=dtype
        ).to(device)
        model.eval()

        processor = KugelAudioProcessor.from_pretrained(args.model)

        # Process inputs
        inputs = processor(
            text=args.text,
            voice=args.voice,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        print("Generating speech...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                cfg_scale=args.cfg_scale,
                max_new_tokens=args.max_tokens,
            )

        audio = outputs.speech_outputs[0]

        # Save
        processor.save_audio(audio, args.output)
        print(f"✅ Audio saved to {args.output}")

    elif args.command == "verify":
        import numpy as np
        import soundfile as sf
        from kugelaudio_open.watermark import AudioWatermark

        print(f"🔍 Checking watermark in: {args.audio}")
        print()

        audio, sr = sf.read(args.audio)

        watermark = AudioWatermark()
        result = watermark.detect(audio, sample_rate=sr)

        if result.detected:
            print(f"✅ Watermark DETECTED")
            print(f"   Confidence: {result.confidence:.1%}")
            print("   This audio was generated by KugelAudio.")
        else:
            print(f"❌ No watermark detected")
            print(f"   Confidence: {result.confidence:.1%}")
            print("   This audio does not appear to be generated by KugelAudio.")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

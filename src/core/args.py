# core/args.py
import argparse

def parse_args():
    """Parses and returns the command-line arguments for the application."""
    parser = argparse.ArgumentParser(description="goan: FramePack-based Video Generation UI")
    parser.add_argument('--share', action='store_true', default=False, help="Enable Gradio sharing link.")
    parser.add_argument("--server", type=str, default='127.0.0.1', help="Server name to bind to.")
    parser.add_argument("--port", type=int, required=False, help="Port to run the server on.")
    parser.add_argument("--inbrowser", action='store_true', default=False, help="Launch in browser automatically.")
    parser.add_argument("--allowed_output_paths", type=str, default="", help="Comma-separated list of additional output folders Gradio is allowed to access.")
    args = parser.parse_args()
    print(f"goan launching with args: {args}")
    return args
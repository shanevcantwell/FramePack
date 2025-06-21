# goan.py
# Main application orchestrator for the goan video generation UI.

import os
import sys
import atexit
import gradio as gr

# Add project root and ui directory to sys.path for module discovery
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Local Application Imports
from core import args as args_manager
from core import model_loader
from ui import (
    layout as layout_manager,
    queue as queue_manager,
    workspace as workspace_manager,
    lora as lora_manager,
    shared_state,
    switchboard
)

# Environment Setup & Model Loading
os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))
args = args_manager.parse_args()
model_loader.load_and_configure_models()

# UI Creation and Component Mapping
print("Creating UI layout...")
gr.processing_utils.video_is_playable = lambda video_filepath: True
ui_components = layout_manager.create_ui()
block = ui_components['block']

# --- Event Wiring is now delegated to the switchboard ---
# This replaces the entire previous "Event Wiring" and "with block:" section.
switchboard.wire_all_events(ui_components)

# --- Application Load/Startup Events ---
# These are still initiated from the main file.
atexit.register(queue_manager.autosave_queue_on_exit_action, shared_state.global_state_for_autosave)

# --- Application Launch ---
if __name__ == "__main__":
    print("Starting goan FramePack UI...")

    initial_output_folder_path = workspace_manager.get_initial_output_folder_from_settings()
    expanded_outputs_folder_for_launch = os.path.abspath(initial_output_folder_path)

    final_allowed_paths = [expanded_outputs_folder_for_launch]
    if args.allowed_output_paths:
        custom_paths = [os.path.abspath(os.path.expanduser(p.strip())) for p in args.allowed_output_paths.split(',') if p.strip()]
        final_allowed_paths.extend(custom_paths)

    # Ensure the LoRA directory is always allowed for uploads/access
    final_allowed_paths.append(lora_manager.LORA_DIR)
    final_allowed_paths = list(set(final_allowed_paths))

    print(f"Gradio allowed paths: {final_allowed_paths}")
    block.launch(server_name=args.server, server_port=args.port, share=args.share, inbrowser=args.inbrowser, allowed_paths=final_allowed_paths)
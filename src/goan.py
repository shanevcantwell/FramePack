# goan.py
# Main application orchestrator for the goan video generation UI.

import os
import sys
# import atexit
import gradio as gr
import logging
import logging.handlers
import warnings

# Add project root and ui directory to sys.path for module discovery
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Suppress the specific torchvision video deprecation warning that is spammed on each segment save.
warnings.filterwarnings(
    "ignore",
    message="The video decoding and encoding capabilities of torchvision are deprecated",
    category=UserWarning,
    module="torchvision.io._video_deprecation_warning"
)

# Local Application Imports
from core import args as args_manager
from core import model_loader
from ui import (
    layout as layout_manager,
    queue as queue_manager,
    workspace as workspace_manager,
    lora as lora_manager,
    shared_state as shared_state_module, # Import module for access to instance
    switchboard
)

def setup_logging(debug_mode=False):
    """Configures a root logger to output to console and a daily rotating file."""
    log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_filepath = os.path.join(log_dir, 'goan.log')

    # --- Define Log Levels ---
    # If debug mode is on, the file gets DEBUG messages and the console gets INFO.
    # Otherwise, the file gets INFO and the console gets WARNING.
    file_log_level = logging.DEBUG if debug_mode else logging.INFO
    console_log_level = logging.INFO if debug_mode else logging.WARNING

    # --- Configure Root Logger ---
    # The root logger must be set to the most verbose level of all its handlers
    # to allow messages to pass through to be filtered by the handlers themselves.
    root_logger = logging.getLogger()
    root_logger.setLevel(min(file_log_level, console_log_level))

    # Remove any existing handlers to avoid duplicate logs if this function is ever called more than once.
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # --- Create and Add Handlers ---
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # File Handler (Timed Rotating)
    file_handler = logging.handlers.TimedRotatingFileHandler(
        log_filepath, when='midnight', interval=1, backupCount=7, encoding='utf-8'
    )
    file_handler.setLevel(file_log_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Console Handler (Stream)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(console_log_level)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

# Environment Setup & Model Loading
os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))
args = args_manager.parse_args()
setup_logging(debug_mode=args.debug)
logger = logging.getLogger(__name__)
model_loader.load_and_configure_models()

# UI Creation and Component Mapping
logger.info("Creating UI layout...")
gr.processing_utils.video_is_playable = lambda video_filepath: True
ui_components = layout_manager.create_ui()
block = ui_components['block']

# --- Event Wiring is delegated to the switchboard ---
switchboard.wire_all_events(ui_components)

# --- Application Load/Startup Events ---
# atexit.register(queue_manager.autosave_queue_on_exit_action, shared_state_module.shared_state_instance.global_state_for_autosave)

# --- Application Launch ---
if __name__ == "__main__":
    logger.info("Starting goan FramePack UI...")

    initial_output_folder_path = workspace_manager.get_initial_output_folder_from_settings()
    expanded_outputs_folder_for_launch = os.path.abspath(initial_output_folder_path)

    final_allowed_paths = [expanded_outputs_folder_for_launch]
    if args.allowed_output_paths:
        custom_paths = [os.path.abspath(os.path.expanduser(p.strip())) for p in args.allowed_output_paths.split(',') if p.strip()]
        final_allowed_paths.extend(custom_paths)

    # Ensure the LoRA directory is always allowed for uploads/access
    final_allowed_paths.append(lora_manager.LORA_DIR)
    final_allowed_paths = list(set(final_allowed_paths))

    logger.info(f"Gradio allowed paths: {final_allowed_paths}")
    block.launch(server_name=args.server, server_port=args.port, share=args.share, inbrowser=args.inbrowser, allowed_paths=final_allowed_paths)
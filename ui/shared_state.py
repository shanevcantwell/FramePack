# ui/shared_state.py
import threading

# --- Application-wide Threading Controls ---
# Lock for thread-safe queue modifications.
queue_lock = threading.Lock()

# Event to signal the abortion of the current processing task.
# This is kept for compatibility and as a simple, overarching abort flag.
abort_event = threading.Event()

# CHANGED: Added state dictionary for the multi-level abort feature.
# 'level' 0: No abort. 1: Graceful abort (1-click). 2: Hard abort (2-click).
# 'last_click_time' tracks clicks to differentiate single vs. double clicks.
abort_state = {'level': 0, 'last_click_time': 0}


# --- Model and Global State Containers ---
# This dictionary will be populated at runtime by the main script after the models are loaded.
models = {}

# This dictionary holds the application state, which is passed to the atexit
# handler to enable the autosave functionality on browser close or exit.
global_state_for_autosave = {}


# --- UI and Parameter Mapping Constants ---
CREATIVE_PARAM_KEYS = [
    'prompt', 'n_prompt', 'total_second_length', 'seed', 'preview_frequency_ui',
    'segments_to_decode_csv', 'gs_ui', 'gs_schedule_shape_ui', 'gs_final_ui', 'steps', 'cfg', 'rs'
]
ENVIRONMENT_PARAM_KEYS = [
    'use_teacache', 'use_fp32_transformer_output_ui', 'gpu_memory_preservation',
    'mp4_crf', 'output_folder_ui', 'latent_window_size'
]
ALL_TASK_UI_KEYS = CREATIVE_PARAM_KEYS + ENVIRONMENT_PARAM_KEYS
UI_TO_WORKER_PARAM_MAP = {
    'prompt': 'prompt',
    'n_prompt': 'n_prompt',
    'total_second_length': 'total_second_length',
    'seed': 'seed',
    'use_teacache': 'use_teacache',
    'preview_frequency_ui': 'preview_frequency',
    'segments_to_decode_csv': 'segments_to_decode_csv',
    'gs_ui': 'gs',
    'gs_schedule_shape_ui': 'gs_schedule_active',
    'gs_final_ui': 'gs_final',
    'steps': 'steps',
    'cfg': 'cfg',
    'latent_window_size': 'latent_window_size',
    'gpu_memory_preservation': 'gpu_memory_preservation',
    'use_fp32_transformer_output_ui': 'use_fp32_transformer_output',
    'rs': 'rs',
    'mp4_crf': 'mp4_crf',
    'output_folder_ui': 'output_folder'
}
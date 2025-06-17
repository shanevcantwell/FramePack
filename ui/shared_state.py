# ui/shared_state.py
import threading

# --- Application-wide Threading Controls ---
# Lock for thread-safe queue modifications.
queue_lock = threading.Lock()

# Event to signal the abortion of the current processing task.
# This is kept for compatibility and as a simple, overarching abort flag.
# MODIFIED START: Renamed abort_event to interrupt_flag
interrupt_flag = threading.Event()
# MODIFIED END

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

# ADDED: Dictionary to hold system-level information detected at startup.
system_info = {
    'is_legacy_gpu': False,
}


# --- UI and Parameter Mapping Constants ---
# ADDED: Centralized lists of UI component keys to ensure consistency across modules.
CREATIVE_UI_KEYS = [
    'prompt_ui', 'n_prompt_ui', 'total_second_length_ui', 'seed_ui', 'preview_frequency_ui',
    'segments_to_decode_csv_ui', 'gs_ui', 'gs_schedule_shape_ui', 'gs_final_ui', 'steps_ui', 'cfg_ui', 'rs_ui'
]
ENVIRONMENT_UI_KEYS = [
    'use_teacache_ui', 'use_fp32_transformer_output_checkbox_ui', 'gpu_memory_preservation_ui',
    'mp4_crf_ui', 'output_folder_ui_ctrl', 'latent_window_size_ui'
]
ALL_TASK_UI_KEYS = CREATIVE_UI_KEYS + ENVIRONMENT_UI_KEYS

# CHANGED: Corrected the keys of this map to be the actual UI component keys.
# This map is the single source of truth for converting UI component names to worker parameter names.
UI_TO_WORKER_PARAM_MAP = {
    'prompt_ui': 'prompt',
    'n_prompt_ui': 'n_prompt',
    'total_second_length_ui': 'total_second_length',
    'seed_ui': 'seed',
    'preview_frequency_ui': 'preview_frequency',
    'segments_to_decode_csv_ui': 'segments_to_decode_csv',
    'gs_ui': 'gs',
    'gs_schedule_shape_ui': 'gs_schedule_active',
    'gs_final_ui': 'gs_final',
    'steps_ui': 'steps',
    'cfg_ui': 'cfg',
    'rs_ui': 'rs',
    'use_teacache_ui': 'use_teacache',
    'use_fp32_transformer_output_checkbox_ui': 'use_fp32_transformer_output',
    'gpu_memory_preservation_ui': 'gpu_memory_preservation',
    'mp4_crf_ui': 'mp4_crf',
    'output_folder_ui_ctrl': 'output_folder',
    'latent_window_size_ui': 'latent_window_size'
}

# The CREATIVE_PARAM_KEYS list defines the canonical names for parameters that are saved
# within image metadata and is used when loading that metadata back into the UI.
# CHANGED: This is now built dynamically to guarantee its order and content match the UI keys and the map.
CREATIVE_PARAM_KEYS = [UI_TO_WORKER_PARAM_MAP[key] for key in CREATIVE_UI_KEYS]
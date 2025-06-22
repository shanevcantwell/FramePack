# ui/shared_state.py
import threading
from .enums import ComponentKey as K

# --- Application-wide Threading Controls ---
# Lock for thread-safe queue modifications.
queue_lock = threading.Lock()

# Event to signal the abortion of the current processing task.
# This is kept for compatibility and as a simple, overarching abort flag.
interrupt_flag = threading.Event()

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
# ADDED: Centralized list of UI component keys for LoRA management.
LORA_UI_KEYS = [K.LORA_UPLOAD_BUTTON_UI, K.LORA_ROW_0]

# MODIFIED: Added the Roll-off UI component keys to the list of creative parameters.
CREATIVE_UI_KEYS = [
    K.PROMPT_UI, K.N_PROMPT_UI, K.TOTAL_SECOND_LENGTH_UI, K.SEED_UI, K.PREVIEW_FREQUENCY_UI,
    K.SEGMENTS_TO_DECODE_CSV_UI, K.GS_UI, K.GS_SCHEDULE_SHAPE_UI, K.GS_FINAL_UI,
    K.ROLL_OFF_START_UI, K.ROLL_OFF_FACTOR_UI,
    K.STEPS_UI, K.CFG_UI, K.RS_UI
]
ENVIRONMENT_UI_KEYS = [
    K.USE_TEACACHE_UI, K.USE_FP32_TRANSFORMER_OUTPUT_CHECKBOX_UI, K.GPU_MEMORY_PRESERVATION_UI,
    K.MP4_CRF_UI, K.OUTPUT_FOLDER_UI_CTRL, K.LATENT_WINDOW_SIZE_UI
]
ALL_TASK_UI_KEYS = CREATIVE_UI_KEYS + ENVIRONMENT_UI_KEYS

# MODIFIED: Added mappings for the Roll-off parameters.
# This is the single source of truth for converting UI component names to worker parameter names.
UI_TO_WORKER_PARAM_MAP = {
    K.PROMPT_UI: 'prompt',
    K.N_PROMPT_UI: 'n_prompt',
    K.TOTAL_SECOND_LENGTH_UI: 'total_second_length',
    K.SEED_UI: 'seed',
    K.PREVIEW_FREQUENCY_UI: 'preview_frequency',
    K.SEGMENTS_TO_DECODE_CSV_UI: 'segments_to_decode_csv',
    K.GS_UI: 'gs',
    K.GS_SCHEDULE_SHAPE_UI: 'gs_schedule_shape',
    K.GS_FINAL_UI: 'gs_final',
    K.ROLL_OFF_START_UI: 'roll_off_start',
    K.ROLL_OFF_FACTOR_UI: 'roll_off_factor',
    K.STEPS_UI: 'steps',
    K.CFG_UI: 'cfg',
    K.RS_UI: 'rs',
    K.USE_TEACACHE_UI: 'use_teacache',
    K.USE_FP32_TRANSFORMER_OUTPUT_CHECKBOX_UI: 'use_fp32_transformer_output',
    K.GPU_MEMORY_PRESERVATION_UI: 'gpu_memory_preservation',
    K.MP4_CRF_UI: 'mp4_crf',
    K.OUTPUT_FOLDER_UI_CTRL: 'output_folder',
    K.LATENT_WINDOW_SIZE_UI: 'latent_window_size'
}

# The CREATIVE_PARAM_KEYS list defines the canonical names for parameters that are saved
# within image metadata and is used when loading that metadata back into the UI.
# This logic is now correct as it iterates over a list of enums.
CREATIVE_PARAM_KEYS = [UI_TO_WORKER_PARAM_MAP[key] for key in CREATIVE_UI_KEYS]

# ADDED: Centralized constant for the queue state JSON filename inside the zip.
QUEUE_STATE_JSON_IN_ZIP = "queue_state.json"
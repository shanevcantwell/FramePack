# ui/shared_state.py
import threading
from .enums import ComponentKey as K

class SharedState:
    _instance = None
    _initialized = False

    def __new__(cls):
        """Ensures only one instance of SharedState exists (singleton pattern)."""
        if cls._instance is None:
            cls._instance = super(SharedState, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initializes the SharedState attributes only once."""
        if self._initialized:
            return

        # --- Application-wide Threading Controls ---
        # Lock for thread-safe queue modifications.
        self.queue_lock: threading.Lock = threading.Lock()

# Event to signal the abortion of the current processing task.
# This is kept for compatibility and as a simple, overarching abort flag.
        self.interrupt_flag: threading.Event = threading.Event()

# Event to signal that a stop has been requested, for immediate UI feedback.
        self.stop_requested_flag: threading.Event = threading.Event()

# Event to signal that a preview should be generated for the current segment.
        self.preview_request_flag: threading.Event = threading.Event()

# State dictionary for the multi-level abort feature.
# 'level' 0: No abort.
# 'level' 2: Hard abort (triggered by the Stop Processing button).
        self.abort_state: dict = {'level': 0, 'last_click_time': 0}

# --- Model and Global State Containers ---
# This dictionary will be populated at runtime by the main script after the models are loaded.
        self.models: dict = {}
# This dictionary holds the application state, which is passed to the atexit
# handler to enable the autosave functionality on browser close or exit.
        self.global_state_for_autosave: dict = {}
# Dictionary to hold system-level information detected at startup.
        self.system_info: dict = {'is_legacy_gpu': False}

        self._initialized = True

# Instantiate the singleton instance. Other modules will import 'shared_state_instance'.
shared_state_instance = SharedState()


# --- UI and Parameter Mapping Constants ---
# Centralized list of UI component keys for LoRA management.
LORA_UI_KEYS = [K.LORA_UPLOAD_BUTTON_UI, K.LORA_ROW_0]

CREATIVE_UI_KEYS = [
    K.PROMPT_UI, K.N_PROMPT_UI, K.TOTAL_SECOND_LENGTH_UI, K.SEED_UI, K.PREVIEW_FREQUENCY_UI,
    K.SEGMENTS_TO_DECODE_CSV_UI, K.FPS_UI, K.GS_UI, K.GS_SCHEDULE_SHAPE_UI, K.GS_FINAL_UI,
    K.ROLL_OFF_START_UI, K.ROLL_OFF_FACTOR_UI,
    K.STEPS_UI, K.CFG_UI, K.RS_UI
]
ENVIRONMENT_UI_KEYS = [
    K.USE_TEACACHE_UI, K.USE_FP32_TRANSFORMER_OUTPUT_CHECKBOX_UI, K.GPU_MEMORY_PRESERVATION_UI,
    K.MP4_CRF_UI, K.OUTPUT_FOLDER_UI_CTRL, K.LATENT_WINDOW_SIZE_UI
]
ALL_TASK_UI_KEYS = CREATIVE_UI_KEYS + ENVIRONMENT_UI_KEYS

# This is the single source of truth for converting UI component names to worker parameter names.
UI_TO_WORKER_PARAM_MAP = {
    K.PROMPT_UI: 'prompt',
    K.N_PROMPT_UI: 'n_prompt',
    K.TOTAL_SECOND_LENGTH_UI: 'total_second_length',
    K.SEED_UI: 'seed',
    K.PREVIEW_FREQUENCY_UI: 'preview_frequency',
    K.SEGMENTS_TO_DECODE_CSV_UI: 'segments_to_decode_csv',
    K.FPS_UI: 'fps',
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
CREATIVE_PARAM_KEYS = [UI_TO_WORKER_PARAM_MAP[key] for key in CREATIVE_UI_KEYS]

# Centralized constant for the queue state JSON filename inside the zip.
QUEUE_STATE_JSON_IN_ZIP = "queue_state.json"
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

# Event to signal that the current task should be paused and its state saved.
        self.pause_request_flag: threading.Event = threading.Event()

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
# (Update or remove as needed based on your new enums.py)
LORA_UI_KEYS = [K.LORA_UPLOAD_BUTTON, K.LORA_ROW]

CREATIVE_UI_KEYS = [
    K.POSITIVE_PROMPT, K.NEGATIVE_PROMPT, K.VIDEO_LENGTH_SLIDER, K.SEED, K.PREVIEW_FREQUENCY_SLIDER,
    K.PREVIEW_SPECIFIED_SEGMENTS_TEXTBOX, K.FPS_SLIDER, K.DISTILLED_CFG_START_SLIDER, K.VARIABLE_CFG_SHAPE_RADIO,
    K.DISTILLED_CFG_END_SLIDER, K.ROLL_OFF_START_SLIDER, K.ROLL_OFF_FACTOR_SLIDER,
    K.STEPS_SLIDER, K.REAL_CFG_SLIDER, K.GUIDANCE_RESCALE_SLIDER
]
ENVIRONMENT_UI_KEYS = [
    K.USE_TEACACHE_CHECKBOX, K.USE_FP32_TRANSFORMER_OUTPUT_CHECKBOX, K.GPU_MEMORY_PRESERVATION_SLIDER,
    K.MP4_CRF_SLIDER, K.OUTPUT_FOLDER_TEXTBOX, K.LATENT_WINDOW_SIZE_SLIDER
]
ALL_TASK_UI_KEYS = CREATIVE_UI_KEYS + ENVIRONMENT_UI_KEYS

# This is the single source of truth for converting UI component names to worker parameter names.
UI_TO_WORKER_PARAM_MAP = {
    K.POSITIVE_PROMPT: 'prompt',
    K.NEGATIVE_PROMPT: 'negative_prompt',
    K.VIDEO_LENGTH_SLIDER: 'video_length',
    K.SEED: 'seed',
    K.PREVIEW_FREQUENCY_SLIDER: 'preview_frequency',
    K.PREVIEW_SPECIFIED_SEGMENTS_TEXTBOX: 'preview_specified_segments',
    K.FPS_SLIDER: 'fps',
    K.DISTILLED_CFG_START_SLIDER: 'distilled_cfg_start',
    K.VARIABLE_CFG_SHAPE_RADIO: 'variable_cfg_shape',
    K.DISTILLED_CFG_END_SLIDER: 'distilled_cfg_end',
    K.ROLL_OFF_START_SLIDER: 'roll_off_start',
    K.ROLL_OFF_FACTOR_SLIDER: 'roll_off_factor',
    K.STEPS_SLIDER: 'steps',
    K.REAL_CFG_SLIDER: 'real_cfg',
    K.GUIDANCE_RESCALE_SLIDER: 'guidance_rescale',
    K.USE_TEACACHE_CHECKBOX: 'use_teacache',
    K.USE_FP32_TRANSFORMER_OUTPUT_CHECKBOX: 'use_fp32_transformer_output',
    K.GPU_MEMORY_PRESERVATION_SLIDER: 'gpu_memory_preservation',
    K.MP4_CRF_SLIDER: 'mp4_crf',
    K.OUTPUT_FOLDER_TEXTBOX: 'output_folder',
    K.LATENT_WINDOW_SIZE_SLIDER: 'latent_window_size'
}

# The CREATIVE_PARAM_KEYS list defines the canonical names for parameters that are saved
# within image metadata and is used when loading that metadata back into the UI.
CREATIVE_PARAM_KEYS = [UI_TO_WORKER_PARAM_MAP[key] for key in CREATIVE_UI_KEYS]

# Centralized constant for the queue state JSON filename inside the zip.
QUEUE_STATE_JSON_IN_ZIP = "queue_state.json"
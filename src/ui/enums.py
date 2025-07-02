# ui/enums.py
from enum import StrEnum, auto

class ComponentKey(StrEnum):
    """
    Enumeration of all Gradio component keys.
    Using StrEnum means the value of each member is its own name,
    making it easy to debug and use while providing type-safety.
    """
    # High-level Components
    BLOCK = auto()
    APP_STATE = auto()

    # --- State Components ---
    LORA_NAME_STATE = auto()
    EXTRACTED_METADATA_STATE = auto()
    METADATA_MODAL_TRIGGER_STATE = auto()

    # --- Main UI Columns & Controls ---
    IMAGE_FILE_INPUT = auto()
    INPUT_IMAGE_DISPLAY = auto()
    ADD_TASK_BUTTON = auto()
    CLEAR_IMAGE_BUTTON = auto()
    DOWNLOAD_IMAGE_BUTTON = auto()
    PROCESS_QUEUE_BUTTON = auto()
    CREATE_PREVIEW_BUTTON = auto()
    CANCEL_EDIT_TASK_BUTTON = auto()
    POSITIVE_PROMPT = auto()
    NEGATIVE_PROMPT = auto()
    VIDEO_LENGTH_SLIDER = auto()
    SEED = auto()
    IMAGE_DOWNLOADER = auto()
    QUEUE_DOWNLOADER = auto()

    # --- Metadata Modal ---
    METADATA_MODAL = auto()
    METADATA_PROMPT_PREVIEW = auto()
    CANCEL_METADATA_BUTTON = auto()
    CONFIRM_METADATA_BUTTON = auto()

    # --- Task Queue ---
    QUEUE_DF = auto()
    SAVE_QUEUE_BUTTON = auto()
    LOAD_QUEUE_BUTTON = auto()
    CLEAR_QUEUE_BUTTON = auto()

    # --- Live Preview & Output ---
    CURRENT_TASK_PREVIEW_IMAGE = auto()
    CURRENT_TASK_PROGRESS_DESCRIPTION = auto()
    CURRENT_TASK_PROGRESS_BAR = auto()
    LAST_FINISHED_VIDEO = auto()

    # --- Accordions & Advanced Settings ---
    TOTAL_SEGMENTS_DISPLAY = auto()
    PREVIEW_FREQUENCY_SLIDER = auto()
    PREVIEW_SPECIFIED_SEGMENTS_TEXTBOX = auto()
    FPS_SLIDER = auto()

    # Renamed from Hunyuan abbreviations (GS, CFG, RS) for clarity.
    DISTILLED_CFG_START_SLIDER = auto()         # GS
    VARIABLE_CFG_SHAPE_RADIO = auto()           # GS_SCHEDULE_SHAPE
    DISTILLED_CFG_END_SLIDER = auto()           # GS_FINAL
    REAL_CFG_SLIDER = auto()                    # CFG
    GUIDANCE_RESCALE_SLIDER = auto()            # RS

    ROLL_OFF_START_SLIDER = auto()
    ROLL_OFF_FACTOR_SLIDER = auto()
    STEPS_SLIDER = auto()

    # --- LoRA Settings ---
    # The singular names here are placeholders for a future system that will support multiple LoRAs dynamically.
    LORA_ACCORDION = auto()
    LORA_UPLOAD_BUTTON = auto()
    LORA_ROW = auto()
    LORA_NAME = auto()
    LORA_WEIGHT = auto()
    LORA_TARGETS = auto()

    # --- Debug Settings & Workspace ---
    USE_TEACACHE_CHECKBOX = auto()
    USE_FP32_TRANSFORMER_OUTPUT_CHECKBOX = auto()
    FORCE_STANDARD_FPS_CHECKBOX = auto()
    GPU_MEMORY_PRESERVATION_SLIDER = auto()
    MP4_CRF_SLIDER = auto()
    LATENT_WINDOW_SIZE_SLIDER = auto()
    OUTPUT_FOLDER_TEXTBOX = auto()
    SAVE_AS_DEFAULT_BUTTON = auto()
    RELAUNCH_NOTIFICATION_MD = auto()

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
    MODAL_TRIGGER_BOX = auto()

    # --- Main UI Columns & Controls ---
    IMAGE_FILE_INPUT_UI = auto()
    INPUT_IMAGE_DISPLAY_UI = auto()
    ADD_TASK_BUTTON = auto()
    CLEAR_IMAGE_BUTTON_UI = auto()
    DOWNLOAD_IMAGE_BUTTON_UI = auto()
    PROCESS_QUEUE_BUTTON = auto()
    ABORT_TASK_BUTTON = auto()
    CANCEL_EDIT_TASK_BUTTON = auto()
    PROMPT_UI = auto()
    N_PROMPT_UI = auto()
    TOTAL_SECOND_LENGTH_UI = auto()
    SEED_UI = auto()
    IMAGE_DOWNLOADER_UI = auto()

    # --- Metadata Modal ---
    METADATA_MODAL = auto()
    METADATA_PROMPT_PREVIEW_UI = auto()
    CANCEL_METADATA_BTN = auto()
    CONFIRM_METADATA_BTN = auto()

    # --- Task Queue ---
    QUEUE_DF_DISPLAY_UI = auto()
    SAVE_QUEUE_BUTTON_UI = auto()
    LOAD_QUEUE_BUTTON_UI = auto()
    CLEAR_QUEUE_BUTTON_UI = auto()

    # --- Live Preview & Output ---
    CURRENT_TASK_PREVIEW_IMAGE_UI = auto()
    CURRENT_TASK_PROGRESS_DESC_UI = auto()
    CURRENT_TASK_PROGRESS_BAR_UI = auto()
    LAST_FINISHED_VIDEO_UI = auto()

    # --- Accordions & Advanced Settings ---
    TOTAL_SEGMENTS_DISPLAY_UI = auto()
    PREVIEW_FREQUENCY_UI = auto()
    SEGMENTS_TO_DECODE_CSV_UI = auto()
    GS_UI = auto()
    GS_SCHEDULE_SHAPE_UI = auto()
    GS_FINAL_UI = auto()
    ROLL_OFF_START_UI = auto()
    ROLL_OFF_FACTOR_UI = auto()
    CFG_UI = auto()
    STEPS_UI = auto()
    RS_UI = auto()

    # --- LoRA Settings ---
    LORA_UPLOAD_BUTTON_UI = auto()
    LORA_ROW_0 = auto()
    LORA_NAME_0 = auto()
    LORA_WEIGHT_0 = auto()
    LORA_TARGETS_0 = auto()

    # --- Debug Settings & Workspace ---
    USE_TEACACHE_UI = auto()
    USE_FP32_TRANSFORMER_OUTPUT_CHECKBOX_UI = auto()
    GPU_MEMORY_PRESERVATION_UI = auto()
    MP4_CRF_UI = auto()
    LATENT_WINDOW_SIZE_UI = auto()
    OUTPUT_FOLDER_UI_CTRL = auto()
    SAVE_AS_DEFAULT_BUTTON = auto()
    RELAUNCH_NOTIFICATION_MD = auto()
    RELAUNCH_BUTTON = auto()
    RESET_UI_BUTTON = auto()
    WORKSPACE_DOWNLOADER_UI = auto()
    SAVE_WORKSPACE_BUTTON = auto()
    LOAD_WORKSPACE_BUTTON = auto()
    SHUTDOWN_BUTTON = auto()
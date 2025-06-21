# ui/enums.py
from enum import StrEnum, auto

class ComponentKey(StrEnum):
    """
    Enumeration of all Gradio component keys.
    Using StrEnum means the value of each member is its own name,
    e.g., ComponentKey.PROMPT_UI.value == 'PROMPT_UI'.
    This provides type-safe, autocompletable, and refactor-friendly keys.
    """
    # High-level
    BLOCK = auto()
    APP_STATE = auto()
    
    # States
    LORA_NAME_STATE = auto()
    EXTRACTED_METADATA_STATE = auto()
    MODAL_TRIGGER_BOX = auto()

    # Image & Metadata
    IMAGE_FILE_INPUT_UI = auto()
    INPUT_IMAGE_DISPLAY_UI = auto()
    ADD_TASK_BUTTON = auto()
    CLEAR_IMAGE_BUTTON_UI = auto()
    DOWNLOAD_IMAGE_BUTTON_UI = auto()
    PROCESS_QUEUE_BUTTON = auto()
    ABORT_TASK_BUTTON = auto()
    CANCEL_EDIT_TASK_BUTTON = auto()
    METADATA_MODAL = auto()
    METADATA_PROMPT_PREVIEW_UI = auto()
    CANCEL_METADATA_BTN = auto()
    CONFIRM_METADATA_BTN = auto()

    # Creative Params
    PROMPT_UI = auto()
    N_PROMPT_UI = auto()
    TOTAL_SECOND_LENGTH_UI = auto()
    SEED_UI = auto()

    # Queue
    QUEUE_DF_DISPLAY_UI = auto()
    SAVE_QUEUE_BUTTON_UI = auto()
    LOAD_QUEUE_BUTTON_UI = auto()
    CLEAR_QUEUE_BUTTON_UI = auto()
    
    # --- etc. Add ALL keys from your layout here ---
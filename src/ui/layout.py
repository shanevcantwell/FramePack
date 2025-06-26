# ui/layout.py
# This file defines the Gradio UI layout for the goan application.

import gradio as gr
from gradio_modal import Modal

# Import ComponentKey enum and workspace manager
from .enums import ComponentKey as K
from . import workspace as workspace_manager

def create_ui():
    """
    Creates the Gradio UI layout and returns a dictionary of all UI components.
    This separation of layout from logic makes the main script cleaner.
    """

    css = """
    #queue_df { font-size: 0.85rem; }
    #queue_df th, #queue_df td { text-align: center; }
    .gradio-container { max-width: 95% !important; margin: auto !important; }
    :root {
        --color-accent-soft: #4CAF50; --color-accent-50: #e8f5e9;
        --color-accent-100: #c8e6c9; --color-accent-200: #a5d6a7;
        --color-accent-300: #81c784; --color-accent-400: #66bb6a;
        --color-accent-500: #4CAF50; --color-accent-600: #43A047;
        --color-accent-700: #388E3C; --color-accent-800: #2E7D32;
        --color-accent-900: #1B5E20;
    }
    .gr-button-primary { background-color: var(--color-accent-500) !important; color: white !important; }
    .gr-button-primary:hover { background-color: var(--color-accent-600) !important; }

    /* Custom blue color for specific action buttons */
    #clear_image_button:not([disabled]), #download_image_button:not([disabled]) {
        background-color: #2563eb !important; /* A standard blue */
        color: white !important;
        border-color: #2563eb !important;
    }
    #clear_image_button:not([disabled]):hover, #download_image_button:not([disabled]):hover {
        background-color: #1d4ed8 !important; /* A darker blue for hover */
    }

    /* Highlight the Image Input Box on Load */
    #image_file_input_ui {
        background-color: var(--color-accent-800); /* A dark green fill */
        border-radius: 5px;
        border: 2px dashed var(--color-accent-200); /* A light green dashed border */
    }
    #image_file_input_ui .text-gray-500 { /* This targets the default text color class */
        color: var(--color-accent-100) !important;
    }

    /* --- NEW, MORE ROBUST FIX for fullscreen images --- */
    /* This targets any image inside a fixed-position container, which is
       how Gradio implements the fullscreen view. Using a space instead of '>'
       makes it work even if the image is nested inside other divs. */
    div.fixed img {
        width: auto !important;
        height: auto !important;
        max-width: 95vw !important;  /* Max width is 95% of the viewport width */
        max-height: 95vh !important; /* Max height is 95% of the viewport height */
        object-fit: contain !important; /* This preserves the aspect ratio */
    }
    """

    components = {}
    components[K.BLOCK] = gr.Blocks(css=css, title="goan").queue()

    with components[K.BLOCK]:
        components[K.APP_STATE] = gr.State({
            "queue_state": {"queue": [], "next_id": 1, "processing": False, "editing_task_id": None},
            "last_completed_video_path": None,
            "lora_state": {"loaded_loras": {}}
        })
        components[K.LORA_NAME_STATE] = gr.Textbox(visible=False, label="LoRA Names State")
        components[K.EXTRACTED_METADATA_STATE] = gr.State({})
        components[K.MODAL_TRIGGER_BOX] = gr.Textbox(visible=False)

        gr.Markdown('# goan (Powered by FramePack)')

        with Modal(visible=False) as metadata_modal:
            components[K.METADATA_MODAL] = metadata_modal
            gr.Markdown("Image has saved parameters. Overwrite current creative settings?")
            components[K.METADATA_PROMPT_PREVIEW_UI] = gr.Textbox(label="Detected Prompt", interactive=False, lines=5, max_lines=10)
            with gr.Row():
                components[K.CANCEL_METADATA_BTN] = gr.Button("No")
                components[K.CONFIRM_METADATA_BTN] = gr.Button("Yes, Apply", variant="primary")

        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                components[K.IMAGE_FILE_INPUT_UI] = gr.File(label="Drop Image Here or Click to Upload", file_types=["image"], elem_id="image_file_input_ui")
                components[K.INPUT_IMAGE_DISPLAY_UI] = gr.Image(type="pil", label="Current Input Image", interactive=False, visible=False, height=220, show_download_button=False)
                with gr.Row():
                    components[K.ADD_TASK_BUTTON] = gr.Button("Add to Queue", variant="secondary", interactive=False)
                    components[K.CANCEL_EDIT_TASK_BUTTON] = gr.Button("Cancel Edit", visible=False, variant="secondary")
                with gr.Row():
                    components[K.CLEAR_IMAGE_BUTTON_UI] = gr.Button("Clear Image", variant="secondary", interactive=False, elem_id="clear_image_button")
                    # Changed from DownloadButton to Button to enable one-click download via JS.
                    components[K.DOWNLOAD_IMAGE_BUTTON_UI] = gr.Button("Download Image", variant="secondary", interactive=False, elem_id="download_image_button")
                components[K.PROCESS_QUEUE_BUTTON] = gr.Button("▶️ Process Queue", variant="primary", interactive=False)
                components[K.CREATE_PREVIEW_BUTTON] = gr.Button("📸 Create Preview", variant="secondary", interactive=False, elem_id="create_preview_button")

            with gr.Column(scale=2, min_width=600):
                components[K.PROMPT_UI] = gr.Textbox(label="Prompt", lines=10)
                components[K.N_PROMPT_UI] = gr.Textbox(label="Negative Prompt", lines=4)
                with gr.Row():
                    components[K.TOTAL_SECOND_LENGTH_UI] = gr.Slider(label="Video Length (s)", minimum=0.1, maximum=120, value=5.0, step=0.1)
                    # Corrected and single definition for SEED_UI
                    components[K.SEED_UI] = gr.Number(label="Seed", value=-1, precision=0, minimum=-1, maximum=2**32 - 1)

        with gr.Group():
            # These hidden file components are the targets for one-click downloads.
            components[K.IMAGE_DOWNLOADER_UI] = gr.File(visible=False, elem_id="image_downloader_hidden_file")
            components[K.QUEUE_DOWNLOADER_UI] = gr.File(visible=False, elem_id="queue_downloader_hidden_file")
            gr.Markdown("## Task Queue") # Removed "Steps" header, reordered action buttons
            components[K.QUEUE_DF_DISPLAY_UI] = gr.DataFrame(headers=["ID", "Status", "✖", "✎", "Prompt", "Length", "Input", "↑", "↓"], datatype=["number","markdown","markdown","markdown","markdown","str","markdown","markdown","markdown"], col_count=(9,"fixed"), interactive=False, elem_id="queue_df")
            with gr.Row():
                # Changed from DownloadButton to Button to enable one-click download via JS.
                components[K.SAVE_QUEUE_BUTTON_UI] = gr.Button("Save Queue", size="sm", interactive=False)
                components[K.LOAD_QUEUE_BUTTON_UI] = gr.UploadButton("Load Queue", file_types=[".zip"], size="sm", variant="primary")
                components[K.CLEAR_QUEUE_BUTTON_UI] = gr.Button("Clear Pending", size="sm", variant="stop", interactive=False)

        components[K.CURRENT_TASK_PREVIEW_IMAGE_UI] = gr.Image(label="Live Latent Preview", interactive=False, visible=False, show_download_button=False)

        with gr.Row(equal_height=False):
            with gr.Column(scale=1):
                with gr.Accordion("Advanced Settings", open=False):
                    components[K.TOTAL_SEGMENTS_DISPLAY_UI] = gr.Markdown("Calculated Total Segments: N/A")
                    components[K.PREVIEW_FREQUENCY_UI] = gr.Slider(label="Preview Freq.", minimum=0, maximum=100, value=5, step=1)
                    components[K.SEGMENTS_TO_DECODE_CSV_UI] = gr.Textbox(label="Preview Segments CSV", value="")
                    with gr.Row():
                        components[K.GS_UI] = gr.Slider(label="Distilled CFG Start", minimum=1.0, maximum=32.0, value=10.0, step=0.01)
                        components[K.GS_SCHEDULE_SHAPE_UI] = gr.Radio(["Off", "Linear", "Roll-off"], label="Variable CFG", value="Off")
                        components[K.GS_FINAL_UI] = gr.Slider(label="Distilled CFG End", minimum=1.0, maximum=32.0, value=10.0, step=0.01, interactive=False)
                    components[K.ROLL_OFF_START_UI] = gr.Slider(label="Roll-off Start %", minimum=0, maximum=100, value=75, step=1, visible=False)
                    components[K.ROLL_OFF_FACTOR_UI] = gr.Slider(label="Roll-off Curve Factor", minimum=0.25, maximum=4.0, value=1.0, step=0.05, visible=False)
                    components[K.CFG_UI] = gr.Slider(label="CFG (Real)", minimum=1.0, maximum=8.0, value=1.5, step=0.01)
                    components[K.STEPS_UI] = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1)
                    components[K.RS_UI] = gr.Slider(label="RS", minimum=0.0, maximum=32.0, value=0.0, step=0.01, visible=False)

                with gr.Accordion("LoRA Settings", open=False, visible=True) as lora_accordion:
                    components[K.LORA_ACCORDION] = lora_accordion
                    gr.Markdown("🧪 Experimental LoRA support. Upload a `.safetensors` file. Applied before generation.")
                    components[K.LORA_UPLOAD_BUTTON_UI] = gr.UploadButton("Upload LoRA", file_types=[".safetensors"], file_count="single", size="sm")
                    with gr.Row(visible=False, variant="panel") as lora_row_0_ctx:
                        components[K.LORA_ROW_0] = lora_row_0_ctx
                        components[K.LORA_NAME_0] = gr.Textbox(label="LoRA Name", interactive=False, scale=2)
                        components[K.LORA_WEIGHT_0] = gr.Slider(label="Weight", minimum=-2.0, maximum=2.0, step=0.05, value=1.0, scale=3)
                        components[K.LORA_TARGETS_0] = gr.CheckboxGroup(label="Target Models", choices=["transformer", "text_encoder", "text_encoder_2"], value=["text_encoder"], scale=3)

                with gr.Accordion("Debug Settings", open=False):
                    components[K.USE_TEACACHE_UI] = gr.Checkbox(label='Use TeaCache', value=True)
                    components[K.USE_FP32_TRANSFORMER_OUTPUT_CHECKBOX_UI] = gr.Checkbox(label="Use FP32 Transformer Output", value=False)
                    components[K.GPU_MEMORY_PRESERVATION_UI] = gr.Slider(label="GPU Preserved (GB)", minimum=4, maximum=128, value=6.0, step=0.1)
                    components[K.FPS_UI] = gr.Slider(label="MP4 Framerate (FPS)", minimum=1, maximum=60, value=30, step=1)
                    components[K.MP4_CRF_UI] = gr.Slider(label="MP4 CRF", minimum=0, maximum=51, value=18, step=1)
                    components[K.LATENT_WINDOW_SIZE_UI] = gr.Slider(label="Latent Window Size", minimum=1, maximum=33, value=9, step=1, visible=False)
                    components[K.OUTPUT_FOLDER_UI_CTRL] = gr.Textbox(label="Output Folder", value=workspace_manager.outputs_folder)
                    components[K.SAVE_AS_DEFAULT_BUTTON] = gr.Button("Save as Default", variant="secondary")
                    components[K.RELAUNCH_NOTIFICATION_MD] = gr.Markdown("ℹ️ **Restart required** for new output path to take effect.", visible=False)
                    components[K.RELAUNCH_BUTTON] = gr.Button("Save Current State & Relaunch", variant="primary", visible=False)
                    components[K.RESET_UI_BUTTON] = gr.Button("Save & Refresh UI", variant="secondary")

                with gr.Row():
                    # These buttons are being hidden for now, pending future functionality.
                    components[K.WORKSPACE_DOWNLOADER_UI] = gr.File(visible=False, file_count="single", elem_id="workspace_downloader_hidden_file",)
                    components[K.SAVE_WORKSPACE_BUTTON] = gr.Button("Save Workspace", variant="secondary", visible=False)
                    components[K.LOAD_WORKSPACE_BUTTON] = gr.UploadButton("Load Workspace", file_types=[".json"], visible=False)
                # This button is being hidden for now, pending future functionality.
                components[K.RESET_UI_BUTTON] = gr.Button("Save & Refresh UI", variant="secondary", visible=False)
                components[K.SHUTDOWN_BUTTON] = gr.Button("Save All & Exit", variant="stop", visible=False)

            with gr.Column(scale=1):
                gr.Markdown("## Live Preview & Output")
                components[K.CURRENT_TASK_PROGRESS_DESC_UI] = gr.Markdown('')
                components[K.CURRENT_TASK_PROGRESS_BAR_UI] = gr.HTML('')
                components[K.LAST_FINISHED_VIDEO_UI] = gr.Video(interactive=True, autoplay=False, height=540)

    return components
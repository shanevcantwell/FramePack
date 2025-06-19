# ui/layout.py
# This file defines the Gradio UI layout for the goan application.

import gradio as gr
from gradio_modal import Modal

# Import workspace manager to get the default output folder path
from . import workspace as workspace_manager

def create_ui():
    """
    Creates the Gradio UI layout and returns a dictionary of all UI components.
    This separation of layout from logic makes the main script cleaner.
    """
    # Define a dictionary to hold all UI components, which will be returned.
    components = {}

    # Applying the green primary button color directly in CSS
    css = """
    #queue_df { font-size: 0.85rem; }
    #queue_df th, #queue_df td { text-align: center; }
    .gradio-container { max-width: 95% !important; margin: auto !important; }

    /* NEW CSS: Force primary buttons to green */
    :root {
        --color-accent-soft: #4CAF50; /* A shade of green */
        --color-accent-50: #e8f5e9; /* Light green for hover/active backgrounds */
        --color-accent-100: #c8e6c9;
        --color-accent-200: #a5d6a7;
        --color-accent-300: #81c784;
        --color-accent-400: #66bb6a;
        --color-accent-500: #4CAF50; /* Your primary green */
        --color-accent-600: #43A047;
        --color-accent-700: #388E3C;
        --color-accent-800: #2E7D32;
        --color-accent-900: #1B5E20;
    }
    .gr-button-primary {
        background-color: var(--color-accent-500) !important;
        color: white !important; /* Ensure text is readable */
    }
    .gr-button-primary:hover {
        background-color: var(--color-accent-600) !important;
    }
    """

    block = gr.Blocks(css=css, title="goan").queue()
    components['block'] = block

    with block:
        # MODIFIED: Added lora_state to the initial application state
        app_state = gr.State({
            "queue_state": {"queue": [], "next_id": 1, "processing": False, "editing_task_id": None},
            "last_completed_video_path": None,
            "lora_state": {"loaded_loras": {}} # Format: { lora_name: { path, weight } }
        })
        components['app_state'] = app_state

        components['lora_name_state'] = gr.Textbox(visible=False, label="LoRA Names State")

        extracted_metadata_state = gr.State({})
        components['extracted_metadata_state'] = extracted_metadata_state

        modal_trigger_box = gr.Textbox(visible=False)
        components['modal_trigger_box'] = modal_trigger_box

        gr.Markdown('# goan (Powered by FramePack)')

        with Modal(visible=False) as metadata_modal:
            components['metadata_modal'] = metadata_modal
            gr.Markdown("Image has saved parameters. Overwrite current creative settings?")
            components['metadata_prompt_preview_ui'] = gr.Textbox(label="Detected Prompt", interactive=False, lines=5, max_lines=10)
            with gr.Row():
                components['cancel_metadata_btn'] = gr.Button("No")
                components['confirm_metadata_btn'] = gr.Button("Yes, Apply", variant="primary")

        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                # This File input is the primary, always-visible drop zone when empty.
                components['image_file_input_ui'] = gr.File(label="Drop Image Here or Click to Upload", file_types=["image"])
                # This Image display is hidden by default and has a fixed height.
                components['input_image_display_ui'] = gr.Image(type="pil", label="Current Input Image", interactive=False, visible=False, height=220, show_download_button=False)

                components['add_task_button'] = gr.Button("Add to Queue", variant="secondary")
                with gr.Row():
                    components['clear_image_button_ui'] = gr.Button("Clear Image", variant="secondary", visible=False)
                    components['download_image_button_ui'] = gr.DownloadButton("Download Image", variant="secondary", visible=False) # Changed to DownloadButton

                components['process_queue_button'] = gr.Button("▶️ Process Queue", variant="primary")
                components['abort_task_button'] = gr.Button("⏹️ Abort", variant="stop", interactive=True)
                components['cancel_edit_task_button'] = gr.Button("Cancel Edit", visible=False, variant="secondary")

            with gr.Column(scale=2, min_width=600):
                components['prompt_ui'] = gr.Textbox(label="Prompt", lines=10)
                components['n_prompt_ui'] = gr.Textbox(label="Negative Prompt", lines=4)
                # MOVED: Video Length and Seed are now here for better workflow.
                with gr.Row():
                    components['total_second_length_ui'] = gr.Slider(label="Video Length (s)", minimum=0.1, maximum=120, value=5.0, step=0.1)
                    components['seed_ui'] = gr.Number(label="Seed", value=-1, precision=0)

        with gr.Group():
            components['image_downloader_ui'] = gr.File(visible=False)
            gr.Markdown("## Task Queue")
            components['queue_df_display_ui'] = gr.DataFrame(headers=["ID", "Status", "Prompt", "Length", "Steps", "Input", "↑", "↓", "✖", "✎"], datatype=["number","markdown","markdown","str","number","markdown","markdown","markdown","markdown","markdown"], col_count=(10,"fixed"), interactive=False, elem_id="queue_df")
            with gr.Row():
                components['save_queue_button_ui'] = gr.DownloadButton("Save Queue", size="sm")
                components['load_queue_button_ui'] = gr.UploadButton("Load Queue", file_types=[".zip"], size="sm")
                components['clear_queue_button_ui'] = gr.Button("Clear Pending", size="sm", variant="stop")

        # NEW LOCATION: Latent preview image is now here, after the task queue group
        # This will make it full-width, scaled to maintain aspect ratio by default.
        components['current_task_preview_image_ui'] = gr.Image(
            label="Live Latent Preview", # Added a label for clarity in new position
            interactive=False,
            visible=False,
            show_download_button=False # Ensure it doesn't have its own download button
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=1):
                # REMOVED: Video Length and Seed were here.
                with gr.Accordion("Advanced Settings", open=False):
                    components['total_segments_display_ui'] = gr.Markdown("Calculated Total Segments: N/A")
                    components['preview_frequency_ui'] = gr.Slider(label="Preview Freq.", minimum=0, maximum=100, value=5, step=1)
                    components['segments_to_decode_csv_ui'] = gr.Textbox(label="Preview Segments CSV", value="")
                    with gr.Row():
                        components['gs_ui'] = gr.Slider(label="Distilled CFG Start", minimum=1.0, maximum=32.0, value=10.0, step=0.01)
                        components['gs_schedule_shape_ui'] = gr.Radio(["Off", "Linear"], label="Variable CFG", value="Off")
                        components['gs_final_ui'] = gr.Slider(label="Distilled CFG End", minimum=1.0, maximum=32.0, value=10.0, step=0.01, interactive=False)
                    components['cfg_ui'] = gr.Slider(label="CFG (Real)", minimum=1.0, maximum=32.0, value=1.0, step=0.01)
                    components['steps_ui'] = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1)
                    components['rs_ui'] = gr.Slider(label="RS", minimum=0.0, maximum=32.0, value=0.0, step=0.01, visible=False)
                
                # ADDED: LoRA Settings Accordion
                with gr.Accordion("LoRA Settings", open=True):
                    gr.Markdown("🧪 Experimental LoRA support. Upload `.safetensors` files. Applied before generation.")
                    components['lora_upload_button_ui'] = gr.UploadButton(
                        "Upload LoRA(s)",
                        file_types=[".safetensors"],
                        file_count="multiple",
                        size="sm"
                    ) 
                    
                # Pre-define 5 static, hidden slots for LoRA controls.
                # This creates the components that goan.py is looking for.
                for i in range(5):
                    with gr.Row(visible=False, variant="panel") as lora_row:
                        # Add each component to the dictionary with a unique key
                        components[f'lora_row_{i}'] = lora_row
                        components[f'lora_name_{i}'] = gr.Textbox(
                            label="LoRA Name", interactive=False, scale=2
                        )
                        components[f'lora_weight_{i}'] = gr.Slider(
                            label="Weight", minimum=-2.0, maximum=2.0, step=0.05, value=1.0, scale=3
                        )
                        components[f'lora_targets_{i}'] = gr.CheckboxGroup(
                            label="Target Models", choices=["transformer", "text_encoder", "text_encoder_2"],
                            value=["transformer"], scale=3
                        )


                with gr.Accordion("Debug Settings", open=False):
                    components['use_teacache_ui'] = gr.Checkbox(label='Use TeaCache', value=True)
                    components['use_fp32_transformer_output_checkbox_ui'] = gr.Checkbox(label="Use FP32 Transformer Output", value=False)
                    components['gpu_memory_preservation_ui'] = gr.Slider(label="GPU Preserved (GB)", minimum=4, maximum=128, value=6.0, step=0.1)
                    components['mp4_crf_ui'] = gr.Slider(label="MP4 CRF", minimum=0, maximum=51, value=18, step=1)
                    components['latent_window_size_ui'] = gr.Slider(label="Latent Window Size", minimum=1, maximum=33, value=9, step=1, visible=False)
                    components['output_folder_ui_ctrl'] = gr.Textbox(label="Output Folder", value=workspace_manager.outputs_folder)
                    components['save_as_default_button'] = gr.Button("Save as Default", variant="secondary")
                    components['relaunch_notification_md'] = gr.Markdown("ℹ️ **Restart required** for new output path to take effect.", visible=False)
                    components['relaunch_button'] = gr.Button("Save Current State & Relaunch", variant="primary", visible=False)
                    components['reset_ui_button'] = gr.Button("Save & Refresh UI", variant="secondary")

                with gr.Row():
                    components['workspace_downloader_ui'] = gr.File(visible=False, file_count="single", elem_id="workspace_downloader_hidden_file") # For save
                    components['save_workspace_button'] = gr.Button("Save Workspace", variant="secondary")
                    components['load_workspace_button'] = gr.UploadButton("Load Workspace", file_types=[".json"]) # For load
                components['shutdown_button'] = gr.Button("Save All & Exit", variant="stop")

            with gr.Column(scale=1): # This column now contains only video, progress desc, and progress bar
                gr.Markdown("## Live Preview & Output")
                # components['current_task_preview_image_ui'] was moved from here
                components['current_task_progress_desc_ui'] = gr.Markdown('')
                components['current_task_progress_bar_ui'] = gr.HTML('')
                components['last_finished_video_ui'] = gr.Video(interactive=True, autoplay=False, height=540)

    return components
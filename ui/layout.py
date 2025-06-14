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

    css = """
    #queue_df { font-size: 0.85rem; } 
    #queue_df th:nth-child(1), #queue_df td:nth-child(1) { width: 5%; } 
    #queue_df th:nth-child(2), #queue_df td:nth-child(2) { width: 10%; } 
    #queue_df th:nth-child(3), #queue_df td:nth-child(3) { width: 40%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;} 
    #queue_df th:nth-child(4), #queue_df td:nth-child(4) { width: 8%; } 
    #queue_df th:nth-child(5), #queue_df td:nth-child(5) { width: 8%; } 
    #queue_df th:nth-child(6), #queue_df td:nth-child(6) { width: 10%; text-align: center; } 
    #queue_df th:nth-child(7), #queue_df td:nth-child(7), 
    #queue_df th:nth-child(8), #queue_df td:nth-child(8), 
    #queue_df th:nth-child(9), #queue_df td:nth-child(9), 
    #queue_df th:nth-child(10), #queue_df td:nth-child(10) { width: 4%; cursor: pointer; text-align: center; } 
    #queue_df td:hover { background-color: #f0f0f0; } 
    .gradio-container { max-width: 95% !important; margin: auto !important; }
    """
    
    block = gr.Blocks(css=css, title="goan").queue()
    components['block'] = block

    with block:
        # The app_state dictionary holds all transient state for the UI
        app_state = gr.State({
            "queue_state": {"queue": [], "next_id": 1, "processing": False, "editing_task_id": None},
            "last_completed_video_path": None
        })
        components['app_state'] = app_state

        gr.Markdown('# goan (Powered by FramePack)')

        with Modal(visible=False) as metadata_modal:
            components['metadata_modal'] = metadata_modal
            gr.Markdown("Image has saved parameters. Overwrite current creative settings?")
            with gr.Row():
                components['cancel_metadata_btn'] = gr.Button("No")
                components['confirm_metadata_btn'] = gr.Button("Yes, Apply", variant="primary")

        # --- Start of layout structure ---

        # TOP ZONE: A row with two columns.
        # Column 1 (scale=1): Image input and primary queue actions.
        # Column 2 (scale=2): Prompting.
        with gr.Row():
            # Column 1: Input Image & Primary Actions
            with gr.Column(scale=1, min_width=300):
                components['input_image_gallery_ui'] = gr.Gallery(type="pil", label="Input Image(s)", height=220)
                components['add_task_button'] = gr.Button("Add to Queue", variant="secondary")
                components['process_queue_button'] = gr.Button("▶️ Process Queue", variant="primary")
                components['abort_task_button'] = gr.Button("⏹️ Abort", variant="stop", interactive=False)
                components['cancel_edit_task_button'] = gr.Button("Cancel Edit", visible=False, variant="secondary")

            # Column 2: Prompts
            with gr.Column(scale=2, min_width=600):
                components['prompt_ui'] = gr.Textbox(label="Prompt", lines=10)
                components['n_prompt_ui'] = gr.Textbox(label="Negative Prompt", lines=4)

        # MIDDLE ZONE: Full-width task queue display and file operations
        with gr.Group():
            gr.Markdown("## Task Queue")
            components['queue_df_display_ui'] = gr.DataFrame(headers=["ID", "Status", "Prompt", "Length", "Steps", "Input", "↑", "↓", "✖", "✎"], datatype=["number","markdown","markdown","str","number","markdown","markdown","markdown","markdown","markdown"], col_count=(10,"fixed"), interactive=False, visible=False, elem_id="queue_df")
            with gr.Row():
                components['save_queue_zip_b64_output'] = gr.Text(visible=False)
                components['save_queue_button_ui'] = gr.DownloadButton("Save Queue", size="sm")
                components['load_queue_button_ui'] = gr.UploadButton("Load Queue", file_types=[".zip"], size="sm")
                components['clear_queue_button_ui'] = gr.Button("Clear Pending", size="sm", variant="stop")

        # BOTTOM ZONE: A row split into two columns for Settings and Output
        with gr.Row(equal_height=False):
            # Column 1: All settings and parameters
            with gr.Column(scale=1):
                with gr.Row():
                    components['total_second_length_ui'] = gr.Slider(label="Video Length (s)", minimum=0.1, maximum=120, value=5.0, step=0.1)
                    components['seed_ui'] = gr.Number(label="Seed", value=-1, precision=0)
                with gr.Accordion("Advanced Settings", open=True):
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
                with gr.Accordion("Debug Settings", open=True):
                    components['use_teacache_ui'] = gr.Checkbox(label='Use TeaCache', value=True)
                    components['use_fp32_transformer_output_checkbox_ui'] = gr.Checkbox(label="Use FP32 Transformer Output", value=False)
                    components['gpu_memory_preservation_ui'] = gr.Slider(label="GPU Preserved (GB)", minimum=4, maximum=128, value=6.0, step=0.1)
                    components['mp4_crf_ui'] = gr.Slider(label="MP4 CRF", minimum=0, maximum=51, value=18, step=1)
                    components['latent_window_size_ui'] = gr.Slider(label="Latent Window Size", minimum=1, maximum=33, value=9, step=1, visible=False)
                    components['output_folder_ui_ctrl'] = gr.Textbox(label="Output Folder", value=workspace_manager.outputs_folder)
                
                # UI and Workspace Management Buttons
                with gr.Row():
                    components['save_as_default_button'] = gr.Button("Save as Default", variant="secondary")
                    components['reset_ui_button'] = gr.Button("Save & Refresh UI", variant="secondary")
                with gr.Row():
                    components['save_workspace_button'] = gr.Button("Save Workspace", variant="secondary")
                    components['load_workspace_button'] = gr.Button("Load Workspace", variant="secondary")

            # Column 2: Live Preview and Final Output
            with gr.Column(scale=1):
                gr.Markdown("## Live Preview & Output")
                components['current_task_preview_image_ui'] = gr.Image(interactive=False, visible=False)
                components['current_task_progress_desc_ui'] = gr.Markdown('')
                components['current_task_progress_bar_ui'] = gr.HTML('')
                # Set a height on the video player to help with layout stability
                components['last_finished_video_ui'] = gr.Video(interactive=True, autoplay=False, height=540)
                
    return components
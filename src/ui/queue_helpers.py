# ui/queue_helpers.py
# Contains helper functions for queue state management and UI data formatting.

import gradio as gr
import numpy as np
from PIL import Image
import base64
import io
import logging
import html

from .queue_manager import queue_manager_instance

PROMPT_DISPLAY_TRUNCATION_LENGTH = 77
logger = logging.getLogger(__name__)

def np_to_base64_uri(np_array_or_tuple, format="png"):
    """Converts a NumPy array to a base64 data URI for embedding in HTML/Markdown."""
    if np_array_or_tuple is None:
        return None
    try:
        # Handle cases where the input might be a raw array or a tuple from other components
        if isinstance(np_array_or_tuple, tuple) and len(np_array_or_tuple) > 0 and isinstance(np_array_or_tuple[0], np.ndarray):
            np_array = np_array_or_tuple[0]
        elif isinstance(np_array_or_tuple, np.ndarray):
            np_array = np_array_or_tuple
        else:
            return None

        pil_image = Image.fromarray(np_array.astype(np.uint8))
        if format.lower() == "jpeg" and pil_image.mode == "RGBA":
            pil_image = pil_image.convert("RGB")

        buffer = io.BytesIO()
        pil_image.save(buffer, format=format.upper())
        img_bytes = buffer.getvalue()
        return f"data:image/{format.lower()};base64,{base64.b64encode(img_bytes).decode('utf-8')}"
    except Exception as e:
        logger.error(f"Error converting NumPy to base64: {e}", exc_info=True)
        return None

def update_queue_df_display():
    """Formats the current queue state into a Gradio DataFrame update object for display."""
    queue_state = queue_manager_instance.get_state()
    queue = queue_state.get("queue", [])
    processing = queue_state.get("processing", False)
    editing_task_id = queue_state.get("editing_task_id")
    total_tasks = len(queue)
    data = []

    def _button_markdown(icon: str, enabled: bool) -> str:
        """Generates markdown for an action button, styled as enabled or disabled."""
        if enabled:
            return f"<a href='#' style='text-decoration: none; font-size: 1.2em;'>{icon}</a>"
        else:
            return f"<span style='color: #999; font-size: 1.2em; cursor: not-allowed;'>{icon}</span>"

    for i, task in enumerate(queue):
        params = task['params']
        task_id = task['id']
        status = task.get("status", "pending")

        is_processing_current_task = processing and i == 0
        is_pending = status == 'pending'

        up_enabled = is_pending and i > 0 and not is_processing_current_task
        down_enabled = is_pending and i < (total_tasks - 1) and not is_processing_current_task
        pause_enabled = is_processing_current_task
        edit_enabled = is_pending and not is_processing_current_task
        cancel_enabled = is_pending or is_processing_current_task

        # Generate the markdown for each button.
        up_arrow = _button_markdown('⬆️', up_enabled)
        down_arrow = _button_markdown('⬇️', down_enabled)
        pause_button = _button_markdown('⏸️', pause_enabled)
        edit_button = _button_markdown('✎', edit_enabled)
        cancel_button = _button_markdown('✖️', cancel_enabled)

        # Create a truncated prompt for display. The 80-character limit is arbitrary for UI neatness.
        prompt_display = (params['prompt'][:PROMPT_DISPLAY_TRUNCATION_LENGTH] + '...') if len(params['prompt']) > 80 else params['prompt']

        # --- CORRECTED ---
        # Use html.escape() for a robust and secure tooltip.
        prompt_title = html.escape(params['prompt'], quote=True)
        prompt_cell = f'<span title="{prompt_title}">{prompt_display}</span>'

        img_uri = np_to_base64_uri(params.get('input_image'), format="png")
        thumbnail_size = "50px"
        img_md = f'<img src="{img_uri}" alt="Input" style="max-width:{thumbnail_size}; max-height:{thumbnail_size}; display:block; margin:auto; object-fit:contain;" />' if img_uri else ""

        # Consolidated status display logic
        is_editing_this_task = editing_task_id == task_id

        status_display = ""
        if is_processing_current_task: status_display = "⏳ Processing"
        elif is_editing_this_task: status_display = "✏️ Editing"
        elif status == "done": status_display = "✅ Done"
        elif status == "error": status_display = f"❌ Error: {html.escape(task.get('error_message', 'Unknown'))}"
        elif status == "aborted": status_display = "⏹️ Aborted"
        else: status_display = "⏸️ Pending"

        data.append([
            up_arrow, down_arrow, pause_button, edit_button, cancel_button,
            status_display, prompt_cell, img_md, f"{params.get('video_length', 0):.1f}s", task_id
        ])

    return gr.update(value=data) if data else gr.update(value=[], headers=["↑", "↓", "⏸️", "✎", "✖", "Status", "Prompt", "Image", "Length", "ID"], datatype=["markdown", "markdown", "markdown", "markdown", "markdown", "markdown", "markdown", "markdown", "str", "number"], col_count=(10, "dynamic"))

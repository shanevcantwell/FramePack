# ui/queue_actions.py
# Handles all user-facing queue management logic and event handling for the UI.

import gradio as gr
import numpy as np
from PIL import Image
import os
import json
import io
import zipfile # Keep this for save
import tempfile # Keep this for save
import shutil
import logging
import queue # For queue.Empty exception

from .queue_manager import queue_manager_instance
from . import shared_state as shared_state_module
from .enums import ComponentKey as K
from . import workspace as workspace_manager
from . import queue_helpers
from .agents import ProcessingAgent

logger = logging.getLogger(__name__)

AUTOSAVE_FILENAME = "goan_autosave_queue.zip"

def add_or_update_task_in_queue(*args_from_ui_controls_tuple):
    """
    Adds a new task to the queue or updates an existing one if in edit mode.
    This function is now independent of the old gr.State object.
    """
    # The first argument is always the input image PIL object.
    input_image_pil = args_from_ui_controls_tuple[0]
    
    if not input_image_pil:
        gr.Warning("Input image is required!")
        # Return updates for the outputs defined in the switchboard for this event.
        # We only need to return gr.update() for each output to signify no change.
        return [gr.update()] * (len(shared_state_module.ALL_TASK_UI_KEYS) + 8)

    # The rest of the arguments are the UI control values.
    all_ui_values_tuple = args_from_ui_controls_tuple[1:]
    default_keys_map = workspace_manager.get_default_values_map()
    enum_keys = list(default_keys_map.keys())
    params_from_ui = dict(zip(enum_keys, all_ui_values_tuple))
    base_params_for_worker_dict = {
        worker_key: params_from_ui.get(ui_key) for ui_key, worker_key in shared_state_module.UI_TO_WORKER_PARAM_MAP.items()
    }
    img_np_data = np.array(input_image_pil) # type: ignore

    editing_task_id = queue_manager_instance.get_state().get("editing_task_id")
    if editing_task_id is not None:
        queue_manager_instance.update_task(editing_task_id, base_params_for_worker_dict, img_np_data)
        # After updating a task, exit edit mode, which correctly resets the UI.
        return cancel_edit_mode_action()
    else:
        queue_manager_instance.add_task(base_params_for_worker_dict, img_np_data)
        # When adding a new task, we only need to update the queue display.
        # All other UI components remain as they are.
        # The switchboard expects a specific number of outputs, so we provide gr.update() placeholders.
        num_outputs = len(shared_state_module.ALL_TASK_UI_KEYS) + 8
        updates = [gr.update()] * num_outputs
        updates[1] = queue_helpers.update_queue_df_display() # Index 1 is the queue dataframe
        return updates

def cancel_edit_mode_action(from_ui=True):
    """Resets the UI to its default state and exits edit mode."""
    queue_manager_instance.set_editing_task(None)
    default_values_map = workspace_manager.get_default_values_map()
    ui_updates = [gr.update(value=default_values_map.get(key)) for key in shared_state_module.ALL_TASK_UI_KEYS]
    
    # The number of outputs depends on where this function is called from.
    # The switchboard has different output lists for different events.
    if from_ui:
        # This is for the "Cancel Edit" button click
        num_outputs = len(shared_state_module.ALL_TASK_UI_KEYS) + 8
        updates = [gr.update()] * num_outputs
        updates[1] = queue_helpers.update_queue_df_display()
        updates[2] = gr.update(value=None, visible=False) # INPUT_IMAGE_DISPLAY_UI
        updates[3] = gr.update(visible=True, value=None) # IMAGE_FILE_INPUT_UI
        for i, key in enumerate(shared_state_module.ALL_TASK_UI_KEYS):
            updates[4 + i] = ui_updates[i]
        updates[-4] = gr.update(interactive=False, variant="secondary") # CLEAR_IMAGE_BUTTON_UI
        updates[-3] = gr.update(interactive=False, variant="secondary") # DOWNLOAD_IMAGE_BUTTON_UI
        updates[-2] = gr.update(value="Add Task to Queue", variant="secondary") # ADD_TASK_BUTTON
        updates[-1] = gr.update(visible=True) # CANCEL_EDIT_TASK_BUTTON
        return updates
    else:
        # This is for when called internally, like after updating a task.
        # It matches the output signature of the add/update task event.
        return cancel_edit_mode_action(from_ui=True)
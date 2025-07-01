# ui/switchboard_workspace.py
import gradio as gr
import logging

from .enums import ComponentKey as K
from . import workspace as workspace_manager
from . import event_handlers

logger = logging.getLogger(__name__)

def wire_events(components: dict):
    """Wires up the workspace save/load events."""
    logger.info("Wiring workspace events...")
    button_state_outputs = [
        components[K.ADD_TASK_BUTTON],
        components[K.PROCESS_QUEUE_BUTTON],
        components[K.CREATE_PREVIEW_BUTTON],
        components[K.CLEAR_IMAGE_BUTTON_UI],
        components[K.DOWNLOAD_IMAGE_BUTTON_UI],
        components[K.SAVE_QUEUE_BUTTON_UI],
        components[K.CLEAR_QUEUE_BUTTON_UI],
    ]
    # Use the workspace's default map as the single source of truth for UI components.
    default_keys_map = workspace_manager.get_default_values_map()
    full_workspace_ui_components = [components[K[key.upper()]] for key in default_keys_map.keys()]
    (components[K.SAVE_WORKSPACE_BUTTON].click(
        fn=workspace_manager.save_workspace,
        inputs=full_workspace_ui_components,
        outputs=components[K.WORKSPACE_DOWNLOADER_UI]
    ).then(
        fn=None, inputs=[components[K.WORKSPACE_DOWNLOADER_UI]], outputs=None,
        js="(file) => { document.getElementById('workspace_downloader_hidden_file').querySelector('a[download]').click(); }"
    ))
    (components[K.LOAD_WORKSPACE_BUTTON].upload(
        fn=workspace_manager.load_workspace,
                inputs=[components[K.LOAD_WORKSPACE_BUTTON]],
        outputs=full_workspace_ui_components
    ).then(
        fn=event_handlers.update_button_states,
        inputs=[components[K.APP_STATE], components[K.INPUT_IMAGE_DISPLAY_UI], components[K.QUEUE_DF_DISPLAY_UI]],
        outputs=button_state_outputs
    ))
    components[K.SAVE_AS_DEFAULT_BUTTON].click(
        fn=workspace_manager.save_as_default_workspace,
        inputs=full_workspace_ui_components,
        outputs=[components[K.RELAUNCH_NOTIFICATION_MD], components[K.RELAUNCH_BUTTON]]
    )
# ui/switchboard_image.py
import gradio as gr
import logging

from .enums import ComponentKey as K
from . import (
    metadata as metadata_manager,
    event_handlers,
    shared_state as shared_state_module,
    workspace as workspace_manager,
)

logger = logging.getLogger(__name__)

def wire_events(components: dict):
    """Wires up the main image input and metadata modal events."""
    logger.info("Wiring image and metadata events...")

    button_state_outputs = [
        components[K.ADD_TASK_BUTTON],
        components[K.PROCESS_QUEUE_BUTTON],
        components[K.CREATE_PREVIEW_BUTTON],
        components[K.CLEAR_IMAGE_BUTTON],
        components[K.DOWNLOAD_IMAGE_BUTTON],
        components[K.SAVE_QUEUE_BUTTON],
        components[K.CLEAR_QUEUE_BUTTON],
    ]
    creative_ui_components = [components[key] for key in shared_state_module.CREATIVE_UI_KEYS]

    clear_button_outputs = [
        components[K.IMAGE_FILE_INPUT],
        components[K.INPUT_IMAGE_DISPLAY],
        components[K.CLEAR_IMAGE_BUTTON],
        components[K.DOWNLOAD_IMAGE_BUTTON],
        components[K.ADD_TASK_BUTTON],
        components[K.EXTRACTED_METADATA_STATE]
    ]

    upload_outputs = [
        components[K.IMAGE_FILE_INPUT],
        components[K.INPUT_IMAGE_DISPLAY],
        components[K.CLEAR_IMAGE_BUTTON],
        components[K.DOWNLOAD_IMAGE_BUTTON],
        components[K.ADD_TASK_BUTTON],
        components[K.METADATA_PROMPT_PREVIEW],
        components[K.EXTRACTED_METADATA_STATE],
        components[K.METADATA_MODAL_TRIGGER_STATE],
    #     components[K.RESUME_LATENT_PATH_STATE]
    ] + creative_ui_components

    (components[K.IMAGE_FILE_INPUT].upload(
        fn=workspace_manager.handle_file_drop,
        inputs=[components[K.IMAGE_FILE_INPUT]],
        outputs=upload_outputs
    ).then(
        fn=event_handlers.update_button_states,
        # Corrected: Pass K.QUEUE_DF as the third argument as expected by update_button_states [svc: then removed again]
        inputs=[components[K.APP_STATE], components[K.INPUT_IMAGE_DISPLAY],
                # components[K.QUEUE_DF]
                ],
        outputs=button_state_outputs
    ))

    (components[K.CLEAR_IMAGE_BUTTON].click(
        fn=event_handlers.clear_image_action, inputs=None, outputs=clear_button_outputs
    ).then(
        fn=event_handlers.update_button_states,
        # Corrected: Pass K.QUEUE_DF as the third argument as expected by update_button_states [svc: then removed again]
        inputs=[components[K.APP_STATE], components[K.INPUT_IMAGE_DISPLAY],
                # components[K.QUEUE_DF]
                ],
        outputs=button_state_outputs
    ))

    (components[K.DOWNLOAD_IMAGE_BUTTON].click(
        fn=event_handlers.prepare_image_for_download,
        inputs=([components[K.INPUT_IMAGE_DISPLAY], components[K.APP_STATE], gr.State(shared_state_module.CREATIVE_UI_KEYS)] + creative_ui_components),
        outputs=components[K.IMAGE_DOWNLOADER], show_progress=True, api_name="download_image_with_metadata"
    ).then(
        fn=None, inputs=None, outputs=None,
        js="() => { document.getElementById('image_downloader_hidden_file').querySelector('a[download]').click(); }"
    ))

    components[K.METADATA_MODAL_TRIGGER_STATE].change(
        fn=lambda x: gr.update(visible=True) if x else gr.update(visible=False),
        inputs=[components[K.METADATA_MODAL_TRIGGER_STATE]],
        outputs=[components[K.METADATA_MODAL]],
        api_name=False, queue=False
    )
    (components[K.CONFIRM_METADATA_BUTTON].click(
        fn=metadata_manager.ui_load_params_from_image_metadata,
        inputs=[components[K.EXTRACTED_METADATA_STATE]],
        outputs=creative_ui_components
    ).then(fn=event_handlers.ui_update_total_segments, inputs=[components[K.VIDEO_LENGTH_SLIDER], components[K.LATENT_WINDOW_SIZE_SLIDER], components[K.FPS_SLIDER]], outputs=[components[K.TOTAL_SEGMENTS_DISPLAY]]
    ).then(fn=lambda: gr.update(value=None), inputs=None, outputs=[components[K.METADATA_MODAL_TRIGGER_STATE]]))
    components[K.CANCEL_METADATA_BUTTON].click(fn=lambda: gr.update(value=None), inputs=None, outputs=[components[K.METADATA_MODAL_TRIGGER_STATE]])

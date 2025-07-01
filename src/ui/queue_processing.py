# ui/queue_processing.py
# Handles interactions with the ProcessingAgent and queue processing logic.

import gradio as gr
import logging
import queue  # For queue.Empty exception
from .queue_manager import queue_manager_instance
from . import shared_state as shared_state_module
from . import queue_helpers
from .agents import ProcessingAgent, ui_update_queue

logger = logging.getLogger(__name__)

def process_task_queue_and_listen(*lora_control_values):
    """Starts the ProcessingAgent, listens for UI updates, and handles stop requests."""
    agent = ProcessingAgent()

    # If processing is already active, this button click is a "stop" request.
    if queue_manager_instance.get_state().get("processing", False):
        # Set the flag for immediate UI feedback via update_button_states
        shared_state_module.shared_state_instance.stop_requested_flag.set()
        agent.send({"type": "stop"})
        gr.Info("Stop requested. The current task will be stopped.")
        # Return minimal updates. The .then() call in the switchboard will call
        # update_button_states, which will see the flag and update the UI correctly.
        return [gr.update()] * 9

    # If not processing, this is a "start" request.
    agent.send({
        "type": "start",
        "lora_controls": lora_control_values
    })

    # The listener loop. It doesn't manage state, just streams updates from the agent.
    while True:
        try:
            # Block until an update is available from the agent's UI queue.
            flag, data = ui_update_queue.get(timeout=1.0)

            if flag == "processing_started":
                # This is the first signal from the agent that it has started.
                # Update the UI to the "processing" state.
                yield (  # The first output (APP_STATE) is gr.update() as we don't modify it here.
                    gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(value="Queue processing started..."),  # Progress description
                    gr.update(value=None, visible=True),  # Progress bar
                    gr.update(interactive=True, value="⏹️ Stop Processing", variant="stop"),  # PROCESS_QUEUE_BUTTON
                    gr.update(interactive=True),  # CREATE_PREVIEW_BUTTON
                    gr.update(interactive=False)  # CLEAR_QUEUE_BUTTON_UI
                )
            elif flag == "progress":
                # Unpack data: task_id, preview_np, desc, html
                _, preview_np, desc, html = data  # type: ignore
                yield (gr.update(), gr.update(), gr.update(), gr.update(value=preview_np), desc, html, gr.update(), gr.update(), gr.update())
            elif flag == "file":
                # Unpack data: task_id, new_video_path, _
                _, new_video_path, _ = data  # type: ignore
                yield (gr.update(), gr.update(), gr.update(value=new_video_path), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update())
            elif flag == "task_starting":
                task = data  # type: ignore
                yield (gr.update(), queue_helpers.update_queue_df_display(), gr.update(), gr.update(), f"Processing Task {task['id']}...", gr.update(), gr.update(), gr.update(), gr.update())
            elif flag == "task_finished":
                yield (gr.update(), queue_helpers.update_queue_df_display(), gr.update(), gr.update(), f"Task {data['id']} {data['status']}.", gr.update(), gr.update(), gr.update(), gr.update())
            elif flag == "info":
                gr.Info(data)
            elif flag == "queue_finished":
                # The agent has signaled the end of all processing.
                logger.info("UI listener received 'queue_finished' signal. Exiting loop.")
                break

        except queue.Empty:
            # If the queue is empty, we check if the agent is still processing.
            # If not, it means the process finished or was stopped without a final signal.
            if not queue_manager_instance.get_state().get("processing", False):
                logger.info("UI listener detected processing has stopped. Exiting loop.")
                break
            continue  # Continue waiting for updates.

    # The .then() call in the switchboard will handle the final button state update.
    # We just need to yield one last time to ensure the final queue state is displayed.
    logger.info("UI listener loop finished. Yielding final queue display.")
    yield (  # Yield a final update to refresh the queue display.
        gr.update(),
        queue_helpers.update_queue_df_display(),
        gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
    )

def request_preview_generation_action():
    """Triggers a preview generation request to the ProcessingAgent."""
    ProcessingAgent().send({"type": "preview"})


# ui/agents.py
import threading
import queue
import logging
import traceback
import numpy as np

from core.generation_core import worker
from diffusers_helper.thread_utils import AsyncStream, async_run
from . import shared_state as shared_state_module
from .lora import LoRAManager
from .queue_manager import queue_manager_instance

logger = logging.getLogger(__name__)

# This queue is the bridge from the agent back to the UI thread.
ui_update_queue = queue.Queue()

def worker_wrapper(output_queue_ref, **kwargs):
    """
    A wrapper that calls the real worker in a try-except block
    to catch and report any backend exceptions.
    """
    try:
        worker(output_queue_ref=output_queue_ref, **kwargs)
    except Exception as e:
        tb_str = traceback.format_exc()
        logger.error(f"--- BACKEND WORKER CRASHED ---\n{tb_str}\n--------------------------", exc_info=True)
        output_queue_ref.push(('crash', tb_str))

class ProcessingAgent(threading.Thread):
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ProcessingAgent, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return

        with self._lock:
            # Double-check inside the lock to ensure initialization happens only once.
            if hasattr(self, '_initialized') and self._initialized:
                return

            super().__init__(daemon=True)
            self.mailbox = queue.Queue()
            self.is_processing = False
            self.start()
            self._initialized = True

    def send(self, message):
        self.mailbox.put(message)

    def run(self):
        """The agent's main loop, waiting for messages."""
        while True:
            message = self.mailbox.get()
            if message.get("type") == "start":
                self._handle_start(message)
            elif message.get("type") == "stop":
                self._handle_stop()

    def _handle_start(self, message):
        if self.is_processing:
            return

        if not queue_manager_instance.has_pending_tasks():
            ui_update_queue.put(("info", "Queue is empty. Add tasks to process."))
            return

        self.is_processing = True
        queue_manager_instance.set_processing(True)
        ui_update_queue.put(("processing_started", None))
        shared_state_module.shared_state_instance.interrupt_flag.clear()

        # Run the actual processing in a separate thread to not block the agent's mailbox
        processing_thread = threading.Thread(target=self._processing_loop, args=(message,))
        processing_thread.start()

    def _handle_stop(self):
        """Handles any stop request by setting the interrupt flag."""
        if not self.is_processing:
            return
        shared_state_module.shared_state_instance.interrupt_flag.set()
        logger.info("Stop signal sent to worker. Worker will stop and finalize the current task.")

    def _processing_loop(self, start_message):
        lora_controls = start_message.get("lora_controls")

        lora_handler = LoRAManager()
        try:
            if lora_controls:
                lora_handler.apply_lora(*lora_controls)

            while not shared_state_module.shared_state_instance.interrupt_flag.is_set():
                task = queue_manager_instance.get_and_start_next_task()

                if task is None:  # No more pending tasks
                    ui_update_queue.put(("info", "All tasks processed."))
                    break

                ui_update_queue.put(("task_starting", task))

                output_stream = AsyncStream()
                worker_args = {**task["params"], "task_id": task["id"], **shared_state_module.shared_state_instance.models}
                worker_args.pop('transformer', None)
                async_run(worker_wrapper, output_queue_ref=output_stream.output_queue, **worker_args)

                task_final_status = "error"
                final_output_path = None
                error_message = "Worker exited unexpectedly."

                while True:
                    if shared_state_module.shared_state_instance.interrupt_flag.is_set():
                        break

                    flag, data = output_stream.output_queue.next()
                    ui_update_queue.put((flag, data))

                    if flag == "end":
                        _, success, final_path = data
                        task_final_status = "done" if success else "error"
                        final_output_path = final_path
                        break
                    elif flag == "crash":
                        task_final_status = "error"
                        error_message = "Worker process crashed."
                        break
                    elif flag == "aborted":
                        task_final_status = "aborted"
                        error_message = None
                        break
                    elif flag == "file":
                        _, new_video_path, _ = data
                        final_output_path = new_video_path

                queue_manager_instance.complete_task(
                    task_id=task["id"],
                    status=task_final_status,
                    final_path=final_output_path,
                    error_msg=error_message
                )
                ui_update_queue.put(("task_finished", {"id": task["id"], "status": task_final_status}))

                if shared_state_module.shared_state_instance.interrupt_flag.is_set():
                    ui_update_queue.put(("info", "Queue processing stopped by user."))
                    break
        finally:
            logger.info("Processing finished. Reverting all LoRAs to clean up.")
            lora_handler.revert_all_loras()
            logger.info("All LoRAs reverted. Processing agent is now idle.")
            self.is_processing = False
            queue_manager_instance.set_processing(False)
            shared_state_module.shared_state_instance.interrupt_flag.clear()
            shared_state_module.shared_state_instance.stop_requested_flag.clear()
            ui_update_queue.put(("queue_finished", None))
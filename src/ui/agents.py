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
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
        super().__init__(daemon=True)
        with self._lock:
            if self._initialized: return
            self._initialized = True
        self.mailbox = queue.Queue()
        self.is_processing = False
        self.start() # Start the agent's own thread

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
        if self.is_processing: return
        self.is_processing = True
        
        tasks_to_process = queue_manager_instance.get_pending_tasks()
        if not tasks_to_process:
            self.is_processing = False
            return
        queue_manager_instance.set_processing(True)
        
        # Run the actual processing in a separate thread to not block the agent's mailbox
        processing_thread = threading.Thread(target=self._processing_loop, args=(message,))
        processing_thread.start()

    def _handle_stop(self):
        if not self.is_processing: return
        shared_state_module.shared_state_instance.interrupt_flag.set()
        shared_state_module.shared_state_instance.abort_state['level'] = 2
        logger.info("Stop signal sent to worker. Interrupt Level: 2.")

    def _processing_loop(self, start_message):
        tasks, lora_controls = start_message["tasks"], start_message["lora_controls"]
        
        lora_handler = LoRAManager()
        try:
            if lora_controls:
                lora_handler.apply_lora(*lora_controls)

            for task in tasks:
                if shared_state_module.shared_state_instance.interrupt_flag.is_set():
                    ui_update_queue.put(("info", "Queue processing stopped by user."))
                    break

                if task.get("params", {}).get("seed") == -1:
                    task["params"]["seed"] = np.random.randint(0, 2**32 - 1)
                
                ui_update_queue.put(("task_starting", task))
                
                output_stream = AsyncStream()
                worker_args = {**task["params"], "task_id": task["id"], **shared_state_module.shared_state_instance.models}
                worker_args.pop('transformer', None)
                async_run(worker_wrapper, output_queue_ref=output_stream.output_queue, **worker_args)

                for flag, data in output_stream.output_queue:
                    ui_update_queue.put((flag, data))
                    if flag in ["end", "crash"]: break
        finally:
            logger.info("Processing finished. Reverting all LoRAs to clean up.")
            lora_handler.revert_all_loras()
            self.is_processing = False
            queue_manager_instance.set_processing(False)
            ui_update_queue.put(("queue_finished", None))
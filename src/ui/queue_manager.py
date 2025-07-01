# ui/queue_manager.py
import threading
import numpy as np
from PIL import Image
import gradio as gr

class QueueManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(QueueManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        with self._lock:
            self.state = {
                "queue": [],
                "next_id": 1,
                "processing": False,
                "editing_task_id": None
            }
            self.queue_lock = threading.Lock()
            self._initialized = True

    def get_state(self):
        with self.queue_lock:
            return self.state.copy()

    def set_processing(self, is_processing: bool):
        with self.queue_lock:
            self.state["processing"] = is_processing

    def add_task(self, params: dict, input_image: np.ndarray):
        with self.queue_lock:
            next_id = self.state["next_id"]
            task = {
                "id": next_id,
                "params": {**params, 'input_image': input_image},
                "status": "pending"
            }
            self.state["queue"].append(task)
            self.state["next_id"] += 1
            gr.Info(f"Added task {next_id} to the queue.")

    def update_task(self, task_id: int, params: dict, input_image: np.ndarray):
        with self.queue_lock:
            for task in self.state["queue"]:
                if task["id"] == task_id:
                    task["params"] = {**params, 'input_image': input_image}
                    task["status"] = "pending"
                    gr.Info(f"Task {task_id} updated.")
                    break
            self.state["editing_task_id"] = None

    def remove_task(self, task_index: int):
        removed_task_id = None
        with self.queue_lock:
            if 0 <= task_index < len(self.state["queue"]):
                removed_task = self.state["queue"].pop(task_index)
                removed_task_id = removed_task['id']
                gr.Info(f"Removed task {removed_task_id}.")
            else:
                gr.Warning("Invalid index for removal.")
        return removed_task_id

    def move_task(self, direction: str, task_index: int):
        with self.queue_lock:
            queue = self.state["queue"]
            if direction == 'up' and task_index > 0:
                queue[task_index], queue[task_index-1] = queue[task_index-1], queue[task_index]
            elif direction == 'down' and task_index < len(queue) - 1:
                queue[task_index], queue[task_index+1] = queue[task_index+1], queue[task_index]

    def clear_pending_tasks(self):
        with self.queue_lock:
            initial_count = len(self.state["queue"])
            self.state["queue"] = [task for task in self.state["queue"] if task.get("status", "pending") != "pending"]
            cleared_count = initial_count - len(self.state["queue"])
            if cleared_count > 0:
                gr.Info(f"Cleared {cleared_count} pending tasks.")
            else:
                gr.Info("No pending tasks to clear.")

    def set_editing_task(self, task_id: int | None):
        with self.queue_lock:
            self.state["editing_task_id"] = task_id
            if task_id is not None:
                gr.Info(f"Editing Task {task_id}.")
            else:
                gr.Info("Edit cancelled.")

    def get_task_to_edit(self, task_index: int):
        with self.queue_lock:
            if 0 <= task_index < len(self.state["queue"]):
                task_to_edit = self.state["queue"][task_index]
                self.set_editing_task(task_to_edit['id'])
                return task_to_edit
        return None

    def complete_task(self, task_id: int, status: str, final_path: str | None = None, error_msg: str | None = None):
        with self.queue_lock:
            if self.state["queue"] and self.state["queue"][0]["id"] == task_id:
                task = self.state["queue"][0]
                task["status"] = status
                if final_path: task["final_output_filename"] = final_path
                if error_msg: task["error_message"] = error_msg
                self.state["queue"].pop(0)

    def load_queue(self, new_queue: list, next_id: int):
        """Safely replaces the current queue with a new one."""
        with self.queue_lock:
            self.state["queue"] = new_queue
            self.state["next_id"] = max(next_id, self.state.get("next_id", 1))

    def get_and_start_next_task(self):
        """
        Finds the first pending task at the top of the queue, marks it as 'processing',
        and returns it. This is an atomic operation for the agent to use.
        """
        with self.queue_lock:
            if self.state["queue"]:
                task = self.state["queue"][0]
                if task.get("status", "pending") == "pending":
                    task["status"] = "processing"
                    if task.get("params", {}).get("seed") == -1:
                        task["params"]["seed"] = np.random.randint(0, 2**32 - 1)
                    return task.copy()  # Return a copy to avoid race conditions
            return None

    def has_pending_tasks(self):
        """Checks if there are any tasks with 'pending' status."""
        with self.queue_lock:
            return any(task.get("status", "pending") == "pending" for task in self.state["queue"])

# Singleton instance
queue_manager_instance = QueueManager()
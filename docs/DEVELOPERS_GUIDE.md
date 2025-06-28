# Developer's Guide to Extending 'goan'

## 1. Introduction & Core Philosophy

This document provides a technical overview of the `goan` application architecture. It is intended for developers looking to understand, maintain, or extend the codebase.

The application is built on three core principles:

1.  **Separation of Concerns**: The user interface (UI) is strictly separated from the backend generation engine. The UI's role is to capture user intent and display results, while the backend's role is to execute generation tasks.
2.  **Centralized State Management**: All critical application state (threading events, loaded models, system info) is managed by a thread-safe singleton. UI-facing state (like the task queue) is managed within a single Gradio `gr.State` component, providing a single source of truth for UI event handlers.
3.  **Declarative Event Wiring**: All UI event handling (e.g., button clicks, slider changes) is wired in a single, dedicated module (`ui/switchboard.py`). This makes it easy to understand the cause-and-effect relationships in the UI without searching through multiple files.

---

## 2. Core Architecture Overview

The application can be understood as three main layers:

*   **Frontend (Gradio View Layer)**: Defined in `ui/layout.py`. This layer is responsible only for creating and arranging the Gradio components. It is a "dumb" layer with no event logic.

*   **Backend (Worker Engine)**: The `worker` function in `core/generation_core.py` is the heart of the generation process. It runs in a separate thread (managed by `diffusers_helper/thread_utils.py`) and is completely decoupled from Gradio. It accepts a dictionary of parameters and communicates its progress back to the UI via a message queue (`AsyncStream`).

*   **UI Logic (The "Glue" Layer)**: This is the largest and most complex layer, residing in the `src/ui/` directory. It bridges the gap between the Frontend and the Backend.
    *   **`ui/shared_state.py`**: A singleton that holds global, thread-safe state like threading locks, events (`interrupt_flag`), and references to the loaded PyTorch models. It also contains critical mappings that define which UI controls correspond to which backend parameters.
    *   **`ui/switchboard.py`**: The central nervous system of the UI. It connects components from `layout.py` to their corresponding event handler functions. If you want to know what a button does, you look here.
    *   **`ui/queue.py`**: Contains the primary UI-side logic for managing the task queue. The `process_task_queue_main_loop` function orchestrates the entire generation process, calling the backend worker and listening for progress updates to stream back to the UI.
    *   **`ui/event_handlers.py`**: Contains general-purpose UI event logic, such as updating button states (`update_button_states`) or handling image uploads.

---

## 3. Key Process Flows

### Application Startup

1.  **`goan.py`**: The main entry point.
2.  **`core/args.py`**: Parses command-line arguments.
3.  **`core/model_loader.py`**: Detects GPU capabilities, loads all necessary models to the CPU, configures them for the environment (e.g., VRAM, dtype), and populates the `shared_state_instance.models` dictionary. The large transformer model is loaded lazily on first use.
4.  **`ui/layout.py`**: `create_ui()` is called to build the Gradio component tree.
5.  **`ui/switchboard.py`**: `wire_all_events()` is called, which registers all `.click()`, `.change()`, and `.load()` event listeners.
6.  **`block.launch()`**: The Gradio server is started.
7.  **On UI Load**: The `.load()` event wired in `_wire_app_startup_events` triggers a chain of functions:
    *   `workspace.load_workspace_on_start()` finds the path to the last session's settings (`goan_unload_save.json`) or the default settings (`goan_settings.json`).
    *   `workspace.load_settings_from_file()` reads the JSON file and populates the UI controls.
    *   `workspace.load_image_from_path()` loads the last-used input image, if any.
    *   `event_handlers.update_button_states()` sets the initial interactive state of all buttons based on the loaded state.

### Adding a Task to the Queue

1.  A user clicks the "Add to Queue" button (`K.ADD_TASK_BUTTON`).
2.  The `switchboard` catches the `.click()` event and calls `queue.add_or_update_task_in_queue`.
3.  This function receives the main `APP_STATE` dictionary and the current values of all task-related UI controls.
4.  It packages these values into a new task dictionary, assigns it a unique ID, and appends it to the list at `APP_STATE["queue_state"]["queue"]`.
5.  It returns the modified `APP_STATE` and a `gr.update` object to refresh the queue's visual display (`K.QUEUE_DF_DISPLAY_UI`).
6.  The `.then()` clause in the `switchboard` calls `event_handlers.update_button_states` to re-evaluate button interactivity (e.g., enabling the "Process Queue" button).

### Processing the Queue

1.  A user clicks the "▶️ Process Queue" button (`K.PROCESS_QUEUE_BUTTON`).
2.  The `switchboard` calls `queue.process_task_queue_main_loop`.
3.  The function checks if `APP_STATE["queue_state"]["processing"]` is `False`.
    *   If `False`, it sets the flag to `True` to begin processing and prevent another start command.
    *   If `True`, it means the user is clicking the "⏹️ Stop Processing" button. The function sets the `interrupt_flag` and returns, stopping the process.
4.  The function enters a `while` loop that continues as long as there are tasks in the queue and the `interrupt_flag` is not set.
5.  Inside the loop, it takes the first task, prepares its arguments, and calls `async_run(worker_wrapper, ...)`. This starts the `core/generation_core.py:worker` function in a separate, non-blocking thread.
6.  The `worker` performs the entire generation, pushing progress updates (images, text, files) to a shared message queue (`AsyncStream`).
7.  Simultaneously, the `process_task_queue_main_loop` listens for messages on this queue and `yield`s `gr.update` objects to the UI in real-time, creating the live preview.
8.  When the `worker` pushes an "end" message, the `process_task_queue_main_loop` removes the completed task from the queue and proceeds to the next one.
9.  When the loop finishes (or is interrupted), the `finally` block cleans up, sets `processing` back to `False`, and yields the final UI state.

---

## 4. Module Contracts (File-by-File)

### `src/core/` - The Backend Engine

*   **`generation_core.py`**: **Contract**: Defines the `worker` function, the main entry point for video generation. It must be self-contained and must not import `gradio` or any `ui/` modules except for `shared_state` (for flags) and `metadata` (for saving). It communicates *only* through the `output_queue_ref`.
*   **`model_loader.py`**: **Contract**: Responsible for loading all PyTorch models from HuggingFace, configuring them for the detected hardware (VRAM, GPU architecture), and populating the `shared_state_instance.models` dictionary.
*   **`args.py`**: **Contract**: Defines and parses command-line arguments using `argparse`.
*   **`generation_utils.py`**: **Contract**: Contains helper functions extracted from `generation_core.py` for clarity and reuse (e.g., saving previews, calculating CFG schedules).

### `src/ui/` - The Frontend and Glue Logic

*   **`enums.py`**: **Contract**: Defines the `ComponentKey` `StrEnum`. This is a critical file that provides unique, typo-proof identifiers for every UI component. All new components **must** have a key added here.
*   **`shared_state.py`**: **Contract**: Defines the `SharedState` singleton and several constant mappings. This is the single source of truth for global state, threading primitives, and the mapping between UI component keys and their corresponding backend worker parameter names (`UI_TO_WORKER_PARAM_MAP`).
*   **`layout.py`**: **Contract**: Defines the entire Gradio UI component tree. Its only job is to create and return a dictionary of all `gr.` components, keyed by `ComponentKey`. It contains no logic.
*   **`switchboard.py`**: **Contract**: The central hub for all UI events. Its only job is to connect components from `layout.py` to handler functions in other `ui/` modules. It contains all `.click()`, `.change()`, `.load()`, etc., event bindings.
*   **`event_handlers.py`**: **Contract**: Contains general-purpose UI event logic that is not specific to one feature area (e.g., `update_button_states`, `process_upload_and_show_image`).
*   **`queue.py`**: **Contract**: Manages all high-level logic for the task queue: adding, removing, reordering, and processing tasks. It contains the main UI-side processing loop.
*   **`queue_helpers.py`**: **Contract**: Contains helper functions for `queue.py`, primarily for formatting queue data for display in the Gradio DataFrame.
*   **`workspace.py`**: **Contract**: Handles saving and loading of UI settings to/from JSON files. Manages the default workspace and session-restore functionality.
*   **`metadata.py`**: **Contract**: Handles reading and writing generation parameters to and from PNG image metadata.
*   **`lora.py`**: **Contract**: Manages the application and reversion of LoRA weights to the models. Contains logic for translating key names from different LoRA training formats.
*   **`legacy_support.py`**: **Contract**: Contains functions to convert old parameter formats from saved files to the current format, ensuring backward compatibility.

### `src/diffusers_helper/`

*   **Contract**: This directory contains a collection of lower-level utility functions, many from the original FramePack author, for interacting with `diffusers` models, managing memory, and handling tensors. These are generally stable and should not require frequent modification.

---

## 5. How to Extend 'goan'

This section provides a practical guide for common development tasks.

### How to Add a New UI Control

Let's add a hypothetical "Sharpness" slider to the "Advanced Settings" accordion.

1.  **`ui/enums.py`**: Add a new key to the `ComponentKey` enum.
    ```python
    class ComponentKey(StrEnum):
        # ... existing keys
        SHARPNESS_UI = auto()
    ```

2.  **`ui/layout.py`**: Create the `gr.Slider` in the `create_ui` function, placing it within the "Advanced Settings" accordion.
    ```python
    with gr.Accordion("Advanced Settings", open=False):
        # ... existing controls
        components[K.SHARPNESS_UI] = gr.Slider(label="Sharpness", minimum=0.0, maximum=5.0, value=1.0, step=0.1)
    ```

3.  **`ui/shared_state.py`**: Register the new control so the application knows it's part of a task's parameters.
    *   Add the key to the appropriate list (e.g., `CREATIVE_UI_KEYS`).
        ```python
        CREATIVE_UI_KEYS = [
            K.PROMPT_UI, # ...
            K.RS_UI,
            K.SHARPNESS_UI # Add new key here
        ]
        ALL_TASK_UI_KEYS = CREATIVE_UI_KEYS + ENVIRONMENT_UI_KEYS
        ```
    *   Map the UI key to a backend parameter name in `UI_TO_WORKER_PARAM_MAP`.
        ```python
        UI_TO_WORKER_PARAM_MAP = {
            # ... existing mappings
            K.LATENT_WINDOW_SIZE_UI: 'latent_window_size',
            K.SHARPNESS_UI: 'sharpness' # Add new mapping here
        }
        ```

4.  **`core/generation_core.py`**: Update the `worker` function to accept and use the new parameter.
    ```python
    def worker(
        # ... existing parameters
        mp4_crf,
        sharpness, # Add new parameter
        # ... model objects
    ):
        # ... use the 'sharpness' variable in your generation logic ...
    ```

5.  **`ui/workspace.py`**: Add a default value for the new control in `get_default_values_map`. This ensures that saving/loading workspaces functions correctly.
    ```python
    def get_default_values_map():
        return {
            # ... existing defaults
            K.LATENT_WINDOW_SIZE_UI: 9,
            K.SHARPNESS_UI: 1.0, # Add new default here
        }
    ```

6.  **`ui/switchboard.py`**: **Verification Step**. Because the switchboard uses the lists from `shared_state.py` (like `ALL_TASK_UI_KEYS`) to build its input/output lists, the new control should be automatically included in events like adding a task or saving a workspace. It is good practice to double-check the `inputs` list for `queue_manager.add_or_update_task_in_queue` to confirm the new component is present.
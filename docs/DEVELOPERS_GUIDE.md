# Developer's Guide to Extending 'goan'

## 1. Introduction & Core Philosophy

This document provides a technical overview of the `goan` application architecture. It is intended for developers looking to understand, maintain, or extend the codebase.

The application is built on three core principles:

1.  **Separation of Concerns**: The user interface (UI) is strictly separated from the backend generation engine. The UI's role is to capture user intent and display results, while the backend's role is to execute generation tasks.
2.  **Centralized State Management**: All critical application state (threading events, loaded models, system info) is managed by a thread-safe singleton. UI-facing state (like the task queue) is now managed by the `QueueManager` singleton, providing a single source of truth accessible to both the UI and the agent. This replaces the previous reliance on Gradio's `gr.State` component for UI state.
3.  **Agent-Driven UI Updates**: The UI is primarily updated by messages sent from the `ProcessingAgent` rather than direct manipulation of Gradio state within event handlers. This ensures that UI changes are synchronized with the backend's processing state and avoids potential race conditions.
4.  **Declarative Event Wiring**: All UI event handling (e.g., button clicks, slider changes) is wired in a single, dedicated module (`ui/switchboard.py`). This makes it easy to understand the cause-and-effect relationships in the UI without searching through multiple files.

---

## 2. Core Architecture Overview

The application can be understood as three main layers:

*   **Frontend (Gradio View Layer)**: Defined in `ui/layout.py`. This layer is responsible only for creating and arranging the Gradio components. It is a "dumb" layer with no event logic.

*   **Backend (Worker Engine)**: The `worker` function in `core/generation_core.py` is the heart of the generation process. It runs in a separate thread (managed by `diffusers_helper/thread_utils.py`) and is completely decoupled from Gradio. It accepts a dictionary of parameters and communicates its progress back to the UI via a message queue (`AsyncStream`).

*   **UI Logic (The "Glue" Layer)**: This layer bridges the gap between the Frontend and the Backend and resides in the `src/ui/` directory.
    *   **`ui/shared_state.py`**: A singleton that holds global, thread-safe state like threading locks, events (`interrupt_flag`), and references to the loaded PyTorch models. It also contains critical mappings that define which UI controls correspond to which backend parameters.
    *   **`ui/switchboard.py`**: The central hub for all UI events. It connects components from `layout.py` to their corresponding event handler functions. If you want to know what a UI element does, look here.
    *   **`ui/queue_manager.py`**: This singleton class manages the task queue itself, providing thread-safe operations for adding, updating, removing, reordering, and retrieving tasks.  It acts as the single source of truth for the queue's state.
    *   **`ui/queue.py`**: Contains the UI-side logic for interacting with the task queue.  It houses functions for handling user actions on the queue (e.g., adding, editing, deleting tasks) and for starting/stopping the processing of the queue.  It communicates with the `ProcessingAgent` to control the backend worker.
    *   **`ui/event_handlers.py`**: Contains general-purpose UI event logic, such as updating button states (`update_button_states`), handling image uploads, and managing UI state transitions.
    *   **`ui/queue_helpers.py`**: Provides helper functions for `queue.py`, primarily for formatting queue data for display in the Gradio DataFrame and converting data types for UI elements.
    *   **`ui/agents.py`**: Defines the `ProcessingAgent` class, which manages the interaction with the backend worker. It handles starting, stopping, and monitoring tasks, and uses message queues to communicate bidirectionally with both the UI and the worker thread.

---

## 3. Key Process Flows

### Application Startup

1.  **`goan.py`**: The main entry point.
2.  **`core/args.py`**: Parses command-line arguments.
3.  **`core/model_loader.py`**: Detects GPU capabilities, loads all necessary models to the CPU, configures them for the environment (e.g., VRAM, dtype), and populates the `shared_state_instance.models` dictionary. The large transformer model is loaded lazily on first use.
4.  **`ui/layout.py`**: `create_ui()` is called to build the Gradio component tree.
5.  **`ui/switchboard.py`**: `wire_all_events()` is called, which registers all `.click()`, `.change()`, and `.load()` event listeners, connecting UI components to their corresponding handler functions.
6.  **`block.launch()`**: The Gradio server is started, making the UI accessible in a web browser.
7.  **On UI Load**: The `.load()` event wired in `_wire_app_startup_events` triggers a chain of functions:
    *   `workspace.load_workspace_on_start()` finds the path to the last session's settings (`goan_unload_save.json`) or the default settings (`goan_settings.json`).
    *   `workspace.load_settings_from_file()` reads the JSON file and populates the UI controls with the saved settings.
    *   `workspace.load_image_from_path()` loads the last-used input image, if any, restoring the visual state.
    *   `event_handlers.update_button_states()` sets the initial interactive state of all buttons based on the loaded workspace, image, and queue state, ensuring a consistent and responsive UI.

### Adding a Task to the Queue

1.  A user interacts with the UI, providing necessary inputs (image, prompt, parameters).
2.  The user clicks the "Add to Queue" button (`K.ADD_TASK_BUTTON`).
3.  The `switchboard` catches the `.click()` event and calls `queue.add_or_update_task_in_queue`. This function now receives only the values of the task-related UI controls directly as arguments, not a global state dictionary.
4.  The `add_or_update_task_in_queue` function uses these inputs to create a new task dictionary. It interacts with the `QueueManager` instance to add the new task to the queue, assigning it a unique ID. If in "edit" mode (updating an existing task), it updates the corresponding task in the queue.
5.  The function returns a list of `gr.update` objects to refresh the queue's visual display (`K.QUEUE_DF_DISPLAY_UI`) and any other relevant UI elements, providing immediate feedback to the user.
6.  The `.then()` clause in the `switchboard` calls `event_handlers.update_button_states` to re-evaluate button interactivity (e.g., enabling the "Process Queue" button if the queue is no longer empty), maintaining UI consistency.

### Processing the Queue

1.  A user clicks the "▶️ Process Queue" button (`K.PROCESS_QUEUE_BUTTON`).
2.  The `switchboard` calls `queue.process_task_queue_and_listen`.
3.  This function interacts with the `ProcessingAgent` to either start or stop the queue processing.
    *   **Start**: If processing is not active, it sends a "start" message to the `ProcessingAgent` with LoRA control values.
    *   **Stop**: If processing is active, it sets the `stop_requested_flag` in `shared_state` and sends a "stop" message to the `ProcessingAgent`. The UI also receives immediate feedback that the stop was requested via `gr.Info`.
4.  The function enters a `while` loop that continues as long as the `ProcessingAgent` is running. This loop acts as a listener for updates from the agent.
5.  The `ProcessingAgent` manages the actual processing:
    *   It retrieves tasks from the `QueueManager` using `get_and_start_next_task`, which atomically gets the next pending task and marks it as processing.
    *   It launches the `worker` function (in `core/generation_core.py`) in a separate thread using `async_run`.
    *   The `worker` performs the video generation and sends progress updates (e.g., preview images, progress descriptions, file paths) back to the `ProcessingAgent` via a queue. The `worker_wrapper` ensures that any unhandled exceptions in the `worker` are caught, logged, and reported back to the agent as a "crash" event.
    *   The `ProcessingAgent` then relays these updates to the UI by putting them into the `ui_update_queue`.
6.  Back in `queue.process_task_queue_and_listen`, the loop retrieves updates from the `ui_update_queue` and `yield`s `gr.update` objects to modify the UI in real-time (e.g., displaying previews, updating progress messages, showing the generated video).
7.  When the agent signals that the queue is finished (by sending a `queue_finished` flag) or a stop is requested (indicated by the `interrupt_flag` in `shared_state`), the loop breaks. `process_task_queue_and_listen` then yields a final update to refresh the UI state and ensure the queue display is accurate.

**Reasoning for Agents over States:** The shift to an agent-based architecture, with the `ProcessingAgent` managing the worker thread and communicating with the UI via queues, offers several advantages over the previous approach that relied heavily on directly manipulating Gradio state within event handlers:

*   **Improved Responsiveness:** By decoupling the UI event handlers from the long-running generation process, the UI remains responsive even when a task is being processed.  The user can continue to interact with the UI, add tasks, or even cancel the processing, without the UI freezing.
*   **Reduced Complexity:** The agent handles the intricate details of thread management and communication, simplifying the logic within the UI event handlers. The event handlers now primarily focus on responding to user actions and updating the UI based on messages from the agent, rather than directly managing the generation process.
*   **Clearer Separation of Concerns:** The responsibilities are more clearly defined: the UI handles user interaction, the agent manages the processing workflow, and the worker performs the generation. This separation makes the code easier to understand, maintain, and extend.
*   **Enhanced Thread Safety:** The use of queues for communication between the agent and the UI eliminates potential race conditions and ensures thread-safe updates to the UI. All UI updates are now serialized through the `ui_update_queue`, preventing conflicts that could arise from multiple threads modifying the UI state concurrently.
*   **More Robust Error Handling:** The `ProcessingAgent` and `worker_wrapper` provide a structured way to handle errors that occur during generation.  Exceptions in the worker are caught, logged, and communicated back to the UI, allowing for more informative error messages and potential recovery mechanisms.

---

## 4. Module Contracts (File-by-File)

### `src/core/` - The Backend Engine

*   **`generation_core.py`**: **Contract**: Defines the `worker` function, the main entry point for video generation. It must be self-contained and must not import `gradio` or any `ui/` modules except for `shared_state` (for flags) and `metadata` (for saving). It communicates *only* through the `output_queue_ref`, sending updates about its progress, including preview images, progress descriptions, and the path to the final generated video.  It should handle all aspects of the generation process, from initializing the pipeline to saving the final output.
*   **`model_loader.py`**: **Contract**: Responsible for loading all PyTorch models from HuggingFace, configuring them for the detected hardware (VRAM, GPU architecture, compute capability), and storing references to them in the `shared_state_instance.models` dictionary.  It should optimize model loading for the available resources and handle potential errors gracefully, such as insufficient VRAM.
*   **`args.py`**: **Contract**: Defines and parses command-line arguments using `argparse`, providing a consistent way to configure the application's behavior.
*   **`generation_utils.py`**: **Contract**: Contains helper functions extracted from `generation_core.py` for clarity and reuse. These functions perform tasks such as saving previews, calculating CFG schedules, and other calculations or operations used within the generation process.

### `src/ui/` - The Frontend and Glue Logic

*   **`enums.py`**: **Contract**: Defines the `ComponentKey` `StrEnum`. This is a critical file that provides unique, typo-proof identifiers for every UI component. All new components **must** have a key added here to ensure consistent referencing throughout the UI code.
*   **`shared_state.py`**: **Contract**: Defines the `SharedState` singleton and various constant mappings. This serves as the single source of truth for global application state, threading primitives (locks, events), and mappings between UI component keys and their corresponding backend worker parameter names (`UI_TO_WORKER_PARAM_MAP`).  Any globally accessible data or synchronization primitives should be managed here.
*   **`layout.py`**: **Contract**: Defines the entire Gradio UI component tree. Its only job is to create and return a dictionary of all `gr.` components, keyed by `ComponentKey`. It should focus solely on the structure and arrangement of UI elements, without containing any event handling logic.
*   **`switchboard.py`**: **Contract**: The central hub for all UI events. Its primary responsibility is to connect components from `layout.py` to handler functions in other `ui/` modules. It uses Gradio's event binding mechanisms (`.click()`, `.change()`, `.load()`, etc.) to wire up UI interactions to their corresponding logic.  All event wiring should be centralized here to improve readability and maintainability.
*   **`event_handlers.py`**: **Contract**: Contains general-purpose UI event logic that is not specific to a single feature area. This includes functions for:
    *   Updating the interactive states and visual properties of buttons based on the application's state (`update_button_states`).
    *   Handling file uploads from the user, processing the uploaded image, and potentially extracting metadata (`process_upload_and_show_image`).
    *   Clearing the input image and resetting related UI components (`clear_image_action`).
    *   Preparing an image for download, injecting current settings as metadata (`prepare_image_for_download`).
    *   Calculating and displaying the number of segments based on video length and other parameters (`ui_update_total_segments`).
    *   Performing cleanup and save operations before the application shuts down (`safe_shutdown_action`).
*   **`queue_manager.py`**: **Contract**: Implements the `QueueManager` class as a thread-safe singleton. This class is the single source of truth for the task queue and provides methods for:
    *   Managing the queue's state (tasks, processing status, editing status).
    *   Adding new tasks to the queue (`add_task`).
    *   Updating existing tasks (`update_task`).
    *   Removing tasks from the queue (`remove_task`).
    *   Reordering tasks within the queue (`move_task`).
    *   Clearing pending tasks (`clear_pending_tasks`).
    *   Setting and retrieving the currently editing task (`set_editing_task`, `get_task_to_edit`).
    *   Marking a task as complete with a given status (done, error, aborted) and optionally a final output path or error message (`complete_task`).
    *   Loading a new queue, potentially from a saved state (`load_queue`).
    *   Atomically retrieving and marking the next pending task as processing (`get_and_start_next_task`).
    *   Checking if there are any pending tasks in the queue (`has_pending_tasks`).
    The `QueueManager` ensures that all operations on the queue are thread-safe using a `threading.Lock`.
*   **`queue.py`**: **Contract**: Manages the UI-side interactions with the task queue. It contains functions for handling user actions related to the queue and for communicating with the `ProcessingAgent`. Key functions include:
    *   `add_or_update_task_in_queue`: Adds a new task to the queue or updates an existing one based on the user's input. It extracts parameters from the UI controls and uses the `QueueManager` to modify the queue.
    *   `process_task_queue_and_listen`: This is the core function for processing the queue. It starts the `ProcessingAgent` (if not already running), sends a "start" or "stop" signal as needed, and then enters a loop to listen for updates from the agent.  It processes these updates (e.g., progress, previews, task completion) and uses `yield` to send `gr.update` objects back to the UI, enabling real-time feedback.
    *   `cancel_edit_mode_action`: Resets the UI to its default state and exits "edit" mode, clearing any task being edited.
    *   `handle_queue_action_on_select`: Handles actions triggered by selections in the queue display (e.g., moving tasks up or down, removing tasks, or selecting a task to edit). It uses the `QueueManager` to perform the requested action and returns UI updates to reflect the changes.
    *   `save_queue_to_zip`: Saves the current queue state to a zip file, including task parameters and input images.
    *   `load_queue_from_zip`: Loads a queue from a previously saved zip file, restoring the task list and associated data.
    *   `autosave_queue_on_exit_action`:  Automatically saves the queue to a zip file (`AUTOSAVE_FILENAME`) when the application exits, ensuring no work is lost.
*   **`queue_helpers.py`**: **Contract**: Contains helper functions for `queue.py`, primarily focused on formatting the queue data for display in the Gradio DataFrame (`update_queue_df_display`) and converting NumPy arrays to base64 URIs for image display (`np_to_base

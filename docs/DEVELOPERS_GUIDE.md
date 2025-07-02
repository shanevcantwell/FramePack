# Developer's Guide to Extending 'goan'

## 1. Introduction & Core Philosophy

This document provides a technical overview of the `goan` application architecture. It is intended for developers looking to understand, maintain, or extend the codebase.

The application is built on four core principles:

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
    *   **`ui/queue_manager.py`**: This singleton class manages the task queue itself, providing thread-safe operations for adding, updating, removing, reordering, and retrieving tasks. It acts as the single source of truth for the queue's state.
    *   **`ui/queue.py`**: Contains the UI-side logic for interacting with the task queue. It houses functions for handling user actions on the queue (e.g., adding, editing, deleting tasks) and for starting/stopping the processing of the queue. It communicates with the `ProcessingAgent` to control the backend worker.
    *   **`ui/event_handlers.py`**: Contains general-purpose UI event logic, such as updating button states (`update_button_states`), handling image uploads, and managing UI state transitions.
    *   **`ui/queue_helpers.py`**: Provides helper functions for `queue.py`, primarily for formatting queue data for display in the Gradio DataFrame and converting data types for UI elements.
    *   **`ui/agents.py`**: Defines the `ProcessingAgent` class, which manages the interaction with the backend worker. It handles starting, stopping, pausing, and monitoring tasks, and uses message queues to communicate bidirectionally with both the UI and the worker thread.

---

## 3. Key Process Flows

### Application Startup

1.  **`goan.py`**: The main entry point.
2.  **`core/args.py`**: Parses command-line arguments.
3.  **`core/model_loader.py`**: Detects GPU capabilities, loads all necessary models to the CPU, configures them for the environment (e.g., VRAM, dtype), and populates the `shared_state_instance.models` dictionary. The large transformer model is loaded lazily on first use.
4.  **`ui/layout.py`**: `create_ui()` is called to build the Gradio component tree.
5.  **`ui/switchboard.py`**: `wire_all_events()` is called, which registers all `.click()`, `.change()`, and `.load()` event listeners, connecting UI components to their corresponding handler functions.
6.  **`block.launch()`**: The Gradio server is started, making the UI accessible in a web browser.
7.  **On UI Load**: The `.load()` event wired in `switchboard_startup.py` triggers a chain of functions:
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

### Processing the Queue (Start, Stop, and Pause)

1.  A user clicks the "▶️ Process Queue" button (`K.PROCESS_QUEUE_BUTTON`), the "⏹️ Stop Processing" button, or the "⏸️" icon in the queue.
2.  The `switchboard` calls the appropriate handler (`queue_processing.process_task_queue_and_listen` for start/stop, `queue.handle_queue_action_on_select` for pause).
3.  The handler sends a message (`"start"`, `"stop"`, or `"pause"`) to the `ProcessingAgent`'s mailbox.
4.  The `ProcessingAgent`'s main loop receives the message and acts accordingly:
    *   **On "start"**: It begins its `_processing_loop`, retrieving tasks from the `QueueManager` and launching the `worker` function in a separate thread.
    *   **On "stop"**: It sets the global `interrupt_flag`. The `worker` checks this flag between segments and inside the diffusion loop, causing it to exit gracefully.
    *   **On "pause"**: It sets a `self.pause_requested` flag to `True` and then sets the global `interrupt_flag`, causing the worker to exit gracefully just like a stop.
5.  The `worker` performs the video generation. When it is interrupted (by the `interrupt_flag`), it enters its `except (InterruptedError, KeyboardInterrupt)` block.
6.  Inside this block, it sends a final message, `('interrupted_with_state', (task_id, history_latents_for_abort))`, to the agent, reporting its final latent state before exiting. It then signals `('aborted', ...)` to the UI.
7.  The `ProcessingAgent`'s `_processing_loop` receives these messages from the worker.
8.  When it receives the `aborted` signal, it checks its internal `self.pause_requested` flag.
    *   **If `True`**: The agent knows this was a pause. It takes the `latents_for_pause` it received earlier and calls `generation_utils.save_resume_state()` to package the `.goan_resume` file. It then sends a `('paused', (task_id, resume_file_path))` message to the UI.
    *   **If `False`**: It was a normal stop. It does nothing further.
9.  The UI listener (`queue_processing.process_task_queue_and_listen`) receives messages from the agent via the `ui_update_queue` and `yield`s `gr.update` objects to modify the UI in real-time (e.g., displaying previews, updating progress, or triggering the download of a resume file).

**Reasoning for Agents over States:** The shift to an agent-based architecture, with the `ProcessingAgent` managing the worker thread and communicating with the UI via queues, offers several advantages over the previous approach that relied heavily on directly manipulating Gradio state within event handlers:

*   **Improved Responsiveness:** By decoupling the UI event handlers from the long-running generation process, the UI remains responsive even when a task is being processed. The user can continue to interact with the UI, add tasks, or even cancel the processing, without the UI freezing.
*   **Reduced Complexity:** The agent handles the intricate details of thread management and communication, simplifying the logic within the UI event handlers. The event handlers now primarily focus on responding to user actions and updating the UI based on messages from the agent, rather than directly managing the generation process.
*   **Clearer Separation of Concerns:** The responsibilities are more clearly defined: the UI handles user interaction, the agent manages the processing workflow, and the worker performs the generation. This separation makes the code easier to understand, maintain, and extend.
*   **Enhanced Thread Safety:** The use of queues for communication between the agent and the UI eliminates potential race conditions and ensures thread-safe updates to the UI. All UI updates are now serialized through the `ui_update_queue`, preventing conflicts that could arise from multiple threads modifying the UI state concurrently.
*   **More Robust Error Handling:** The `ProcessingAgent` and `worker_wrapper` provide a structured way to handle errors that occur during generation. Exceptions in the worker are caught, logged, and communicated back to the UI, allowing for more informative error messages and potential recovery mechanisms.

---

## 4. Module Contracts (File-by-File)

### `src/core/` - The Backend Engine

*   **`generation_core.py`**: **Contract**: Defines the `worker` function. It must be self-contained and must not import `gradio`. It communicates *only* through the `output_queue_ref`. When interrupted, it **must** send a final `('interrupted_with_state', ...)` message containing its last known latent state before signaling `('aborted', ...)`.
*   **`model_loader.py`**: **Contract**: Responsible for loading all PyTorch models, configuring them for the detected hardware, and storing references in `shared_state_instance.models`.
*   **`args.py`**: **Contract**: Defines and parses command-line arguments.
*   **`generation_utils.py`**: **Contract**: Contains helper functions extracted from `generation_core.py`. This includes the `save_resume_state` function, which packages the `.goan_resume` file.

### `src/ui/` - The Frontend and Glue Logic

*   **`enums.py`**: **Contract**: Defines the `ComponentKey` `StrEnum`. All new components **must** have a key added here.
*   **`shared_state.py`**: **Contract**: Defines the `SharedState` singleton, global threading primitives, and the `UI_TO_WORKER_PARAM_MAP`.
*   **`layout.py`**: **Contract**: Defines the entire Gradio UI component tree. It should contain no event handling logic.
*   **`switchboard.py`**: **Contract**: The central hub for all UI events. All event wiring (`.click()`, `.change()`, etc.) should be centralized here.
*   **`agents.py`**: **Contract**: Defines the `ProcessingAgent`. It is responsible for the entire lifecycle of a generation task: starting, stopping, and pausing. It receives simple commands from the UI and orchestrates the complex interactions with the `worker` thread. It is responsible for deciding whether an interruption is a "stop" or a "pause" and for initiating the save of a resume file.
*   **`queue_manager.py`**: **Contract**: Implements the `QueueManager` class as a thread-safe singleton. This is the single source of truth for the task queue's data and state.
*   **`queue.py`**: **Contract**: Manages the UI-side interactions with the task queue. It sends commands to the `ProcessingAgent` and handles user actions like editing, reordering, and deleting tasks.
*   **`queue_processing.py`**: **Contract**: Contains the `process_task_queue_and_listen` generator function. Its sole purpose is to listen for messages from the `ProcessingAgent`'s `ui_update_queue` and `yield` `gr.update` objects to the UI.
*   **`event_handlers.py`**: **Contract**: Contains general-purpose UI event logic not specific to a single feature, like `update_button_states`.
*   **`workspace.py`**: **Contract**: Manages saving and loading of settings. Contains the `handle_file_drop` function, which is the entry point for restoring a job from a `.goan_resume` file.
*   **`queue_helpers.py`**: **Contract**: Contains helper functions for `queue.py`, primarily for formatting the queue data for display.

# Developer's Guide to Extending 'goan'

## 1. Introduction & Core Philosophy

This document provides a technical overview of the `goan` application architecture. It is intended for developers looking to understand, maintain, or extend the codebase.

The application is built on four core principles:

1.  **Separation of Concerns**: The user interface (UI) is strictly separated from the backend generation engine. The UI's role is to capture user intent and display results, while the backend's role is to execute generation tasks.
2.  **Centralized State Management**: Critical application state (threading events, loaded models, system info) is managed by the `SharedState` singleton. The state of the task queue itself is managed exclusively by the `QueueManager` singleton, providing a single, thread-safe source of truth.
3.  **Decoupled UI Updates**: The UI is updated through two distinct patterns:
    * **Asynchronous Updates**: During long-running generation, the `ProcessingAgent` sends messages to the UI via a dedicated queue to report progress, display previews, and signal completion.
    * **Synchronous Updates**: For immediate user actions (e.g., adding a task, loading a file), event handlers return a dictionary of `gradio.update` objects, which the switchboard then applies to the UI.
4.  **Declarative Event Wiring**: All UI event handling (e.g., button clicks, file uploads) is wired in dedicated `switchboard_*.py` modules. This makes it easy to understand the cause-and-effect relationships in the UI without searching through multiple files.

---

## 2. Core Architecture Overview

The application can be understood as three main layers:

* **Frontend (Gradio View Layer)**: Defined in `ui/layout.py`. This layer is responsible only for creating and arranging the Gradio components. It is a "dumb" layer with no event logic.

* **Backend (Worker Engine)**: The `worker` function in `core/generation_core.py` is the heart of the generation process. It runs in a separate thread and is completely decoupled from Gradio. It accepts a dictionary of parameters and communicates its progress back to the `ProcessingAgent` via a message queue.

* **UI Logic (The "Glue" Layer)**: This layer bridges the gap between the Frontend and the Backend and resides in the `src/ui/` directory.
    * **`ui/shared_state.py`**: A singleton that holds global, thread-safe state like threading locks, events (`interrupt_flag`), and references to the loaded PyTorch models. It also contains critical mappings (`UI_TO_WORKER_PARAM_MAP`) that define how UI controls correspond to backend parameters.
    * **`ui/switchboard.py`**: The central hub that orchestrates the wiring of all UI events by calling the specialized `switchboard_*.py` modules.
    * **`ui/queue_manager.py`**: A thread-safe singleton class that exclusively manages the task queue's data and state (adding, removing, reordering, etc.).
    * **`ui/queue.py`**: Contains the *handler functions* for user actions on the queue, such as `add_or_update_task_in_queue`. These handlers interact with the `QueueManager` singleton.
    * **`ui/agents.py`**: Defines the `ProcessingAgent`, a singleton that manages the entire lifecycle of the backend worker (starting, stopping). It communicates with the UI via the global `ui_update_queue`.
    * **`ui/switchboard_helpers.py`**: A new module containing helper functions for the switchboards, primarily `apply_updates`, which maps dictionaries to UI component updates.

---

## 3. Key Process Flow: The "Handler-Returns-Dict" Pattern

All synchronous UI events now follow a robust, decoupled pattern. The file drop event is a perfect example:

1.  A user drops an image file onto the `gr.File` component.
2.  The `.upload()` event in `ui/switchboard_image.py` is triggered.
3.  The event calls the handler function, `workspace.handle_file_drop()`.
4.  The `handle_file_drop()` function performs its logic (opens the image, checks metadata) and **returns a dictionary** where keys are `ComponentKey` enums and values are `gr.update` objects (e.g., `{K.INPUT_IMAGE_DISPLAY: gr.update(value=pil_image)}`).
5.  The switchboard receives this dictionary into a temporary `gr.State` object.
6.  A `.then()` block calls the `switchboard_helpers.apply_updates` function, which takes the dictionary and maps its values to the full list of expected UI output components.

This pattern ensures the handler function is simple and decoupled, while the switchboard remains the single source of truth for UI wiring.

---

## 4. Module Contracts (File-by-File)

### `src/core/` - The Backend Engine

* **`generation_core.py`**: **Contract**: Defines the `worker` function. It must be self-contained and must not import `gradio`. It communicates *only* through the `output_queue_ref` passed to it.
* **`model_loader.py`**: **Contract**: Responsible for loading all PyTorch models, detecting hardware capabilities (`is_legacy_gpu`), setting model `dtype` appropriately, and storing references in `shared_state_instance`.
* **`generation_utils.py`**: **Contract**: Contains helper functions for the `worker`, such as `handle_segment_saving`.

### `src/ui/` - The Frontend and Glue Logic

* **`enums.py`**: **Contract**: Defines the `ComponentKey` `StrEnum`. All new UI components **must** have a key added here.
* **`shared_state.py`**: **Contract**: Defines the `SharedState` singleton, global threading primitives, parameter maps, and the `IS_LEGACY_GPU_KEY` constant.
* **`layout.py`**: **Contract**: Defines the entire Gradio UI component tree. Contains no event handling logic.
* **`switchboard.py`**: **Contract**: The central hub that calls all other `switchboard_*.py` modules to wire their respective events.
* **`switchboard_*.py` (all of them)**: **Contract**: Define the `outputs` list for each event and use the "Handler-Returns-Dict" pattern with `switchboard_helpers.apply_updates` to wire events to handlers.
* **`switchboard_helpers.py`**: **Contract**: Provides helper functions like `apply_updates` to be used by all switchboard modules to reduce code duplication.
* **`agents.py`**: **Contract**: Defines the `ProcessingAgent`. Its responsibility is to manage the lifecycle of the `worker` thread. It should not be involved in synchronous UI updates.
* **`queue_manager.py`**: **Contract**: Implements the `QueueManager` singleton. This is the **only** class that should directly modify the queue's internal state (`self.state["queue"]`).
* **`queue.py`, `workspace.py`, `event_handlers.py`**: **Contract**: These modules contain the handler functions for UI events. Any function called directly by a switchboard event **must** return a `dict` of updates.
# Developer's Guide to Extending 'goan'

## 1. Introduction & Core Philosophy

This document provides a technical overview of the `goan` application architecture. It is intended for developers looking to understand, maintain, or extend the codebase.

The application is built on four core principles:

1.  **Separation of Concerns**: The user interface (UI) is strictly separated from the backend generation engine. The UI's role is to capture user intent and display results, while the backend's role is to execute generation tasks.
2.  **Centralized State Management**: Critical application state (threading events, loaded models, system info) is managed by the `SharedState` singleton. The state of the task queue itself is managed exclusively by the `QueueManager` singleton, providing a single, thread-safe source of truth.
3.  **Decoupled UI Updates**: The UI is updated through two distinct patterns:
    * **Asynchronous Updates**: During long-running generation, the `ProcessingAgent` sends messages to the UI via a dedicated queue to report progress and display previews.
    * **Synchronous Updates**: For immediate user actions (e.g., adding a task), event handlers return a tuple of `gradio.update` objects, which the switchboard then applies to the UI.
4.  **Declarative Event Wiring**: All UI event logic is defined in a single, declarative `switchboard.json` file. This blueprint is processed at startup by a central runner in `switchboard.py`, which programmatically builds all event listeners.

---

## 2. Core Architecture Overview

The application can be understood as three main layers:

* **Frontend (Gradio View Layer)**: Defined in `ui/layout.py`. This layer is responsible only for creating and arranging the Gradio components. It contains no event logic. At the end of its `create_ui` function, it makes a single call to `switchboard.wire_all_events()` to build the UI's interactivity.

* **Backend (Worker Engine)**: The `worker` function in `core/generation_core.py` is the heart of the generation process. It runs in a separate thread and is completely decoupled from Gradio. It communicates its progress back to the `ProcessingAgent` via a message queue.

* **UI Logic (The "Glue" Layer)**: This layer bridges the gap between the Frontend and the Backend.
    * **`ui/shared_state.py`**: A singleton that holds global, thread-safe state, parameter-to-UI-key mappings, and system information.
    * **`ui/switchboard.json`**: A declarative JSON file that serves as the **single source of truth** for all UI event wiring. It defines every component, event, handler, input, and output.
    * **`ui/switchboard.py`**: The central **runner** that parses `switchboard.json` at startup and dynamically builds the entire graph of Gradio event listeners.
    * **`ui/queue_manager.py`**: A thread-safe singleton class that exclusively manages the task queue's data and state.
    * **`ui/agents.py`**: Defines the `ProcessingAgent`, a singleton that manages the lifecycle of the backend `worker` thread.
    * **Handler Modules (`ui/queue.py`, `ui/workspace.py`, `ui/event_handlers.py`, etc.)**: These modules contain the Python functions that are executed when UI events are triggered.

---

## 3. Key Process Flow: Declarative Event Wiring

The application's interactivity is built dynamically at startup.

1.  The main `goan.py` script calls `ui.layout.create_ui()`.
2.  The `create_ui()` function defines all Gradio components and stores them in a `components` dictionary.
3.  As its final step inside the `with gr.Blocks()` context, `create_ui()` calls `switchboard.wire_all_events(components)`.
4.  The `wire_all_events()` function in `switchboard.py` loads and parses the `switchboard.json` blueprint.
5.  The runner iterates through the event definitions in the JSON. For each definition, it:
    * Finds the trigger component in the `components` dictionary.
    * Resolves the handler function string (e.g., `"workspace_manager.handle_file_drop"`) to a callable Python function.
    * Resolves the `inputs` and `outputs` string lists into lists of actual Gradio components.
    * Dynamically constructs the event listener (e.g., `.click()`) and its entire `.then()` chain.
6.  Once the loop is complete, the entire UI is interactive.

---

## 4. Module Contracts (File-by-File)

### `src/core/` - The Backend Engine

* **`generation_core.py`**: **Contract**: Defines the `worker` function. Must not import `gradio`. Communicates *only* through the `output_queue_ref` passed to it.
* **`model_loader.py`**: **Contract**: Loads all models, detects hardware, sets `dtype`, and populates `shared_state_instance`.
* **`generation_utils.py`**: **Contract**: Contains pure helper functions for the `worker`.

### `src/ui/` - The Frontend and Glue Logic

* **`enums.py`**: **Contract**: Defines the `ComponentKey` `StrEnum`. All UI components must have a key here.
* **`shared_state.py`**: **Contract**: Defines the `SharedState` singleton and constants for UI-to-worker parameter mapping.
* **`layout.py`**: **Contract**: Defines the Gradio component tree and makes the single call to `switchboard.wire_all_events()` at the end of the `with gr.Blocks()` context.
* **`switchboard.json`**: **Contract**: Defines the entire UI event graph in a declarative JSON format. It is the single source of truth for all event wiring.
* **`switchboard.py`**: **Contract**: Contains the runner that parses `switchboard.json` and dynamically builds all Gradio event listeners at startup.
* **`agents.py`**: **Contract**: Defines the `ProcessingAgent` for managing the `worker` thread lifecycle. Not involved in synchronous UI updates.
* **`queue_manager.py`**: **Contract**: Implements the `QueueManager` singleton. The only class that should directly modify the queue state.
* **Handler Modules (`queue.py`, `workspace.py`, etc.)**: **Contract**: Contain the handler functions. Any function called by the switchboard **must** return a `tuple` whose length exactly matches the number of components in the corresponding `outputs` list in `switchboard.json`.
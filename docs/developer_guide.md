# Goan UI Architecture Developer's Guide

This document outlines the architecture of the Goan Gradio UI. The primary goal of this architecture is to manage complexity through modularity and a clear separation of concerns, making the codebase easier to understand, maintain, and extend.

## Core Concepts

The architecture is built on a few core concepts that work together to create a robust and scalable system.

### 1. The Orchestrator Pattern

Instead of a single, monolithic file that wires up every UI event, we use an orchestrator pattern.

- **`switchboard.py`**: This is the main orchestrator. Its *only* job is to import the specialized `switchboard_*` modules and call their `wire_events()` function. It acts as a clean, high-level dispatcher and contains no event logic itself.

- **`switchboard_*` Modules**: The event wiring logic is broken down into smaller, feature-focused modules (e.g., `switchboard_queue.py`, `switchboard_image.py`). Each module is responsible for wiring up the events for a specific panel or feature of the UI. This keeps related logic together and makes it easy to find and modify.

### 2. ComponentKeys and the `components` Dictionary

- **`enums.py`**: Defines the `ComponentKey` `StrEnum`. This is the single source of truth for all UI component identifiers. Using an enum prevents the use of "magic strings," provides type-safety, and makes the code self-documenting.

- **`layout.py`**: This module defines the visual structure of the UI. It instantiates all Gradio components and stores them in a single `components` dictionary, where the keys are the `ComponentKey` enums. This dictionary is then passed to all switchboard modules.

### 3. Declarative State Logic

We strive to make UI updates a function of the application's state, rather than imperatively updating components from dozens of different places.

- **`event_handlers.update_button_states`**: This function is a prime example of the declarative approach. It implements a "rules engine" that evaluates the current application state (e.g., *is processing?*, *is editing?*, *is an image loaded?*) and returns a complete set of UI updates for all major buttons. Event chains simply call this function in a `.then()` block, trusting it to figure out the correct button states. This centralizes all button logic into one place.

## Key Modules and Their Responsibilities

| Module(s) | Responsibility | Description |
| :--- | :--- | :--- |
| **`layout.py`** | **View** | Defines the static layout of all Gradio components and creates the master `components` dictionary. |
| **`enums.py`** | **Identifiers** | Provides the `ComponentKey` enum for safe and consistent access to UI components. |
| **`switchboard.py`** | **Orchestrator** | The main entry point for event wiring. Delegates all work to the specialized `switchboard_*` modules. |
| **`switchboard_*`** | **Wiring** | Connects user actions (e.g., `button.click`) to handler functions. Defines the `inputs` and `outputs` for each event. |
| **`event_handlers.py`** | **UI-Centric Logic** | Contains functions that are directly called by the switchboard. These functions often deal with UI state changes, like the button rules engine. |
| **`queue.py`, `workspace.py`, `lora.py`, `metadata.py`** | **Business Logic** | Contain the core application logic for their respective domains. They handle data manipulation, file I/O, and interaction with state managers. |
| **`queue_manager.py`, `shared_state.py`** | **State Management** | Singleton instances that hold the application's state in a structured, thread-safe manner (e.g., the task queue, threading events). |
| **`agents.py`, `queue_processing.py`** | **Async Backend** | Manages the long-running generation process. The `ProcessingAgent` runs in a separate thread, and `queue_processing.py` uses a `queue` to stream updates back to the UI for live previews and progress bars. |

## Anatomy of an Event: Adding a Task

To understand how these pieces fit together, let's trace the "Add Task to Queue" event:

1.  **User Action**: The user clicks the "Add Task to Queue" button.

2.  **Switchboard Wiring**: `switchboard_queue.py` has wired this `click` event to the `queue.add_or_update_task_in_queue` function. It specifies all the necessary UI controls as `inputs` and the components that need to be updated as `outputs`.

3.  **Business Logic**: The `add_or_update_task_in_queue` function in `queue.py` receives the values from the UI. It packages them into a task dictionary and calls `queue_manager_instance.add_task()`.

4.  **State Mutation**: The `QueueManager` singleton locks its state and appends the new task to its internal `queue` list.

5.  **Direct UI Update**: The `add_or_update_task_in_queue` function returns a list of `gr.update()` objects. A key one is a call to `queue_helpers.update_queue_df_display()`, which redraws the queue table with the new task.

6.  **Chained UI Updates (`.then()`)**: The event chain in `switchboard_queue.py` doesn't stop there. It uses `.then()` to trigger subsequent updates:
    - It calls `event_handlers.update_button_states`. This function reads the new state from the `QueueManager`, sees that the queue now has tasks, and returns updates to enable the "Process Queue" and "Save Queue" buttons.
    - It calls `event_handlers.ui_update_total_segments` to ensure the segment count display is refreshed based on the newly added task's settings.

This flow ensures that logic is handled in the appropriate layer and that the UI remains consistent after every action.

## How to Add a New Feature (e.g., a New Slider)

1.  **Add the Key**: Add a new entry for your slider in `ui/enums.py`.
    ```python
    class ComponentKey(StrEnum):
        ...
        MY_NEW_SLIDER = auto()
    ```

2.  **Add to Layout**: Instantiate the `gr.Slider` in `ui/layout.py` and add it to the `components` dictionary.
    ```python
    # in create_ui()
    components[K.MY_NEW_SLIDER] = gr.Slider(label="My New Setting", ...)
    ```

3.  **Wire the Event**: Decide which switchboard module is most appropriate. If the slider's value needs to be recalculated on change, add a `.change()` event in that module (e.g., `switchboard_misc.py`).
    ```python
    # in switchboard_misc.py
    components[K.MY_NEW_SLIDER].change(
        fn=my_new_handler_function,
        inputs=[...],
        outputs=[...]
    )
    ```

4.  **Create the Handler**: Implement `my_new_handler_function` in the appropriate logic module (e.g., `event_handlers.py` if it's purely a UI calculation).

5.  **Include in State**: If the slider's value needs to be saved in workspaces or tasks, add its `ComponentKey` to the relevant key lists in `ui/shared_state.py` (e.g., `ALL_TASK_UI_KEYS`) and map it to a worker parameter name in `UI_TO_WORKER_PARAM_MAP`.

By following this pattern, new features can be integrated cleanly without disrupting the existing structure.
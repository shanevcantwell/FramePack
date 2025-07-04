# Refactoring Plan: Migrating to a "Handler-Returns-Dict" Architecture

## 1. Objective

The goal of this refactor is to overhaul the Gradio event handling logic across the entire `goan` application. We will replace the current brittle system, where handler functions must return perfectly ordered tuples of updates, with a robust and maintainable **"Handler-Returns-Dict"** pattern.

This will eliminate `ValueError` exceptions caused by return signature mismatches, remove "magic numbers," and formally decouple the business logic (handlers) from the UI wiring (switchboards).

---

## 2. The Architectural Pattern

The new architecture consists of two parts:

* **The Handler Function**:
    * **Contract**: A function responsible for a specific action (e.g., processing a file, adding a task).
    * **Return Value**: It **must** return a single Python `dict`. The keys of this dictionary are `ComponentKey` enums, and the values are the corresponding `gr.update()` objects. The handler no longer needs to know the number or order of UI components.

* **The Switchboard Wiring**:
    * **Contract**: The switchboard module is the **single source of truth** for which UI components an event affects.
    * **Implementation**: It defines the list of component keys for an event's outputs. It uses a helper function in a `.then()` block to map the dictionary returned by the handler to this list of components.

---

## 3. Action Plan: Module-by-Module Refactor

### Step 3.1: Create a New `switchboard_helpers.py`

This new file will contain a helper function to keep our switchboard code DRY (Don't Repeat Yourself).

* **File**: `src/ui/switchboard_helpers.py`
* **Action**: Create the file and add the following function.

```python
# src/ui/switchboard_helpers.py
import gradio as gr
from typing import Dict, List

def apply_updates(update_dict: Dict, output_keys: List) -> List:
    """
    Takes a dictionary of updates and a list of output keys, and returns a list
    of gr.update() objects in the correct order for the UI.
    """
    if not isinstance(update_dict, dict):
        # If the handler returned something other than a dict, return no-op updates
        # to prevent a crash, and log a warning.
        print(f"WARNING: apply_updates expected a dict but got {type(update_dict)}. Returning no-op updates.")
        return [gr.update() for _ in output_keys]
    return [update_dict.get(key, gr.update()) for key in output_keys]
# Refactoring Summary & Decoder Ring

## 1. Project Summary & Philosophy

We've performed a significant refactoring of the UI's "contract" (`enums.py`) to improve code quality, clarity, and maintainability. The core philosophy is that a UI for power users should be powerful and explicit, not confusing or cluttered.

**Key Changes:**
- **Descriptive Naming:** All UI component keys now have clear, descriptive names that indicate their function and type (e.g., `_SLIDER`, `_CHECKBOX`). Ambiguous abbreviations (`GS`, `CFG`, `DF`) have been eliminated.
- **Consistency:** The inconsistent `_UI` suffix has been removed, and component types have been added for clarity.
- **Decluttering:** All non-functional or placeholder UI elements (like `RELAUNCH_BUTTON`) have been removed from the contract and will be removed from the layout.

---

## 2. The Decoder Ring: Old vs. New Enum Keys

This table maps the old, often confusing, component keys to their new, explicit counterparts.

| Old Name | New Name | Rationale |
| :--- | :--- | :--- |
| `PROMPT_UI` | `POSITIVE_PROMPT` | More explicit. |
| `N_PROMPT_UI` | `NEGATIVE_PROMPT` | More explicit. |
| `TOTAL_SECOND_LENGTH_UI` | `VIDEO_LENGTH_SLIDER` | Clearer function and type. |
| `GS_UI` | `DISTILLED_CFG_START_SLIDER` | Hunyuan abbreviation -> Descriptive name. |
| `GS_SCHEDULE_SHAPE_UI` | `VARIABLE_CFG_SHAPE_RADIO` | Hunyuan abbreviation -> Descriptive name. |
| `GS_FINAL_UI` | `DISTILLED_CFG_END_SLIDER` | Hunyuan abbreviation -> Descriptive name. |
| `CFG_UI` | `REAL_CFG_SLIDER` | Hunyuan abbreviation -> Descriptive name. |
| `RS_UI` | `GUIDANCE_RESCALE_SLIDER` | Hunyuan abbreviation -> Descriptive name. |
| `QUEUE_DF_DISPLAY_UI` | `QUEUE_DATAFRAME` | "DF" was ambiguous. |
| `SEGMENTS_TO_DECODE_CSV_UI`| `PREVIEW_SPECIFIED_SEGMENTS_TEXTBOX` | More descriptive name and type. |
| `OUTPUT_FOLDER_UI_CTRL` | `OUTPUT_FOLDER_TEXTBOX` | Standardized naming and added type. |
| `USE_TEACACHE_UI` | `USE_TEACACHE_CHECKBOX` | Added component type for clarity. |
| `USE_STANDARD_FPS_CHECKBOX_UI`| `FORCE_STANDARD_FPS_CHECKBOX` | More user-friendly name. |
| `RELAUNCH_BUTTON_UI` | *(Removed)* | Non-functional and not feasible. |
| `RESET_UI_BUTTON` | *(Removed)* | Non-functional. |
| `SAVE_WORKSPACE_BUTTON` | *(Removed)* | Non-functional. |
| `LOAD_WORKSPACE_BUTTON` | *(Removed)* | Non-functional. |
| `SHUTDOWN_BUTTON` | *(Removed)* | Non-functional. |

---

## 3. Loose Ends & Design Decisions

Here's a summary of the two complex behaviors we discussed and the decided path forward.

### Re-awakening the UI after an Error
- **Problem:** If a task fails, the UI controls remain disabled, forcing a manual refresh.
- **Solution:** The agent-driven architecture is the key. When the `ProcessingAgent` finishes its work (on success, failure, or abort), it sends a `queue_finished` message. The UI listener (`queue_processing.py`) that receives this message must trigger a final, comprehensive UI state update.
- **Action:** Ensure the `.then()` clause after the `process_task_queue_and_listen` call in the switchboard *always* calls `event_handlers.update_button_states`. This will re-evaluate the state of all buttons based on the final queue state, effectively "re-awakening" the UI.

### Server Restart for Settings Changes
- **Problem:** Changing the output folder requires a server restart to take effect, which is not feasible to trigger from the UI.
- **Solution:** We will not attempt a programmatic server restart. Instead, we will adopt the standard "reboot necessary" UX pattern.
- **Action:** When the "Save as Default" button is clicked and the output folder has changed, we will make the `RELAUNCH_NOTIFICATION_MD` component visible. This markdown element will inform the user that a manual restart is required for the change to apply, just like a standard application installer.

---

## 4. Immediate TODOs

The next step is to propagate the new, clean `enums.py` contract throughout the entire application. This involves a systematic search-and-replace across the following key files:

1.  **`ui/layout.py`**: Update all `components[K.OLD_NAME]` to `components[K.NEW_NAME]`. Remove the non-functional buttons.
2.  **`ui/shared_state.py`**: Update the `CREATIVE_UI_KEYS`, `ENVIRONMENT_UI_KEYS`, and `UI_TO_WORKER_PARAM_MAP` lists to use the new enum keys.
3.  **`ui/workspace.py`**: Update `get_default_values_map` and all other functions that reference UI keys.
4.  **`core/generation_core.py`**: Update the `worker` function signature to use the new, more descriptive parameter names (e.g., `negative_prompt` instead of `n_prompt`).
5.  **`ui/event_handlers.py`**: Update all functions, especially `update_button_states`, to use the new keys.
6.  **All `switchboard_*.py` files**: Update all `inputs` and `outputs` lists in the event wiring to use the new keys.

This systematic update will bring the entire codebase in line with our new, much-improved contract.

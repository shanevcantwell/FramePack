# ui/switchboard.py
# This module dynamically wires all Gradio UI events from a declarative JSON map.

import json
import logging
import gradio as gr
import os

# Import all modules that contain handler functions to make them discoverable.
from . import (
    event_handlers,
    workspace,
    queue as queue_actions,
    queue_processing,
    lora as lora_manager,
    metadata as metadata_manager,
    shared_state
)
from .enums import ComponentKey as K

logger = logging.getLogger(__name__)

# A centralized map to resolve handler strings to their module locations.
# This makes the JSON blueprint cleaner and avoids repetitive imports.
HANDLER_MODULES = {
    "event_handlers": event_handlers,
    "workspace_manager": workspace,
    "queue_actions": queue_actions,
    "queue_processing": queue_processing,
    "lora_manager": lora_manager,
    "metadata_manager": metadata_manager,
    # A special entry for miscellaneous, one-off lambda functions
    "switchboard_misc": workspace # Placeholder, as update_scheduler_visibility is in workspace for now
}

def _resolve_handler(handler_str: str | None):
    """
    Resolves a handler string (e.g., 'workspace_manager.load_workspace') to a callable function.
    Handles regular functions and simple, one-line lambda expressions from the JSON map.
    """
    if not handler_str:
        return None

    # Handle lambda expressions defined directly in the JSON.
    # eval() is used here in a controlled context where the JSON source is trusted.
    if handler_str.startswith("lambda"):
        return eval(handler_str, {"gr": gr})

    # Resolve standard module.function handlers.
    try:
        module_name, func_name = handler_str.split('.')
        module = HANDLER_MODULES.get(module_name)
        if not module:
            # A fallback for simple function calls that may not be in a module
            if func_name in globals():
                return globals()[func_name]
            raise ValueError(f"Module '{module_name}' not found in HANDLER_MODULES.")
        return getattr(module, func_name)
    except (ValueError, AttributeError) as e:
        logger.error(f"Failed to resolve handler: {handler_str}. Error: {e}")
        raise

def _resolve_io_list(io_str_list: list | None, components: dict, temp_states: dict, collections: dict):
    """
    Resolves a list of input/output strings from the JSON map into actual Gradio
    components, temporary states, or expanded collections.
    """
    if io_str_list is None:
        return None

    resolved_list = []
    for item_str in io_str_list:
        if isinstance(item_str, str) and item_str.startswith("K."):
            key = K[item_str.split('.')[1]]
            resolved_list.append(components[key])
        
        # This block now correctly receives the 'collections' map to look up references.
        elif isinstance(item_str, str) and item_str.startswith("coll:"):
            collection_name = item_str.split(':')[1]
            key_list = collections.get(collection_name, [])
            resolved_list.extend(_resolve_io_list(key_list, components, temp_states, collections))
            
        elif isinstance(item_str, str) and item_str.startswith("gr.State"):
            resolved_list.append(eval(item_str, {"gr": gr, "shared_state_module": shared_state}))
        elif isinstance(item_str, str) and item_str.startswith("ref:"):
            resolved_list.append(temp_states[item_str])
        else:
            raise ValueError(f"Could not resolve IO item: {item_str}")

    return resolved_list

def wire_all_events(components: dict):
    """
    Dynamically wires all Gradio UI events based on the declarative switchboard.json.
    This function reads the map, resolves all components and handlers, and builds
    the complete event listener graph with all `.then()` chains.
    """
    # Build a path to the JSON file relative to this script's location
    script_dir = os.path.dirname(__file__)
    json_path = os.path.join(script_dir, "switchboard.json")
    
    try:
        # Use the full path to open the file
        with open(json_path, "r", encoding='utf-8') as f:
            event_map = json.load(f)
    except FileNotFoundError:
        logger.error(f"FATAL: {json_path} not found. UI events will not be wired.")
        return
    except json.JSONDecodeError as e:
        logger.error(f"FATAL: Could not parse switchboard.json. Check for syntax errors. Error: {e}")
        return

    # FIX: Extract the collections map from the event map before looping.
    collections_map = event_map.get("collections", {})

    logger.info("Starting to wire events from event_map.json...")

    for switchboard_name, event_definitions in event_map.items():
        # FIX: Skip the top-level 'collections' key so it is not processed as a list of events.
        if switchboard_name == "collections":
            continue

        for i, event_def in enumerate(event_definitions):
            try:
                component_key_str = event_def["component"]
                component_key = K[component_key_str.split('.')[1]]
                trigger_component = components[component_key]
                event_name = event_def["event"]
                chained_calls = event_def["chained_calls"]

                if not chained_calls:
                    continue

                temp_states_for_chain = {}
                for call_idx, call_info in enumerate(chained_calls):
                    outputs = call_info.get("outputs")
                    if not outputs: continue
                    for output_idx, _ in enumerate(outputs):
                        ref_key = f"ref:chained_calls[{call_idx}].outputs[{output_idx}]"
                        is_referenced = any(
                            ref_key in str(future_call.get("inputs", []))
                            for future_call in chained_calls[call_idx + 1:]
                        )
                        if is_referenced:
                            temp_states_for_chain[ref_key] = gr.State()

                listener = None
                for call_idx, call_info in enumerate(chained_calls):
                    # Pass the collections_map to the resolver functions
                    resolved_inputs = _resolve_io_list(call_info.get("inputs"), components, temp_states_for_chain, collections_map)
                    resolved_outputs = _resolve_io_list(call_info.get("outputs"), components, temp_states_for_chain, collections_map)

                    kwargs = {
                        "fn": _resolve_handler(call_info.get("handler")),
                        "inputs": resolved_inputs,
                        "outputs": resolved_outputs,
                        "js": call_info.get("js"),
                        "show_progress": call_info.get("show_progress"),
                        "api_name": call_info.get("api_name"),
                        "queue": call_info.get("queue")
                    }
                    kwargs = {k: v for k, v in kwargs.items() if v is not None}

                    if call_idx == 0:
                        listener = getattr(trigger_component, event_name)(**kwargs)
                    else:
                        if listener is None: raise ValueError("Listener not initialized for a .then() call.")
                        listener = listener.then(**kwargs)

            except Exception as e:
                logger.error(
                    f"Failed to wire event {i} in '{switchboard_name}'. "
                    f"Component: {event_def.get('component', 'N/A')}, "
                    f"Event: {event_def.get('event', 'N/A')}. Error: {e}",
                    exc_info=True
                )

    logger.info("Finished wiring all events.")

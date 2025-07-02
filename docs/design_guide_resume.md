# Design Doc: Resumable Video Generation

## 1. Feature Overview

**Feature:** Interactive Pause & Resume for Video Generation
**Author:** Gemini Code Assist
**Status:** Implemented

## 2. Problem Statement

If a long video generation process is interrupted—either by a user-initiated stop, a system error (like CUDA out of memory), or an application crash—all progress is lost. This is inefficient and creates a frustrating user experience. An interactive "pause" gives the user direct control to free up system resources or save their progress at will.

## 3. Goals and Non-Goals

### Goals

*   **Interactive Pause:** Allow a user to pause the currently processing task directly from the UI queue.
*   **Automatic State Save:** Periodically and automatically save a complete, resumable state file during long generations as a fail-safe.
*   **Rolling Save Retention:** Automatically manage the number of saved resume files per job to prevent disk space exhaustion.
*   **State Export:** When paused, the system will export the complete generation state into a single, portable file.
*   **Automatic Download:** The exported state file will be automatically downloaded by the user's browser.
*   **Intuitive Resume:** The user can resume a job—even after an application restart—by simply dragging and dropping the exported file onto the main image input area.
*   **Seamless Continuation:** The resumed job will continue from the exact point it was paused, preserving all generated progress and settings.

### Non-Goals

*   **Real-time Editing of Paused Tasks:** This design does not include changing parameters (e.g., prompt, seed) of a job mid-way through. The resume file restores the state *as it was*.
*   **Automatic Cloud Sync:** The resume state is managed locally via the downloaded file.

## 4. Proposed Architecture

The feature is implemented through a clear, agent-driven workflow: **Request Pause**, **Interrupt & Report**, **Package & Download**, and **Restore & Resume**.

### 4.1. The Pause-Package-Download Workflow

1.  **UI Action (Request Pause):**
    *   A "Pause" icon (⏸️) is available for the currently "Processing" task in the queue.
    *   Clicking this icon sends a `"pause"` message to the `ProcessingAgent`.

2.  **Agent Action (Interrupt & Report):**
    *   The `ProcessingAgent` receives the `"pause"` message. It sets an internal `pause_requested` flag to `True` and then sets the global `interrupt_flag`.
    *   The `worker` function, running in its own thread, detects the `interrupt_flag` and enters its exception handling block.
    *   Before exiting, the `worker` sends its final latent state, `history_latents_for_abort`, back to the agent via a `('interrupted_with_state', ...)` message.
    *   The worker then sends its final `('aborted', ...)` signal and terminates.

3.  **Agent Action (Package & Download):**
    *   The `ProcessingAgent` receives the `aborted` signal. Because its `pause_requested` flag is `True`, it knows this was a pause, not a stop.
    *   It takes the latent state it received and calls `generation_utils.save_resume_state()`.
    *   This function assembles a zip archive with the extension `.goan_resume`.

4.  **The `.goan_resume` Archive:**
    *   This zip file is the core of the feature. It contains:
        *   `params.json`: A JSON dump of all UI creative settings (prompt, sliders, seed, etc.).
        *   `source_image.png`: The original input image.
        *   `latent_history.pt`: A `torch.save()` export of the `history_latents` tensor, containing all generated data.
        *   `job_info.json`: Metadata including the original `job_id` and the number of completed segments.

5.  **UI Response (Download):**
    *   After successfully saving the archive, the agent sends a `('paused', path_to_archive)` message to the UI listener.
    *   The UI listener receives this message and triggers a browser download of the `.goan_resume` file.
    *   The task's status in the queue is updated to "Paused".

### 4.2. The Restore-Resume Workflow

1.  **Unified UI Action (Restore):**
    *   The user drags and drops a file onto the main image input area (`Drop Image or .goan_resume File Here`).
    *   This action triggers the unified handler function, `workspace.handle_file_drop`.

2.  **Backend Handler Logic (Identify & Restore):**
    *   The handler function inspects the dropped file's extension.
    *   **If it's an image:** It proceeds with the standard image upload workflow.
    *   **If it's a `.goan_resume` archive:** The handler:
        1.  Unzips the archive to a temporary location.
        2.  **Performs a signature check:** It verifies the presence of `params.json` and `latent_history.pt`.
        3.  Reads `params.json` and uses its contents to populate all the UI controls, restoring the creative settings.
        4.  Loads `source_image.png` and displays it in the input image preview.
        5.  Stores the path to the extracted `latent_history.pt` in a hidden `gr.State` component (`K.RESUME_LATENT_PATH_STATE`).
        6.  **Dynamically updates the UI:** The "Video Length" slider's label is changed to "Additional Video Length (s)" to clarify its function.
        7.  Provides feedback to the user (e.g., a `gr.Info` message) that a resume state has been loaded.
    *   The UI is now "primed" for resumption. The user simply clicks "Add to Queue".

3.  **Backend Action (Resume):**
    *   When the new task is added, the path to the `latent_history.pt` file is included in its parameters.
    *   The `worker` function is called with the `resume_latent_path`.
    *   The worker loads the `history_latents` tensor, performs an initial VAE decode for soft-blending, and continues generation from the next segment.

## 5. UI/UX Considerations

*   **Controls First Layout:** To prevent horizontal scrolling for essential actions, the queue columns are ordered with all interactive controls on the far left (`↑`, `↓`, `⏸️`, `✎`, `✖`).
*   **Disable, Don't Hide:** For the currently processing task, inapplicable controls (like 'Edit' or 'Move') are visually disabled (greyed out) rather than hidden. This maintains a stable UI layout and clearly communicates which actions are available.
*   **Clear Feedback:** The UI provides clear feedback at each stage: "Pausing task...", "Resume state loaded. Add to queue to continue."
*   **Intuitive Interaction:** Using the same drag-and-drop area for both starting a new job and resuming an old one creates a simple and discoverable user experience.
*   **Portability & Simplicity:** The resume zip file is a self-contained archive automatically downloaded to the user's machine upon pausing a task. This makes it trivial to save progress, move a task to another computer, or simply resume at a later time by dragging and dropping the file back into the application.

## 6. Key Code Changes

*   **`ui/layout.py` & `ui/enums.py`**: Add `K.RESUME_DOWNLOADER_UI` and `K.RESUME_LATENT_PATH_STATE`. Modify the main file input to accept `.goan_resume`.
*   **`ui/shared_state.py`**: Add `pause_request_flag` event (now used by the agent).
*   **`core/generation_core.py`**: Modify the `worker` to send its latent state upon interruption and to handle the initial loading of a `resume_latent_path`.
*   **`core/generation_utils.py`**: Add the `save_resume_state()` function to create the zip archive.
*   **`ui/agents.py`**: The `ProcessingAgent` now has logic to handle a `"pause"` message, orchestrate the interruption, and trigger the save of the resume state.
*   **`ui/queue.py` & `ui/switchboard_queue.py`**: Wire up the pause button click.
*   **`ui/queue_processing.py`**: The listener loop now handles the `'paused'` event to trigger the download.
*   **`ui/workspace.py`**: Implement the unified `handle_file_drop` function that inspects the uploaded file and routes it to the appropriate logic.

---

## 8. Legacy GPU (<SM80, Turing and Older) Precision Handling

### Problem

On Turing and earlier GPUs, the Hunyuan Packed model's default BF16 precision is not supported by CUDA. This causes PyTorch to fall back to the "slow_conv3d_forward" kernel, which is not implemented for CUDA, resulting in runtime errors such as:

```
NotImplementedError: Could not run 'aten::slow_conv3d_forward' with arguments from the 'CUDA' backend. ...
```

### Solution

**Force all tensors and Conv3d weights to FP32 before any Conv3d operation in the model.**
- This avoids the unsupported kernel and ensures correct execution on legacy GPUs.
- This is implemented in the `forward` methods of the following files:

- **File:** `w:\home\shane\github\shanevcantwell\goan\src\diffusers_helper\models\hunyuan_video_packed.py`
    - **Class:** `HunyuanVideoPatchEmbed`
    - **Class:** `HunyuanVideoPatchEmbedForCleanLatents`

#### Example Implementation

```python
# filepath: w:\home\shane\github\shanevcantwell\goan\src\diffusers_helper\models\hunyuan_video_packed.py

class HunyuanVideoPatchEmbed(nn.Module):
    def __init__(self, patch_size, in_chans, embed_dim):
        super().__init__()
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # Force input to float32, contiguous, and on the same device as weights
        x = x.to(dtype=torch.float32, device=self.proj.weight.device)
        if not x.is_contiguous():
            x = x.contiguous()
        return self.proj(x)

class HunyuanVideoPatchEmbedForCleanLatents(nn.Module):
    def __init__(self, inner_dim):
        super().__init__()
        self.proj = nn.Conv3d(16, inner_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.proj_2x = nn.Conv3d(16, inner_dim, kernel_size=(2, 4, 4), stride=(2, 4, 4))
        self.proj_4x = nn.Conv3d(16, inner_dim, kernel_size=(4, 8, 8), stride=(4, 8, 8))

    def forward(self, x, proj_type='proj'):
        proj_layer = getattr(self, proj_type)
        x = x.to(dtype=torch.float32, device=proj_layer.weight.device)
        if not x.is_contiguous():
            x = x.contiguous()
        return proj_layer(x)
```

### Notes

- No UI/config changes are needed; the backend already forces FP32 and hides the UI toggle for legacy GPUs.
- This ensures legacy GPU compatibility without affecting newer GPUs or user experience.

---

## 9. Open Questions & Future Considerations

1. **User Control Over Auto-Saves:**
    - Should users have control over the frequency and conditions of auto-saves? For instance, only on significant progress (e.g., every 10 seconds of video) or based on time intervals?
    - Consider a setting like "Auto-save interval" with options for "Off", "Low", "Medium", "High" (affecting both frequency and granularity of auto-saves).

2. **Storage Management:**
    - As resume files can be large, especially for long videos, should the system provide warnings or automatic management when disk space is low?
    - Options could include automatic deletion of older resume files, compression of resume files, or alerts to the user with recommended actions.

3. **Advanced Resume Options:**
    - Future iterations could allow resuming with modified parameters (e.g., changing the prompt or seed) or even swapping out the source image.
    - This would require a more sophisticated handling of the latent space and a potential re-encoding of the initial frames to ensure continuity.

4. **Multi-Scene Video Generation:**
    - Building on the advanced resume options, enabling users to define multiple scenes or segments in a single video generation task.
    - The user could specify different settings or source images for each segment, and the system would manage the transitions and continuity.

5. **In-Place Editing of Paused Tasks:**
    - Allowing users to modify parameters of a paused task and resume it with the updated settings.
    - This would create a more fluid and interactive editing experience, akin to refining a work-in-progress.

6. **Cloud Integration:**
    - While not part of the current scope, integrating cloud storage options for resume files could provide users with more flexibility and prevent data loss.
    - Consider partnerships with cloud storage providers or leveraging existing cloud infrastructure.

7. **User Testing & Feedback:**
    - Conduct user testing sessions to gather feedback on the pause/resume functionality and identify any pain points or areas for improvement.
    - Consider a feedback mechanism within the UI to report issues or suggest enhancements related to this feature.

8. **Documentation & Tutorials:**
    - Update the documentation and tutorials to include the new pause/resume feature, ensuring users are aware of how to use it effectively.
    - Consider creating video tutorials or interactive guides to demonstrate the feature in action.

9. **Performance Monitoring:**
    - Monitor the performance impact of the new feature, especially in terms of CPU/GPU usage and memory consumption during pause/resume operations.
    - Optimize the implementation as needed to minimize any negative impact on system performance.

10. **Security Considerations:**
    - Ensure that the new feature does not introduce any security vulnerabilities, especially related to file handling and user data privacy.
    - Conduct a security review and testing to validate the safety of the implementation.

---

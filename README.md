# goan: A Power-User UI for FramePack

Welcome to `goan`, an enhanced user interface designed for creative professionals and power users of FramePack. This project builds upon the brilliant FramePack video generation engine created by lllyasviel (of Fooocus and the Stable Diffusion Forge fork of A1111), exposing useful controls to reach further into FramePack's functionality through a robust and intuitive interface.

The base FramePack provides a powerful core model. `goan` extends it with a suite of tools designed for serious workflow, experimentation, and reproducibility. Unlock fine-grained control over your video generations with batch processing, parameter editing, effortless recipe sharing, and complete workspace management.

---

### Key Features for a Creative Workflow

The features in `goan` are designed to directly support two important aspects for any serious use of FramePack: deep control over the diffusion process and recovery handling for long-running jobs.

* **Uninterrupted Sessions & True Resilience:** Long-running jobs are the norm for video generation, and nothing is more frustrating than a UI disconnect or a browser crash wiping out hours of progress. `goan` is architected specifically to combat this fragility, a common pain point in many Gradio-based UIs.
    * **Backend-Driven State:** The entire state of your session—the task queue, processing status, and live updates—is managed persistently on the backend. This means your job keeps running safely on the server, completely independent of the browser tab's status. Whether the tab is minimized, the screen is locked, or the connection is temporarily lost, your progress is secure.
    * **Intelligent Session Re-attachment:** If your browser tab becomes disconnected for any reason, `goan`'s UI is designed to intelligently and automatically reconnect to the active backend process upon being re-focused. It finds the existing session and its live update stream, allowing the UI to seamlessly catch up to the real-time status of your render queue. This completely mitigates the dreaded "Error" boxes that plague typical Gradio interfaces during long sessions.
    * **Robust Task Queue & Crash Recovery:** Beyond network stability, the entire task queue is automatically saved to disk. If there's a system crash or you need to restart, you can simply relaunch `goan`, and your queue will be right where you left it, ready to continue processing. No more lost work.

* **Limit MP4 Preview Generation:** The base FramePack functionality writes a VAE-decoded `.mp4` preview for every single segment. This creates a tremendous amount of potentially unneeded compute, causing video generation to take much longer than necessary. `goan` introduces the ability to restrict these previews to only the segments you care about, using a combination of a periodic slider and a comma-separated list of individual segments to dramatically speed up your workflow.

* **Advanced Diffusion Controls:** The base FramePack is a unique approach to video generation. `goan` exposes advanced controls like **Variable CFG**, which allows you to change the prompt adherence over the course of the video. This can be used to correct for a tendency for FramePack to "burn in" or oversaturate the final video as total length increases, giving you greater artistic control.

* **Effortless "Recipe" Sharing:** `goan`'s **Drop-in Parameter Loading** allows you to save all creative settings directly into a generated PNG. Share the image, and anyone using `goan` can drop it into their UI to instantly load your exact "recipe," making collaboration and experimentation simple and repeatable.

* **Complete Workspace Management:** For more complex projects, you can save your *entire UI state*—every slider, checkbox, and text prompt—into a single `.json` file. This ensures you can always get back to a specific setup for consistent results.

### Deeper Dive: Functional & UI Control Comparison

For those curious about the specifics, here's a more detailed breakdown of what's new.

#### Diffusion Controls (CFG, Guidance Scale)

**Understanding CFG:** Classifier-Free Guidance (CFG) is a critical technique in diffusion models. Think of it as a knob that controls how strongly the model should adhere to your text prompt versus how much creative freedom it has.
* A **low CFG** value allows the model to be more imaginative, potentially straying from the prompt.
* A **high CFG** value forces the model to follow the prompt more strictly, which can sometimes reduce creativity or lead to artifacts if pushed too high.

In this model, there are two main guidance controls:

* **`Distilled CFG Scale` (`gs`):** This is the primary control you will use.
    * Recommended settings to begin with are:
    * Always start at 10.
    * If bright colors contrast too harshly, experiment with levels around 7-9.
    * Variable CFG has been added but useful suggestions here are still pending.
* **`CFG Scale` (`cfg`):** This controls the standard guidance, which is essential for negative prompts.
    * **Important:** For your **Negative Prompt** to have any effect, you must set the `CFG Scale` to a value greater than 1.0 (e.g., 1.1 or 2.0).
    * **Performance Trade-Off:** Be aware that setting `CFG Scale` to any value other than 1.0 will roughly **double the generation time** for your video, as it requires a second pass for each step. Use it only when you need the control of a negative prompt.

**Comparison:**
* **Base FramePack:** Presented a very simplified interface. Key controls like `CFG Scale` and `CFG Re-Scale` were hidden (`visible=False`), and the `info` text for `Distilled CFG Scale` explicitly said, "Changing this value is not recommended." This was effective for a simple demo but limited experimentation.
* **`goan`:** Exposes all guidance controls for the power user. It introduces the concept of **Variable CFG**, allowing the `Distilled CFG Scale` to change linearly over the course of the generation. This provides advanced control over the video's evolution, letting a user start with high prompt adherence and gradually decrease it, for example.

#### New Functionality: Workspace & Metadata

This entire feature set is new in `goan` to extend power user functionality to FramePack.

* **Drop-in Parameter Loading:** This is the core of the new workflow. You can take a PNG generated by `goan`, drop it into the image input, and the UI will automatically detect the embedded settings. A modal will ask if you want to apply them. This makes sharing and reusing "recipes" effortless.
* **Workspace Management:** Users can now save the *entire state* of the UI—all sliders, text boxes, and checkboxes—to a `.json` file. This "workspace" can be reloaded at any time, which is invaluable for complex projects or for ensuring consistent settings across sessions.
* **Session Persistence & Autosave:** The task queue is automatically saved when the application is closed and reloaded on startup. This prevents the loss of a long list of batched jobs. The UI also attempts to restore its last state after a page refresh.
* **Full Task Queue Control:** `goan` includes a full-featured task queue where you can add, remove, reorder, and *edit* jobs before you start processing. This is a massive improvement over the original's single-task processing model.

### For Developers: A Look Under the Hood (pending review)

### Code Example: A Glimpse at the New Architecture (pending review)

### Acknowledgements

* **@lllyasviel** for creating the groundbreaking FramePack engine.
* **@Tophness** for the super-useful queueing system architecture introduced in FramePack PR #150, from which forms the foundation of goan's task management, background processing, and progress update features.

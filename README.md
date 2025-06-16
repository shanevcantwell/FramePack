# **goan: A Power-User UI for FramePack**

\[\!IMPORTANT\]  
An absolute must-read before even touching goan is the official FramePack documentation, as it contains critical scripts and instructions for setting up the correct environment, as well as beautiful examples just from the most basic controls: https://github.com/lllyasviel/FramePack

Welcome to goan, an enhanced user interface designed for creative professionals and power users of FramePack. This project builds upon the brilliant FramePack video generation engine created by lllyasviel (of Fooocus and the Stable Diffusion Forge fork of A1111), exposing useful controls to reach further into FramePack's functionality through a robust and intuitive interface.

The base FramePack provides a powerful core model. goan extends it with a suite of tools designed for serious workflow, experimentation, and reproducibility. Unlock fine-grained control over your video generations with batch processing, parameter editing, effortless recipe sharing, and complete workspace management.

### **Installation**

goan is designed to run from a cloned repository with its Python dependencies managed via pip. Before running, ensure you have the necessary system-level dependencies.

* **PyTorch and CUDA:** A compatible version of PyTorch and the CUDA Toolkit is required. This project was developed and tested on **PyTorch 2.7.0+cu128**. It should be compatible with other PyTorch 2.x versions, but this has not been tested. Your specific version will depend on your GPU and drivers. You can find the correct installation command on the [official PyTorch website](https://pytorch.org/get-started/locally/).  
* **Tkinter Requirement:** The user interface relies on tkinter for file dialogs. On Debian-based systems like Ubuntu, you must install it separately. You can do this by running:  
  sudo apt install python3-tk

### **Key Features for a Creative Workflow**

The features in goan are designed to directly support two important aspects for any serious use of FramePack: deep control over the diffusion process and recovery handling for long-running jobs.

* **Uninterrupted Sessions & True Resilience:** Long-running jobs are the norm for video generation, and nothing is more frustrating than a UI disconnect or a browser crash wiping out hours of progress. goan is architected specifically to combat this fragility, a common pain point in many Gradio-based UIs.  
  * **Backend-Driven State:** The entire state of your session—the task queue, processing status, and live updates—is managed persistently on the backend. This means your job keeps running safely on the server, completely independent of the browser tab's status (minimized, screen locked, etc.). Your progress is secure.  
  * **Intelligent Session Re-attachment:** If your browser tab becomes disconnected for any reason, goan's UI is designed to intelligently and **automatically reconnect** to the active backend process upon being re-focused. It finds the existing session and its live update stream (AsyncStream), allowing the UI to seamlessly catch up to the real-time status of your render queue. This significantly mitigates the dreaded "Error" boxes that plague typical Gradio interfaces during long sessions.  
  * **Robust Task Queue & Crash Recovery:** Beyond network stability, the entire task queue is automatically saved to disk. If there's a system crash or you need to restart, you can simply relaunch goan, and your queue will be right where you left it, ready to continue processing. No more lost work.  
* **Limit MP4 Preview Generation:** The base FramePack functionality writes a VAE-decoded .mp4 preview for every single segment. This creates a tremendous amount of potentially unneeded compute, causing video generation to take much longer than necessary. goan introduces the ability to restrict these previews to only the segments you care about, using a combination of a periodic slider (Preview Frequency) and a comma-separated list of individual segments (Preview Segments CSV) to dramatically speed up your workflow. The Preview Frequency now correctly applies to segments generated within the main loop (starting from Segment 1).  
* **Advanced Diffusion Controls:** The base FramePack is a unique approach to video generation. goan exposes advanced controls like **Variable CFG**, which allows you to change the prompt adherence over the course of the video. This can be used to correct for a tendency for FramePack to "burn in" or oversaturate the final video as total length increases, giving you greater artistic control.  
* **Effortless "Recipe" Sharing and Management:** goan provides a robust, stateful image input system for managing creative parameters.  
  * **Reliable Image Input:** Instead of a simple gallery, goan uses a stateful component that begins as a clean "Drop Image Here" area. Once an image is provided, the drop zone is replaced by the image preview and a new set of context-aware buttons ("Clear Image," "Download"). This provides a stable, intuitive, and error-free workflow.  
  * **Drop-in Parameter Loading:** You can take a PNG generated by goan, drop it into the input area, and the UI will automatically detect the embedded settings. A modal dialog will appear, showing you a preview of the prompt from the image's metadata and asking if you want to apply the settings. This makes sharing and reusing "recipes" effortless and transparent.  
  * **Download with Fresh Metadata:** The "Download" button that appears with a loaded image allows you to save a new copy of it at any time. This new PNG will be embedded with the *current* creative settings from the UI, making it a perfect, portable snapshot of your work-in-progress.  
* **Complete Workspace Management:** For more complex projects, you can save your *entire UI state*—every slider, checkbox, and text prompt—into a single .json file. This "workspace" can be reloaded at any time, which is invaluable for complex projects or for ensuring consistent settings across sessions. Crucially, the Output Folder path specified in the UI is now correctly loaded from saved settings at application startup and seamlessly integrated with Gradio's allowed\_paths, preventing file access errors for custom output locations.

### **Deeper Dive: Functional & UI Control Comparison**

For those curious about the specifics, here's a more detailed breakdown of what's new.

#### **Diffusion Controls (CFG, Guidance Scale)**

**Understanding CFG:** Classifier-Free Guidance (CFG) is a critical technique in diffusion models. Think of it as a knob that controls how strongly the model should adhere to your text prompt versus how much creative freedom it has.

* A **low CFG** value allows the model to be more imaginative, potentially straying from the prompt.  
* A **high CFG** value forces the model to follow the prompt more strictly, which can sometimes reduce creativity or lead to artifacts if pushed too high.

In this model, there are two main guidance controls:

* **Distilled CFG Scale (gs):** This is the primary control you will use.  
  * Recommended settings are based on general diffusion behavior, as their specific effects on FramePack video are still being discovered.  
  * Always start at 10\.  
  * If bright colors contrast too harshly, experiment with levels around 7-9.  
  * Variable CFG has been added but useful suggestions here are still pending.  
* **CFG Scale (cfg):** This controls the standard guidance, which is essential for negative prompts.  
  * **Important:** For your **Negative Prompt** to have any effect, you must set the CFG Scale to a value greater than 1.0 (e.g., 1.1 or 2.0).  
  * **Performance Trade-Off:** Be aware that setting CFG Scale to any value other than 1.0 will roughly **double the generation time** for your video, as it requires a second pass for each step. Use it only when you need the control of a negative prompt.

**Comparison:**

* **Base FramePack:** Presented a perfectly minimalist interface. Key controls like CFG Scale and CFG Re-Scale were hidden (visible=False), and the info text for Distilled CFG Scale explicitly said, "Changing this value is not recommended." This was effective for many uses but limited experimentation.  
* **goan:** Exposes all guidance controls for the power user. It introduces the concept of **Variable CFG**, allowing the Distilled CFG Scale to change linearly over the course of the generation. This provides advanced control over the video's evolution, letting a user start with high prompt adherence and gradually decrease it, for example.

#### **New Functionality: Workspace & Metadata**

This entire feature set is new in goan to extend power user functionality to FramePack.

* **Drop-in Parameter Loading:** This is the core of the new workflow. You can take a PNG generated by goan, drop it into the image input, and the UI will automatically detect the embedded settings. For each image, you are presented the prompt and the option of applying the image's settings. This makes sharing and reusing "recipes" via the base images themselves effortless.  
* **Workspace Management:** Users can now save the *entire state* of the UI—all sliders, text boxes, and checkboxes—to a .json file. This "workspace" can be reloaded at any time, which is invaluable for complex projects or for ensuring consistent settings across sessions.  
* **Session Persistence & Autosave:** The task queue is automatically saved when the application is closed and reloaded on startup. This prevents the loss of a long list of batched jobs. The UI also successfully restores its last state after a page refresh, ensuring continuous live progress updates.  
* **Full Task Queue Control:** goan includes a full-featured task queue where you can add, remove, reorder, and *edit* jobs before or even during the task processing.

### **For Developers: A Look Under the Hood (pending review)**

### **Code Example: A Glimpse at the New Architecture (pending review)**

### **Acknowledgements**

* **@lllyasviel** for creating the groundbreaking FramePack engine.  
* **@Tophness** for the super-useful queueing system architecture introduced in FramePack PR \#150, from which forms the foundation of goan's task management, background processing, and progress update features.
* **@freely-boss** for the clever work in enabling "legacy" Turing and earlier (compute rating SM75 and below) devices to generate.

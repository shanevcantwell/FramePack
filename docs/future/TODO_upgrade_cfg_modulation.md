# Design Doc: Endcap CFG Modulation (Revision 2)

-   **Author**: Gemini
-   **Date**: 2025-07-03
-   **Status**: Proposed
-   **Replaces**: Design Doc: Contextual Damping for Initial Segment Generation

---

## 1. Summary

This document proposes **Endcap CFG Modulation** to fix the high-contrast artifact at the beginning of videos. This new design is based on the correct understanding of FramePack's inverted temporal generation order.

The solution modifies the CFG strength specifically during the final "endcap" generation pass—the pass that creates the first second of the video—to counteract the instability caused by bridging the static input image to the rest of the generated video.

---

## 2. Problem

The FramePack architecture generates the beginning of the video last. This final "endcap" pass must create motion that connects the static input image (at time `t=0`) to the start of the already-generated video bulk.

With a high or steady CFG setting, this difficult task creates a contextual conflict for the model, resulting in unstable, high-contrast frames at the start of the final video output.

---

## 3. Proposed Solution

We will introduce a specific CFG modifier that applies **only** to the endcap generation pass. This allows for a targeted reduction (or increase) in guidance strength to produce a smoother, more stable start to the video without affecting the rest of the generation.

This can be exposed to the user through a new UI control, like a "Start-up CFG Trim" slider, allowing them to precisely tune the initial contrast by dampening or boosting the CFG for the first second of video.

---

## 4. Implementation Details

We will add a logic block inside the main worker loop in `src/core/generation_core.py`. This code will detect the final segment and apply the CFG modification before calling the sampler.

### Code

This block should be inserted inside the `for` loop in `src/core/generation_core.py`, after `current_segment_gs_to_use` has been calculated by the scheduling logic.

```python
            # ... after current_segment_gs_to_use is calculated from the schedule ...

            # --- PROPOSED ENDCAP CFG MODULATION ---
            # This parameter could be a new UI slider, e.g., from -50 to 50 (%)
            startup_cfg_trim_percent = -25.0

            # The final iteration of the loop generates the first second of the video.
            # In FramePack, this is the "endcap" pass.
            is_endcap_pass = is_last_section # is_last_section is already calculated

            if is_endcap_pass and startup_cfg_trim_percent != 0:
                modifier = 1.0 + (startup_cfg_trim_percent / 100.0)
                original_cfg = current_segment_gs_to_use
                current_segment_gs_to_use = original_cfg * modifier
                logger.info(f"Applying Endcap CFG Trim. Original CFG: {original_cfg:.2f}, New CFG: {current_segment_gs_to_use:.2f}")

            # --- END OF PROPOSED LOGIC ---

            generated_latents = sample_hunyuan(
                transformer=transformer,
                # ...
                distilled_guidance_scale=current_segment_gs_to_use,
                # ...
            )
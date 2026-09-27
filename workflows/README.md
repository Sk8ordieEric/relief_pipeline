# ComfyUI workflows

These are the version-controlled copies. ComfyUI does **not** read them from here — it
reads from:

    ComfyUI_windows_portable\ComfyUI\user\default\workflows\

So the routine is: edit in ComfyUI, then copy back into this folder and commit, or
copy from here into ComfyUI to restore a known-good version.

| File | Purpose |
|---|---|
| `Create_Portrait_Relief.json` | Pet/family portraits from a photo. 5 gated stages. |
| `Create_Relief_Image.json` | The raven. Bas-relief from generated art. |
| `Create_Sunken_Relief.json` | Cavo-rilievo — figure cut *into* the panel face. |
| `Create_HeightMap.json` | Standalone Depth Anything 3 pass. |
| `Klein_Text2Image_Generator.json` | Plain FLUX.2 Klein text-to-image. |
| `CNC_Image_Creater.json` | Earlier gated img2img experiment. |
| `Base_Image_Generator.json` | SDXL equivalent of the Klein generator. |

All of them follow the same rule: **preview before saving, name the file descriptively,
and never let a stage auto-save.** Each stage reads a file from disk, so an approved
image is frozen and cannot drift.

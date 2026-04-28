# Samples

End-to-end output from the pipeline as it has shipped, plus the base character image used as the seed input. Useful as:

- **Sanity check** — drop `inputs/source_idle_breathing.png` into ComfyUI's `input/` folder, run the pipeline, compare your output to `base_animations/idle/idle.png`. If they look close, your install is good.
- **Reference layout** — confirm cell size, frame count, and channel handling match what the runtime expects before you wire your own asset loader.

## `inputs/`

| File | Purpose |
|---|---|
| `source_idle_breathing.png` | The base character image used to seed every base animation in this sample set. Faceless on purpose — the runtime layers facial features and cosmetics on top. |

## `base_animations/`

Greyscale RGBA sprite sheets produced by [W3](../workflows/w3-spritesheet/). Each is a horizontal strip of 128×128 cells.

| Animation | Frames | Loop mode |
|---|---|---|
| `idle/idle.png` | 33 | loop |
| `walking/walking.png` | 33 | loop |
| `in-air/in-air.png` | 9 | once (jump apex) |

The greyscale format lets a runtime tint each pixel at draw time (skin tone, etc.). If your runtime needs full-colour sprites instead, set `GREYSCALE = False` in the W3 driver.

## `layers/`

RGBA cosmetic sheets produced by [W5](../workflows/w5-cosmetic-spritesheet/). Each cosmetic has its own folder with one sub-folder per base animation and a sprite sheet inside that aligns pixel-for-pixel with the corresponding base animation.

| Cosmetic | Type |
|---|---|
| `hair_style_1/` | Hair |
| `hair_style_2/` | Hair |
| `eye_style_1/` | Eyes (no paired iris — bakes default iris) |
| `eye_style_2/` + `iris_style_2/` | Eyes paired with tintable iris |
| `eye_style_3/` + `iris_style_3/` | Eyes paired with tintable iris |
| `face_1/` | Face details (mouth, nose, blush, …) |
| `shirt/` | Body — top |
| `pants/` | Body — bottom |

The eye + iris pairs were produced by W5 (eye sheet) plus the [iris green-key](../docs/iris-greenkey.md) post-processing script (iris sheet). The iris is authored bright green in the cosmetic reference image so a global colour-key isolates it; the runtime then tints the iris layer to the player's chosen eye colour.

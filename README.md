# relief_pipeline

Turns a generated or photographed image into a machinable carving and a verified STL,
for the Anolex 3030 Evo Max. **Every length in this codebase is millimetres.**

The pipeline is three stages, and they live in three different places:

| Stage | Where it runs | What it produces |
|---|---|---|
| 1. Image | ComfyUI, in your browser | depth map PNG, silhouette PNG, normalized photo |
| 2. Geometry | **this folder** | height field, verified STL, DXF, preview PNG |
| 3. Toolpaths | Fusion 360 | NC programs, one per tool |

Stage 2 is the only part that is code, and it is entirely offline. Nothing here needs
the internet or a connection to Claude — if the WiFi is down you can still run it.

---

## First: check your Python

There are several Pythons on this machine and they are not interchangeable. Run this:

```
run.bat check_setup.py
```

`run.bat` looks for an interpreter that works, preferring ComfyUI's embedded Python
(`..\ComfyUI_windows_portable\python_embeded\python.exe`) because ComfyUI already
requires all three of our dependencies, so it is the safest bet on this machine.

If you'd rather use Anaconda or a system Python, install the deps into it:

```
pip install -r requirements.txt
```

In VS Code, **Ctrl+Shift+P → Python: Select Interpreter** and pick whichever one
`check_setup.py` reported as OK. `.vscode/settings.json` is already set up so `lib/`
resolves and you get completions.

---

## Folder map

```
relief_pipeline/
├── README.md            this file
├── requirements.txt     numpy, opencv-python, scipy
├── check_setup.py       run first - verifies the environment
├── run.bat              launcher that finds a working Python
│
├── relief_build.py      ENTRY POINT - depth map -> relief STL
├── inlay_build.py       ENTRY POINT - silhouette -> pocket + plug STL + DXF
│
├── lib/                 shared internals, read these to understand the method
│   ├── tooling.py       the cutter's real geometry; reach filter; slope clamp
│   ├── flow.py          coherence-enhancing diffusion (fur, feathers, hair)
│   ├── fields.py        sub-pixel distance fields, Poisson, gradient compression
│   └── mesh.py          watertight solids, verification, STL and DXF writers
│
├── presets/             tuning snapshots you can reuse
└── projects/            per-job working folders (outputs land here)
```

---

## relief_build.py — reliefs

```
run.bat relief_build.py ^
  --depth  projects\rex\Rex_Depth.png ^
  --mask   projects\rex\Rex_Silhouette.png ^
  --photo  projects\rex\Rex_Normalized.png ^
  --mode   portrait ^
  --preset presets\portrait_pet.json ^
  --panel  203x254 --relief 5.0 --base 10.0 ^
  --out    projects\rex\rex_v1
```

Three modes:

- `bas` — figure stands proud of a flat panel. The raven.
- `sunken` — figure cut down into the panel face (cavo-rilievo). **Not an inverted
  bas-relief** — the modelling keeps its normal sense, the whole figure just sits below
  the surface. Invert a height map instead and you get a mould.
- `portrait` — bas-relief tuned for faces and fur.

`--photo` is optional but worth supplying: it adds surface detail the depth map alone
cannot see, and it steers the fur/feather flow direction.

Reproduce the raven exactly:

```
run.bat relief_build.py --depth ... --mask ... --photo ... ^
  --mode bas --preset presets\raven_v5.json ^
  --panel 180x180 --relief 7.0 --base 10.05 --out projects\raven\raven_v5
```

## inlay_build.py — two-wood press-fit inlay

```
run.bat inlay_build.py ^
  --mask projects\oak_raven\Raven_Silhouette.png ^
  --panel 180x180 --vbit 60 --pocket 3.0 --offset 0.4 ^
  --board-a 19.0 --board-b 6.35 ^
  --out projects\oak_raven\inlay
```

Writes `_pocket.stl`, `_plug.stl`, `_contour.dxf` and two previews. The plug is
undersized laterally so it seats **deeper** rather than looser:

```
epsilon = S * tan(delta)        lateral undersize per side
Delta   = epsilon / tan(delta) = S
```

Those cancel, so the plug sinks exactly `S` and you face off exactly `S` past board A's
surface. One number to control. Watch the **coverage** line in the output — it tells you
what fraction of the inlay actually reaches full depth. Below ~65% means the artwork is
too fine for the bit.

---

## The numbers that drive everything

The finishing tool is a 1/16" tapered ball nose: tip radius 0.794 mm, taper 5.36° per
side. **It is not a ball.** Its effective radius grows with depth:

| depth | effective R | narrowest feature |
|---|---|---|
| 2 mm | 0.910 mm | 1.82 mm |
| 3 mm | 1.004 mm | 2.01 mm |
| 5 mm | 1.192 mm | 2.38 mm |
| 7 mm | 1.380 mm | 2.76 mm |

Anything narrower than that simply cannot be cut. `lib/tooling.py:reach_filter` removes
those features from the model rather than leaving them as uncut material, which is what
made earlier versions read as rough.

Other constants worth knowing:

- Machine feed ceiling **2000 mm/min** (GRBL `$110`/`$111`). Programs asking for more
  get silently clamped, which thins the chip and burnishes.
- Work envelope about **300 × 300 mm**. This rules out standard wooden flags (20–40").
- **GRBL ignores `M6`.** Tool changes must be separate NC programs, each ending `M5 M30`.
- Slope clamp default **65°**, comfortably inside the taper's 84.6° reach limit.

---

## Troubleshooting

**"no usable contour found"** — the silhouette is empty or nearly so. Re-check Stage 4's
threshold in ComfyUI.

**Ribbed comb along the silhouette** — that was OpenCV's chamfer distance transform
quantising. `fields.signed_field` supersamples an exact Euclidean transform instead;
don't swap it back for `cv2.distanceTransform`.

**Mirrored carving** — `mesh.grid_solid(flip_y=...)`. Image row 0 is the TOP of the
picture and must map to MAX Y in CAD.

**Fusion crawling on the STL** — drop `--quads`. 760 gives ~0.33 mm facets on an 8×10;
600 gives ~0.42 mm, still far finer than the tool can cut.

**WATERTIGHT: FAIL** — nothing is written. That check is there because a non-closed mesh
imports into Fusion looking fine and then produces silently wrong toolpaths.

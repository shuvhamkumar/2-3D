# Project Guide — Photogrammetry Engine (`2-3D`)

> **Audience:** a new intern joining the project. No prior photogrammetry
> knowledge assumed. By the end of this doc you should understand *what* the
> project does, *why* it's built the way it is, *every tool and component*
> involved, and *how to run it*.
>
> **Status (2026-06-03):** all **nine** pipeline stages are implemented and
> **verified end-to-end** on real data, with **two interchangeable MVS
> backends** *and* a new **object-isolation subsystem** (masking + geometric
> OBB crop) for turntable captures. **92 unit tests pass.** See
> [§10](#10-current-status--roadmap).

---

## 1. The one-sentence summary

**This project turns a folder of overlapping photos of an object into a
textured 3D model — by orchestrating two industry-standard reconstruction
engines (COLMAP and OpenMVS) into a robust, resumable, instrumented pipeline.**

The single most important thing to understand:

> 🚫 **This project does NOT implement any 3D reconstruction math itself.**
> It does not write structure-from-motion, bundle adjustment, multi-view
> stereo, meshing, or texturing code. Those are *hard, solved* problems owned
> by COLMAP and OpenMVS. **Our code is the *conductor*, not the *orchestra*.**

```
                  ┌──────────────────────────────────────────────┐
                  │            OUR CODE  ("the conductor")          │
                  │  validate · sequence · resume · diagnose ·      │
                  │  isolate object · parse metrics · gate · export │
                  └───────────────┬──────────────────┬─────────────┘
              shells out to       │                  │      shells out to
                          ┌───────▼──────┐   ┌────────▼────────┐
                          │   COLMAP     │   │    OpenMVS      │   ← "the orchestra"
                          │  (SfM, CPU)  │   │ (dense/mesh/tex,│      (the hard math
                          │              │   │   GPU/CUDA)     │       we never rewrite)
                          └──────────────┘   └─────────────────┘
```

Think of it like a CI/CD pipeline: Jenkins doesn't compile your code — `gcc`
does. Jenkins decides *what runs, in what order, validates each step, retries,
and reports*. We are "Jenkins for photogrammetry."

### Primary use case (this drives a lot of the defaults)
- **Scanning small objects for 3D printing** (think: a figurine on a turntable).
- Captures are **ordered / video frames** (turntable or orbit), often **500+ frames**.
- Output is **`.glb`** (a single-file 3D model with embedded texture, web/AR friendly).
- Because it's *one object on a turntable*, the engine can **isolate the object
  from its background** (table, room) — see [§4.4](#44-object-isolation-masking--obb-crop).

---

## 2. Background concepts (read this if photogrammetry is new to you)

**Photogrammetry** = reconstructing 3D geometry from 2D photographs. If you
take many overlapping photos of an object from different angles, the geometry
that was "flattened" by each camera can be recovered by comparing the photos.

The full photo → model journey, and which engine owns each phase:

```
  PHOTOS                 SfM                 MVS              MESH            TEXTURE          FILE
 ┌──────┐   ┌────────────────────────┐  ┌──────────┐   ┌────────────┐  ┌────────────┐   ┌────────┐
 │ 📷📷 │──▶│ where was each camera? │─▶│ millions │──▶│ connect    │─▶│ paint photo│──▶│  .glb  │
 │ 📷📷 │   │ + a sparse point cloud │  │ of dense │   │ points into│  │ colors onto│   │ + .ply │
 └──────┘   └────────────────────────┘  │  points  │   │ a surface  │  │ the surface│   └────────┘
              ▲ COLMAP (CPU here)        └──────────┘   └────────────┘  └────────────┘     ▲ trimesh
                                          ▲────────── OpenMVS (GPU) ──────────────▲          /open3d
```

| Phase | Plain-English meaning | Analogy |
|-------|----------------------|---------|
| **SfM** (Structure-from-Motion) | Figure out *where each camera was* + a sparse cloud of 3D points | Reverse-engineering camera positions from the photos |
| **MVS** (Multi-View Stereo / "dense") | Fill in *millions* of 3D points using the known camera positions | Going from a dot-sketch to a dense point cloud |
| **Meshing** | Connect the dense points into a continuous surface (triangles) | Shrink-wrapping a surface over the points |
| **Texturing** | Paint the original photo colors back onto the mesh surface | Wrapping the 3D shape in its real-world skin |
| **Export** | Clean + save as a standard file (`.glb`, `.ply`) | Saving your work in a format others can open |

A few key terms you'll see everywhere in the code:

- **Feature / keypoint** — a distinctive little patch in an image (a corner, a
  speck of texture) that can be re-found in other photos. COLMAP uses **SIFT**.
- **Matching** — finding which features in photo A are the same physical point
  as features in photo B. This is how photos get "linked" together.
- **Bundle adjustment (BA)** — a big optimization that jointly refines all
  camera positions *and* all 3D points to minimize error. Runs inside SfM.
- **Reprojection error** — if you take a reconstructed 3D point and project it
  back into a photo, how far (in pixels) is it from where the feature actually
  was? **Lower = better.** Under ~1px is excellent.
- **Registered image** — a photo whose camera position was successfully solved.
  If only 60% of images register, 40% were "lost" — a red flag.
- **Track length** — how many photos a single 3D point is seen in. Longer
  tracks = more reliable points.
- **Intrinsics** — a camera's internal parameters (focal length, lens
  distortion). "Shared intrinsics" means we assume every photo came from the
  *same* physical camera (true for a turntable video).
- **Mask** — a black/white image (0 = ignore, 255 = use) marking *which pixels
  are the object*. Feeding masks to SfM/MVS keeps the table & room out of the model.
- **OBB** — oriented bounding box; the tightest (possibly tilted) box around the
  object's sparse points. We crop to it as a background backstop.
- **Watertight / manifold** — is the mesh a closed, well-formed solid (no holes,
  no self-intersecting edges)? Matters for 3D printing. We **report** it, we
  never auto-repair it.

---

## 3. The external tools (with technical specifications)

The engine **shells out** to these tools' command-line programs. It detects
them at startup and **fails loudly with install instructions** if they're
missing (run `engine doctor` to see status).

### 3.1 COLMAP — the Structure-from-Motion engine

| Spec | Value (this machine) |
|------|---------------------|
| Version | **3.9.1**, installed via `apt` |
| Location | `/usr/bin/colmap` |
| Build | **CPU-only** (⚠️ no CUDA — see note below) |
| Role | Feature extraction, matching, sparse reconstruction (SfM), undistortion, sparse→PLY export; *optionally* dense/mesh (colmap backend) |
| Interface | We call its CLI subcommands and read its outputs |

COLMAP subcommands the engine drives:

| COLMAP subcommand | What it does | Used by |
|-------------------|-------------|---------|
| `feature_extractor` | Detects SIFT keypoints (honors `--ImageReader.mask_path`) → SQLite DB | `features` |
| `exhaustive_matcher` / `sequential_matcher` / `vocab_tree_matcher` / `spatial_matcher` | Matches features between image pairs | `matching` |
| `mapper` | Incremental SfM: solves camera poses + sparse points, runs bundle adjustment | `sfm` |
| `model_analyzer` | Prints quality stats (registered images, reproj error, track length) — we *parse* this | `sfm` |
| `image_undistorter` | Undistorts images + writes a COLMAP dense workspace (feeds **both** MVS backends) | `undistortion` |
| `model_converter` | Exports the sparse model's 3D points to PLY (for the OBB crop) | `mesh` |
| `patch_match_stereo` → `stereo_fusion` | Dense MVS (⚠️ **needs CUDA**) | `dense` (colmap backend) |
| `poisson_mesher` / `delaunay_mesher` | Build a vertex-colored mesh from the fused cloud | `mesh` (colmap backend) |

> ⚠️ **CPU-only COLMAP is the current performance bottleneck *and* a hard limit
> on the colmap dense backend.** The machine *has* a CUDA GPU (see 3.3), but the
> apt COLMAP binary wasn't compiled with CUDA. Consequences:
> - SIFT extraction & matching run on CPU (slow — matching took ~24 min on
>   south-building). Detected by `detect_colmap_cuda` in [binaries.py](src/engine/binaries.py).
> - `patch_match_stereo` (the colmap *dense* backend) **requires CUDA**, so it
>   cannot run here. The engine detects COLMAP's "requires CUDA" message and
>   **fails loud + actionable**, telling you to use `--mvs-backend openmvs`.

COLMAP writes a **SQLite database** (`database.db`) holding images, keypoints,
and matches, plus a **sparse model** (`cameras.bin`, `images.bin`,
`points3D.bin`) — a binary format we read for metrics.

### 3.2 OpenMVS — the dense / mesh / texture engine

| Spec | Value (this machine) |
|------|---------------------|
| Version | Built from source (master) |
| Build | **CUDA-enabled**, compiled for `sm_75` (Turing) |
| Location | binaries in `~/.local/bin/OpenMVS/`, libs in `~/.local/lib/OpenMVS/` |
| Role | Dense reconstruction, meshing, mesh refinement, texturing |
| PATH | persisted via `~/.bashrc`; the engine *also* derives the lib dir at call time (see [openmvs.py](src/engine/openmvs.py) `openmvs_lib_env`) |

OpenMVS ships **five** binaries the engine expects (all must be present for
`engine doctor` to report "OpenMVS ready: yes"):

| OpenMVS binary | What it does | Used by stage |
|----------------|-------------|---------------|
| `InterfaceCOLMAP` | Converts COLMAP's dense workspace into OpenMVS's `.mvs` format | `dense` (entry) |
| `DensifyPointCloud` | MVS dense cloud (millions of points) — CUDA; honors `<stem>.mask.png` + `--ignore-mask-label` | `dense` |
| `ReconstructMesh` | Builds a triangle mesh surface from the dense cloud | `mesh` |
| `RefineMesh` | Refines mesh detail against the photos (expensive; on for `balanced`/`high`) | `mesh` |
| `TextureMesh` | Projects photo colors onto the mesh to create a UV texture atlas | `texture` |

> ✅ **OpenMVS CUDA densify is FIXED (2026-06-02).** A local patch bug in
> `PatchMatchCUDA.cu` passed a `reinterpret_cast` macro (instead of the real
> `__constant__` symbol) to `cudaMemcpyToSymbolAsync`, yielding
> `cudaErrorInvalidDeviceSymbol`. With that fixed and rebuilt for `sm_75`,
> `DensifyPointCloud` now runs CUDA depth-maps on the GTX 1660 Ti.

> ⚠️ **The `mesh.refine` path (balanced/high) is not yet verified.** `RefineMesh`
> needs a mesh-bearing `.mvs`, but this `ReconstructMesh` build emits only the
> mesh `.ply`. The mesh stage **fails loud** if refine is on and the `.mvs` is
> missing. The verified path is **draft (no refine)**.

### 3.3 CUDA / GPU

| Spec | Value |
|------|-------|
| GPU | NVIDIA GTX 1660 Ti |
| Architecture | Turing, compute capability **`sm_75`** |
| OS | Ubuntu 24.04 |
| Detection | via `nvidia-smi -L` (best-effort, never crashes if absent) |

```
                       ┌─────────────────────────────────────────────┐
   DIVISION OF LABOR   │  CPU  ──▶ COLMAP   ingest→features→matching→ │
   BY HARDWARE         │            sfm→undistortion                   │
                       │  GPU  ──▶ OpenMVS  dense→mesh→texture (CUDA)  │
                       │  CPU  ──▶ Python   masks/OBB/export           │
                       └─────────────────────────────────────────────┘
```

The engine auto-detects CUDA and passes GPU flags to the tools that can use
them (`--cuda-device 0` / `gpu_index 0`), with a **clear CPU-fallback** path
(`--mvs-cpu` → `--cuda-device -1`) and **actionable diagnostics** for CUDA
out-of-memory vs CUDA build-mismatch errors (see [openmvs.py](src/engine/openmvs.py)
`looks_like_oom` / `looks_like_cuda_error`).

### 3.4 Python stack

Declared in [pyproject.toml](pyproject.toml). Requires **Python ≥ 3.11**.

| Library | Why it's here | Where declared |
|---------|--------------|----------------|
| **typer** | Builds the `engine` command-line interface | core |
| **pydantic** (v2) | The config model — validation, presets, type safety | core |
| **rich** | Pretty terminal logging, tables, colored output | core |
| **pillow** (PIL) | Reading images, dimensions, EXIF; mask I/O; texture dims | core |
| **numpy** | Sharpness / blur math; mesh + mask array ops | core |
| **trimesh** | Mesh container + `.glb`/`.ply` conversion (export) | core |
| **pyyaml** | Loading YAML config files | core |
| **open3d** | Watertight/manifold checks, decimation, OBB crop | `[full]` extra |
| **scipy** | `cKDTree` outlier removal (export); EDT mask dilation | imported at runtime |
| **rembg** + **onnxruntime** | `--auto-mask` background removal (U²-Net) | imported at runtime |
| pytest, pytest-cov | Testing | `[dev]` extra |

> 📦 The back half of the pipeline (`export`) needs **open3d** and **scipy**;
> `--auto-mask` needs **rembg**. These are *lazy-imported* so the unit tests and
> the SfM-only path run without them — but they must be installed for a full run.
> All of the above are installed in the project venv. Missing-dependency paths
> fail with an actionable `pip install …` message, never a bare `ImportError`.

The package installs a single console command: **`engine`** (`engine.cli:app`).

---

## 4. What the project does — step by step

The pipeline is an ordered list of **stages**. Each stage takes the previous
stage's output, runs a tool (or pure-Python check), validates the result,
writes a metrics file, and either succeeds or **stops the pipeline loudly**.

### 4.1 The full pipeline (all nine stages — ALL BUILT & VERIFIED ✅)

```
            ┌──── masks (external dir / rembg --auto-mask / sibling images/../masks) ────┐
            │  optional · restrict SfM + dense to the OBJECT only · 0=ignore 255=use      │
            ▼                                                          ▼
 ┌────────┐ ┌──────────┐ ┌──────────┐ ┌───────┐ ┌─────────────┐ ┌────────┐ ┌──────┐ ┌─────────┐ ┌────────┐
 │ ingest │▶│ features │▶│ matching │▶│  sfm  │▶│ undistortion│▶│ dense  │▶│ mesh │▶│ texture │▶│ export │▶ .glb
 └────────┘ └──────────┘ └──────────┘ └───────┘ └─────────────┘ └────────┘ └──────┘ └─────────┘ └────────┘
  pure Py    COLMAP        COLMAP       COLMAP    COLMAP          OpenMVS/   OpenMVS/ OpenMVS/    trimesh+
             +mask_path                                          COLMAP     COLMAP   COLMAP      open3d
                                                                 +.mask.png ▲OBB crop
            └──────────── backend-agnostic (always COLMAP) ──────┘└──── branches on --mvs-backend ────┘
                                          ✅ all stages BUILT & VERIFIED end-to-end
```

The **first five stages are backend-agnostic** (always COLMAP). Only
`dense → mesh → texture → export` branch on `mvs.backend`:

```
                      ┌──────────────── dense ─────────────┬──────── mesh ────────┬──── texture ────┬──── export ────┐
   --mvs-backend      │                                    │                      │                 │                │
   ───────────────────┼────────────────────────────────────┼──────────────────────┼─────────────────┼────────────────┤
   colmap   (CPU      │ patch_match_stereo → stereo_fusion │ OBB-crop cloud, then  │ pass-through    │ trimesh →      │
            here:     │ ⚠️ NEEDS CUDA → fails loud here     │ poisson/delaunay      │ (vertex colors, │ .glb w/ vertex │
            blocked)  │ → use openmvs                      │ → vertex-colored ply  │  no UV atlas)   │ colors         │
   ───────────────────┼────────────────────────────────────┼──────────────────────┼─────────────────┼────────────────┤
   openmvs  (default, │ InterfaceCOLMAP → (.mask.png) →    │ ReconstructMesh, then │ TextureMesh →   │ trimesh →      │
            GPU)      │ DensifyPointCloud → scene_dense.ply │ OBB-crop the mesh     │ UV atlas        │ .glb w/        │
                      │   + scene_dense.mvs                │ (+RefineMesh if on)   │ (OBJ+MTL+PNG)   │ embedded tex   │
                      └────────────────────────────────────┴──────────────────────┴─────────────────┴────────────────┘
```

### 4.2 The nine stages in detail

**Stage 1 — `ingest`** ([stages/ingest.py](src/engine/stages/ingest.py)) · *pure Python*
- Scans the image folder, opens every image.
- **Rejects the job early** if it's unlikely to reconstruct: too few images
  (`min_images`), inconsistent resolutions (`max_resolution_ratio`), mostly-blurry
  frames (`max_blurry_fraction`), or (when required) missing EXIF focal data.
- Measures **sharpness** via *variance of the Laplacian* on a downscaled grayscale
  copy. Philosophy: **catch a doomed job in seconds, not after an hour of compute.**

**Stage 2 — `features`** ([stages/features.py](src/engine/stages/features.py)) · *COLMAP `feature_extractor`*
- Detects SIFT keypoints in each image, stores them in `database.db`.
- **Resolves masks** ([masks.py](src/engine/masks.py)) and, if present, passes
  `--ImageReader.mask_path` so keypoints come **only from the object** → an
  object-only sparse cloud. Records mask source + coverage; warns if `<100%`.

**Stage 3 — `matching`** ([stages/matching.py](src/engine/stages/matching.py)) · *COLMAP `*_matcher`*
- Matches features between images. **Auto-selects** the matcher by job
  type/size (or honors `--matcher`): `sequential` (video/ordered — our default),
  `exhaustive` (small unordered), `vocab_tree` (large unordered), `spatial` (geo).
- Fails loudly if **zero** geometrically-verified pairs result.

**Stage 4 — `sfm`** ([stages/sfm.py](src/engine/stages/sfm.py)) · *COLMAP `mapper`*
- Camera poses + sparse 3D points + bundle adjustment. Keeps the **largest** of
  any disconnected models; warns about discarded ones.
- **Quality gates** flagged as warnings (never hidden): `<60%` registered,
  reproj `>1.5px`.
- ⚠️ **Registration fraction is measured against *all* input images** (the DB
  image count), not the largest model's count — so a partial reconstruction
  (e.g. 32/41) can no longer masquerade as "100% registered." Emits a clear
  "N of M did not register" warning. *(This bug was caught on the monstree set.)*

**Stage 5 — `undistortion`** ([stages/undistort.py](src/engine/stages/undistort.py)) · *COLMAP `image_undistorter`*
- Removes lens distortion and writes a **COLMAP dense workspace** (the shared
  entry point for **both** MVS backends). Fails if zero images came out.

**Stage 6 — `dense`** ([stages/dense.py](src/engine/stages/dense.py)) · *OpenMVS (default) or COLMAP*
- **openmvs:** `InterfaceCOLMAP` → (optional `<stem>.mask.png` + `--ignore-mask-label 0`)
  → `DensifyPointCloud` → `scene_dense.ply` (+ `.mvs`).
- **colmap:** `patch_match_stereo` → `stereo_fusion` → `fused.ply` (needs CUDA).
- **VRAM is the first thing to blow on a 6 GB GPU.** CUDA-OOM → actionable message
  (lower `mvs.max_image_size` / raise `mvs.resolution_level`); `--auto-downscale`
  retries coarser instead of aborting. A non-OOM CUDA error gets its own message
  pointing to `--mvs-cpu`.

**Stage 7 — `mesh`** ([stages/mesh.py](src/engine/stages/mesh.py)) · *OpenMVS (default) or COLMAP*
- **openmvs:** `ReconstructMesh` (+ `RefineMesh` if `mesh.refine`) → untextured mesh.
- **colmap:** `poisson_mesher` (default) or `delaunay_mesher` → vertex-colored mesh.
- **OBB crop backstop** ([geometry.py](src/engine/geometry.py)): exports the sparse
  points to PLY, computes their oriented bounding box (+ `bbox_margin`), and crops
  the **dense cloud** (colmap, *before* meshing) or the **reconstructed mesh**
  (openmvs, *after* — `ReconstructMesh` has no external-cloud input). Fails loud if
  the crop removes everything; warns if it removes `>60%`.
- Fails if the mesh is empty (dense cloud too sparse).

**Stage 8 — `texture`** ([stages/texture.py](src/engine/stages/texture.py)) · *OpenMVS or pass-through*
- **openmvs:** `TextureMesh` builds a UV atlas + texture image (OBJ + MTL + PNG/JPG).
- **colmap:** no-op pass-through — the poisson/delaunay mesh already carries vertex
  colors that export embeds into the GLB.

**Stage 9 — `export`** ([stages/export.py](src/engine/stages/export.py)) · *pure Python (`trimesh` + `open3d`)*
- **Cleans** the mesh — *order matters*: remove statistical outliers **first**
  (a kd-tree distance test with a safety cap so it never shatters a uniform
  surface), then **keep the largest connected component** (welding coincident
  vertices first so a per-corner-split textured OBJ gets correct connectivity).
- **Inspects quality** (watertight / edge-manifold / vertex-manifold via open3d)
  and **reports honestly** — never auto-repairs. Warns if an `object` job isn't
  watertight.
- **Decimates** to a face budget for the `web` use-case only.
- **Converts** to `.glb` (embedded UV texture, or vertex colors for the colmap
  path) + `.ply`, and writes `manifest.json` into `<job>/output/`.

### 4.3 Object isolation (masking + OBB crop)

Turntable/orbit captures contain a **table and a room** behind the object. Left
alone, MVS happily reconstructs all of it (on monstree, the table slab dwarfed
the figurine). The engine has **three complementary mechanisms** to keep only
the object, layered weakest-coupling-last:

```
  ① MASKED SfM  (reliable)            ② MASKED DENSIFY (best-effort)     ③ OBB CROP (backstop)
  ─────────────────────────           ──────────────────────────────    ─────────────────────────
  masks on ORIGINAL images            <stem>.mask.png beside the         sparse object points
  → COLMAP --ImageReader.mask_path    UNDISTORTED images →               → oriented bbox (+margin)
  → sparse cloud is object-only       OpenMVS --ignore-mask-label 0      → crop dense cloud / mesh
                                                                          (open3d)
  ✓ no undistortion-alignment issue   ⚠ masks aren't undistorted →       ✓ alignment-independent
  ✓ the dependable path                 only accurate for low distortion ✓ works masked OR unmasked
                                                                          default ON (bbox_crop=true)
```

**Where masks come from** (`resolve_mask_dir`): `--masks <dir>` (external,
COLMAP-named `<image>.png`) → else `--auto-mask` (rembg/U²-Net into `<job>/masks`)
→ else a sibling `images/../masks` folder → else none.

**Mask dilation** (`--mask-dilate <frac>`): tight masks on a *small-in-frame*
object starve SfM of features and under-register. Dilation grows the mask by a
fraction of the image's short side (via a Euclidean distance transform) to keep
context around the object. Auto-masks are dilated at generation; external masks
are dilated into a job-local `masks_dilated/` (originals never touched).

### 4.4 Proof it works (two verified live runs)

**(a) Building facade — `south-building` (128 imgs, draft, OpenMVS/GPU)**, full
`ingest → … → export`:

```
 STAGE          WALL TIME   KEY RESULT
 ──────────────────────────────────────────────────────────────────────────────
 features          187 s    SIFT keypoints → database.db
 matching        1 431 s ◀── ~24 min  (CPU-only COLMAP — the bottleneck)
 sfm               313 s    128/128 registered (100%), 0.69 px reproj
 dense             213 s    3,874,865 dense points          (CUDA, GTX 1660 Ti)
 mesh              295 s    1,407,495 verts / 2,814,321 faces
 texture         1 364 s    UV atlas 2048×2048 (draft)
 export            143 s    cleaned 2.81 M → 2.61 M faces; 1 component
 ──────────────────────────────────────────────────────────────────────────────
 OUTPUT: runs/south-building/output/model.glb (230 MB, glTF 2.0) + .ply + manifest
 NOTE:   NOT watertight (open building facade) — reported honestly, never faked.
```

**(b) Turntable object — `monstree` (41 imgs, orbit of a figurine, draft, OpenMVS/GPU)** —
the object-isolation proof, three runs compared:

```
 RUN                         REGISTERED   DENSE PTS   MESH FACES   RESULT
 ──────────────────────────────────────────────────────────────────────────────────
 unmasked                    41/41        1.15 M      ~936 k       ❌ dominated by the TABLE
 masked, no dilation         17/41        —           —            ❌ object-only but under-registered
 masked + --mask-dilate 0.06 32/41 (78%)  90,168      102,621      ✅ object only, NO table
 ──────────────────────────────────────────────────────────────────────────────────
 → masking removes ~12× the points (background); dilation fixes the under-
   registration tradeoff. OBB crop removed 0% here (masks already did the work).
   Output: runs/monstree_dilated/output/model.glb + object_preview.png
```

> Object detail is modest at `draft` (a small-in-frame object gets downscaled) —
> use `balanced`/`high` for detail. Full provenance lives in each job's
> `metrics/` and `output/manifest.json`.

---

## 5. Architecture

### 5.1 Layered design

```
┌─────────────────────────────────────────────────────────────────────┐
│  CLI layer          cli.py        (typer: doctor/show-config/         │  ← what the user types
│                                    validate/run/report)               │
├─────────────────────────────────────────────────────────────────────┤
│  Orchestration      pipeline.py   (sequence stages, --from/--to,      │  ← the "conductor"
│                                    resume, incremental report writes) │
│                     report.py     (aggregate + pretty-print)          │
├─────────────────────────────────────────────────────────────────────┤
│  Stage layer        stages/base.py (Stage ABC + uniform driver)       │  ← uniform contract
│                     ingest·features·matching·sfm·undistort·           │
│                     dense·mesh·texture·export   (9 stages)            │
├─────────────────────────────────────────────────────────────────────┤
│  Tool / IO helpers  binaries.py  (detect + run external, CUDA flags)  │  ← talks to the world
│                     colmap.py    (read COLMAP DB + sparse models)     │
│                     openmvs.py   (lib env, CUDA args, OOM/err detect) │
│                     masks.py     (resolve/generate/dilate masks)      │
│                     geometry.py  (sparse→PLY, OBB, crop cloud/mesh)   │
│                     imaging.py   (PIL/numpy image checks)             │
│                     meshops.py   (trimesh/open3d clean·inspect·export)│
│                     workspace.py (deterministic paths)                │
│                     logging.py   (rich logging + timing)             │
├─────────────────────────────────────────────────────────────────────┤
│  Configuration      config.py    (pydantic model + 3 presets)        │  ← drives everything
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 Control flow — one stage's life (`Stage.execute`)

The driver in [base.py](src/engine/stages/base.py) is identical for every
stage; subclasses only fill in `validate_inputs()` and `run()`:

```
            ┌─────────────────────────────────────────────────────────────┐
            │ Stage.execute(ctx)                                          │
            └─────────────────────────────────────────────────────────────┘
                                   │
              metrics/<stage>.json exists  AND not --force ?
                       │ yes                         │ no
                       ▼                             ▼
              ┌─────────────────┐          ┌──────────────────┐
              │ SKIP — reload   │          │ validate_inputs()│──raise──▶┐
              │ prior result    │          └──────────────────┘          │
              │ (status=SKIPPED)│                   │ ok                  │
              └─────────────────┘                   ▼                     │
                                          ┌──────────────────┐            │
                                          │ run()  (timed)   │──raise──▶  │
                                          │ → metrics,        │           │
                                          │   warnings,outputs│           ▼
                                          └──────────────────┘    ┌───────────────┐
                                                   │              │ FAILED result │
                                                   ▼              │ (full error   │
                                          ┌──────────────────┐    │  text saved)  │
                                          │ COMPLETED result │    └───────┬───────┘
                                          └────────┬─────────┘            │
                                                   ▼                      ▼
                                          ┌────────────────────────────────────┐
                                          │ persist metrics/<stage>.json        │
                                          │ (this file == "stage is complete")  │
                                          └────────────────────────────────────┘
```

After **every** stage the orchestrator rewrites `report.json`, so even an
interrupted or failed job leaves an up-to-date report. A failed stage **stops
the pipeline** and surfaces the error verbatim.

Stages read each other's results through the metrics files, not shared memory:
`prior_outputs(ctx, "sfm")` / `prior_metrics(ctx, "dense")` — which is *why*
resume works and *why* each stage is independently re-runnable.

### 5.3 Why this shape? (design principles you'll be held to)

1. **Orchestrate, never reimplement.** No SfM/MVS/meshing math in our repo.
2. **Fail loud, never fake success.** A missing binary, a degenerate
   reconstruction, a CUDA-OOM, a non-watertight mesh, a partial registration —
   all surfaced with full context. The pipeline *never* "fixes up" a bad stage.
3. **Resumable & idempotent.** A stage is complete *iff* its metrics file exists;
   re-running skips finished work unless `--force`.
4. **Everything is instrumented.** Each external command is logged verbatim,
   each stage is timed and emits metrics, degenerate results are flagged.
5. **Deterministic layout.** Outputs always live at predictable paths
   ([workspace.py](src/engine/workspace.py)); the input image folder is
   **read-only** — never mutated (even mask dilation writes to a job-local dir).

---

## 6. Every component of the codebase

Source uses a **src layout**: the package is [src/engine/](src/engine/).

### Core modules

| File | Responsibility | Key things inside |
|------|---------------|-------------------|
| [config.py](src/engine/config.py) | The single config model that drives everything | `EngineConfig` (pydantic), per-stage sub-configs incl. `MaskingConfig`, 3 presets (`build_preset`), use-case defaults, YAML/JSON loading, deep-merge |
| [cli.py](src/engine/cli.py) | The `engine` command-line interface | `doctor`, `show-config`, `validate`, `run`, `report` (typer) |
| [pipeline.py](src/engine/pipeline.py) | The orchestrator (also the importable Python API) | `run_pipeline()`, `--from`/`--to` range resolution, incremental report writes |
| [binaries.py](src/engine/binaries.py) | Detect external tools + CUDA, run commands safely | `detect_environment`, `require`, `run_command`, `CommandError`, CPU-only-COLMAP detection |
| [colmap.py](src/engine/colmap.py) | Read COLMAP outputs to compute metrics | `read_database_stats` (SQLite), `parse_model_analyzer`, `select_largest_model` |
| [openmvs.py](src/engine/openmvs.py) | OpenMVS runtime glue (CLI-driven) | `openmvs_lib_env`, `cuda_device_arg`, `looks_like_oom`/`looks_like_cuda_error`, PLY-header counts |
| [masks.py](src/engine/masks.py) | Object isolation — masks | `resolve_mask_dir`, `prepare_masks`, `generate_masks_rembg`, `dilate_mask` (EDT), `write_openmvs_masks`; `MaskInfo` |
| [geometry.py](src/engine/geometry.py) | Object isolation — geometric crop | `export_sparse_points`, `compute_obb` (open3d), `crop_cloud`, `crop_mesh`; `CropResult` |
| [imaging.py](src/engine/imaging.py) | Image inspection (PIL + numpy only) | `discover_images`, `inspect_image`, variance-of-Laplacian sharpness, EXIF-focal detection |
| [meshops.py](src/engine/meshops.py) | Mesh cleanup, quality, conversion | `load_mesh`, `keep_largest_component`, `remove_statistical_outliers`, `inspect_quality`, `decimate`, `export`; `MeshQuality` |
| [workspace.py](src/engine/workspace.py) | Deterministic job-directory paths | `Workspace`: `db_path`, `sparse_dir`, `dense_dir`, `output_dir`, `metrics_dir`, `report_path`, … |
| [logging.py](src/engine/logging.py) | Structured logging + per-stage timing | `configure_logging`, `stage_timer`, `format_command` |
| [report.py](src/engine/report.py) | Aggregate stage results → `report.json` + summary | `build_report`, `collect_stage_results`, `print_summary` (shows masking + OBB-crop lines) |

### Stage modules ([src/engine/stages/](src/engine/stages/))

| File | Responsibility | Tool / backend |
|------|---------------|----------------|
| [base.py](src/engine/stages/base.py) | `Stage` ABC: `StageContext`, `StageResult`, `StageStatus`, `StageError`, uniform `execute()` | — |
| [\_\_init\_\_.py](src/engine/stages/__init__.py) | `ALL_STAGES` (ordered registry) + `STAGE_NAMES` | — |
| [ingest.py](src/engine/stages/ingest.py) | Validate the photo set | pure Python |
| [features.py](src/engine/stages/features.py) | Feature extraction (+ mask resolution) | COLMAP |
| [matching.py](src/engine/stages/matching.py) | Feature matching (auto-selects matcher) | COLMAP |
| [sfm.py](src/engine/stages/sfm.py) | Sparse reconstruction + quality gates | COLMAP |
| [undistort.py](src/engine/stages/undistort.py) | Undistortion + dense workspace | COLMAP |
| [dense.py](src/engine/stages/dense.py) | Dense MVS (+ OpenMVS masks, CUDA-OOM, auto-downscale) | OpenMVS / COLMAP |
| [mesh.py](src/engine/stages/mesh.py) | Meshing (+ OBB crop, optional refine) | OpenMVS / COLMAP |
| [texture.py](src/engine/stages/texture.py) | UV texturing (or vertex-color pass-through) | OpenMVS / — |
| [export.py](src/engine/stages/export.py) | Clean, inspect, decimate, convert to `.glb`/`.ply` | trimesh + open3d |

### The Stage contract (every stage looks like this)

```python
class MyStage(Stage):
    name = "mystage"                       # also the metrics filename + --from/--to token

    def validate_inputs(self, ctx):        # raise StageError if prerequisites missing
        require(ctx.env, ["colmap"])       # a binary, or a prior stage's output
        prev = self.prior_outputs(ctx, "sfm")

    def run(self, ctx):                    # do the work; return (metrics, warnings, outputs)
        run_command([...], env_required=ctx.env)
        return {"some_metric": 42}, ["a warning"], {"artifact": "/path"}
```

`ctx` (a `StageContext`) carries the `workspace`, resolved `config`, detected
`env`, and the `force` flag. Then register it in `ALL_STAGES`.

### Tests ([tests/](tests/)) — **92 unit tests**

| File | Covers | Tests |
|------|--------|-------|
| [test_config.py](tests/test_config.py) | Presets, use-case defaults, YAML/JSON loading, deep-merge | 13 |
| [test_binaries.py](tests/test_binaries.py) | Tool detection, `require`, GPU flags, `run_command` | 13 |
| [test_pipeline.py](tests/test_pipeline.py) | Stage sequencing, `--from`/`--to`, resume/skip, reporting | 13 |
| [test_masks.py](tests/test_masks.py) | Mask resolution/naming, coverage, dilation (EDT), OpenMVS mask write | 13 |
| [test_dense_backends.py](tests/test_dense_backends.py) | dense branching, CUDA-OOM/error/no-CUDA-COLMAP, auto-downscale | 9 |
| [test_meshops.py](tests/test_meshops.py) | Component labeling, keep-largest, outlier cap, decimation, export | 9 |
| [test_colmap.py](tests/test_colmap.py) | SQLite stat reading, analyzer parsing, model selection | 8 |
| [test_ingest.py](tests/test_ingest.py) | Image discovery, sharpness, rejection rules | 6 |
| [test_export_stage.py](tests/test_export_stage.py) | export cleanup order, watertight reporting, manifest | 3 |
| [test_geometry.py](tests/test_geometry.py) | OBB compute (+margin), crop cloud/mesh, `CropResult` | 3 |

Unit tests need **no binaries**. An `integration` marker (see
[pyproject.toml](pyproject.toml)) gates the full end-to-end test (auto-skips when
COLMAP/OpenMVS are absent — now installed, so it runs).

---

## 7. The job directory layout (what `--output` looks like)

Defined in [workspace.py](src/engine/workspace.py). Combined real layout:

```
runs/<job>/
├── config.json          # snapshot of the exact resolved config (provenance)
├── database.db          # COLMAP SQLite DB: images, keypoints, matches
├── report.json          # aggregated metrics + per-stage status (rewritten each stage)
├── masks/               # (--auto-mask) generated object masks  '<image>.png'
├── masks_dilated/       # (external masks + --mask-dilate) dilated copies (originals untouched)
├── metrics/             # one JSON per stage — presence == "this stage is complete"
│   ├── ingest.json … export.json   (9 files; also the channel stages pass outputs through)
├── sparse/0/            # COLMAP sparse model: cameras.bin, images.bin, points3D.bin
├── dense/               # COLMAP dense workspace + OpenMVS scene files
│   ├── images/          #   undistorted images (+ <stem>.mask.png if masked densify)
│   ├── sparse/ · stereo/        #   undistorted model · (colmap) depth/normal maps
│   ├── sparse_points.ply        #   sparse points exported for the OBB crop
│   ├── scene_dense.ply          #   (openmvs) dense cloud   ── + scene_dense.mvs
│   ├── fused_cropped.ply        #   (colmap) OBB-cropped dense cloud
│   ├── scene_dense_mesh[_cropped].ply   #   (openmvs) mesh, pre/post OBB crop
│   └── scene_textured.{obj,mtl,*.jpg}   #   (openmvs) UV-textured mesh
├── logs/
└── output/              # ✅ FINAL DELIVERABLES
    ├── model.glb        #   binary glTF, embedded texture
    ├── model.ply
    └── manifest.json    #   files + cleanup stats + quality (watertight/manifold)
```

> 🔑 **Resume in one line:** a stage is "done" *iff* `metrics/<stage>.json` exists
> with `status: completed`. Delete that file (or pass `--force`) to re-run it.

---

## 8. The configuration system

Everything tunable lives in one pydantic model, `EngineConfig`
([config.py](src/engine/config.py)). You rarely build it by hand — you pick a
**preset** and optionally a **use-case**, then override on the CLI or via a file.

```
   build_preset(preset, use_case)                from_dict / from_file
   ┌──────────────┐   ┌─────────────────┐        ┌─────────────────────┐
   │ preset values│ + │ use-case tweaks │  ───▶  │ + file/CLI overrides│ ──▶ EngineConfig
   │ (speed↔qual.)│   │ (matcher/camera)│        │ (deep-merge, wins)  │     (extra:"forbid")
   └──────────────┘   └─────────────────┘        └─────────────────────┘
```

### Presets (speed ⟷ quality)

| Preset | feature img / feats | MVS img / res-level | mesh refine / poisson-depth | texture res / size | use |
|--------|---------------------|---------------------|-----------------------------|--------------------|-----|
| `draft` | 1600 px / 4096 | 1000 px / 3 | off / 9 | 2 / 2048 | fast smoke test |
| `balanced` | 3200 px / 8192 | 2000 px / 1 | **on** / 11 | 1 / 4096 | **default** |
| `high` | 5000 px / 16384 | 3200 px / 0 | on / 13 | 0 / 8192 | best quality, slow |

### Use-cases (capture style → matcher + camera + export defaults)

| Use-case | matcher | intrinsics | extra |
|----------|---------|-----------|-------|
| `object` (**default**) | sequential | shared (`single_camera`) | turntable/orbit video of one object |
| `outdoor` | spatial | per-image | requires EXIF focal length |
| `web` | auto | — | decimates mesh to ~100k faces for light assets |

### Masking config (`MaskingConfig`) — object isolation

| Field | Default | Meaning |
|-------|---------|---------|
| `mask_dir` | `None` | External masks dir (`<image>.png`, 0=ignore/255=use) |
| `auto_mask` | `False` | Generate masks with rembg/U²-Net into `<job>/masks` |
| `mask_dilate_frac` | `0.0` (≤0.25) | Grow masks by this fraction of the image short side (EDT) |
| `bbox_crop` | `True` | Geometric OBB-crop backstop (on by default — safe) |
| `bbox_margin` | `0.15` (≤2.0) | Fractional margin added to each OBB half-extent |

### Other key non-preset defaults
- `mvs.backend = openmvs`, `mesh.backend = openmvs` (the verified path here).
- `mvs.auto_downscale = False` — fail loud on CUDA-OOM rather than silently
  coarsen. `auto_downscale_attempts = 2`.
- Export cleanup: `keep_largest_component`, `outlier_neighbors=20`,
  `outlier_std_ratio=2.0`, `outlier_max_fraction=0.02` (safety cap),
  `min_component_fraction=0.05`.

### Config file example
```yaml
preset: high
mvs:
  resolution_level: 2   # override just this one high-preset value
masking:
  auto_mask: true
  mask_dilate_frac: 0.05
```
Precedence: **preset values first, then file keys override them** (`from_dict` +
`_deep_merge`). The model is `extra: "forbid"`, so a typo'd key is a loud error.

---

## 9. How to run it

```bash
# One-time setup
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,full]"          # open3d/scipy/rembg also needed for a full run

# Sanity-check the environment (are the tools + GPU there?)
engine doctor

# See exactly what a preset resolves to
engine show-config --preset balanced --use-case object

# Validate a photo set without reconstructing (fast — ingest only)
engine validate data/south-building/images

# Run the WHOLE pipeline to a textured .glb (OpenMVS backend is the default)
engine run data/south-building/images -o runs/south-building --preset draft

# Turntable object, isolated from its background (auto masks + dilation)
engine run data/monstree/images -o runs/monstree --preset draft \
    --auto-mask --mask-dilate 0.06

# Limit the range with --from/--to
engine run data/south-building/images -o runs/south-building --to sfm

# Re-print a finished/partial job's report
engine report runs/south-building
```

Useful `run` flags:

| Flag | Effect |
|------|--------|
| `--from STAGE` / `--to STAGE` | Limit the stage range (resume / partial run) |
| `--force` | Re-run completed stages |
| `--matcher` | Override matcher (`auto`/`exhaustive`/`sequential`/`vocab_tree`/`spatial`) |
| `--mvs-backend` | `colmap` or `openmvs` (default) |
| `--auto-downscale` | On CUDA-OOM in dense, retry at a coarser resolution |
| `--mvs-cpu` | Run OpenMVS dense/refine on CPU (`--cuda-device -1`) |
| `--masks DIR` | Use external object masks (`<image>.png`) |
| `--auto-mask` | Generate object masks with rembg |
| `--mask-dilate FRAC` | Grow masks by FRAC of the image short side |
| `--bbox-margin FRAC` | Margin for the sparse-OBB crop backstop |
| `--no-bbox-crop` | Disable the OBB geometric crop backstop |

### Datasets in `data/`
`south-building` (128-img building facade), `gerrard-hall` (building), and
`monstree` (41-img turntable orbit of a figurine — from AliceVision; used to
verify the masking path).

### Dev workflow
```bash
source .venv/bin/activate
pytest                  # 92 unit tests, no binaries needed
pytest -m integration   # full end-to-end run (needs COLMAP/OpenMVS — now installed)
```

---

## 10. Current status & roadmap

Milestones 1–4 built the pipeline; **Milestone 5** (docs) is current; the active
push is **v1.0 (turntable hardening)**, split into four review-gated sub-steps.

| Milestone | Scope | Status |
|-----------|-------|--------|
| 1 | Scaffold, config + presets, binary/CUDA detection, logging | ✅ done |
| 2 | Ingest / validate stage | ✅ done |
| 3 | SfM path, Stage abstraction, orchestrator, reports | ✅ done |
| 4 | Back half: undistort → dense → mesh → texture → export, both backends, GLB export | ✅ done & verified |
| 5 | Polish: presets doc, CLI, README + this guide | ✅ current |
| **v1.0-A** | **Object isolation: masks + OBB crop + dilation** | ✅ **done & verified (monstree)** |
| v1.0-B | Metric scale (known-distance / ArUco → mm) | ⬜ next |
| v1.0-C | Printable export (`.stl`/`.3mf`, printability summary, opt-in `--repair`) | ⬜ planned |
| v1.0-D | Hardening (integration test, `--strict`, batch mode, CI, CHANGELOG → tag v1.0) | ⬜ planned |

### Known gaps / next up
- **`mesh.refine` (balanced/high) is unverified** — this `ReconstructMesh` build
  emits only a mesh `.ply`, but `RefineMesh` needs a mesh `.mvs`; the mesh stage
  fails loud when refine is on and the `.mvs` is missing. **Draft is the verified path.**
- The **colmap dense backend** can't run on this CPU-only COLMAP (it's useful on a
  machine with a CUDA COLMAP build; the code path + tests exist).
- Object **detail at `draft` is modest** for a small-in-frame object — use
  `balanced`/`high` once the refine path is sorted.

---

## 11. Glossary (quick reference)

| Term | Meaning |
|------|---------|
| **SfM** | Structure-from-Motion — recover camera poses + sparse 3D points |
| **MVS** | Multi-View Stereo — the "dense" step; millions of 3D points |
| **BA** | Bundle Adjustment — joint optimization of cameras + points (inside SfM) |
| **SIFT** | The feature detector COLMAP uses (Scale-Invariant Feature Transform) |
| **Sparse / dense cloud** | COLMAP's thin point cloud (`*.bin`) / OpenMVS's thick cloud (`scene_dense.ply`) |
| **Mesh / UV atlas** | Triangle surface / a 2D image + per-vertex UVs painting photo color onto it |
| **Mask** | Binary image (0=ignore, 255=use) marking the object pixels |
| **Dilation** | Growing a mask outward (EDT) to keep context around a small object |
| **OBB** | Oriented bounding box of the sparse object points — cropped to as a backstop |
| **Watertight / manifold** | Closed, well-formed solid (matters for printing) — reported, never auto-fixed |
| **Reprojection error** | Pixel distance between a reprojected 3D point and its feature — lower is better |
| **Registered image** | A photo whose camera pose was solved (fraction is vs. *all* inputs) |
| **`.glb`** | Binary glTF — single-file 3D model with embedded texture (our output) |
| **Preset / Backend** | Speed-quality bundle (`draft`/`balanced`/`high`) / engine for dense-mesh-texture (`openmvs` default, `colmap`) |
| **Stage / Workspace** | One pipeline step (validate→run→record) / the deterministic output-dir layout |
| **CUDA-OOM** | GPU ran out of VRAM — detected from stderr, turned into actionable advice |
| **`sm_75`** | The CUDA compute capability of the GTX 1660 Ti (Turing) |

---

*Generated as an onboarding guide. The source of truth is always the code in
[src/engine/](src/engine/) and the metrics in each job's `report.json`.*

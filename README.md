# Photogrammetry engine

Orchestration pipeline that turns a folder of overlapping photos into a textured
3D model. The reconstruction algorithms come from **COLMAP** (structure-from-motion)
and **COLMAP/OpenMVS** (dense, mesh, texture) — this project *orchestrates* those
engines into a robust, resumable, instrumented pipeline. It does **not** reimplement
SfM, bundle adjustment, MVS, meshing, or texturing.

- **Primary use case:** small-object scanning for 3D printing
- **Primary output:** `.glb` (binary glTF, embedded texture)
- **Typical jobs:** ordered / video captures (turntable or orbit), 500+ frames

> **Status:** all nine stages are implemented and **verified end to end** —
> `ingest → features → matching → sfm → undistortion → dense → mesh → texture →
> export` — with two MVS backends (`--mvs-backend colmap|openmvs`), VRAM/OOM and
> CUDA-error diagnosis, mesh cleanup, watertight/manifold reporting, and
> `.glb`+`.ply`+manifest export. **92 unit tests pass.**
>
> _Verified full run_ — `data/south-building` (128 imgs, draft, OpenMVS/GPU):
> SfM **128/128 registered, 0.69 px reproj** → dense **3.87 M points** → mesh
> **2.81 M faces** → UV-textured → export cleaned to a single component
> (**2.61 M faces**), producing a real textured `model.glb` (glTF 2.0, embedded
> texture) + `model.ply` + `manifest.json`. Not watertight (open building
> facade) — reported honestly, never auto-repaired.
>
> _Verified turntable run (object isolation)_ — `data/monstree` (41 imgs, draft,
> OpenMVS/GPU) with masking + `--mask-dilate 0.06`: masked SfM registered
> **32/41** with an object-only sparse cloud → OBB-cropped dense **~90 k points**
> → mesh **~101 k faces** → textured `model.glb` (~7.6 MB). Side-by-side with the
> unmasked run (41/41 registered, **~937 k faces**, ~81 MB of object **plus**
> background) confirms the masks + OBB crop isolate the object as intended.
>
> ⚠️ **Backend notes for this machine:** COLMAP is a CPU-only build, so the
> `colmap` dense backend (`patch_match_stereo` needs CUDA) fails *loudly and
> actionably* — use `--mvs-backend openmvs`. The `mesh.refine` path (balanced/
> high) needs a mesh `.mvs` this ReconstructMesh build doesn't emit, so it is
> not yet verified; draft (no refine) is the verified path.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"        # add ".[full]" for pycolmap + open3d
```

### External binaries (required at run time)

The engine detects these at startup and **fails loudly with guidance** if missing.
Check what's available:

```bash
engine doctor   # reports all 6 binaries + CUDA; expect "COLMAP ready / OpenMVS ready"
```

- **COLMAP** — `apt install colmap` / `brew install colmap`, or build from
  <https://colmap.github.io/install.html>. A CUDA build is recommended.
- **OpenMVS** (`InterfaceCOLMAP`, `DensifyPointCloud`, `ReconstructMesh`,
  `RefineMesh`, `TextureMesh`) — build from
  <https://github.com/cdcseacave/openMVS/wiki/Building>.

CUDA is auto-detected via `nvidia-smi`; GPU flags are passed to COLMAP/OpenMVS
when a device is present, with a CPU fallback warning otherwise. Note that an
apt/`brew` COLMAP is typically a **CPU-only** build, which makes exhaustive
matching the bottleneck on large jobs — prefer the sequential matcher for
ordered/video captures.

## CLI

```bash
engine doctor                       # report detected binaries + CUDA
engine show-config --preset draft   # resolve a preset and print the full config
engine validate <image_dir>         # ingest/validation only
engine run <image_dir> -o <out> --preset balanced --use-case object \
    [--from STAGE] [--to STAGE] [--force] \
    [--matcher auto|exhaustive|sequential|vocab_tree|spatial] \
    [--mvs-backend colmap|openmvs] [--auto-downscale] [--mvs-cpu] \
    [--masks <dir>] [--auto-mask] [--mask-dilate 0.06] [--bbox-margin 0.15] [--no-bbox-crop]
engine report <job_dir>             # print the metrics summary for a job
```

`-o/--output` is required for `run`. Stage names for `--from`/`--to` are
`ingest`, `features`, `matching`, `sfm`, `undistortion`, `dense`, `mesh`,
`texture`, `export`. `--auto-downscale` retries dense at a coarser resolution on
CUDA-OOM; `--mvs-cpu` runs OpenMVS dense/refine on CPU. The pipeline is also
importable as a Python API (`engine.pipeline.run_pipeline`) so the engine can be
embedded in another app.

## Object isolation (turntable/orbit captures)

On a turntable capture the static background and turntable surface pollute the
cloud and mesh. Two complementary mechanisms isolate the object:

1. **Masks** — binary PNGs (`0` = ignore, `255` = use). Supply a directory with
   `--masks <dir>` (COLMAP naming: `<image_name>.png`, e.g. `frame001.JPG.png`),
   or generate them with `--auto-mask` (needs the optional `rembg` package). Masks
   restrict COLMAP feature extraction (`--ImageReader.mask_path`) so the **sparse
   cloud is object-only**, and are passed to OpenMVS densify as a best-effort
   refinement. Mask coverage is reported in `report.json`.
   - **Mask dilation** (`--mask-dilate <frac>`): tight masks on a *small-in-frame*
     object starve SfM (few features → poor registration). Growing the mask by a
     fraction of the image's short side keeps surrounding context for matching
     while still excluding most background. On the monstree sample this lifted
     registration from 17/41 → 32/41 images.
2. **OBB crop backstop** (on by default) — before meshing, the dense cloud
   (COLMAP backend) / reconstructed mesh (OpenMVS backend) is cropped to the
   oriented bounding box of the sparse object points plus a margin
   (`--bbox-margin`, default 0.15). This removes surviving background even when
   masks are imperfect or absent, and is alignment-independent. Disable with
   `--no-bbox-crop`. Point/face counts before and after the crop are reported.

The reliable path is *masked SfM → object-only sparse → OBB crop of the dense
result*, which needs no mask/undistortion alignment.

## Configuration & presets

A `pydantic` `EngineConfig` drives everything. Load from YAML/JSON and override
on the CLI. A file may set `preset:` as its base; other keys override it:

```yaml
preset: high
mvs:
  resolution_level: 2   # overrides the high-preset value
```

| Preset     | feature img | MVS img / res-level | mesh refine | texture size | use |
|------------|-------------|---------------------|-------------|--------------|-----|
| `draft`    | 1600 px     | 1000 px / 3         | off         | 2048         | fast smoke test |
| `balanced` | 3200 px     | 2000 px / 1         | on (light)  | 4096         | default |
| `high`     | 5000 px     | 3200 px / 0         | on          | 8192         | best quality, slow |

Use-case defaults (`--use-case object|outdoor|web`):

- **object** (default): sequential matcher, shared intrinsics — turntable/orbit video.
- **outdoor**: spatial matcher, per-image intrinsics, EXIF focal required.
- **web**: auto matcher, decimate to ~100k faces for lightweight assets.

## Pipeline stages

`ingest/validate → feature extraction → matching → SfM → undistortion →
dense (MVS) → meshing → texturing → export/convert (.glb)`.

The `colmap` backend does `patch_match_stereo → stereo_fusion → poisson_mesher`
(vertex-colored mesh); the `openmvs` backend does `InterfaceCOLMAP →
DensifyPointCloud → ReconstructMesh → [RefineMesh] → TextureMesh` (UV-textured).
The export stage keeps the largest connected component, strips statistical
outliers, reports watertight/manifold honestly (no auto-repair), optionally
decimates for `--use-case web`, and writes `model.glb` + `model.ply` +
`manifest.json` to `<job>/output/`.

Each stage validates its inputs, writes outputs to a deterministic location,
records a metrics JSON under `<job>/metrics/`, and is skipped on re-run unless
`--force` is passed (resumable). An interrupted job leaves an up-to-date
`report.json` behind.

## Quality / verification

Photogrammetry produces plausible-but-wrong results when something is subtly
off, so each stage emits metrics into a per-job `report.json`. Captured across
the full pipeline: discovered/readable image counts, resolution spread,
sharpness, EXIF-focal coverage, keypoint counts, matched pairs/inliers,
registered vs total images, mean reprojection error, track length, and sparse
point counts; then dense point counts, mesh face counts, and — after export —
final face count, watertight/edge-manifold checks, and the written `.glb`
path/size. Degenerate results (e.g. <60% images registered) are flagged as
warnings, and a mesh is reported watertight-or-not honestly — never
auto-repaired.

## Tests

```bash
pytest          # 92 unit tests; no external binaries needed
```

The unit tests mock the external tools and cover preset/config resolution,
binary & CUDA detection, COLMAP/OpenMVS command construction, mesh cleanup, the
export stage, and pipeline resume/skip/force logic. End-to-end behavior is
verified with live runs on real datasets (see **Status** above); the
`integration` pytest marker is registered in `pyproject.toml` and reserved for
an automated full-pipeline test.

## Out of scope

Reimplementing reconstruction algorithms; NeRF / Gaussian Splatting (future
extension); a GUI. The pipeline never hides or "fixes up" a failed stage to
appear successful.

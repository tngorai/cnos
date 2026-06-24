# part2_visualiser.py — Complete Line-by-Line Explainer

Welcome! This document explains every single line of `part2_visualiser.py`, assuming you know **nothing** about the code or the maths behind it. By the end, you will understand exactly what each line does, how the maths formulas work, and why we chose them.

---

## Table of Contents

1. [What does this program do?](#1-what-does-this-program-do)
2. [Imports — the tools we use](#2-imports--the-tools-we-use)
3. [Configuration — paths and colours](#3-configuration--paths-and-colours)
4. [Math Helpers — the error formulas](#4-math-helpers--the-error-formulas)
    - [Rotation Error](#rotation_error_deg)
    - [Translation Error](#translation_error_mm)
    - [IoU (Intersection over Union)](#compute_iou)
    - [Matrix to Quaternion](#matrix_to_quat_wxyz)
5. [Mesh Cache — loading 3D models](#5-mesh-cache--loading-3d-models)
6. [Data Loading — reading JSON and mask files](#6-data-loading--reading-json-and-mask-files)
7. [Mock Predictions — faking a model's output](#7-mock-predictions--faking-a-models-output)
    - [Pose perturbation](#generate_mock_predictions)
    - [Mask perturbation](#generate_mock_mask_prediction)
8. [2D Rendering — drawing bounding boxes and masks on images](#8-2d-rendering--drawing-bounding-boxes-and-masks-on-images)
9. [Error Graphs — interactive Plotly charts](#9-error-graphs--interactive-plotly-charts)
10. [Main Dashboard — the Viser server and GUI](#10-main-dashboard--the-viser-server-and-gui)
11. [Event Callbacks — making the dashboard interactive](#11-event-callbacks--making-the-dashboard-interactive)
12. [Full Data Flow Diagram](#12-full-data-flow-diagram)

---

## 1. What does this program do?

This program is a **web-based dashboard** that lets you visualise how well an AI model predicts the position of 3D objects in photographs. It shows:

- **2D View**: The actual photograph with coloured boxes and overlays showing where the object *really* is (Ground Truth, green) vs where the model *thinks* it is (Prediction, red).
- **3D View**: The real 3D CAD model rotated and positioned at the GT and predicted locations, colour-coded green and red.
- **Error Graphs**: Interactive charts showing how the errors change across hundreds of frames. Hover to see exact values.
- **Controls**: Slider, buttons, and checkboxes to switch frames and toggle visibility.

Since we don't have a real AI model yet, we **fake** the predictions by adding random noise to the ground truth. This lets us build and test the dashboard before the model exists.

---

## 2. Imports — the tools we use

```python
import json          # Read/write JSON files (our data is in .json format)
import os            # File paths, check if files exist
import time          # sleep() to keep the server alive, and small pauses
from typing import Optional, Dict  # Type hints for function signatures

import numpy as np           # Arrays, matrices, and maths (np = numpy)
import cv2                   # OpenCV — read images, image processing (erosion/dilation)
from PIL import Image, ImageDraw  # Pillow — draw rectangles and text on images
import plotly.graph_objects as go # Plotly — interactive graphs with hover tooltips
import trimesh                # Load and handle 3D mesh files (.ply CAD models)
import viser                  # The main 3D web dashboard server
import viser.transforms as tf # Viser's math library for rotations (3D → quaternions)
```

| Library | What it does |
|---------|-------------|
| `json` | Reads our BOP dataset files (scene_gt.json, etc.) |
| `numpy` | All the maths: matrix multiplication, norms, random numbers |
| `opencv (cv2)` | Reads .png images and masks, does erosion/dilation |
| `PIL (Pillow)` | Draws coloured rectangles (bounding boxes) and text on images |
| `plotly` | Creates interactive charts you can hover over to see exact values |
| `trimesh` | Loads 3D CAD models from .ply files |
| `viser` | Runs a web server at localhost:8080 with a 3D scene and GUI panel |

---

## 3. Configuration — paths and colours

```python
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/000001"))
```

Let's unpack this step by step:
- `__file__` is the path to this Python file, e.g. `/home/aryan/Documents/cnos/src/poses/part2_visualiser.py`
- `os.path.dirname(__file__)` gives the folder: `/home/aryan/Documents/cnos/src/poses/`
- `os.path.join(..., "../../data/000001")` goes up two folders then into `data/000001`
- `os.path.abspath(...)` makes it absolute: `/home/aryan/Documents/cnos/data/000001`

```python
RGB_DIR = os.path.join(DATA_DIR, "rgb")       # e.g. data/000001/rgb/
MASK_DIR = os.path.join(DATA_DIR, "mask")     # e.g. data/000001/mask/
MASK_VISIB_DIR = os.path.join(DATA_DIR, "mask_visib")  # e.g. data/000001/mask_visib/
MODEL_DIR = "/home/aryan/Downloads/tudl_models/models"  # Where the .ply 3D models live
```

**Colours** use RGBA format: `(Red, Green, Blue, Alpha)`. Alpha controls transparency (0=invisible, 255=opaque).

```python
COLOR_GT   = (0, 255, 0, 200)    # Semi-transparent green  → Ground Truth
COLOR_PRED = (255, 60, 60, 200)  # Semi-transparent red    → Prediction
```

---

## 4. Math Helpers — the error formulas

### `rotation_error_deg(R_gt, R_pred)`

**What it computes**: The angular difference between two 3D rotations, in degrees.

**The formula**:
```
θ = arccos( (trace(R_gtᵀ × R_pred) - 1) / 2 )
```

**Why this works — explained from the ground up:**

#### What is a Rotation Matrix?

A **rotation matrix** is a 3×3 grid of numbers that describes how an object is oriented in 3D space. For example, this matrix:

```
R = [ 1  0  0 ]
    [ 0  1  0 ]
    [ 0  0  1 ]
```

means "no rotation at all" (the identity matrix — every axis points exactly along itself).

This matrix:

```
R = [ 1  0  0 ]
    [ 0  0 -1 ]
    [ 0  1  0 ]
```

means "rotate 90° around the X-axis" (the Y and Z axes are swapped and one is negated).

#### The Trace of a Matrix

The **trace** of a matrix is simply the sum of its diagonal elements (top-left to bottom-right):

```
trace(R) = R[0,0] + R[1,1] + R[2,2]
```

For any 3D rotation matrix, the trace has a special property:

```
trace(R) = 1 + 2·cos(θ)
```

where θ (theta) is the rotation angle in radians. This is a fundamental identity from **linear algebra** — it comes from the fact that any rotation can be expressed as a rotation of θ degrees around some axis.

#### The Difference Rotation

To find how different two rotations are, we compute the **difference rotation**:

```
R_diff = R_gtᵀ × R_pred
```

- `R_gtᵀ` is the **transpose** (flipped diagonally) of the ground truth rotation. The transpose of a rotation matrix is its **inverse** — it "undoes" the rotation.
- `R_gtᵀ × R_pred` means: "first un-rotate by the ground truth, then rotate by the prediction". The result is a rotation matrix that represents the *net* difference.

#### Putting it together

```
trace(R_diff) = 1 + 2·cos(θ_diff)
```

We know `trace(R_diff)`, so we solve for `θ_diff`:

```
cos(θ_diff) = (trace(R_diff) - 1) / 2
θ_diff = arccos(cos(θ_diff))   # arccos = inverse cosine, in radians
θ_diff_degrees = θ_diff × 180 / π   # convert radians to degrees
```

#### Numerical safety — `np.clip`

Due to floating-point rounding, `(trace - 1) / 2` might be slightly outside [-1, 1], which would make `arccos` fail. `np.clip(value, -1.0, 1.0)` forces the value to stay in the valid range.

#### Worked Example

Suppose:
```
R_gt  = [1  0  0]     (no rotation)
        [0  1  0]
        [0  0  1]

R_pred = [ 0  0  1]   (90° around Y)
         [ 0  1  0]
         [-1  0  0]
```

1. `R_diff = R_gtᵀ × R_pred = I × R_pred = R_pred` (since GT is identity)
2. `trace(R_diff) = 0 + 1 + 0 = 1`
3. `cos(θ) = (1 - 1) / 2 = 0`
4. `θ = arccos(0) = π/2 = 90°`
5. Output: `90.0°` ✓

---

### `translation_error_mm(t_gt, t_pred)`

**What it computes**: The straight-line distance between two 3D positions, in millimetres.

**The formula**:
```
error = ||t_gt - t_pred||₂  = √((x₁-x₂)² + (y₁-y₂)² + (z₁-z₂)²)
```

This is the **Euclidean norm** or **L2 norm** — the same formula you'd use to measure the straight-line distance between two points in any dimension.

#### Worked Example

```
t_gt   = [0, 0, 850]      # Object is 850mm directly ahead
t_pred = [12, -8, 847]    # Model thinks it's 12mm right, 8mm down, 3mm closer

error = √(12² + (-8)² + (-3)²)
      = √(144 + 64 + 9)
      = √217
      ≈ 14.73 mm
```

---

### `compute_iou(mask_a, mask_b)`

**What it computes**: Intersection over Union — how much two binary masks overlap.

**What is a binary mask?** Imagine a black-and-white image where white pixels are "the object" and black pixels are "not the object". A binary mask stores this as 1s (object) and 0s (background).

**The formula**:
```
IOU = |A ∩ B| / |A ∪ B|
```
- `|A ∩ B|` = number of pixels that are 1 in **both** masks (the overlap area)
- `|A ∪ B|` = number of pixels that are 1 in **either** mask (the total covered area)

**What IOU means**:
- `IOU = 1.0` → perfect match (masks are identical)
- `IOU = 0.5` → half the total area overlaps
- `IOU = 0.0` → no overlap at all

#### Worked Example with a Diagram

Imagine two simple 5×5 masks:

```
   Mask A              Mask B          Intersection (A∩B)     Union (A∪B)
[0 0 0 0 0]        [0 0 0 0 0]        [0 0 0 0 0]         [0 0 0 0 0]
[0 1 1 1 0]        [0 0 1 1 1]        [0 0 1 1 0]         [0 1 1 1 1]
[0 1 1 1 0]        [0 0 1 1 1]        [0 0 1 1 0]         [0 1 1 1 1]
[0 1 1 1 0]        [0 0 1 1 1]        [0 0 1 1 0]         [0 1 1 1 1]
[0 0 0 0 0]        [0 0 0 0 0]        [0 0 0 0 0]         [0 0 0 0 0]
 pixels=9              pixels=9           pixels=4            pixels=14

IOU = 4 / 14 ≈ 0.286
```

In code:
```python
intersection = np.logical_and(mask_a, mask_b).sum()  # Count pixels where BOTH are 1
union = np.logical_or(mask_a, mask_b).sum()           # Count pixels where EITHER is 1
```

**Edge case**: If `union == 0` (both masks are completely empty), we return `1.0` — two empty masks are in perfect agreement.

---

### `matrix_to_quat_wxyz(R)`

**What it does**: Converts a 3×3 rotation matrix into a **quaternion** in `(w, x, y, z)` order.

**Why?** Viser (the 3D engine) uses **quaternions** to represent rotations, not matrices. A quaternion is a 4-number system `(w, x, y, z)` that avoids a problem called **gimbal lock** (where two rotation axes align and you lose a degree of freedom). Quaternions are standard in all 3D graphics and games.

**How?** `tf.SO3.from_matrix(R)` creates a special object from the `viser.transforms` library that represents the rotation. `.wxyz` extracts the quaternion in Viser's preferred `(w, x, y, z)` order.

```
Input:  [[1, 0, 0],        (identity rotation = no rotation)
         [0, 1, 0],
         [0, 0, 1]]

Output: [1.0, 0.0, 0.0, 0.0]   (w=1 means no rotation)
```

---

## 5. Mesh Cache — loading 3D models

```python
_mesh_cache: Dict[int, trimesh.Trimesh] = {}
```

This is a **dictionary** (like a lookup table) that maps object IDs (e.g. `1`, `2`, `3`) to loaded 3D meshes. The underscore `_` at the start is a Python convention meaning "this is private/internal — don't use it from outside".

```python
def get_mesh(obj_id: int) -> Optional[trimesh.Trimesh]:
```

`Optional[X]` means the function returns either `X` or `None`.

```
LINE 94: if obj_id in _mesh_cache:
              return _mesh_cache[obj_id]
```
If we already loaded this mesh before, just return it immediately. This is called **caching** — it avoids loading the same .ply file thousands of times (once per frame).

```
LINE 96: ply_path = os.path.join(MODEL_DIR, f"obj_{obj_id:06d}.ply")
```
`f"obj_{obj_id:06d}"` formats the ID with leading zeros to 6 digits. For example:
- `obj_id=1` → `"obj_000001.ply"`
- `obj_id=42` → `"obj_000042.ply"`

This matches how BOP datasets name their model files.

```
LINE 101: mesh = trimesh.load(ply_path)
```
`trimesh.load()` reads the .ply file and returns a `Trimesh` object containing:
- `mesh.vertices` — a list of `(x, y, z)` 3D points (31,707 of them for object 1)
- `mesh.faces` — triangles connecting those points (10,833 triangles for object 1)

A 3D model is made of triangles. The vertices are the corners, and the faces say which three vertices form each triangle. Together they define the shape.

```
LINE 102-103: _mesh_cache[obj_id] = mesh
              print(f"Loaded mesh for obj_{obj_id:06d}: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
```
Store in cache and print a confirmation message.

---

## 6. Data Loading — reading JSON and mask files

### `load_json(filename)`

Standard JSON file reader. Opens a file, parses the JSON text, returns a Python dictionary.

### `load_mask(frame_id, obj_id)`

Loads a binary mask image from the `mask/` folder.

```
LINE 118: fname = f"{frame_id:06d}_{obj_id:06d}.png"
```
Mask filenames look like: `000042_000001.png` (frame 42, object 1).

```
LINE 122: img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
```
OpenCV reads the image as a 2D array of pixel values (0-255). `IMREAD_GRAYSCALE` means we load it as a single-channel grayscale image.

```
LINE 123: return (img > 0).astype(np.uint8)
```
`img > 0` creates a **boolean array** — `True` for every pixel brighter than 0, `False` for black pixels. `.astype(np.uint8)` converts `True`→`1` and `False`→`0`. This gives us a clean binary mask: 1=object, 0=background.

### `load_mask_visib(frame_id, obj_id)`

Same as `load_mask` but from the `mask_visib/` folder. The "visib" masks only show the *visible* part of the object (parts hidden behind other objects are removed), while the regular mask shows the whole object even if partially occluded.

---

## 7. Mock Predictions — faking a model's output

### `generate_mock_predictions(scene_gt)`

Since we don't have a real AI model, we need to **pretend** we have one. We take the ground truth and add random noise to create a "prediction" that is slightly wrong — exactly what a real model would produce.

#### Structure of a BOP pose entry:
```json
{
    "cam_R_m2c": [list of 9 floats],   // 3×3 rotation matrix, flattened
    "cam_t_m2c": [x, y, z],            // translation vector in mm
    "obj_id": 1
}
```

`cam_R_m2c` = "Camera Rotation from Model to Camera" — how the object is rotated relative to the camera.
`cam_t_m2c` = "Camera Translation from Model to Camera" — where the object is in 3D space relative to the camera (in millimetres).

#### Reshaping the rotation matrix:
```python
R_gt = np.array(obj["cam_R_m2c"]).reshape(3, 3)
```
The JSON stores 9 numbers in a flat list. `reshape(3, 3)` turns them into a proper 3×3 matrix.

#### Translation noise — Gaussian:
```python
t_noise = np.random.normal(0, 15, 3)   # mean=0, std=15mm, 3 values (x,y,z)
t_pred = t_gt + t_noise
```
`np.random.normal(0, 15, 3)` generates 3 random numbers from a **Gaussian (bell-curve) distribution**:
- Mean = 0 (equally likely to be positive or negative)
- Standard deviation = 15mm (68% of values fall within ±15mm)
- This simulates a model that is usually within ~15mm of the true position

```
Example: t_gt = [0, 0, 850]
         t_noise = [12.3, -7.8, 2.1]
         t_pred = [12.3, -7.8, 852.1]
```

#### Rotation noise — Rodrigues' Formula:
```python
axis = np.random.randn(3)     # Random 3D direction
axis /= np.linalg.norm(axis)  # Normalise to length 1 (unit vector)
angle_rad = np.random.uniform(0.05, 0.18)  # Random angle between ~3° and ~10°
```

**Rodrigues' rotation formula** lets you build a rotation matrix from any axis and angle. It uses the **skew-symmetric matrix** K:

```
For axis (ax, ay, az):

K = [  0   -az   ay ]
    [  az   0   -ax ]
    [ -ay   ax   0  ]
```

The rotation matrix for angle θ is:
```
R = I + sin(θ)·K + (1 - cos(θ))·K²
```
where I is the 3×3 identity matrix (= no rotation).

**Why this works**: This formula decomposes a rotation into three parts:
1. `I` — keep the original position
2. `sin(θ)·K` — the "first-order" rotation (dominant for small angles, since sin(θ)≈θ for small θ)
3. `(1-cos(θ))·K²` — the "second-order" correction (small for small angles, since 1-cos(θ)≈θ²/2)

This is derived from the **Taylor series** expansion of the exponential of K. If you've done Further Maths A-Level, it's related to Maclaurin series and matrix exponentials.

```python
R_noise = np.eye(3) + np.sin(angle_rad) * K + (1 - np.cos(angle_rad)) * (K @ K)
R_pred = R_noise @ R_gt    # Apply the noise rotation ON TOP OF the ground truth rotation
```

**Worked Example** (simplified 2D case for clarity):
```
axis = [0, 0, 1]    # rotate around Z axis
θ = 0.1 radians ≈ 5.7°

K = [ 0 -1  0 ]
    [ 1  0  0 ]
    [ 0  0  0 ]

K² = [-1  0  0]
     [ 0 -1  0]
     [ 0  0  0]

R = I + 0.0998·K + 0.005·K²
  = [1 0 0]   [ 0 -0.1 0]   [-0.005  0  0]
    [0 1 0] + [0.1  0  0] + [ 0 -0.005 0]
    [0 0 1]   [ 0   0  0]   [ 0   0   0]

  ≈ [ 0.995 -0.1  0 ]
    [ 0.1   0.995 0 ]    ← This is a 5.7° rotation in 2D!
    [ 0     0     1 ]
```

### `generate_mock_mask_prediction(gt_mask, frame_id, obj_id)`

Fakes a predicted binary mask by slightly eroding or dilating the ground truth mask.

```
LINE 182: rng = np.random.RandomState(frame_id * 1000 + obj_id)
```
`RandomState(seed)` creates a random number generator with a **fixed seed**. The seed is `frame_id * 1000 + obj_id`, which means:
- Frame 42, Object 1 → seed 42001
- Frame 42, Object 1 → **always** seed 42001 (deterministic!)
- Frame 42, Object 2 → seed 42002 (different seed, different "prediction")

This is called **deterministic pseudo-randomness** — it looks random but is actually repeatable. The same (frame, object) always gives the same prediction.

```
LINE 186-189:
if rng.rand() > 0.5:
    pred = cv2.erode(gt_mask, kernel, iterations=1)   # Shrink the mask
else:
    pred = cv2.dilate(gt_mask, kernel, iterations=1)  # Grow the mask
```

**Erosion** removes pixels from the edges of the white area (shrinks it).
**Dilation** adds pixels to the edges (expands it).

```
Original:         Eroded:           Dilated:
[0 0 0 0 0]     [0 0 0 0 0]      [0 0 0 0 0]
[0 1 1 1 0]     [0 0 1 0 0]      [0 1 1 1 0]
[0 1 1 1 0]     [0 0 1 0 0]      [1 1 1 1 1]
[0 1 1 1 0]     [0 0 1 0 0]      [0 1 1 1 0]
[0 0 0 0 0]     [0 0 0 0 0]      [0 0 0 0 0]
```

```
LINE 192-196: Random pixel shift (translation)
    shift = rng.randint(0, 4)          # 0 to 3 pixel shift
    M = np.float32([[1, 0, dx], [0, 1, dy]])  # Affine transform matrix
    pred = cv2.warpAffine(pred, M, (pred.shape[1], pred.shape[0]))
```

`warpAffine` applies a 2D transformation. The matrix `[[1, 0, dx], [0, 1, dy]]` means "keep the pixels as they are, just shift by (dx, dy)". This simulates a model that slightly misaligns the mask.

---

## 8. 2D Rendering — drawing bounding boxes and masks on images

### `render_2d_composite(frame_id, scene_gt, scene_gt_info, predictions, show_gt, show_pred)`

This function takes a frame number and returns that frame's RGB photo with all annotations drawn on top.

```
LINE 213: rgb_path = os.path.join(RGB_DIR, f"{frame_id:06d}.png")
```
RGB images are named like `000042.png`.

```
LINE 214-215: if file doesn't exist → return a black 480×640 RGBA image
```

```
LINE 217: img = cv2.imread(rgb_path)          # OpenCV loads image
LINE 218: img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR, convert to RGB
```
**Important**: OpenCV loads images in BGR (Blue-Green-Red) order for historical reasons. Most other libraries use RGB. `cvtColor` fixes this.

```
LINE 220: pil_img = Image.fromarray(img).convert("RGBA")
```
Convert the numpy array to a PIL Image with an alpha (transparency) channel. RGBA = Red, Green, Blue, Alpha.

```
LINE 221: overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
```
Create a completely transparent image the same size as the photo. We'll draw bounding boxes and masks on this overlay, then blend it with the original photo.

```
LINE 222: draw = ImageDraw.Draw(overlay)
```
Get a "drawing pen" to draw on the overlay.

#### Drawing bounding boxes:
```python
def draw_bbox(draw_obj, bbox, colour, label):
    x, y, w, h = bbox          # BOP format: [left, top, width, height]
    draw_obj.rectangle([x, y, x + w, y + h], outline=colour, width=2)
    # Draw a label background rectangle
    draw_obj.rectangle([tx, ty, tx + 70, ty + 14], fill=colour)
    draw_obj.text((tx + 2, ty), label, fill=(255, 255, 255, 255))
```

The bounding box format in BOP is `[x, y, w, h]`:
- `(x, y)` = top-left corner
- `w` = width extending right from x
- `h` = height extending down from y
- Bottom-right corner = `(x + w, y + h)`

This is different from the more common `[x1, y1, x2, y2]` format!

#### Drawing mask overlays:
```python
mask_colour = (0, 255, 0, 60)   # green with low opacity (60/255 ≈ 24%)
mask_rgba = np.zeros((height, width, 4), dtype=np.uint8)
mask_rgba[gt_mask > 0] = mask_colour    # Set colour ONLY where mask=1
mask_pil = Image.fromarray(mask_rgba, "RGBA")
overlay = Image.alpha_composite(overlay, mask_pil)  # Blend with existing overlay
```

`Image.alpha_composite` blends two RGBA images, respecting the alpha channel. Where the mask is 0, it's fully transparent and doesn't affect the photo.

```
LINE 280: draw = ImageDraw.Draw(overlay)  # reacquire after composite
```
After `alpha_composite`, we need to re-get the drawing pen because the overlay object was replaced.

```
LINE 282: result = Image.alpha_composite(pil_img, overlay)
LINE 283: return np.array(result)
```
Blend the overlay (with all our drawings) onto the original photo, then convert back to a numpy array for Viser.

---

## 9. Error Graphs — interactive Plotly charts

### `build_error_plotly(frame_ids, scene_gt, predictions, current_frame)`

This builds a 3-panel interactive chart.

**Panel layout**:
```
┌─────────────────────────────────────┐
│  Rotation Error (degrees)           │  ← Orange line  (#ff9944)
│  .  .  .  .   .  .   .  . . . .    │     domain: top 32% of figure
├─────────────────────────────────────┤
│  Translation Error (mm)             │  ← Blue line   (#44aaff)
│  . .  .  . .  .  . .  .  .  . .    │     domain: middle 30%
├─────────────────────────────────────┤
│  Mask Error (1 - IOU)               │  ← Green line  (#44ff88)
│  . .. . .  . . . .. .  . .. . .     │     domain: bottom 32%
│                                     │
│  Frame ID ──────────────────────────│  ← Shared X-axis
└─────────────────────────────────────┘
```

#### How the data is collected:
```python
for f_id in frame_ids:
    for gt_obj, pred_obj in zip(scene_gt[f_str], predictions[f_str]):
        R_gt = np.array(gt_obj["cam_R_m2c"]).reshape(3, 3)
        R_pred = np.array(pred_obj["cam_R_m2c"]).reshape(3, 3)
        rot_errors.append((f_id, rotation_error_deg(R_gt, R_pred), obj_id))
```

We loop through every frame, and for every object in that frame, we:
1. Get the GT and predicted rotation matrices
2. Compute the rotation error using our formula
3. Store it as a tuple: `(frame_number, error_value, object_id)`

#### Unpacking the data:
```python
def _unpack(data):
    if not data: return [], [], []
    xs, ys, ids = zip(*data)
    return list(xs), list(ys), list(ids)
```

`zip(*data)` is a Python trick that transposes a list of tuples. For example:
```python
data = [(0, 1.5, 1), (1, 3.2, 1), (2, 2.1, 1)]
xs, ys, ids = zip(*data)
# xs  = (0, 1, 2)
# ys  = (1.5, 3.2, 2.1)
# ids = (1, 1, 1)
```

#### Hover tooltips:
```python
hovertemplate="Frame %{x}<br>Rot Err: %{y:.2f}°<br>Obj %{customdata[0]}<extra></extra>"
```

When you hover over a data point, Plotly shows:
```
Frame 42
Rot Err: 5.37°
Obj 1
```

`%{x}` = the x value (frame number), `%{y:.2f}` = y value formatted to 2 decimal places, `customdata[0]` = the object ID we attached.

#### The current-frame indicator:
```python
shapes.append(dict(
    type="line", x0=current_frame, x1=current_frame,
    y0=0, y1=1, yref="paper",
    line=dict(color="#ffdd00", width=1.5, dash="dash"),
))
```

This draws a **dashed yellow vertical line** across all three panels at the current frame. `yref="paper"` means y0=0 to y1=1 spans the entire figure height regardless of data values.

---

## 10. Main Dashboard — the Viser server and GUI

```python
server = viser.ViserServer()
```
Starts a web server at `http://localhost:8080`. Any browser that opens this URL will see the dashboard.

### GUI Controls (left sidebar)
```python
with server.gui.add_folder("🎮 Controls"):    # Creates a collapsible folder
    frame_slider = server.gui.add_slider(...)  # Numbered slider 0..1826
    btn_prev = server.gui.add_button("◀ Prev Frame")
    btn_next = server.gui.add_button("Next Frame ▶")
    show_gt_cb = server.gui.add_checkbox("Show Ground Truth", initial_value=True)
    show_pred_cb = server.gui.add_checkbox("Show Predictions", initial_value=True)
    stats_md = server.gui.add_markdown(...)    # Text area for error stats
```

### 2D View Panel
```python
with server.gui.add_folder("📷 2D View (RGB + Boxes + Masks)"):
    image_gui = server.gui.add_image(composite_2d, format="png")
```
Displays the annotated RGB image. Updated each frame via `image_gui.image = new_image`.

### Error Graph Panel
```python
with server.gui.add_folder("📈 Error Graphs"):
    graph_gui = server.gui.add_plotly(error_plotly_fig, config={...})
```
Displays the interactive Plotly chart. Updated via `graph_gui.figure = new_fig`.

### 3D Scene
```python
server.scene.add_icosphere("/origin", radius=8.0, position=(0,0,0), color=(150,150,150))
```
Adds a small grey sphere at the origin (0,0,0) as a reference point.

### Drawing 3D objects
```python
def draw_3d_objects(frame_id):
    for h in scene_handles: h.remove()      # Delete all previous objects
    scene_handles.clear()

    # Camera reference frame at origin
    server.scene.add_frame("/camera_origin", show_axes=True, ...)

    for each object in this frame:
        if show_gt_cb.value:                 # Only if GT checkbox is checked
            # Place the CAD mesh at the GT position with the GT rotation
            server.scene.add_mesh_simple(
                f"/gt_mesh_{obj_id}",
                vertices=mesh.vertices,      # The 3D points from the .ply file
                faces=mesh.faces,            # The triangles
                color=(0, 180, 0),           # Solid green
                position=t_gt,               # Where to place it (from scene_gt)
                wxyz=wxyz_gt,                # How to rotate it (matrix→quaternion)
            )
            # Also draw the coordinate axes (red=X, green=Y, blue=Z)
            server.scene.add_frame(..., position=t_gt, wxyz=wxyz_gt)

        if show_pred_cb.value:               # Same but for prediction
            server.scene.add_mesh_simple(
                ...,
                color=(180, 20, 20),         # Solid red
                position=t_pred,             # Slightly different from GT
                wxyz=wxyz_pred,
            )
```

**How the 3D placement works**:

The data in the JSON gives `cam_t_m2c` (the translation) and `cam_R_m2c` (the rotation). These together form the **6D pose**:

```
Pose = (Position in 3D space, Rotation in 3D space)
```

When we call `add_mesh_simple(position=t_gt, wxyz=wxyz_gt)`:
1. Viser translates (moves) the mesh so its center is at `t_gt`
2. Viser rotates the mesh by the quaternion `wxyz_gt`
3. The mesh is rendered in solid green

**Green mesh at GT pose** — shows where the object actually is.
**Red mesh at Pred pose** — shows where the model thinks it is.
**The visual difference between them IS the pose error.**

---

## 11. Event Callbacks — making the dashboard interactive

```python
@frame_slider.on_update
def _(_):
    update_all(int(frame_slider.value))
```

`@frame_slider.on_update` is a **decorator** — it tells Viser: "when the user moves this slider, run this function". The function `_(_)` ignores both its arguments (hence the `_` names) and calls `update_all()` with the current slider value.

The same pattern applies to buttons and checkboxes:

```python
@btn_prev.on_click             # When "Prev Frame" button is clicked
def _(_):
    frame_slider.value = max(min_frame, int(frame_slider.value) - 1)
    # Changing frame_slider.value triggers the @frame_slider.on_update callback!

@show_gt_cb.on_update          # When GT checkbox is toggled
def _(_):
    update_all(int(frame_slider.value))   # Redraw everything
```

Notice how the Prev/Next buttons work: they **change the slider value**, which then **triggers the slider callback**, which calls `update_all()`. This is a chain reaction:
```
Button click → slider value changes → slider callback fires → update_all() runs
```

### The `update_all(frame_id)` function:
```python
def update_all(frame_id):
    # 1. Re-render the 2D composite image for this frame
    comp = render_2d_composite(frame_id, ...)
    image_gui.image = comp              # Update the viser image element

    # 2. Re-draw all 3D objects
    draw_3d_objects(frame_id)

    # 3. Re-draw the Plotly graph with the yellow line at this frame
    eg = build_error_plotly(frame_ids, ..., current_frame=frame_id)
    graph_gui.figure = eg

    # 4. Update the stats text (rotation error, translation error, IOU)
    stats_md.content = f"### Frame {frame_id}\n..."
```

### Server keep-alive:
```python
try:
    while True:
        time.sleep(0.5)       # Sleep 0.5 seconds, repeat forever
except KeyboardInterrupt:     # User pressed Ctrl+C
    print("Shutting down.")
```

The `while True` loop prevents the Python script from ending. If it ended, the Viser server would shut down and your browser would lose the dashboard.

---

## 12. Full Data Flow Diagram

```
                  ┌─────────────────────────────────────┐
                  │        BOP JSON Files                │
                  │  scene_gt.json (3D poses)            │
                  │  scene_camera.json (camera info)     │
                  │  scene_gt_info.json (2D boxes)       │
                  └───────────┬─────────────────────────┘
                              │ load_json()
                              ▼
                  ┌─────────────────────────────────────┐
                  │         Load & Validate              │
                  │  Parse 1827 frames, extract R & t    │
                  └───────────┬─────────────────────────┘
                              │
          ┌───────────────────┴───────────────────┐
          ▼                                       ▼
┌─────────────────────┐              ┌─────────────────────┐
│  Generate Mock      │              │  Load 3D CAD Meshes  │
│  Predictions        │              │  (.ply files)        │
│                     │              │  obj_000001.ply, etc │
│  Pose: Gaussian +   │              │  ↓ trimesh.load()    │
│    Rodrigues noise  │              │  Stored in cache     │
│  Mask: erosion/     │              └──────────┬──────────┘
│    dilation noise   │                         │
└──────────┬──────────┘                         │
           │                                    │
           └────────────────┬───────────────────┘
                            │
                            ▼
                  ┌─────────────────────────────────────┐
                  │         Viser Dashboard              │
                  │         http://localhost:8080        │
                  │                                     │
                  │  ┌──────────────────────────────┐   │
                  │  │ 🎮 Controls                   │   │
                  │  │  Frame slider, buttons,       │   │
                  │  │  GT/Pred checkboxes, stats    │   │
                  │  └──────────────────────────────┘   │
                  │  ┌──────────────────────────────┐   │
                  │  │ 📷 2D View                    │   │
                  │  │  RGB photo + bounding boxes   │   │
                  │  │  + mask overlays              │   │
                  │  │  (render_2d_composite())      │   │
                  │  └──────────────────────────────┘   │
                  │  ┌──────────────────────────────┐   │
                  │  │ 📈 Error Graphs               │   │
                  │  │  Interactive Plotly 3-panel   │   │
                  │  │  (build_error_plotly())       │   │
                  │  └──────────────────────────────┘   │
                  │                                     │
                  │  ┌──────────────────────────────┐   │
                  │  │  🖼 3D Viewport               │   │
                  │  │  Green mesh at GT pose        │   │
                  │  │  Red mesh at Pred pose        │   │
                  │  │  (draw_3d_objects())          │   │
                  │  └──────────────────────────────┘   │
                  │                                     │
                  │  User moves slider                  │
                  │        │                            │
                  │        ▼                            │
                  │  update_all(frame_id)               │
                  │  ┌──────────────────────────────┐   │
                  │  │ 1. image_gui.image = new_img │   │
                  │  │ 2. draw_3d_objects(frame)    │   │
                  │  │ 3. graph_gui.figure = new_fig│   │
                  │  │ 4. stats_md.content = text   │   │
                  │  └──────────────────────────────┘   │
                  └─────────────────────────────────────┘
```

---

## Summary

This program is a **full-stack interactive dashboard** that:

1. **Reads** 3D pose data from BOP-format JSON files
2. **Generates** fake predictions by adding noise to ground truth
3. **Renders** a 2D view (annotated photos) and a 3D view (CAD models at their poses)
4. **Computes** three types of error (rotation angle, translation distance, mask IOU)
5. **Displays** everything in an interactive web dashboard via Viser
6. **Updates** in real-time as the user scrubs through frames

Every piece connects: the slider triggers `update_all()`, which re-renders the 2D composite, re-draws the 3D meshes, rebuilds the Plotly graph, and updates the stats panel — all in a fraction of a second per frame.
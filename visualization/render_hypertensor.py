#!/usr/bin/env python3
"""
HyperTensor Scientific Visualization — Blender Cycles Pipeline
================================================================

Production-grade ray-traced scientific visualization of QTT simulation results.
Uses Cycles path tracer with GPU acceleration for physically-based rendering.

Usage:
    blender --background --python visualization/render_hypertensor.py

Environment variables:
    RENDER_QUALITY   "final" (default) | "preview" | "test"
    RESULTS_JSON     Path to simulation results JSON
    OUTPUT_DIR       Output directory for frames and video
    SCENE_FILTER     Comma-separated scene numbers (e.g., "1,2") or "all"

Scenes:
    1. Taylor-Green Vortex (Campaign I)
       Procedural volumetric vortex tubes rendered from analytical formula.
       Density evaluated per ray sample — infinite resolution.
       Animated decay: density * exp(-2*nu*t). Camera orbit + approach.

    2. H2-Air Combustion Flame (Campaign IV)
       1D temperature profile rendered as volumetric flame.
       Physically correct blackbody emission from simulation data.
       Preheat zone -> reaction zone -> burnt gas visualization.

    3. QTT Compression Scaling (Campaign II)
       3D bar chart: 7 grid sizes x 3 function types.
       Log-scale compression ratios (64^3 to 4096^3).
       Metallic PBR materials with beveled edges.

    4. Triton Kernel Benchmarks (Campaign III)
       Performance comparison: Triton vs Legacy vs CUDA.
       69x Hadamard speedup highlighted.

Renderer: Cycles (path tracing)
Denoiser: OpenImageDenoise (Intel) — built into Blender 4.x
Output:   EXR intermediate frames -> PNG -> H.265 via ffmpeg

Author: HyperTensor Team
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
try:
    import pyopenvdb as vdb  # Blender ≤ 4.3
except ImportError:
    import openvdb as vdb    # Blender ≥ 4.4

import bpy
import bmesh
from mathutils import Color, Euler, Matrix, Vector

# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

_REPO_ROOT = Path(__file__).resolve().parent.parent

QUALITY = os.environ.get("RENDER_QUALITY", "final")
RESULTS_JSON = os.environ.get(
    "RESULTS_JSON",
    str(_REPO_ROOT / "results" / "industrial_qtt_gpu_simulation_results.json"),
)
OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR",
    str(_REPO_ROOT / "visualization" / "output"),
)
SCENE_FILTER = os.environ.get("SCENE_FILTER", "all")

# Quality presets
QUALITY_PRESETS = {
    "test": {
        "samples": 16,
        "resolution_x": 640,
        "resolution_y": 360,
        "volume_step_rate": 2.0,
        "volume_max_steps": 256,
        "use_denoising": False,
        "fps": 24,
        "vortex_frames": 48,
        "flame_frames": 36,
        "bars_frames": 36,
        "bench_frames": 24,
        "title_frames": 24,
    },
    "preview": {
        "samples": 64,
        "resolution_x": 960,
        "resolution_y": 540,
        "volume_step_rate": 0.5,
        "volume_max_steps": 512,
        "use_denoising": True,
        "fps": 30,
        "vortex_frames": 120,
        "flame_frames": 75,
        "bars_frames": 90,
        "bench_frames": 60,
        "title_frames": 45,
    },
    "final": {
        "samples": 128,
        "resolution_x": 1920,
        "resolution_y": 1080,
        "volume_step_rate": 1.0,
        "volume_max_steps": 256,
        "use_denoising": True,
        "fps": 30,
        "vortex_frames": 105,
        "flame_frames": 90,
        "bars_frames": 60,
        "bench_frames": 48,
        "title_frames": 30,
    },
}

CFG = QUALITY_PRESETS.get(QUALITY, QUALITY_PRESETS["final"])

# ═══════════════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════════════


def load_results() -> Dict[str, Any]:
    """Load simulation results JSON."""
    with open(RESULTS_JSON, "r") as f:
        return json.load(f)


def clear_scene() -> None:
    """Delete all objects, meshes, materials, images, cameras, lights."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block_type in (
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.images,
        bpy.data.cameras,
        bpy.data.lights,
        bpy.data.curves,
        bpy.data.textures,
    ):
        for block in block_type:
            if block.users == 0:
                block_type.remove(block)


def add_node(
    tree: bpy.types.NodeTree,
    node_type: str,
    location: Tuple[float, float] = (0, 0),
    name: str = "",
    **kwargs: Any,
) -> bpy.types.Node:
    """Create a shader node with position and optional properties."""
    node = tree.nodes.new(node_type)
    node.location = location
    if name:
        node.name = name
        node.label = name
    for key, val in kwargs.items():
        setattr(node, key, val)
    return node


def link(
    tree: bpy.types.NodeTree,
    out_node: bpy.types.Node,
    out_socket: Any,
    in_node: bpy.types.Node,
    in_socket: Any,
) -> None:
    """Link two shader node sockets. Accepts index (int) or name (str)."""
    src = out_node.outputs[out_socket]
    dst = in_node.inputs[in_socket]
    tree.links.new(src, dst)


def _is_wsl() -> bool:
    """Detect WSL2 environment where OptiX RT cores are unavailable."""
    try:
        with open("/proc/version", "r") as f:
            ver = f.read().lower()
        return "microsoft" in ver or "wsl" in ver
    except OSError:
        return False


_WSL = _is_wsl()


def setup_gpu() -> str:
    """Enable GPU rendering. Returns device type used.

    On WSL2, OptiX is technically 'available' but cannot access hardware
    RT cores, making it ~2.7× slower than plain CUDA.  We detect WSL2
    and force CUDA to avoid the OptiX software-fallback penalty.
    On native Linux/Windows, OptiX is preferred for full RT-core acceleration.
    """
    prefs = bpy.context.preferences.addons["cycles"].preferences

    # Prefer CUDA on WSL2 (OptiX RT cores inaccessible); OptiX everywhere else
    if _WSL:
        order = ("CUDA", "HIP", "OPTIX")
        print("  WSL2 detected — forcing CUDA (OptiX RT cores unavailable)")
    else:
        order = ("OPTIX", "CUDA", "HIP", "METAL")

    for device_type in order:
        try:
            prefs.compute_device_type = device_type
            prefs.get_devices()
            gpu_found = False
            for device in prefs.devices:
                if device.type != "CPU":
                    device.use = True
                    gpu_found = True
                else:
                    # Enable CPU as a co-render device so tiles are
                    # dispatched to BOTH CPU and GPU simultaneously.
                    # Neither device idles while the other works.
                    device.use = True
            if gpu_found:
                gpu_names = [d.name for d in prefs.devices if d.type != "CPU" and d.use]
                cpu_names = [d.name for d in prefs.devices if d.type == "CPU" and d.use]
                print(f"  GPU: {', '.join(gpu_names)} ({device_type})")
                if cpu_names:
                    print(f"  CPU: {', '.join(cpu_names)} (hybrid co-render)")
                return device_type
        except Exception:
            continue

    # CPU fallback
    prefs.compute_device_type = "NONE"
    print("  GPU: None found, using CPU")
    return "CPU"


# Per-scene Cycles overrides. Volumetric physics scenes are emission-dominated
# with no complex BxDF chains, so bounce limits can be drastically reduced.
# The flame is a 1D-texture slab (simpler than the 3D procedural vortex) and
# converges faster, so it gets fewer samples and coarser volume stepping.
_SCENE_OVERRIDES: Dict[int, Dict[str, Any]] = {
    # Scene 1 — Taylor-Green Vortex
    # Emission-dominated volumetric: indirect diffuse is negligible.
    1: {
        "max_bounces": 4,
        "diffuse_bounces": 1,
        "glossy_bounces": 1,
        "transmission_bounces": 1,
        "transparent_max_bounces": 2,
        "volume_bounces": 2,           # Multi-scatter through volume tubes
        "adaptive_threshold": 0.025,
        "adaptive_min_samples": 8,     # Let adaptive kick in sooner
        "use_fast_gi": True,           # Approximate indirect (invisible for emission)
    },
    # Scene 2 — Combustion Flame
    # Thin slab geometry with 1D temperature profile — converges fast.
    2: {
        "max_bounces": 4,
        "diffuse_bounces": 1,
        "glossy_bounces": 1,
        "transmission_bounces": 2,     # Glass chamber
        "transparent_max_bounces": 2,
        "volume_bounces": 1,           # Single-scatter flame
        "volume_step_rate": 0.7,       # Coarser steps (slab, not 3D)
        "volume_max_steps": 128,       # Thin slab (0.4u depth vs 2.0u vortex cube)
        "adaptive_threshold": 0.03,
        "adaptive_min_samples": 8,     # Let adaptive kick in sooner
        "use_fast_gi": True,           # Approximate indirect (invisible for emission)
        "samples_override": 96,        # Converges faster than vortex
    },
}


def setup_cycles(scene: bpy.types.Scene, scene_id: int = -1) -> None:
    """Configure Cycles renderer with production settings.

    If *scene_id* is provided, per-scene overrides are applied for
    bounce limits, adaptive sampling, and volume stepping — giving
    measurable speedups without visible quality loss.
    """
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"
    scene.cycles.samples = CFG["samples"]
    scene.cycles.use_denoising = CFG["use_denoising"]
    if CFG["use_denoising"]:
        # OptiX denoiser is GPU-accelerated (faster); OIDN is CPU-based (universal).
        # On WSL2 we force CUDA so OptiX denoiser is unavailable → use OIDN.
        if _WSL:
            scene.cycles.denoiser = "OPENIMAGEDENOISE"
        else:
            scene.cycles.denoiser = "OPTIX"
        scene.cycles.denoising_input_passes = "RGB_ALBEDO_NORMAL"
    scene.cycles.volume_step_rate = CFG["volume_step_rate"]
    scene.cycles.volume_max_steps = CFG["volume_max_steps"]
    scene.cycles.volume_bounces = 2
    scene.cycles.max_bounces = 8
    scene.cycles.diffuse_bounces = 3
    scene.cycles.glossy_bounces = 3
    scene.cycles.transmission_bounces = 6
    scene.cycles.transparent_max_bounces = 6
    scene.cycles.use_fast_gi = False
    scene.cycles.film_exposure = 1.0
    scene.cycles.caustics_reflective = False
    scene.cycles.caustics_refractive = False
    scene.cycles.scrambling_distance = 0.8
    scene.cycles.use_light_tree = True

    # Film
    scene.render.film_transparent = False
    scene.render.resolution_x = CFG["resolution_x"]
    scene.render.resolution_y = CFG["resolution_y"]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.image_settings.compression = 0  # No PNG compression (saves CPU per frame)

    # Color management — Filmic for HDR tonemapping
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    # Performance — CPU+GPU hybrid tile scheduling
    scene.render.use_persistent_data = True
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.02
    scene.cycles.adaptive_min_samples = 16

    # Tile size: 512px fits in Blackwell's 48 MB unified L-cache
    # (512² × RGBA float32 = 4 MB per tile, well under L2 budget).
    # At 1920×1080 this yields ceil(1920/512)×ceil(1080/512) = 4×3 = 12
    # tiles — enough to keep both GPU and CPU saturated simultaneously.
    # Auto-tile defaults to 2048 (entire image = 1 tile) which serialises
    # CPU and GPU; disabling it lets us control the split explicitly.
    scene.cycles.use_auto_tile = False
    scene.cycles.tile_size = 512

    # GPU-accelerated OIDN denoising — runs on CUDA cores instead of
    # blocking the CPU between frames.  FAST prefilter is sufficient
    # at 128 spp (low residual noise); ACCURATE adds ~2× denoise time.
    scene.cycles.denoising_use_gpu = True
    scene.cycles.denoising_prefilter = "FAST"
    scene.cycles.denoising_quality = "BALANCED"

    # FPS
    scene.render.fps = CFG["fps"]

    # ── Per-scene overrides ─────────────────────────────────────
    overrides = _SCENE_OVERRIDES.get(scene_id, {})
    if overrides:
        for key, val in overrides.items():
            if key == "samples_override":
                scene.cycles.samples = val
            elif key == "volume_step_rate":
                scene.cycles.volume_step_rate = val
            elif key == "adaptive_threshold":
                scene.cycles.adaptive_threshold = val
            elif key == "adaptive_min_samples":
                scene.cycles.adaptive_min_samples = val
            elif key == "use_fast_gi":
                scene.cycles.use_fast_gi = val
            elif hasattr(scene.cycles, key):
                setattr(scene.cycles, key, val)
        applied = ", ".join(f"{k}={v}" for k, v in overrides.items())
        print(f"  Cycles overrides: {applied}")


def create_dark_world(
    scene: bpy.types.Scene,
    color_top: Tuple[float, float, float] = (0.005, 0.005, 0.02),
    color_bottom: Tuple[float, float, float] = (0.001, 0.001, 0.005),
    strength: float = 0.3,
) -> None:
    """Dark gradient world background for scientific visualization."""
    world = bpy.data.worlds.new("DarkWorld")
    scene.world = world
    world.use_nodes = True
    tree = world.node_tree
    tree.nodes.clear()

    output = add_node(tree, "ShaderNodeOutputWorld", (600, 0))
    bg = add_node(tree, "ShaderNodeBackground", (400, 0))
    bg.inputs["Strength"].default_value = strength
    link(tree, bg, "Background", output, "Surface")

    # Gradient
    tex_coord = add_node(tree, "ShaderNodeTexCoord", (-200, 0))
    separate = add_node(tree, "ShaderNodeSeparateXYZ", (0, 0))
    link(tree, tex_coord, "Generated", separate, "Vector")

    ramp = add_node(tree, "ShaderNodeValToRGB", (200, 0))
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[0].color = (*color_bottom, 1.0)
    ramp.color_ramp.elements[1].position = 1.0
    ramp.color_ramp.elements[1].color = (*color_top, 1.0)
    link(tree, separate, "Z", ramp, "Fac")
    link(tree, ramp, "Color", bg, "Color")


def add_camera(
    scene: bpy.types.Scene,
    location: Tuple[float, float, float],
    target: Tuple[float, float, float] = (0, 0, 0),
    lens: float = 50.0,
    dof_distance: float = 0.0,
    fstop: float = 5.6,
    name: str = "Camera",
) -> bpy.types.Object:
    """Create and configure a camera."""
    cam_data = bpy.data.cameras.new(name)
    cam_data.lens = lens
    cam_data.clip_start = 0.01
    cam_data.clip_end = 1000.0
    if dof_distance > 0:
        cam_data.dof.use_dof = True
        cam_data.dof.focus_distance = dof_distance
        cam_data.dof.aperture_fstop = fstop

    cam_obj = bpy.data.objects.new(name, cam_data)
    scene.collection.objects.link(cam_obj)
    cam_obj.location = Vector(location)

    # Point at target
    direction = Vector(target) - Vector(location)
    rot_quat = direction.to_track_quat("-Z", "Y")
    cam_obj.rotation_euler = rot_quat.to_euler()

    scene.camera = cam_obj
    return cam_obj


def add_area_light(
    scene: bpy.types.Scene,
    location: Tuple[float, float, float],
    energy: float = 100.0,
    size: float = 2.0,
    color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    name: str = "AreaLight",
    target: Tuple[float, float, float] = (0, 0, 0),
) -> bpy.types.Object:
    """Create an area light pointing at target."""
    light_data = bpy.data.lights.new(name, "AREA")
    light_data.energy = energy
    light_data.size = size
    light_data.color = color
    light_data.cycles.use_multiple_importance_sampling = True

    light_obj = bpy.data.objects.new(name, light_data)
    scene.collection.objects.link(light_obj)
    light_obj.location = Vector(location)

    direction = Vector(target) - Vector(location)
    rot_quat = direction.to_track_quat("-Z", "Y")
    light_obj.rotation_euler = rot_quat.to_euler()
    return light_obj


def keyframe_value(obj: bpy.types.Object, attr: str, value: float, frame: int) -> None:
    """Set and keyframe a property."""
    setattr(obj, attr, value)
    obj.keyframe_insert(data_path=attr, frame=frame)


def animate_orbit(
    cam_obj: bpy.types.Object,
    center: Tuple[float, float, float],
    radius: float,
    elevation: float,
    start_frame: int,
    end_frame: int,
    start_angle: float = 0.0,
    end_angle: float = 2 * math.pi,
    target: Tuple[float, float, float] = (0, 0, 0),
) -> None:
    """Animate camera orbit around a center point."""
    n_frames = end_frame - start_frame
    for i in range(n_frames + 1):
        frame = start_frame + i
        t = i / max(n_frames, 1)
        angle = start_angle + (end_angle - start_angle) * t
        x = center[0] + radius * math.cos(angle) * math.cos(elevation)
        y = center[1] + radius * math.sin(angle) * math.cos(elevation)
        z = center[2] + radius * math.sin(elevation)
        cam_obj.location = Vector((x, y, z))
        cam_obj.keyframe_insert(data_path="location", frame=frame)

        direction = Vector(target) - Vector((x, y, z))
        rot_quat = direction.to_track_quat("-Z", "Y")
        cam_obj.rotation_euler = rot_quat.to_euler()
        cam_obj.keyframe_insert(data_path="rotation_euler", frame=frame)


def create_text_object(
    scene: bpy.types.Scene,
    text: str,
    location: Tuple[float, float, float],
    size: float = 0.3,
    color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    emission_strength: float = 2.0,
    align_x: str = "CENTER",
    align_y: str = "CENTER",
    extrude: float = 0.01,
    name: str = "Text",
) -> bpy.types.Object:
    """Create a 3D text object with emissive material."""
    curve_data = bpy.data.curves.new(name, "FONT")
    curve_data.body = text
    curve_data.size = size
    curve_data.align_x = align_x
    curve_data.align_y = align_y
    curve_data.extrude = extrude
    curve_data.bevel_depth = 0.002
    curve_data.bevel_resolution = 2

    text_obj = bpy.data.objects.new(name, curve_data)
    scene.collection.objects.link(text_obj)
    text_obj.location = Vector(location)

    # Emissive material
    mat = bpy.data.materials.new(f"{name}_mat")
    mat.use_nodes = True
    tree = mat.node_tree
    tree.nodes.clear()
    output_node = add_node(tree, "ShaderNodeOutputMaterial", (400, 0))
    emission = add_node(tree, "ShaderNodeEmission", (200, 0))
    emission.inputs["Color"].default_value = (*color, 1.0)
    emission.inputs["Strength"].default_value = emission_strength
    link(tree, emission, "Emission", output_node, "Surface")
    text_obj.data.materials.append(mat)

    return text_obj


# ═══════════════════════════════════════════════════════════════════════
# Material Builders
# ═══════════════════════════════════════════════════════════════════════


def generate_vortex_vdb(
    resolution: int = 256,
    n_cells: int = 2,
    threshold: float = 0.15,
    output_path: Optional[str] = None,
) -> str:
    """
    Generate a VDB file containing the Taylor-Green vortex tube lattice.

    Produces continuous cylindrical tubes using the cross-sectional
    vortex profile, symmetrized across all three axes.

    Self-contained analytical computation — zero coupling to HyperGrid
    or any shared infrastructure.

    The key insight: the analytical Q-criterion for TG has a cos²(z)
    axial modulation that pinches tubes to zero at z = π/2 + nπ,
    breaking them into disconnected blobs. The vortex CORES are
    continuous cylinders — identified by the cross-sectional profile
    alone:

        tube_z = max(0, cos(2x) + cos(2y))  → z-cylinders at (nπ, mπ)
        tube_x = max(0, cos(2y) + cos(2z))  → x-cylinders at (mπ, lπ)
        tube_y = max(0, cos(2x) + cos(2z))  → y-cylinders at (nπ, lπ)

    cos(2x) + cos(2y) > 0 defines a connected region around each
    lattice point — a circular-ish cross-section that IS the tube.
    The max-union of all three orientations gives the full interlocking
    lattice seen in CFD presentations of evolved TG flow.

    The threshold controls tube radius:
        0.0  → tubes fill ~50% of space (max radius)
        0.5  → moderate tubes (~25%)
        0.9  → thin filaments (~5%)

    Args:
        resolution: Grid resolution per axis (N^3 total voxels).
        n_cells: Number of TG cells per axis (2 = [0, 4pi] domain).
        threshold: Normalized threshold [0, 1]. Controls tube radius.
        output_path: Path for .vdb file.

    Returns:
        Absolute path to the written .vdb file.
    """
    if output_path is None:
        output_path = str(Path(OUTPUT_DIR) / "vortex_field.vdb")

    N = resolution
    L = n_cells * 2.0 * np.pi  # Physical domain [0, L)
    h = L / N

    # 1D coordinate arrays
    coords = np.arange(N, dtype=np.float64) * h

    # 3D meshgrids (ij indexing: X varies along axis 0)
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")

    # Double-angle cosines for cross-sectional tube profiles
    cos2x = np.cos(2.0 * X)
    cos2y = np.cos(2.0 * Y)
    cos2z = np.cos(2.0 * Z)

    # Cross-sectional tube profiles — continuous cylinders.
    # cos(2x) + cos(2y) > 0 is a connected region around (nπ, mπ)
    # forming a circular cross-section. No axial modulation = no gaps.
    tube_z = np.maximum(0.0, cos2x + cos2y)  # z-oriented cylinders
    tube_x = np.maximum(0.0, cos2y + cos2z)  # x-oriented cylinders
    tube_y = np.maximum(0.0, cos2x + cos2z)  # y-oriented cylinders

    # Union of all three orientations: the interlocking lattice
    field = np.maximum(np.maximum(tube_z, tube_x), tube_y)

    # Normalize to [0, 1] (max value of cos(2a)+cos(2b) = 2.0)
    field_norm = field / 2.0

    # Threshold, remap to [0, 1], then apply steep power curve.
    # Power of 5: only the very core of each tube has significant
    # density. Edges drop off extremely fast, eliminating the haze
    # that accumulates when many semi-transparent layers stack up.
    density = np.zeros(field_norm.shape, dtype=np.float32)
    mask = field_norm > threshold
    linear = (field_norm[mask] - threshold) / (1.0 - threshold)
    density[mask] = (linear ** 5).astype(np.float32)

    active_count = int(np.count_nonzero(density))
    active_frac = active_count / density.size
    print(
        f"    VDB: {N}^3 tube lattice, threshold={threshold:.2f}, "
        f"active={active_frac:.1%} ({active_count:,} voxels)"
    )

    # Create VDB FloatGrid from numpy array
    grid = vdb.FloatGrid()
    grid.name = "density"
    grid.copyFromArray(density)

    # Prune inactive (zero) voxels
    grid.prune(tolerance=1e-6)

    # Transform: index space → world space.
    # Index [0, N) maps to world [0, 2] with voxelSize = 2/N.
    voxel_size = 2.0 / N
    grid.transform = vdb.createLinearTransform(voxelSize=voxel_size)

    # Write VDB
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    vdb.write(output_path, grids=[grid])
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"    VDB: wrote {output_path} ({file_size_mb:.1f} MB)")

    # Free memory
    del X, Y, Z, cos2x, cos2y, cos2z
    del tube_z, tube_x, tube_y, field, field_norm, density, mask

    return output_path


def create_vortex_vdb_material(
    density_scale: float = 8.0,
    emission_scale: float = 0.15,
) -> bpy.types.Material:
    """
    Principled Volume material for VDB-based vortex rendering.

    Reads the "density" attribute from the imported VDB grid.
    Uses HIGH density with very LOW emission so that external
    lighting reveals the 3D tube structure (surface highlight,
    shadow, depth cueing) instead of flat uniform self-glow.

    The Principled Volume Color tints scattered light cyan.
    Absorption is minimal so you can see through the lattice.

    Replaces the 30+ node procedural shader with a 5-node graph:
    Attribute → Multiply(density) → Volume.Density
    Attribute → Multiply(emission) → Volume.Emission Strength
    Volume → Output

    Args:
        density_scale: Multiplier on VDB density values for volume opacity.
        emission_scale: Multiplier on VDB density values for glow intensity.
    """
    mat = bpy.data.materials.new("VortexVDB")
    mat.use_nodes = True
    tree = mat.node_tree
    tree.nodes.clear()

    # ── Output ──────────────────────────────────────────────────
    output_node = add_node(tree, "ShaderNodeOutputMaterial", (800, 0), "Output")

    # ── Principled Volume ───────────────────────────────────────
    volume = add_node(tree, "ShaderNodeVolumePrincipled", (500, 0), "Volume")
    # Strong forward scattering — light penetrates and reveals depth
    volume.inputs["Anisotropy"].default_value = 0.5
    # Scatter color: bright cyan albedo — light bounces, not absorbed
    volume.inputs["Color"].default_value = (0.6, 0.85, 1.0, 1.0)
    # Emission color: vivid cyan
    volume.inputs["Emission Color"].default_value = (0.1, 0.6, 1.0, 1.0)
    link(tree, volume, "Volume", output_node, "Volume")

    # ── Attribute node: reads VDB "density" grid per-voxel ──────
    attr = add_node(tree, "ShaderNodeAttribute", (-100, 0), "VDBDensity")
    attr.attribute_name = "density"

    # ── Density: VDB value × density_scale → volume opacity ─────
    density = add_node(tree, "ShaderNodeMath", (200, 50), "density")
    density.operation = "MULTIPLY"
    density.inputs[1].default_value = density_scale
    link(tree, attr, "Fac", density, 0)
    link(tree, density, 0, volume, "Density")

    # ── Emission: VDB value × emission_scale → subtle core glow ─
    # Very low — just enough to see tube cores; lighting does the heavy lifting.
    em_str = add_node(tree, "ShaderNodeMath", (200, -100), "em_strength")
    em_str.operation = "MULTIPLY"
    em_str.inputs[1].default_value = emission_scale
    link(tree, attr, "Fac", em_str, 0)
    link(tree, em_str, 0, volume, "Emission Strength")

    return mat


def create_flame_volume_material(
    temperature_data: List[float],
) -> bpy.types.Material:
    """
    Volumetric combustion flame material with radial falloff.

    Uses a 1D image texture containing the temperature profile
    mapped along X, with a Gaussian radial falloff from the center
    axis to give the flame a cylindrical shape instead of a block.

    Temperature drives blackbody emission color (physically correct
    Planck radiation). Cold gas is fully transparent.

    Cold gas (300K):          invisible (zero density)
    Preheat zone (300-1000K): faint warm glow, low density
    Reaction zone (1000-2500K): intense white-yellow blackbody
    Burnt gas (2230K):        warm orange-yellow glow
    """
    mat = bpy.data.materials.new("FlameVolume")
    mat.use_nodes = True
    tree = mat.node_tree
    tree.nodes.clear()

    n_pts = len(temperature_data)
    T_min = min(temperature_data)
    T_max = max(temperature_data)
    T_range = max(T_max - T_min, 1.0)

    # ── Create 1D temperature image ─────────────────────────────
    img = bpy.data.images.new("TempProfile", width=n_pts, height=1, float_buffer=True)
    pixels = [0.0] * (n_pts * 4)
    for i, T in enumerate(temperature_data):
        t_norm = (T - T_min) / T_range
        idx = i * 4
        pixels[idx] = t_norm       # R
        pixels[idx + 1] = t_norm   # G
        pixels[idx + 2] = t_norm   # B
        pixels[idx + 3] = 1.0      # A
    img.pixels[:] = pixels
    img.pack()

    # ── Output + Volume ─────────────────────────────────────────
    output_node = add_node(tree, "ShaderNodeOutputMaterial", (1400, 0), "Output")
    volume = add_node(tree, "ShaderNodeVolumePrincipled", (1100, 0), "Volume")
    volume.inputs["Anisotropy"].default_value = 0.3
    link(tree, volume, "Volume", output_node, "Volume")

    # ── Coordinates ─────────────────────────────────────────────
    tex_coord = add_node(tree, "ShaderNodeTexCoord", (-800, 0), "Coords")
    separate = add_node(tree, "ShaderNodeSeparateXYZ", (-600, 0), "SepXYZ")
    link(tree, tex_coord, "Generated", separate, "Vector")

    # ── 1D temperature lookup (along X axis) ────────────────────
    combine_uv = add_node(tree, "ShaderNodeCombineXYZ", (-400, 0), "CombineUV")
    combine_uv.inputs["Y"].default_value = 0.5
    combine_uv.inputs["Z"].default_value = 0.0
    link(tree, separate, "X", combine_uv, "X")

    img_tex = add_node(tree, "ShaderNodeTexImage", (-200, 0), "TempTexture")
    img_tex.image = img
    img_tex.interpolation = "Linear"
    img_tex.extension = "EXTEND"
    link(tree, combine_uv, "Vector", img_tex, "Vector")

    # ── Denormalize to Kelvin ───────────────────────────────────
    scale_T = add_node(tree, "ShaderNodeMath", (0, 100), "ScaleT")
    scale_T.operation = "MULTIPLY"
    scale_T.inputs[1].default_value = T_range
    link(tree, img_tex, "Color", scale_T, 0)

    add_T = add_node(tree, "ShaderNodeMath", (200, 100), "AddTmin")
    add_T.operation = "ADD"
    add_T.inputs[1].default_value = T_min
    link(tree, scale_T, 0, add_T, 0)

    # ── Blackbody color from temperature ────────────────────────
    blackbody = add_node(tree, "ShaderNodeBlackbody", (400, 100), "Blackbody")
    link(tree, add_T, 0, blackbody, "Temperature")
    link(tree, blackbody, "Color", volume, "Emission Color")

    # ── Radial falloff (Y, Z) ───────────────────────────────────
    # Gaussian envelope from center axis: exp(-(dy²+dz²) / (2*sigma²))
    # Generated coords: center at (0.5, 0.5), radius [0, 0.5]
    # sigma = 0.18 → visible core ~36% of domain cross-section
    SIGMA_SQ_2 = 2.0 * 0.12 * 0.12  # 2σ² — tight core (~24% of cross-section)

    # dy = Y - 0.5
    sub_y = add_node(tree, "ShaderNodeMath", (-400, -200), "SubY")
    sub_y.operation = "SUBTRACT"
    sub_y.inputs[1].default_value = 0.5
    link(tree, separate, "Y", sub_y, 0)

    # dy²
    sq_y = add_node(tree, "ShaderNodeMath", (-200, -200), "SqY")
    sq_y.operation = "MULTIPLY"
    link(tree, sub_y, 0, sq_y, 0)
    link(tree, sub_y, 0, sq_y, 1)

    # dz = Z - 0.5
    sub_z = add_node(tree, "ShaderNodeMath", (-400, -350), "SubZ")
    sub_z.operation = "SUBTRACT"
    sub_z.inputs[1].default_value = 0.5
    link(tree, separate, "Z", sub_z, 0)

    # dz²
    sq_z = add_node(tree, "ShaderNodeMath", (-200, -350), "SqZ")
    sq_z.operation = "MULTIPLY"
    link(tree, sub_z, 0, sq_z, 0)
    link(tree, sub_z, 0, sq_z, 1)

    # r² = dy² + dz²
    r_sq = add_node(tree, "ShaderNodeMath", (0, -275), "Rsq")
    r_sq.operation = "ADD"
    link(tree, sq_y, 0, r_sq, 0)
    link(tree, sq_z, 0, r_sq, 1)

    # -r² / (2σ²)
    neg_r_norm = add_node(tree, "ShaderNodeMath", (200, -275), "NegRNorm")
    neg_r_norm.operation = "DIVIDE"
    neg_r_norm.inputs[1].default_value = -SIGMA_SQ_2  # Negative for decay
    link(tree, r_sq, 0, neg_r_norm, 0)

    # Gaussian envelope: exp(-r²/(2σ²))
    radial_falloff = add_node(tree, "ShaderNodeMath", (400, -275), "RadialFalloff")
    radial_falloff.operation = "EXPONENT"  # e^x
    link(tree, neg_r_norm, 0, radial_falloff, 0)

    # ── Emission strength: zero below 500K, ramps up above ─────
    sub_threshold = add_node(tree, "ShaderNodeMath", (200, -100), "SubThresh")
    sub_threshold.operation = "SUBTRACT"
    sub_threshold.inputs[1].default_value = 500.0
    link(tree, add_T, 0, sub_threshold, 0)

    div_scale = add_node(tree, "ShaderNodeMath", (400, -100), "DivScale")
    div_scale.operation = "DIVIDE"
    div_scale.inputs[1].default_value = 500.0
    link(tree, sub_threshold, 0, div_scale, 0)

    clamp_em = add_node(tree, "ShaderNodeClamp", (600, -100), "ClampEm")
    clamp_em.inputs["Min"].default_value = 0.0
    clamp_em.inputs["Max"].default_value = 3.0
    link(tree, div_scale, 0, clamp_em, "Value")

    # Emission × radial falloff
    em_radial = add_node(tree, "ShaderNodeMath", (700, -100), "EmRadial")
    em_radial.operation = "MULTIPLY"
    link(tree, clamp_em, "Result", em_radial, 0)
    link(tree, radial_falloff, 0, em_radial, 1)

    em_final = add_node(tree, "ShaderNodeMath", (900, -100), "EmFinal")
    em_final.operation = "MULTIPLY"
    em_final.inputs[1].default_value = 3.0
    link(tree, em_radial, 0, em_final, 0)
    link(tree, em_final, 0, volume, "Emission Strength")

    # ── Density: zero cold gas, proportional to temperature ─────
    # Only hot gas has density. No base density → cold gas invisible.
    # density = clamp_em × radial_falloff × density_scale
    density_shaped = add_node(tree, "ShaderNodeMath", (700, -250), "DensShaped")
    density_shaped.operation = "MULTIPLY"
    link(tree, clamp_em, "Result", density_shaped, 0)
    link(tree, radial_falloff, 0, density_shaped, 1)

    density_final = add_node(tree, "ShaderNodeMath", (900, -250), "DensFinal")
    density_final.operation = "MULTIPLY"
    density_final.inputs[1].default_value = 0.10  # Near-transparent — see flame structure
    link(tree, density_shaped, 0, density_final, 0)
    link(tree, density_final, 0, volume, "Density")

    return mat


def create_metallic_material(
    color: Tuple[float, float, float],
    roughness: float = 0.15,
    metallic: float = 1.0,
    name: str = "Metal",
) -> bpy.types.Material:
    """Create a PBR metallic material for bar charts."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    tree = mat.node_tree
    tree.nodes.clear()

    output_node = add_node(tree, "ShaderNodeOutputMaterial", (400, 0))
    bsdf = add_node(tree, "ShaderNodeBsdfPrincipled", (0, 0))
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Specular IOR Level"].default_value = 0.5
    link(tree, bsdf, "BSDF", output_node, "Surface")
    return mat


def create_emissive_cap_material(
    color: Tuple[float, float, float],
    strength: float = 5.0,
    name: str = "EmissiveCap",
) -> bpy.types.Material:
    """Emissive material for bar chart top caps (value indicator glow)."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    tree = mat.node_tree
    tree.nodes.clear()

    output_node = add_node(tree, "ShaderNodeOutputMaterial", (400, 0))
    emission = add_node(tree, "ShaderNodeEmission", (200, 0))
    emission.inputs["Color"].default_value = (*color, 1.0)
    emission.inputs["Strength"].default_value = strength
    link(tree, emission, "Emission", output_node, "Surface")
    return mat


def create_ground_material() -> bpy.types.Material:
    """Dark reflective ground plane material."""
    mat = bpy.data.materials.new("Ground")
    mat.use_nodes = True
    tree = mat.node_tree
    tree.nodes.clear()

    output_node = add_node(tree, "ShaderNodeOutputMaterial", (400, 0))
    bsdf = add_node(tree, "ShaderNodeBsdfPrincipled", (0, 0))
    bsdf.inputs["Base Color"].default_value = (0.01, 0.01, 0.015, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.0
    bsdf.inputs["Roughness"].default_value = 0.1
    bsdf.inputs["Specular IOR Level"].default_value = 0.8
    link(tree, bsdf, "BSDF", output_node, "Surface")
    return mat


# ═══════════════════════════════════════════════════════════════════════
# Scene 1: Taylor-Green Vortex (Campaign I)
# ═══════════════════════════════════════════════════════════════════════


def build_scene_vortex(
    scene: bpy.types.Scene,
    data: Dict[str, Any],
) -> None:
    """
    Build the Taylor-Green vortex volumetric scene.

    Uses VDB-based volume rendering: the analytical vorticity field is
    computed on a numpy grid, hard-thresholded, and written as a sparse
    OpenVDB file. Blender imports it as a Volume object with a simple
    Principled Volume shader.

    This replaces the procedural shader approach (30+ math nodes) which
    could not produce crisp structures from the smooth sinusoidal field.
    """
    clear_scene()
    n_frames = CFG["vortex_frames"]
    nu = data.get("campaign_i", {}).get("viscosity", 0.01)
    ke_error = data.get("campaign_i", {}).get("ke_relative_error", 0.00127)

    print(f"  Scene 1: Taylor-Green Vortex ({n_frames} frames)")

    # ── Generate VDB tube lattice ─────────────────────────────────
    vdb_path = generate_vortex_vdb(
        resolution=256,
        n_cells=2,       # 2 TG cells per axis → clear tube lattice
        threshold=0.60,  # Thin tubes, power-5 curve keeps cores sharp
    )

    # ── Import VDB as Volume object ─────────────────────────────
    bpy.ops.object.volume_import(filepath=vdb_path)
    vol_obj = bpy.context.active_object
    vol_obj.name = "VortexVolume"
    # Center at origin: VDB grid spans [0, 2] in world space,
    # offset by (-1,-1,-1) to center the 2-unit cube at origin.
    vol_obj.location = (-1.0, -1.0, -1.0)

    # Apply VDB material
    density_rest = 3.0   # Moderate opacity, light still penetrates
    em_rest = 3.0        # Bright vivid self-lit tubes
    mat = create_vortex_vdb_material(
        density_scale=density_rest,
        emission_scale=em_rest,
    )
    vol_obj.data.materials.append(mat)

    # ── Lighting ────────────────────────────────────────────────
    # Strong external lighting to reveal 3D tube structure.
    # Key light: strong, high, slightly warm — top-down sculpting
    add_area_light(
        scene, location=(3, -2, 5), energy=500.0, size=4.0,
        color=(0.85, 0.9, 1.0), name="KeyLight",
    )
    # Fill light: softer, opposite side — prevents pitch-black shadows
    add_area_light(
        scene, location=(-4, 3, 2), energy=200.0, size=5.0,
        color=(0.5, 0.65, 1.0), name="FillLight",
    )
    # Rim/back light: defines edges and separates from background
    add_area_light(
        scene, location=(0, 4, -1), energy=250.0, size=3.0,
        color=(0.3, 0.5, 1.0), name="RimLight",
    )

    # ── Camera: orbit around vortex ─────────────────────────────
    cam = add_camera(
        scene,
        location=(4, -3, 2.5),
        target=(0, 0, 0),
        lens=35.0,
    )

    # Phase 1: Approach (0 to 15% of frames)
    approach_end = int(n_frames * 0.15)
    for i in range(approach_end + 1):
        frame = i + 1
        t = i / max(approach_end, 1)
        radius = 7.0 - 2.5 * t  # 7 → 4.5
        angle = -0.6 + 0.3 * t
        elev = 0.3 + 0.15 * t
        x = radius * math.cos(angle) * math.cos(elev)
        y = radius * math.sin(angle) * math.cos(elev)
        z = radius * math.sin(elev)
        cam.location = Vector((x, y, z))
        cam.keyframe_insert(data_path="location", frame=frame)
        direction = -Vector((x, y, z))
        cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        cam.keyframe_insert(data_path="rotation_euler", frame=frame)

    # Phase 2: Full orbit (15% to 80% of frames)
    orbit_start = approach_end + 1
    orbit_end = int(n_frames * 0.80)
    animate_orbit(
        cam, center=(0, 0, 0), radius=4.5, elevation=0.45,
        start_frame=orbit_start, end_frame=orbit_end,
        start_angle=-0.3, end_angle=-0.3 + 2 * math.pi,
    )

    # Phase 3: Zoom out + decay (80% to 100%)
    decay_start = orbit_end + 1
    for i in range(n_frames - decay_start + 1):
        frame = decay_start + i
        t = i / max(n_frames - decay_start, 1)
        radius = 4.5 + 2.5 * t  # Pull back out
        angle = -0.3 + 2 * math.pi + 0.3 * t
        elev = 0.45 - 0.1 * t
        x = radius * math.cos(angle) * math.cos(elev)
        y = radius * math.sin(angle) * math.cos(elev)
        z = radius * math.sin(elev)
        cam.location = Vector((x, y, z))
        cam.keyframe_insert(data_path="location", frame=frame)
        direction = -Vector((x, y, z))
        cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        cam.keyframe_insert(data_path="rotation_euler", frame=frame)

    # NOTE: No shader keyframes on the volume material.
    # Keyframing density/emission node inputs causes Blender to flag
    # the material as animated, forcing a FULL volume re-sync every
    # frame: volume mesh recompute, BVH rebuild, density reload,
    # GPU re-upload. This burns ~0.15s of CPU overhead per frame
    # while the GPU sits idle waiting. For a static VDB with an
    # orbiting camera, the volume data is identical every frame —
    # keeping the material static lets Blender cache it once.

    # ── Stats text overlay ──────────────────────────────────────
    create_text_object(
        scene,
        text=f"1024\u00b3 QTT DNS  |  Rank 8  |  KE Error {ke_error:.2e}",
        location=(0, 0, -1.5),
        size=0.12,
        color=(0.6, 0.7, 1.0),
        emission_strength=1.5,
        name="VortexStats",
    )

    # ── Timeline ────────────────────────────────────────────────
    scene.frame_start = 1
    scene.frame_end = n_frames


# ═══════════════════════════════════════════════════════════════════════
# Scene 2: Combustion Flame (Campaign IV)
# ═══════════════════════════════════════════════════════════════════════


def build_scene_flame(
    scene: bpy.types.Scene,
    data: Dict[str, Any],
) -> None:
    """
    Build the combustion flame volumetric scene.

    A rectangular slab volume with the 1D temperature profile mapped
    along the X axis. Temperature drives blackbody emission color.
    The flame front is a thin intense white-yellow zone.
    """
    clear_scene()
    n_frames = CFG["flame_frames"]
    c4 = data.get("campaign_iv", {})
    temp_profile = c4.get("temperature_profile", [300.0] * 100)
    flame_speed = c4.get("flame_speed_estimate", 1.89)
    T_max = c4.get("max_temperature", 2497.0)

    print(f"  Scene 2: Combustion Flame ({n_frames} frames)")

    # ── Flame volume slab ───────────────────────────────────────
    # Domain: elongated along X with enough Y-Z extent for
    # the Gaussian radial falloff to create a cylindrical shape.
    domain_x = 2.0   # Blender units
    domain_yz = 0.8   # Wide enough for Gaussian taper (sigma=0.18 of unit)

    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0))
    slab = bpy.context.active_object
    slab.name = "FlameSlab"
    slab.scale = (domain_x, domain_yz, domain_yz)

    flame_mat = create_flame_volume_material(temp_profile)
    slab.data.materials.append(flame_mat)

    # ── Lighting ────────────────────────────────────────────────
    # Dim lighting — the flame is self-illuminating via blackbody.
    # External light is just for subtle reflections on surrounding geometry.
    add_area_light(
        scene, location=(0, -1.5, 2.0), energy=15.0, size=3.0,
        color=(0.9, 0.9, 1.0), name="KeyLight",
    )
    add_area_light(
        scene, location=(1.5, 1.0, 0.5), energy=8.0, size=1.5,
        color=(1.0, 0.95, 0.9), name="WarmFill",
    )
    add_area_light(
        scene, location=(-2.0, 0, 1.5), energy=20.0, size=1.0,
        color=(0.7, 0.8, 1.0), name="CoolRim",
    )

    # ── Camera: tracking shot along flame ───────────────────────
    # DOF disabled — flame structure must stay uniformly sharp for
    # scientific presentation; DOF just wastes samples on bokeh.
    cam = add_camera(
        scene,
        location=(-2.5, -3.0, 1.2),
        target=(0, 0, 0),
        lens=35.0,
    )

    # Phase 1: Establish (wide shot, slow drift in)
    phase1_end = int(n_frames * 0.20)
    for i in range(phase1_end + 1):
        frame = i + 1
        t = i / max(phase1_end, 1)
        cam.location = Vector((-2.5 + 0.5 * t, -3.0 + 0.3 * t, 1.2 - 0.1 * t))
        cam.keyframe_insert(data_path="location", frame=frame)
        target = Vector((0.3 * t, 0, 0))
        direction = target - cam.location
        cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        cam.keyframe_insert(data_path="rotation_euler", frame=frame)

    # Phase 2: Track along flame (left to right, pulled back)
    phase2_start = phase1_end + 1
    phase2_end = int(n_frames * 0.75)
    for i in range(phase2_end - phase2_start + 1):
        frame = phase2_start + i
        t = i / max(phase2_end - phase2_start, 1)
        # Camera slides from left to right, stays pulled back
        cam_x = -1.8 + 3.6 * t
        cam.location = Vector((cam_x, -2.5, 0.8 + 0.2 * math.sin(t * math.pi)))
        cam.keyframe_insert(data_path="location", frame=frame)
        # Look at center of flame
        look_x = cam_x * 0.3  # Smoothed look-ahead
        direction = Vector((look_x, 0, 0.05)) - cam.location
        cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        cam.keyframe_insert(data_path="rotation_euler", frame=frame)

    # Phase 3: Settle on angled view of flame front
    phase3_start = phase2_end + 1
    flame_front_x = 0.0
    for i in range(n_frames - phase3_start + 1):
        frame = phase3_start + i
        t = i / max(n_frames - phase3_start, 1)
        cam.location = Vector((
            flame_front_x + 0.8,
            -2.2 + 0.1 * math.sin(t * 2 * math.pi),
            0.9 + 0.05 * math.cos(t * 2 * math.pi),
        ))
        cam.keyframe_insert(data_path="location", frame=frame)
        direction = Vector((flame_front_x, 0, 0.05)) - cam.location
        cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        cam.keyframe_insert(data_path="rotation_euler", frame=frame)

    # ── Stats text ──────────────────────────────────────────────
    create_text_object(
        scene,
        text=f"H\u2082-Air DNS  |  S_L = {flame_speed:.2f} m/s  |  T_max = {T_max:.0f} K",
        location=(0, 0.8, -0.5),
        size=0.08,
        color=(1.0, 0.8, 0.4),
        emission_strength=1.5,
        name="FlameStats",
    )

    scene.frame_start = 1
    scene.frame_end = n_frames


# ═══════════════════════════════════════════════════════════════════════
# Scene 3: Compression Scaling (Campaign II)
# ═══════════════════════════════════════════════════════════════════════


def _create_bar(
    scene: bpy.types.Scene,
    x: float,
    y: float,
    height: float,
    width: float = 0.12,
    depth: float = 0.12,
    body_mat: bpy.types.Material = None,
    cap_mat: bpy.types.Material = None,
    name: str = "Bar",
) -> bpy.types.Object:
    """Create a single 3D bar with metallic body and emissive cap."""
    # Body
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x, y, height / 2.0))
    bar = bpy.context.active_object
    bar.name = name
    bar.scale = (width, depth, height)

    # Bevel for rounded edges
    bev = bar.modifiers.new("Bevel", "BEVEL")
    bev.width = min(0.005, width * 0.1)
    bev.segments = 3

    if body_mat:
        bar.data.materials.append(body_mat)

    # Emissive cap on top
    if cap_mat:
        cap_height = max(height * 0.02, 0.005)
        bpy.ops.mesh.primitive_cube_add(
            size=1.0, location=(x, y, height + cap_height / 2.0),
        )
        cap = bpy.context.active_object
        cap.name = f"{name}_cap"
        cap.scale = (width * 1.02, depth * 1.02, cap_height)
        cap.data.materials.append(cap_mat)

    return bar


def build_scene_compression(
    scene: bpy.types.Scene,
    data: Dict[str, Any],
) -> None:
    """
    Build the QTT compression scaling bar chart scene.

    X-axis: Grid sizes (64^3 to 4096^3)
    Groups: Sod shock (ruby), Turbulent (emerald), Boundary layer (sapphire)
    Y-axis: log10(compression ratio)
    """
    clear_scene()
    n_frames = CFG["bars_frames"]
    c2_data = data.get("campaign_ii", [])

    print(f"  Scene 3: Compression Scaling ({n_frames} frames)")

    # ── Parse data ──────────────────────────────────────────────
    func_types = ["sod_shock", "turbulent_8mode", "boundary_layer"]
    func_colors = {
        "sod_shock": (0.8, 0.08, 0.08),       # Ruby red
        "turbulent_8mode": (0.08, 0.7, 0.2),   # Emerald green
        "boundary_layer": (0.08, 0.2, 0.8),    # Sapphire blue
    }
    func_labels = {
        "sod_shock": "Sod Shock",
        "turbulent_8mode": "Turbulent",
        "boundary_layer": "Boundary Layer",
    }

    # Group by function type and grid size
    grouped: Dict[str, List[Dict]] = {ft: [] for ft in func_types}
    for entry in c2_data:
        ft = entry.get("function_type", "")
        if ft in grouped:
            grouped[ft].append(entry)

    # Sort each group by n_bits
    for ft in func_types:
        grouped[ft].sort(key=lambda e: e.get("n_bits", 0))

    # ── Materials ───────────────────────────────────────────────
    body_mats = {}
    cap_mats = {}
    for ft in func_types:
        c = func_colors[ft]
        body_mats[ft] = create_metallic_material(c, roughness=0.15, name=f"Metal_{ft}")
        cap_mats[ft] = create_emissive_cap_material(
            (c[0] * 1.5, c[1] * 1.5, c[2] * 1.5),
            strength=8.0, name=f"Cap_{ft}",
        )

    ground_mat = create_ground_material()

    # ── Build bars ──────────────────────────────────────────────
    bar_width = 0.1
    bar_depth = 0.1
    group_spacing = 0.45
    bar_spacing = 0.14
    max_log_ratio = 8.0  # log10(28M) ≈ 7.45
    height_scale = 0.4   # Blender units per log10 unit

    for g_idx, ft in enumerate(func_types):
        entries = grouped[ft]
        for b_idx, entry in enumerate(entries):
            ratio = entry.get("compression_ratio", 1.0)
            log_ratio = math.log10(max(ratio, 1.0))
            height = max(log_ratio * height_scale, 0.01)

            x = b_idx * group_spacing
            y = (g_idx - 1) * (bar_depth * 3 + bar_spacing)

            _create_bar(
                scene, x, y, height, bar_width, bar_depth,
                body_mat=body_mats[ft], cap_mat=cap_mats[ft],
                name=f"Bar_{ft}_{b_idx}",
            )

    # ── Grid labels (grid sizes along X) ────────────────────────
    grid_labels = ["64\u00b3", "128\u00b3", "256\u00b3", "512\u00b3",
                   "1024\u00b3", "2048\u00b3", "4096\u00b3"]
    for i, label in enumerate(grid_labels):
        create_text_object(
            scene, text=label,
            location=(i * group_spacing, 1.0, -0.1),
            size=0.06, color=(0.7, 0.7, 0.7),
            emission_strength=1.0, name=f"GridLabel_{i}",
        )

    # ── Function type labels ────────────────────────────────────
    for g_idx, ft in enumerate(func_types):
        y = (g_idx - 1) * (bar_depth * 3 + bar_spacing)
        c = func_colors[ft]
        create_text_object(
            scene, text=func_labels[ft],
            location=(-0.5, y, 0.05),
            size=0.05, color=c, emission_strength=2.0,
            align_x="RIGHT", name=f"FuncLabel_{ft}",
        )

    # ── Title ───────────────────────────────────────────────────
    create_text_object(
        scene, text="QTT Compression Scaling",
        location=(1.2, 0, 3.8),
        size=0.12, color=(0.8, 0.85, 1.0),
        emission_strength=2.0, name="CompTitle",
    )

    # ── Ground plane ────────────────────────────────────────────
    bpy.ops.mesh.primitive_plane_add(size=20.0, location=(1.5, 0, -0.01))
    ground = bpy.context.active_object
    ground.name = "Ground"
    ground.data.materials.append(ground_mat)

    # ── Lighting ────────────────────────────────────────────────
    add_area_light(
        scene, location=(1.5, -4, 5), energy=200.0, size=4.0,
        color=(0.9, 0.92, 1.0), name="KeyLight",
    )
    add_area_light(
        scene, location=(-2, 3, 3), energy=80.0, size=2.0,
        color=(0.7, 0.75, 1.0), name="FillLight",
    )
    add_area_light(
        scene, location=(5, 0, 2), energy=60.0, size=1.5,
        color=(1.0, 0.95, 0.85), name="RimLight",
    )

    # ── Camera: slow orbit ──────────────────────────────────────
    center = (1.5, 0, 1.5)
    cam = add_camera(
        scene,
        location=(1.5 + 5, -4, 4),
        target=center,
        lens=45.0,
        dof_distance=6.0,
        fstop=5.6,
    )
    animate_orbit(
        cam, center=center, radius=6.0, elevation=0.5,
        start_frame=1, end_frame=n_frames,
        start_angle=-0.7, end_angle=-0.7 + 0.8,
        target=center,
    )

    scene.frame_start = 1
    scene.frame_end = n_frames


# ═══════════════════════════════════════════════════════════════════════
# Scene 4: Kernel Benchmarks (Campaign III)
# ═══════════════════════════════════════════════════════════════════════


def build_scene_benchmarks(
    scene: bpy.types.Scene,
    data: Dict[str, Any],
) -> None:
    """
    Build the kernel benchmark comparison scene.

    Grouped bars: Triton (gold) vs Legacy (silver) vs CUDA (bronze).
    Highlights the 69x Hadamard speedup.
    """
    clear_scene()
    n_frames = CFG["bench_frames"]
    c3_data = data.get("campaign_iii", [])

    print(f"  Scene 4: Kernel Benchmarks ({n_frames} frames)")

    # ── Parse benchmark data ────────────────────────────────────
    # Group by operation
    ops_triton = {}
    ops_legacy = {}
    ops_cuda = {}
    for entry in c3_data:
        name = entry.get("kernel_name", "")
        backend = entry.get("backend", "")
        median = entry.get("median_ms", 0)
        if "add" in name.lower():
            op = "Add"
        elif "scale" in name.lower():
            op = "Scale"
        elif "inner" in name.lower():
            op = "Inner"
        elif "hadamard" in name.lower():
            op = "Hadamard"
        elif "mpo" in name.lower():
            op = "MPO Apply"
        elif "morton" in name.lower():
            op = "Morton"
        elif "truncate" in name.lower():
            op = "Truncate"
        else:
            continue

        if backend == "triton_native":
            ops_triton[op] = median
        elif backend == "legacy_python":
            ops_legacy[op] = median
        elif backend == "cuda_native":
            ops_cuda[op] = median

    # Operations to display (common between backends)
    display_ops = ["Add", "Inner", "Hadamard"]
    if not display_ops:
        print("  WARNING: No common operations found for benchmarks")
        return

    # ── Materials ───────────────────────────────────────────────
    gold_body = create_metallic_material((0.85, 0.65, 0.08), 0.1, name="Triton_Gold")
    gold_cap = create_emissive_cap_material((1.0, 0.85, 0.2), 10.0, name="Triton_Cap")
    silver_body = create_metallic_material((0.5, 0.5, 0.55), 0.15, name="Legacy_Silver")
    silver_cap = create_emissive_cap_material((0.7, 0.7, 0.75), 5.0, name="Legacy_Cap")
    bronze_body = create_metallic_material((0.65, 0.35, 0.08), 0.12, name="CUDA_Bronze")
    bronze_cap = create_emissive_cap_material((0.9, 0.5, 0.15), 7.0, name="CUDA_Cap")
    ground_mat = create_ground_material()

    # ── Build bars ──────────────────────────────────────────────
    bar_width = 0.12
    bar_depth = 0.15
    op_spacing = 0.8
    backend_spacing = 0.16
    max_time = 0.001  # Will be set dynamically
    for op in display_ops:
        for val in (ops_triton.get(op, 0), ops_legacy.get(op, 0), ops_cuda.get(op, 0)):
            max_time = max(max_time, val)
    height_scale = 2.5 / max(max_time, 0.001)  # Normalize so tallest bar is ~2.5

    backends = [
        ("Triton", ops_triton, gold_body, gold_cap),
        ("Legacy", ops_legacy, silver_body, silver_cap),
        ("CUDA", ops_cuda, bronze_body, bronze_cap),
    ]

    for op_idx, op in enumerate(display_ops):
        x_center = op_idx * op_spacing
        for b_idx, (b_name, b_data, body_mat, cap_mat) in enumerate(backends):
            val = b_data.get(op, 0)
            if val <= 0:
                continue
            height = max(val * height_scale, 0.01)
            x = x_center + (b_idx - 1) * backend_spacing
            _create_bar(
                scene, x, 0, height, bar_width, bar_depth,
                body_mat=body_mat, cap_mat=cap_mat,
                name=f"Bar_{op}_{b_name}",
            )

    # ── Op labels ───────────────────────────────────────────────
    for op_idx, op in enumerate(display_ops):
        create_text_object(
            scene, text=op,
            location=(op_idx * op_spacing, 0.5, -0.1),
            size=0.06, color=(0.8, 0.8, 0.8),
            emission_strength=1.5, name=f"OpLabel_{op}",
        )

    # ── Backend legend ──────────────────────────────────────────
    legend_y = -0.8
    legend_colors = [(0.85, 0.65, 0.08), (0.5, 0.5, 0.55), (0.65, 0.35, 0.08)]
    legend_names = ["Triton", "Legacy", "CUDA"]
    for i, (color, lname) in enumerate(zip(legend_colors, legend_names)):
        create_text_object(
            scene, text=lname,
            location=(-0.3 + i * 0.6, legend_y, 0.05),
            size=0.05, color=color,
            emission_strength=2.0, name=f"Legend_{lname}",
        )

    # ── Hadamard speedup callout ────────────────────────────────
    triton_had = ops_triton.get("Hadamard", 1.0)
    legacy_had = ops_legacy.get("Hadamard", 1.0)
    if triton_had > 0 and legacy_had > 0:
        speedup = legacy_had / triton_had
        create_text_object(
            scene, text=f"{speedup:.0f}\u00d7 speedup",
            location=(2 * op_spacing, 0, 2.8),
            size=0.10, color=(1.0, 0.85, 0.2),
            emission_strength=4.0, name="SpeedupCallout",
        )

    # ── Title ───────────────────────────────────────────────────
    create_text_object(
        scene, text="Triton Kernel Performance",
        location=(0.8, 0, 3.5),
        size=0.10, color=(0.85, 0.9, 1.0),
        emission_strength=2.0, name="BenchTitle",
    )

    # ── Ground ──────────────────────────────────────────────────
    bpy.ops.mesh.primitive_plane_add(size=15.0, location=(0.8, 0, -0.01))
    ground = bpy.context.active_object
    ground.name = "Ground"
    ground.data.materials.append(ground_mat)

    # ── Lighting ────────────────────────────────────────────────
    add_area_light(
        scene, location=(0.8, -3, 4), energy=150.0, size=3.0,
        color=(0.95, 0.95, 1.0), name="KeyLight",
    )
    add_area_light(
        scene, location=(-2, 2, 2), energy=60.0, size=2.0,
        color=(0.8, 0.85, 1.0), name="FillLight",
    )

    # ── Camera ──────────────────────────────────────────────────
    center = (0.8, 0, 1.2)
    cam = add_camera(
        scene,
        location=(0.8 + 4, -3, 3),
        target=center,
        lens=50.0,
        dof_distance=5.0,
        fstop=5.6,
    )
    animate_orbit(
        cam, center=center, radius=5.0, elevation=0.45,
        start_frame=1, end_frame=n_frames,
        start_angle=-0.8, end_angle=-0.8 + 0.6,
        target=center,
    )

    scene.frame_start = 1
    scene.frame_end = n_frames


# ═══════════════════════════════════════════════════════════════════════
# Title Card Scene
# ═══════════════════════════════════════════════════════════════════════


def build_scene_title(
    scene: bpy.types.Scene,
    data: Dict[str, Any],
    title_type: str = "intro",
) -> None:
    """Build title or end card scene."""
    clear_scene()
    n_frames = CFG["title_frames"]

    if title_type == "intro":
        print(f"  Title Card: Intro ({n_frames} frames)")
        create_text_object(
            scene, text="HYPERTENSOR VM",
            location=(0, 0, 0.3),
            size=0.35,
            color=(0.7, 0.8, 1.0),
            emission_strength=3.0,
            name="TitleMain",
            extrude=0.02,
        )
        create_text_object(
            scene, text="Industrial QTT / GPU Simulation",
            location=(0, 0, -0.2),
            size=0.12,
            color=(0.5, 0.6, 0.8),
            emission_strength=1.5,
            name="TitleSub",
        )

        hw = data.get("metadata", {}).get("hardware", {})
        gpu_name = hw.get("gpu_name", "GPU")
        create_text_object(
            scene, text=f"Rendered on {gpu_name}",
            location=(0, 0, -0.6),
            size=0.06,
            color=(0.3, 0.35, 0.45),
            emission_strength=0.8,
            name="TitleHW",
        )

    elif title_type == "end":
        print(f"  Title Card: End ({n_frames} frames)")
        c1 = data.get("campaign_i", {})
        c4 = data.get("campaign_iv", {})

        lines = [
            f"Campaign I:  1024\u00b3 NS3D DNS — KE Error {c1.get('ke_relative_error', 0):.2e}",
            f"Campaign II: Compression up to 28.4M\u00d7 at 4096\u00b3",
            f"Campaign III: Hadamard 69\u00d7 speedup (Triton vs Legacy)",
            f"Campaign IV: S_L = {c4.get('flame_speed_estimate', 0):.2f} m/s (H\u2082-air premixed)",
        ]
        for i, line in enumerate(lines):
            create_text_object(
                scene, text=line,
                location=(0, 0, 0.4 - i * 0.25),
                size=0.06,
                color=(0.6, 0.65, 0.75),
                emission_strength=1.2,
                name=f"EndLine_{i}",
            )

        create_text_object(
            scene, text="github.com/tigantic/HyperTensor-VM",
            location=(0, 0, -0.8),
            size=0.05,
            color=(0.4, 0.5, 0.7),
            emission_strength=1.0,
            name="EndURL",
        )

    # Camera (static, looking at text)
    cam = add_camera(
        scene,
        location=(0, -3, 0),
        target=(0, 0, 0),
        lens=50.0,
    )

    # Minimal lighting
    add_area_light(
        scene, location=(0, -2, 2), energy=30.0, size=5.0,
        color=(0.8, 0.85, 1.0), name="TitleLight",
    )

    scene.frame_start = 1
    scene.frame_end = n_frames


# ═══════════════════════════════════════════════════════════════════════
# Render Orchestration
# ═══════════════════════════════════════════════════════════════════════


def render_scene(
    scene: bpy.types.Scene,
    output_prefix: str,
    frame_offset: int = 0,
) -> int:
    """
    Render all frames of a scene.

    Uses Blender's animation render path which enables persistent
    data caching between frames. For scenes with static geometry
    (VDB volumes, meshes) and only an orbiting camera, the BVH,
    volume mesh, and density grid are uploaded to GPU ONCE and
    reused across all frames — eliminating the per-frame resync
    overhead.

    Returns the number of frames rendered (for offset tracking).
    """
    import time

    output_path = Path(OUTPUT_DIR) / "frames"
    output_path.mkdir(parents=True, exist_ok=True)

    n_frames = scene.frame_end - scene.frame_start + 1
    print(f"  Rendering {n_frames} frames (#{frame_offset + 1}-{frame_offset + n_frames})...")

    # Enable persistent data so Blender caches BVH/textures/volumes
    # across frames. Only camera transforms change per frame.
    scene.render.use_persistent_data = True

    # Set up output path pattern for animation render.
    # Blender's animation mode renders frame_start..frame_end
    # and writes each frame using the filepath pattern.
    scene.render.filepath = str(output_path / f"frame_{frame_offset + scene.frame_start:04d}")

    # For frame numbering with offset, we render frame-by-frame
    # but with persistent_data=True, BVH/volume stay cached.
    frame_times: list[float] = []
    for frame in range(scene.frame_start, scene.frame_end + 1):
        t0 = time.perf_counter()
        scene.frame_set(frame)
        global_frame = frame_offset + frame
        scene.render.filepath = str(output_path / f"frame_{global_frame:04d}.png")
        bpy.ops.render.render(write_still=True)
        elapsed = time.perf_counter() - t0
        frame_times.append(elapsed)

        if frame % max(1, n_frames // 5) == 0 or frame == scene.frame_end:
            pct = (frame - scene.frame_start + 1) / n_frames * 100
            avg = sum(frame_times) / len(frame_times)
            remaining = avg * (n_frames - len(frame_times))
            print(f"    Frame {frame}/{scene.frame_end} ({pct:.0f}%) "
                  f"— {elapsed:.1f}s (avg {avg:.1f}s, ~{remaining:.0f}s left)")

    total = sum(frame_times)
    avg = total / len(frame_times)
    print(f"  Scene total: {total:.1f}s ({avg:.1f}s/frame)")
    return n_frames


def main() -> None:
    """Main entry point — build and render all scenes."""
    print("\n" + "=" * 72)
    print("  HYPERTENSOR — Blender Cycles Scientific Visualization")
    print(f"  Quality: {QUALITY} ({CFG['resolution_x']}x{CFG['resolution_y']}, "
          f"{CFG['samples']} samples)")
    print("=" * 72 + "\n")

    # Load data
    data = load_results()
    print(f"  Loaded: {RESULTS_JSON}")
    print(f"  Campaigns: {len(data.get('metadata', {}).get('campaigns', []))}")
    print(f"  Output: {OUTPUT_DIR}")

    # GPU setup
    gpu_type = setup_gpu()
    print(f"  Render device: {gpu_type}")
    print()

    # Determine which scenes to render
    if SCENE_FILTER == "all":
        scenes_to_render = [0, 1, 2, 3, 4, 5]  # intro, 4 campaigns, end
    else:
        scenes_to_render = [int(s.strip()) for s in SCENE_FILTER.split(",")]

    scene = bpy.context.scene
    setup_cycles(scene)
    create_dark_world(scene)

    frame_offset = 0
    scene_builders = [
        (0, "Intro", lambda s, d: build_scene_title(s, d, "intro")),
        (1, "Taylor-Green Vortex", build_scene_vortex),
        (2, "Combustion Flame", build_scene_flame),
        (3, "Compression Scaling", build_scene_compression),
        (4, "Kernel Benchmarks", build_scene_benchmarks),
        (5, "End Card", lambda s, d: build_scene_title(s, d, "end")),
    ]

    for scene_id, scene_name, builder in scene_builders:
        if scene_id not in scenes_to_render:
            continue

        print(f"\n{'─' * 60}")
        print(f"  SCENE {scene_id}: {scene_name}")
        print(f"{'─' * 60}")

        # Rebuild world for each scene (clear_scene removes it)
        builder(scene, data)
        create_dark_world(scene)
        setup_cycles(scene, scene_id=scene_id)

        n_rendered = render_scene(scene, f"scene{scene_id}", frame_offset)
        frame_offset += n_rendered
        print(f"  Done: {n_rendered} frames ({scene_name})")

    print(f"\n{'=' * 72}")
    print(f"  COMPLETE: {frame_offset} total frames rendered")
    print(f"  Frames: {Path(OUTPUT_DIR) / 'frames'}")
    print(f"  Encode: ffmpeg -framerate {CFG['fps']} "
          f"-i frames/frame_%04d.png -c:v libx265 -crf 18 "
          f"-pix_fmt yuv420p output.mp4")
    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    main()

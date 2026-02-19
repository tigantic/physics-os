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
        "vortex_frames": 150,
        "flame_frames": 90,
        "bars_frames": 90,
        "bench_frames": 60,
        "title_frames": 45,
    },
    "final": {
        "samples": 512,
        "resolution_x": 1920,
        "resolution_y": 1080,
        "volume_step_rate": 0.1,
        "volume_max_steps": 1024,
        "use_denoising": True,
        "fps": 30,
        "vortex_frames": 300,
        "flame_frames": 180,
        "bars_frames": 150,
        "bench_frames": 120,
        "title_frames": 60,
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


def setup_gpu() -> str:
    """Enable GPU rendering. Returns device type used."""
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for device_type in ("OPTIX", "CUDA", "HIP", "METAL"):
        try:
            prefs.compute_device_type = device_type
            prefs.get_devices()
            found = False
            for device in prefs.devices:
                if device.type == device_type:
                    device.use = True
                    found = True
                elif device.type == "CPU":
                    device.use = False
            if found:
                print(f"  GPU: Using {device_type}")
                return device_type
        except Exception:
            continue
    # CPU fallback
    prefs.compute_device_type = "NONE"
    print("  GPU: None found, using CPU")
    return "CPU"


def setup_cycles(scene: bpy.types.Scene) -> None:
    """Configure Cycles renderer with production settings."""
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"
    scene.cycles.samples = CFG["samples"]
    scene.cycles.use_denoising = CFG["use_denoising"]
    if CFG["use_denoising"]:
        scene.cycles.denoiser = "OPENIMAGEDENOISE"
        scene.cycles.denoising_input_passes = "RGB_ALBEDO_NORMAL"
    scene.cycles.volume_step_rate = CFG["volume_step_rate"]
    scene.cycles.volume_max_steps = CFG["volume_max_steps"]
    scene.cycles.volume_bounces = 4
    scene.cycles.max_bounces = 12
    scene.cycles.diffuse_bounces = 4
    scene.cycles.glossy_bounces = 4
    scene.cycles.transmission_bounces = 8
    scene.cycles.transparent_max_bounces = 8
    scene.cycles.use_fast_gi = False
    scene.cycles.film_exposure = 1.0
    scene.cycles.caustics_reflective = False
    scene.cycles.caustics_refractive = False

    # Film
    scene.render.film_transparent = False
    scene.render.resolution_x = CFG["resolution_x"]
    scene.render.resolution_y = CFG["resolution_y"]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "16"
    scene.render.image_settings.compression = 15

    # Color management — Filmic for HDR tonemapping
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    # Performance
    scene.render.use_persistent_data = True
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.005

    # FPS
    scene.render.fps = CFG["fps"]


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


def create_vortex_volume_material(
    density_scale: float = 8.0,
    emission_scale: float = 15.0,
) -> bpy.types.Material:
    """
    Procedural Taylor-Green vortex volume material.

    The vorticity magnitude |omega(x,y,z)| is computed entirely from shader
    nodes using the analytical formula. This gives INFINITE resolution --
    the density is evaluated at every ray sample point by the renderer.

    TG vortex:
        omega_x =  sin(2pi*x) * cos(2pi*y) * sin(2pi*z)
        omega_y = -cos(2pi*x) * sin(2pi*y) * sin(2pi*z)
        omega_z = -2*cos(2pi*x) * cos(2pi*y) * cos(2pi*z)

        |omega|^2 = sin^2(x)cos^2(y)sin^2(z) + cos^2(x)sin^2(y)sin^2(z)
                  + 4*cos^2(x)*cos^2(y)*cos^2(z)

    Color mapping: dark blue -> ocean blue -> cyan -> white (by vorticity).
    """
    mat = bpy.data.materials.new("VortexVolume")
    mat.use_nodes = True
    tree = mat.node_tree
    tree.nodes.clear()

    TWO_PI = 2.0 * math.pi
    col_x, col_y, col_z = -1200, -200, 200  # layout helpers

    # ── Output + Volume shader ──────────────────────────────────
    output_node = add_node(tree, "ShaderNodeOutputMaterial", (1400, 0), "Output")
    volume = add_node(tree, "ShaderNodeVolumePrincipled", (1100, 0), "Volume")
    volume.inputs["Anisotropy"].default_value = 0.2
    link(tree, volume, "Volume", output_node, "Volume")

    # ── Coordinates ─────────────────────────────────────────────
    tex_coord = add_node(tree, "ShaderNodeTexCoord", (col_x, 0), "Coords")
    separate = add_node(tree, "ShaderNodeSeparateXYZ", (col_x + 200, 0), "SepXYZ")
    link(tree, tex_coord, "Generated", separate, "Vector")

    # Scale [0,1] -> [0, 2*pi] for each axis
    scales = {}
    for i, axis in enumerate(("X", "Y", "Z")):
        s = add_node(
            tree, "ShaderNodeMath",
            (col_x + 400, col_y + col_z * (1 - i)),
            f"Scale_{axis}",
        )
        s.operation = "MULTIPLY"
        s.inputs[1].default_value = TWO_PI
        link(tree, separate, axis, s, 0)
        scales[axis] = s

    # sin/cos of each scaled coordinate
    trig = {}
    for i, axis in enumerate(("X", "Y", "Z")):
        for j, func in enumerate(("SINE", "COSINE")):
            short = f"{'sin' if func == 'SINE' else 'cos'}_{axis.lower()}"
            n = add_node(
                tree, "ShaderNodeMath",
                (col_x + 600, col_y + col_z * (1 - i) + 80 * (1 - j)),
                short,
            )
            n.operation = func
            link(tree, scales[axis], 0, n, 0)
            trig[short] = n

    # ── Compute |omega|^2 ───────────────────────────────────────
    # We need squared trig values
    sq = {}
    x_off = col_x + 800
    for k, (name_key, trig_key) in enumerate((
        ("sin2x", "sin_x"), ("cos2x", "cos_x"),
        ("sin2y", "sin_y"), ("cos2y", "cos_y"),
        ("sin2z", "sin_z"), ("cos2z", "cos_z"),
    )):
        n = add_node(tree, "ShaderNodeMath", (x_off, 300 - k * 100), name_key)
        n.operation = "MULTIPLY"
        link(tree, trig[trig_key], 0, n, 0)
        link(tree, trig[trig_key], 0, n, 1)
        sq[name_key] = n

    # Term 1: sin^2(x) * cos^2(y) * sin^2(z)
    t1a = add_node(tree, "ShaderNodeMath", (x_off + 200, 250), "t1a")
    t1a.operation = "MULTIPLY"
    link(tree, sq["sin2x"], 0, t1a, 0)
    link(tree, sq["cos2y"], 0, t1a, 1)

    t1 = add_node(tree, "ShaderNodeMath", (x_off + 400, 250), "t1")
    t1.operation = "MULTIPLY"
    link(tree, t1a, 0, t1, 0)
    link(tree, sq["sin2z"], 0, t1, 1)

    # Term 2: cos^2(x) * sin^2(y) * sin^2(z)
    t2a = add_node(tree, "ShaderNodeMath", (x_off + 200, 0), "t2a")
    t2a.operation = "MULTIPLY"
    link(tree, sq["cos2x"], 0, t2a, 0)
    link(tree, sq["sin2y"], 0, t2a, 1)

    t2 = add_node(tree, "ShaderNodeMath", (x_off + 400, 0), "t2")
    t2.operation = "MULTIPLY"
    link(tree, t2a, 0, t2, 0)
    link(tree, sq["sin2z"], 0, t2, 1)

    # Term 3: 4 * cos^2(x) * cos^2(y) * cos^2(z)
    t3a = add_node(tree, "ShaderNodeMath", (x_off + 200, -250), "t3a")
    t3a.operation = "MULTIPLY"
    link(tree, sq["cos2x"], 0, t3a, 0)
    link(tree, sq["cos2y"], 0, t3a, 1)

    t3b = add_node(tree, "ShaderNodeMath", (x_off + 400, -250), "t3b")
    t3b.operation = "MULTIPLY"
    link(tree, t3a, 0, t3b, 0)
    link(tree, sq["cos2z"], 0, t3b, 1)

    t3 = add_node(tree, "ShaderNodeMath", (x_off + 500, -250), "t3_x4")
    t3.operation = "MULTIPLY"
    t3.inputs[1].default_value = 4.0
    link(tree, t3b, 0, t3, 0)

    # Sum: |omega|^2 = t1 + t2 + t3
    sum12 = add_node(tree, "ShaderNodeMath", (x_off + 600, 125), "sum12")
    sum12.operation = "ADD"
    link(tree, t1, 0, sum12, 0)
    link(tree, t2, 0, sum12, 1)

    sum_all = add_node(tree, "ShaderNodeMath", (x_off + 700, 0), "omega_sq")
    sum_all.operation = "ADD"
    link(tree, sum12, 0, sum_all, 0)
    link(tree, t3, 0, sum_all, 1)

    # |omega| = sqrt(|omega|^2)
    omega_mag = add_node(tree, "ShaderNodeMath", (x_off + 800, 0), "omega_mag")
    omega_mag.operation = "POWER"
    omega_mag.inputs[1].default_value = 0.5
    link(tree, sum_all, 0, omega_mag, 0)

    # ── Density ─────────────────────────────────────────────────
    density = add_node(tree, "ShaderNodeMath", (x_off + 900, 50), "density")
    density.operation = "MULTIPLY"
    density.inputs[1].default_value = density_scale
    link(tree, omega_mag, 0, density, 0)
    link(tree, density, 0, volume, "Density")

    # ── Emission Color via ColorRamp ────────────────────────────
    ramp = add_node(tree, "ShaderNodeValToRGB", (x_off + 850, -200), "ColorRamp")
    elements = ramp.color_ramp.elements
    # Element 0 already exists at position 0
    elements[0].position = 0.0
    elements[0].color = (0.0, 0.0, 0.02, 1.0)         # Near black

    e1 = elements.new(0.15)
    e1.color = (0.01, 0.02, 0.15, 1.0)                 # Deep navy

    e2 = elements.new(0.35)
    e2.color = (0.02, 0.1, 0.5, 1.0)                   # Ocean blue

    e3 = elements.new(0.55)
    e3.color = (0.05, 0.4, 0.8, 1.0)                   # Bright blue

    e4 = elements.new(0.75)
    e4.color = (0.1, 0.7, 0.95, 1.0)                   # Cyan

    # Element 1 already exists at position 1
    elements[1].position = 1.0
    elements[1].color = (0.9, 0.95, 1.0, 1.0)          # Near white

    ramp.color_ramp.interpolation = "B_SPLINE"
    link(tree, omega_mag, 0, ramp, "Fac")
    link(tree, ramp, "Color", volume, "Emission Color")

    # Emission strength proportional to vorticity
    em_str = add_node(tree, "ShaderNodeMath", (x_off + 900, -100), "em_strength")
    em_str.operation = "MULTIPLY"
    em_str.inputs[1].default_value = emission_scale
    link(tree, omega_mag, 0, em_str, 0)
    link(tree, em_str, 0, volume, "Emission Strength")

    return mat


def create_flame_volume_material(
    temperature_data: List[float],
) -> bpy.types.Material:
    """
    Volumetric combustion flame material.

    Uses a 1D image texture containing the temperature profile.
    Temperature is mapped to blackbody emission color (physically correct
    Planck radiation) and emission strength.

    Cold gas (300K): transparent/dim blue scatter
    Preheat zone (300-1000K): faint warm glow
    Reaction zone (1000-2500K): intense white-yellow blackbody emission
    Burnt gas (2230K): warm orange-yellow glow, settling to equilibrium
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
    output_node = add_node(tree, "ShaderNodeOutputMaterial", (1200, 0), "Output")
    volume = add_node(tree, "ShaderNodeVolumePrincipled", (900, 0), "Volume")
    volume.inputs["Anisotropy"].default_value = 0.3
    link(tree, volume, "Volume", output_node, "Volume")

    # ── Coordinates → 1D lookup ─────────────────────────────────
    tex_coord = add_node(tree, "ShaderNodeTexCoord", (-600, 0), "Coords")
    separate = add_node(tree, "ShaderNodeSeparateXYZ", (-400, 0), "SepXYZ")
    link(tree, tex_coord, "Generated", separate, "Vector")

    # Map X [0,1] → image UV, Y=0.5 (center of 1-pixel-high image)
    combine_uv = add_node(tree, "ShaderNodeCombineXYZ", (-200, 0), "CombineUV")
    combine_uv.inputs["Y"].default_value = 0.5
    combine_uv.inputs["Z"].default_value = 0.0
    link(tree, separate, "X", combine_uv, "X")

    # Sample temperature image
    img_tex = add_node(tree, "ShaderNodeTexImage", (0, 0), "TempTexture")
    img_tex.image = img
    img_tex.interpolation = "Linear"
    img_tex.extension = "EXTEND"
    link(tree, combine_uv, "Vector", img_tex, "Vector")

    # ── Denormalize to Kelvin ───────────────────────────────────
    # t_norm → T = t_norm * T_range + T_min
    scale_T = add_node(tree, "ShaderNodeMath", (200, 100), "ScaleT")
    scale_T.operation = "MULTIPLY"
    scale_T.inputs[1].default_value = T_range
    link(tree, img_tex, "Color", scale_T, 0)

    add_T = add_node(tree, "ShaderNodeMath", (400, 100), "AddTmin")
    add_T.operation = "ADD"
    add_T.inputs[1].default_value = T_min
    link(tree, scale_T, 0, add_T, 0)

    # ── Blackbody color from temperature ────────────────────────
    blackbody = add_node(tree, "ShaderNodeBlackbody", (600, 100), "Blackbody")
    link(tree, add_T, 0, blackbody, "Temperature")
    link(tree, blackbody, "Color", volume, "Emission Color")

    # ── Emission strength: zero below 500K, ramps up above ─────
    # strength = max(0, (T - 500) / 500) * scale
    sub_threshold = add_node(tree, "ShaderNodeMath", (400, -100), "SubThresh")
    sub_threshold.operation = "SUBTRACT"
    sub_threshold.inputs[1].default_value = 500.0
    link(tree, add_T, 0, sub_threshold, 0)

    div_scale = add_node(tree, "ShaderNodeMath", (500, -100), "DivScale")
    div_scale.operation = "DIVIDE"
    div_scale.inputs[1].default_value = 500.0
    link(tree, sub_threshold, 0, div_scale, 0)

    clamp_em = add_node(tree, "ShaderNodeClamp", (600, -100), "ClampEm")
    clamp_em.inputs["Min"].default_value = 0.0
    clamp_em.inputs["Max"].default_value = 10.0
    link(tree, div_scale, 0, clamp_em, "Value")

    em_final = add_node(tree, "ShaderNodeMath", (700, -100), "EmFinal")
    em_final.operation = "MULTIPLY"
    em_final.inputs[1].default_value = 80.0  # Overall emission strength
    link(tree, clamp_em, "Result", em_final, 0)
    link(tree, em_final, 0, volume, "Emission Strength")

    # ── Density: higher in flame zone for opacity ───────────────
    # Base density everywhere (thin), higher near flame front
    density_base = add_node(tree, "ShaderNodeMath", (700, -250), "DensityBase")
    density_base.operation = "ADD"
    density_base.inputs[0].default_value = 0.5   # Background density
    density_base.inputs[1].default_value = 0.0
    # Add extra density near flame (proportional to emission)
    density_add = add_node(tree, "ShaderNodeMath", (750, -200), "DensityAdd")
    density_add.operation = "MULTIPLY"
    density_add.inputs[1].default_value = 2.0
    link(tree, clamp_em, "Result", density_add, 0)

    density_total = add_node(tree, "ShaderNodeMath", (850, -200), "DensityTotal")
    density_total.operation = "ADD"
    link(tree, density_base, 0, density_total, 0)
    link(tree, density_add, 0, density_total, 1)
    link(tree, density_total, 0, volume, "Density")

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

    A cube with a procedural volumetric shader computes |omega(x,y,z)|
    at every ray sample. The vortex tubes glow blue-cyan-white.
    Camera orbits the volume. Vortex decays over time (exp(-2*nu*t)).
    """
    clear_scene()
    n_frames = CFG["vortex_frames"]
    nu = data.get("campaign_i", {}).get("viscosity", 0.01)
    ke_error = data.get("campaign_i", {}).get("ke_relative_error", 0.00127)

    print(f"  Scene 1: Taylor-Green Vortex ({n_frames} frames)")

    # ── Volume cube ─────────────────────────────────────────────
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0, 0, 0))
    cube = bpy.context.active_object
    cube.name = "VortexVolume"

    # Apply vortex material
    mat = create_vortex_volume_material(density_scale=8.0, emission_scale=15.0)
    cube.data.materials.append(mat)

    # ── Lighting ────────────────────────────────────────────────
    # Minimal lighting — the volume is self-illuminating
    add_area_light(
        scene, location=(3, -2, 4), energy=50.0, size=3.0,
        color=(0.8, 0.85, 1.0), name="KeyLight",
    )
    add_area_light(
        scene, location=(-3, 3, 2), energy=20.0, size=2.0,
        color=(0.6, 0.7, 1.0), name="FillLight",
    )

    # ── Camera: orbit around vortex ─────────────────────────────
    cam = add_camera(
        scene,
        location=(4, -3, 2.5),
        target=(0, 0, 0),
        lens=35.0,
        dof_distance=5.0,
        fstop=4.0,
    )

    # Phase 1: Approach (0 to 20% of frames)
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

    # Phase 2: Full orbit (20% to 80% of frames)
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

    # ── Animate vortex decay via density keyframes ──────────────
    # Access the density_scale multiplier node and keyframe it
    vol_nodes = mat.node_tree.nodes
    density_node = vol_nodes.get("density")
    em_node = vol_nodes.get("em_strength")
    if density_node and em_node:
        # Stable phase (0 to 60%)
        stable_end = int(n_frames * 0.6)
        density_node.inputs[1].default_value = 8.0
        density_node.inputs[1].keyframe_insert("default_value", frame=1)
        density_node.inputs[1].keyframe_insert("default_value", frame=stable_end)
        em_node.inputs[1].default_value = 15.0
        em_node.inputs[1].keyframe_insert("default_value", frame=1)
        em_node.inputs[1].keyframe_insert("default_value", frame=stable_end)

        # Decay phase (60% to 100%) — simulate exp(-2*nu*t)
        for i in range(n_frames - stable_end + 1):
            frame = stable_end + i
            t_frac = i / max(n_frames - stable_end, 1)
            # Simulate ~50 time steps of the actual simulation
            sim_time = 50 * 0.000941 * t_frac
            decay = math.exp(-2 * nu * sim_time * 500)  # Exaggerated for visibility
            density_node.inputs[1].default_value = 8.0 * max(decay, 0.05)
            density_node.inputs[1].keyframe_insert("default_value", frame=frame)
            em_node.inputs[1].default_value = 15.0 * max(decay, 0.05)
            em_node.inputs[1].keyframe_insert("default_value", frame=frame)

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
    # Domain: 2cm long, 4mm tall, 4mm deep
    domain_x = 2.0   # Blender units (scaled from 0.02m)
    domain_yz = 0.4

    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0))
    slab = bpy.context.active_object
    slab.name = "FlameSlab"
    slab.scale = (domain_x, domain_yz, domain_yz)

    flame_mat = create_flame_volume_material(temp_profile)
    slab.data.materials.append(flame_mat)

    # ── Glass combustion chamber (outer shell) ──────────────────
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0))
    chamber = bpy.context.active_object
    chamber.name = "CombustionChamber"
    chamber.scale = (domain_x * 1.05, domain_yz * 1.3, domain_yz * 1.3)

    glass_mat = bpy.data.materials.new("Glass")
    glass_mat.use_nodes = True
    glass_tree = glass_mat.node_tree
    glass_tree.nodes.clear()
    g_output = add_node(glass_tree, "ShaderNodeOutputMaterial", (400, 0))
    g_bsdf = add_node(glass_tree, "ShaderNodeBsdfGlass", (0, 0))
    g_bsdf.inputs["Color"].default_value = (0.95, 0.97, 1.0, 1.0)
    g_bsdf.inputs["Roughness"].default_value = 0.02
    g_bsdf.inputs["IOR"].default_value = 1.52
    link(glass_tree, g_bsdf, "BSDF", g_output, "Surface")
    chamber.data.materials.append(glass_mat)

    # ── Lighting ────────────────────────────────────────────────
    add_area_light(
        scene, location=(0, -1.5, 2.0), energy=80.0, size=3.0,
        color=(0.9, 0.9, 1.0), name="KeyLight",
    )
    add_area_light(
        scene, location=(1.5, 1.0, 0.5), energy=30.0, size=1.5,
        color=(1.0, 0.95, 0.9), name="WarmFill",
    )
    add_area_light(
        scene, location=(-2.0, 0, 1.5), energy=20.0, size=1.0,
        color=(0.7, 0.8, 1.0), name="CoolRim",
    )

    # ── Camera: tracking shot along flame ───────────────────────
    cam = add_camera(
        scene,
        location=(-1.5, -1.2, 0.6),
        target=(0, 0, 0),
        lens=50.0,
        dof_distance=1.5,
        fstop=2.8,
    )

    # Phase 1: Establish (wide shot)
    phase1_end = int(n_frames * 0.15)
    for i in range(phase1_end + 1):
        frame = i + 1
        t = i / max(phase1_end, 1)
        cam.location = Vector((-1.5, -1.2 - 0.3 * t, 0.6 + 0.2 * t))
        cam.keyframe_insert(data_path="location", frame=frame)
        target = Vector((0.3 * t, 0, 0))
        direction = target - cam.location
        cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        cam.keyframe_insert(data_path="rotation_euler", frame=frame)

    # Phase 2: Track along flame (left to right)
    phase2_start = phase1_end + 1
    phase2_end = int(n_frames * 0.75)
    for i in range(phase2_end - phase2_start + 1):
        frame = phase2_start + i
        t = i / max(phase2_end - phase2_start, 1)
        # Camera slides from left of flame to right
        cam_x = -1.2 + 2.4 * t
        cam.location = Vector((cam_x, -1.0, 0.5 + 0.15 * math.sin(t * math.pi)))
        cam.keyframe_insert(data_path="location", frame=frame)
        # Look slightly ahead
        look_x = cam_x + 0.3
        direction = Vector((look_x, 0, 0.1)) - cam.location
        cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        cam.keyframe_insert(data_path="rotation_euler", frame=frame)

    # Phase 3: Close-up on flame front
    phase3_start = phase2_end + 1
    flame_front_x = 0.0  # Mid-domain where the steep gradient is
    for i in range(n_frames - phase3_start + 1):
        frame = phase3_start + i
        t = i / max(n_frames - phase3_start, 1)
        cam.location = Vector((
            flame_front_x - 0.3 + 0.1 * t,
            -0.6 + 0.05 * math.sin(t * 2 * math.pi),
            0.3 + 0.05 * math.cos(t * 2 * math.pi),
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

    Returns the number of frames rendered (for offset tracking).
    """
    output_path = Path(OUTPUT_DIR) / "frames"
    output_path.mkdir(parents=True, exist_ok=True)

    n_frames = scene.frame_end - scene.frame_start + 1
    print(f"  Rendering {n_frames} frames (#{frame_offset + 1}-{frame_offset + n_frames})...")

    for frame in range(scene.frame_start, scene.frame_end + 1):
        scene.frame_set(frame)
        global_frame = frame_offset + frame
        scene.render.filepath = str(output_path / f"frame_{global_frame:04d}.png")
        bpy.ops.render.render(write_still=True)

        if frame % max(1, n_frames // 5) == 0 or frame == scene.frame_end:
            pct = (frame - scene.frame_start + 1) / n_frames * 100
            print(f"    Frame {frame}/{scene.frame_end} ({pct:.0f}%)")

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
        setup_cycles(scene)

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

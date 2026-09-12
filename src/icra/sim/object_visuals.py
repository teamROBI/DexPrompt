"""Opt-in OBJ texture materials. Missing visual assets are errors, not grey fallbacks."""

from __future__ import annotations

import io
import shlex
from pathlib import Path

import numpy as np


def diffuse_texture(mesh_path: Path) -> Path:
    """Resolve the single-material OBJ/MTL contract used by our HOPE assets."""
    mesh_path = Path(mesh_path)
    libraries = []
    for line in mesh_path.read_text().splitlines():
        words = shlex.split(line, comments=True)
        if words and words[0] == "mtllib":
            libraries.extend(words[1:])
    if len(libraries) != 1:
        raise ValueError(f"{mesh_path}: texture profile requires one MTL library")
    mtl = mesh_path.parent / libraries[0]
    materials, images = [], []
    for line in mtl.read_text().splitlines():
        words = shlex.split(line, comments=True)
        if words and words[0] == "newmtl":
            materials.append(words[1:])
        if words and words[0] == "map_Kd":
            if len(words) != 2 or words[1].startswith("-"):
                raise ValueError(f"{mtl}: unsupported diffuse texture options")
            images.append(mtl.parent / words[1])
    if len(materials) != 1 or len(images) != 1 or not images[0].is_file():
        raise ValueError(f"{mtl}: exactly one material and existing diffuse image required")
    return images[0].resolve()


def add_material(spec, asset) -> str:
    import mujoco
    import trimesh
    from PIL import Image

    texture = diffuse_texture(asset.mesh)
    mesh = trimesh.load(asset.mesh, force="mesh")
    uv = getattr(mesh.visual, "uv", None)
    if (uv is None or uv.shape != (len(mesh.vertices), 2)
            or not np.isfinite(uv).all()):
        raise ValueError(f"{asset.name}: finite per-vertex OBJ texture coordinates required")
    name = f"{asset.name}_diffuse_v1"
    # MuJoCo 3.3.3 reads PNG textures, not JPEG. Use its virtual filesystem so
    # existing asset directories remain untouched; compiled MJB embeds pixels.
    encoded = io.BytesIO()
    with Image.open(texture) as source:
        source.convert("RGB").save(encoded, format="PNG")
    filename = f"{name}.png"
    spec.assets[filename] = encoded.getvalue()
    spec.add_texture(name=name, type=mujoco.mjtTexture.mjTEXTURE_2D, file=filename)
    material = spec.add_material(name=name, rgba=[1, 1, 1, 1], specular=0.0,
                                 shininess=0.0, reflectance=0.0)
    material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = name
    return name

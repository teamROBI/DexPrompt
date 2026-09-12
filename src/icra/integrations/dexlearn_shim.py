"""Make DexLearn runnable here, without changing a line of it.

DexLearn's only backbone is a Minkowski sparse-convolution U-Net, and
MinkowskiEngine does not build on this machine. It fails before reaching CUDA at
all: its setup imports ``numpy.distutils``, which modern setuptools no longer
supports and which numpy removed outright, so the build ends in

    TypeError: Compiler.__init__() takes from 1 to 3 positional arguments but 4
    were given

Pinning numpy and setuptools old enough to satisfy it conflicts with the torch
build this driver needs, and the package has been unmaintained since CUDA 11. So
two substitutions are made from outside the checkout:

**A backbone.** :class:`DensePointNet` has the interface ``BaseModel`` calls --
``forward(data) -> (global_feature, local_feature)`` over ``data["point_clouds"]``
-- and is plain PyTorch. It is a weaker encoder than a sparse U-Net and should be
read as what makes the pipeline run rather than as a reproduction of the paper's
numbers; swapping it back is one config line if MinkowskiEngine is ever built.

**A stub for the import.** ``dexlearn.dataset`` imports MinkowskiEngine at module
level but only *calls* it to voxelise, and voxelisation is skipped for any
backbone whose name lacks "MinkUNet" -- ``create_dataset`` decides that. The stub
satisfies the import and raises if anything actually reaches for it, so a silent
fallback to wrong behaviour is not possible.

:func:`install` puts both in place; call it before importing anything from
``dexlearn``.
"""

from __future__ import annotations

import sys
import types


class _Unavailable:
    """Anything reached for on the stub, which is an error rather than a no-op."""

    def __init__(self, path: str = "MinkowskiEngine"):
        self._path = path

    def __getattr__(self, name: str) -> "_Unavailable":
        return _Unavailable(f"{self._path}.{name}")

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            f"{self._path} was called, but MinkowskiEngine is not installed here. "
            "That means a sparse-voxel path is live -- check that the backbone's "
            "name does not contain 'MinkUNet', which is what turns voxelisation on."
        )


def _stub_module(name: str) -> types.ModuleType:
    """A module that raises on use but behaves normally as a module.

    Dunder attributes are left alone deliberately. Python's own import
    machinery asks a module for ``__file__``, ``__spec__`` and friends -- torch
    does it while registering custom ops -- and answering those with a raising
    proxy turns an unrelated import into a confusing failure inside this file.
    """
    module = types.ModuleType(name)
    module.__file__ = f"<{name} stub>"
    module.__path__ = []

    def missing(attribute: str):
        if attribute.startswith("__") and attribute.endswith("__"):
            raise AttributeError(attribute)
        return _Unavailable(f"{name}.{attribute}")

    module.__getattr__ = missing
    return module


def _stub_minkowski() -> None:
    if "MinkowskiEngine" in sys.modules:
        return
    names = [
        "MinkowskiEngine",
        "MinkowskiEngine.utils",
        "MinkowskiEngine.modules",
        "MinkowskiEngine.modules.resnet_block",
    ]
    stubs = {name: _stub_module(name) for name in names}
    stubs["MinkowskiEngine"].utils = stubs["MinkowskiEngine.utils"]
    stubs["MinkowskiEngine"].modules = stubs["MinkowskiEngine.modules"]
    stubs["MinkowskiEngine.modules"].resnet_block = (
        stubs["MinkowskiEngine.modules.resnet_block"]
    )
    sys.modules.update(stubs)


def _backbone_class():
    """Built lazily so importing this module does not require torch."""
    import torch
    import torch.nn as nn

    class DensePointNet(nn.Module):
        """A per-point MLP with a pooled summary, in place of the sparse U-Net.

        Point clouds arrive dense here -- ``(B, N, 3)`` -- because voxelisation
        is skipped for any backbone not named for MinkUNet, so no sparse tensor
        machinery is involved. The shape of what comes out is what matters: a
        global feature per cloud and a feature per point, which is what
        ``BaseModel`` concatenates the grasp-type embedding onto.

        Both max and mean pooling are used and summed. Max alone is the usual
        choice and is nearly blind to how *much* of a surface is present, which
        for partial views -- the only kind here -- is information worth keeping.
        """

        def __init__(self, cfg):
            super().__init__()
            width = int(getattr(cfg, "out_feat_dim", 1024))
            hidden = max(64, width // 8)
            self.point_mlp = nn.Sequential(
                nn.Linear(3, hidden), nn.GELU(),
                nn.Linear(hidden, hidden * 2), nn.GELU(),
                nn.Linear(hidden * 2, width),
            )
            # Applied after pooling rather than per point: a summary of the whole
            # cloud is what conditions the flow, and it is cheap to enrich here.
            self.summary = nn.Sequential(
                nn.Linear(width, width), nn.GELU(), nn.Linear(width, width)
            )

        def forward(self, data):
            points = data["point_clouds"]
            if not torch.is_tensor(points):
                points = torch.as_tensor(points)
            points = points.float()
            # Centred per cloud: where the object sits in the frame is carried by
            # the grasp, not by the shape, and leaving it in makes the encoder
            # learn the table's position instead of the object's geometry.
            centre = points.mean(dim=1, keepdim=True)
            local_feature = self.point_mlp(points - centre)
            pooled = local_feature.max(dim=1).values + local_feature.mean(dim=1)
            return self.summary(pooled), local_feature

    return DensePointNet


def install() -> None:
    """Stub MinkowskiEngine and register the replacement backbone.

    Call before importing anything from ``dexlearn``: its dataset module imports
    MinkowskiEngine at import time, so the stub has to be in ``sys.modules``
    first.
    """
    _stub_minkowski()
    import dexlearn.network.backbones as backbones
    import dexlearn.network.models.base as base

    backbone = _backbone_class()
    backbones.DensePointNet = backbone
    # BaseModel resolves the backbone by ``eval(cfg.backbone.name)`` against its
    # own module globals, which star-imports the backbones package -- so the name
    # has to be there as well.
    base.DensePointNet = backbone

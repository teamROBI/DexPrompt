"""Offline full-model forward with real weights/data; optimization is forbidden."""
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]


def main():
    import torch
    from train_hx5_student import dry_run

    from icra.student import LocalDINO

    def forbidden(*args, **kwargs):
        raise RuntimeError("This verification forbids downloads and optimizer/backward calls")

    torch.Tensor.backward = forbidden
    torch.autograd.backward = forbidden
    torch.optim.Adam = forbidden
    torch.optim.AdamW = forbidden
    torch.hub.download_url_to_file = forbidden
    original_init = LocalDINO.__init__
    encoders = []

    def audited_init(self, source, name, weights):
        started = time.monotonic()
        original_init(self, source, name, weights)
        assert not self.model.training
        assert not any(p.requires_grad for p in self.model.parameters())
        record = {"name": name, "weights": str(weights), "strict_loaded": True,
                  "frozen": True, "embedding_width": self.width,
                  "parameters": sum(p.numel() for p in self.model.parameters()),
                  "seconds": time.monotonic() - started}
        encoders.append(record)
        print(json.dumps(record), flush=True)

    LocalDINO.__init__ = audited_init
    args = SimpleNamespace(
        data=ROOT / "out/offset-compare-93/executed-offset",
        cache=ROOT / "out/student-review-20260909/cache",
        full_model=True, device="cpu", out=ROOT / "out/student-dino-full-check-20260909",
    )
    config = json.loads((ROOT / "configs/hx5_student.local.json").read_text())
    started = time.monotonic()
    print("Starting full pretrained-DINO student forward on CPU; no learning.", flush=True)
    dry_run(args, config)
    assert len(encoders) == 2
    report = {"encoders": encoders, "full_check_seconds": time.monotonic() - started,
              "downloads_during_check": 0, "backward_calls": 0, "optimizer_steps": 0,
              "device": "cpu", "fixture_used": False}
    (args.out / "encoder-check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

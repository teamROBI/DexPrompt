"""Small CPU checks against the actual pinned upstream model and DDIM scheduler."""
from __future__ import annotations

# Support direct CLI execution and legacy sibling imports after relocation.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[1]))
from _bootstrap import setup as _layout_setup
_layout_setup()


import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from icra.student import ActionTransformer, RMSNorm, cosine_alphas
from icra.student_upstream import OfficialActionTransformer, optimizer_groups, scheduler


def check(config):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    small = dict(config, embedding_dim=32, layers=1, heads=4, action_horizon=4, dropout=0.0)
    official = OfficialActionTransformer(small, 5).eval()
    native = ActionTransformer(small, 5).eval()
    src, dst = official.model.state_dict(), native.state_dict()
    pairs = {
        'input_emb': 'action_embedding', 'pos_emb': 'position',
        'cond_pos_emb': 'condition_position', 'time_emb.mlp': 'time_embedding',
        'ln_f': 'output.0', 'head': 'output.1',
    }
    for norm in ('norm1', 'norm2', 'norm3'):
        pairs['blocks.0.' + norm] = 'blocks.0.' + norm
    for key, value in {
        'attn.q_norm': 'self_attention.query_norm', 'attn.k_norm': 'self_attention.key_norm',
        'attn.proj': 'self_attention.projection', 'cross_attn.q': 'cross_attention.query',
        'cross_attn.kv': 'cross_attention.key_value', 'cross_attn.q_norm': 'cross_attention.query_norm',
        'cross_attn.k_norm': 'cross_attention.key_norm', 'cross_attn.proj': 'cross_attention.projection',
        'ffn.fc1': 'feed_forward.0', 'ffn.fc2': 'feed_forward.2',
    }.items():
        pairs['blocks.0.' + key] = 'blocks.0.' + value
    assigned = set()
    for old, new in pairs.items():
        for suffix in ('', '.weight', '.bias', '.0.weight', '.0.bias', '.2.weight', '.2.bias'):
            if old + suffix in src and new + suffix in dst:
                dst[new + suffix] = src[old + suffix]
                assigned.add(new + suffix)
    for suffix in ('weight', 'bias'):
        q, kv = src['blocks.0.attn.qkv.' + suffix].split([32, 64])
        for name, val in [('query', q), ('key_value', kv)]:
            key = 'blocks.0.self_attention.' + name + '.' + suffix
            dst[key] = val
            assigned.add(key)
    assert assigned == set(dst), set(dst) - assigned
    native.load_state_dict(dst)
    x, cond = torch.randn(2, 4, 27), torch.randn(2, 5, 32)
    t = torch.tensor([0, 17])
    with torch.no_grad():
        a, b = official(x, t, cond), native(x, t, cond)
    raw_error = float((a-b).abs().max())
    # timm==1.0.12's RmsNorm fallback uses sample variance, whereas the
    # workspace implementation uses mean square. Match only this operation
    # in this disposable comparison model, not in the production workspace.
    for module in native.modules():
        if isinstance(module, RMSNorm):
            module.forward = lambda value, m=module: value * torch.rsqrt(
                value.var(dim=-1, keepdim=True) + 1e-6
            ) * m.weight
    with torch.no_grad():
        b_matched = native(x, t, cond)
    torch.testing.assert_close(a, b_matched, atol=2e-6, rtol=2e-5)
    condition = cond.clone().requires_grad_(True)
    loss = official(x, t, condition).square().mean()
    loss.backward()
    assert torch.isfinite(condition.grad).all() and condition.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in official.parameters() if p.grad is not None)
    sched = scheduler(config)
    torch.testing.assert_close(sched.alphas_cumprod, cosine_alphas(config['diffusion_steps']))
    sched.set_timesteps(config['inference_steps'])
    model_output = torch.full_like(x, 0.2)
    prev = sched.step(model_output, sched.timesteps[-1], x).prev_sample
    if config['prediction_type'] == 'x0':
        torch.testing.assert_close(prev, model_output)
    groups = optimizer_groups(official, 0.01)
    params = [p for g in groups for p in g['params']]
    assert len(params) == len({id(p) for p in params})
    assert {id(p) for p in params} == {id(p) for p in official.parameters() if p.requires_grad}
    assert all(p.ndim >= 2 for p in groups[0]['params'])
    assert all(p.ndim < 2 for p in groups[1]['params'])
    return {'passed': True, 'matched_weight_original_norm_max_error': raw_error,
            'matched_weight_and_norm_max_error': float((a-b_matched).abs().max()),
            'timm_1_0_12_norm': 'sample variance, not mean square; upstream behavior retained',
            'condition_gradient_nonzero': True, 'backward_finite': True,
            'optimizer_coverage': True, 'ddim_timesteps': sched.timesteps.tolist(),
            'alpha_schedule_matches': True, 'optimizer_steps': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = check(json.loads(args.config.read_text()))
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))

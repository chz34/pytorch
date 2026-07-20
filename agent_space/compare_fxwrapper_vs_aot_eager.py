"""Compare, for the same model, the FX graphs produced by:

  (A) aot_eager       -- aten-level fwd/bwd graphs, no Inductor
  (B) inductor fx_wrapper (with the ExprBuffer fix) -- the rebuilt *host* fwd/bwd
      graphs, CPU cpp kernels routed through torch_dispatch_capture.v4

Point of the comparison: the ExprBuffer/buffer is only a codegen-time handle. The
FX GraphModule that finally comes out of the fx_wrapper STILL carries the compound
symint (s0 + 448) as a placeholder meta['val'] -- exactly like aot_eager's backward
graph does. The fix just lets that placeholder be minted; it does not "lower away"
the expression.

Needs the ExprBuffer fix present in the torch being imported. Run:
    python compare_fxwrapper_vs_aot_eager.py
"""
import sys

import torch
import torch._inductor.config as inductor_config
from torch._functorch.compilers import nop

# Force torch_dispatch_capture's pure-Python fallback (stale v1 _C.so aborts).
sys.modules.setdefault("torch_dispatch_capture._C", None)
import torch_dispatch_capture.v4 as tdcv4  # noqa: E402


class M(torch.nn.Module):
    def forward(self, a, b):
        z = torch.cat([a, b], dim=-1)  # last dim = s0 + 448
        return z * z.shape[-1]  # scalar s0 + 448


def _print_gm(title, gm):
    print(f"================= {title} =================")
    gm.print_readable()
    print("---- placeholders (name : meta['val']) ----")
    for n in gm.graph.find_nodes(op="placeholder"):
        print(f"  {n.name:>18} : {n.meta.get('val')}")
    print()


def _fresh_inputs():
    a = torch.randn(1, 8, 448, requires_grad=True)
    b = torch.randn(1, 8, 40, requires_grad=True)
    torch._dynamo.mark_dynamic(b, 2)
    return a, b


def run_aot_eager():
    graphs = {}

    def capture(tag):
        def comp(gm, inputs):
            graphs[tag] = gm
            return nop(gm, inputs)

        return comp

    from torch._dynamo.backends.common import aot_autograd

    backend = aot_autograd(
        fw_compiler=capture("forward"), bw_compiler=capture("backward")
    )
    torch._dynamo.reset()
    a, b = _fresh_inputs()
    out = torch.compile(M(), backend=backend, dynamic=False)(a, b)
    out.sum().backward()
    return graphs["forward"], graphs["backward"]


def run_fx_wrapper():
    host_gms = []

    def gm_backend(gm, example_inputs):
        host_gms.append(gm)
        return gm.forward

    torch._dynamo.reset()
    a, b = _fresh_inputs()
    with inductor_config.patch(force_disable_caches=True), \
            tdcv4.enable_device_with_fusion("cpu", gm_backend):
        out = torch.compile(M(), backend="inductor", dynamic=False)(a, b)
        out.sum().backward()

    # Identify fwd vs bwd host graph by the tell-tale s0 + 448 / tangents input.
    def is_backward(gm):
        return any(
            "tangents" in n.name
            or (n.meta.get("val") is not None and getattr(n.meta["val"], "node", None)
                and "448" in str(n.meta["val"]))
            for n in gm.graph.find_nodes(op="placeholder")
        )

    bwd = next(g for g in host_gms if is_backward(g))
    fwd = next(g for g in host_gms if g is not bwd)
    return fwd, bwd


def main():
    af, ab = run_aot_eager()
    ff, fb = run_fx_wrapper()

    print("\n########## (A) aot_eager: aten-level graphs ##########\n")
    _print_gm("aot_eager FORWARD", af)
    _print_gm("aot_eager BACKWARD", ab)

    print("\n########## (B) inductor fx_wrapper (fixed): host graphs ##########\n")
    _print_gm("fx_wrapper FORWARD host", ff)
    _print_gm("fx_wrapper BACKWARD host", fb)


if __name__ == "__main__":
    main()

"""Show how AOT autograd handles a backward that depends on a forward-computed
sym expr, WITHOUT Inductor (backend="aot_eager").

Same model as the FXIR repro: cat a static-last-dim tensor (448) with a
dynamic-last-dim tensor (s0), then use the combined size s0 + 448 as a scalar.
The backward needs that scalar. We capture the forward and backward FX graphs
that AOT autograd produces and print them, plus the backward placeholders'
symbolic meta vals.

Pure torch -- no Inductor, no torch_dispatch_capture. Run:
    python aot_eager_symexpr_graphs.py
"""
import torch
from torch._functorch.compilers import nop


class M(torch.nn.Module):
    def forward(self, a, b):
        z = torch.cat([a, b], dim=-1)  # last dim = s0 + 448
        return z * z.shape[-1]  # scalar s0 + 448


def main() -> None:
    graphs: dict[str, torch.fx.GraphModule] = {}

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
    a = torch.randn(1, 8, 448, requires_grad=True)
    b = torch.randn(1, 8, 40, requires_grad=True)
    torch._dynamo.mark_dynamic(b, 2)  # dynamic last dim -> symbol s0

    compiled = torch.compile(M(), backend=backend, dynamic=False)
    out = compiled(a, b)
    out.sum().backward()  # forces the backward graph to be compiled

    print("=== NO ERROR: aot_eager compiled + ran fwd/bwd ===\n")

    for tag in ("forward", "backward"):
        gm = graphs[tag]
        print(f"================= {tag} graph =================")
        gm.print_readable()
        print("---- placeholders (name : meta['val']) ----")
        for n in gm.graph.find_nodes(op="placeholder"):
            print(f"  {n.name:>16} : {n.meta.get('val')}")
        print()

    # Numerics sanity: gradients exist and match eager.
    a_ref = a.detach().clone().requires_grad_()
    b_ref = b.detach().clone().requires_grad_()
    z = torch.cat([a_ref, b_ref], dim=-1)
    (z * z.shape[-1]).sum().backward()
    print("grad match eager:",
          torch.allclose(a.grad, a_ref.grad) and torch.allclose(b.grad, b_ref.grad))


if __name__ == "__main__":
    main()

"""Standalone GPU repro of the FXIR compound-symint backward-input bug.

No torch_dispatch_capture needed: on a GPU the fused kernels are Triton, which
the stock fx_wrapper already supports, so a plain ``torch.compile`` under
``config.fx_wrapper=True`` drives the whole forward+backward through the FX
wrapper. The forward compiles fine; the backward receives the saved combined
size ``s0 + 448`` as a standalone scalar symint input, which stock FXIR cannot
build a placeholder for:

    torch._inductor.exc.InductorError:
        NotImplementedError: Unable to extract buffer from node: s0 + 448

Same failure torchtitan hits on Ascend NPU. Requires a Triton-capable device
(CUDA / XPU / NPU); it will not reproduce on CPU (there the fx_wrapper rejects
the non-Triton cpp kernel on the forward before the backward is ever compiled).

Run on a GPU box:
    python fxir_gpu_compound_symint_repro.py
    TORCHINDUCTOR_REPRO_DEVICE=xpu python fxir_gpu_compound_symint_repro.py
"""
import os

import torch
import torch._inductor.config as inductor_config


class M(torch.nn.Module):
    def forward(self, a, b):
        z = torch.cat([a, b], dim=-1)  # last dim = s0 + 448
        return z * z.shape[-1]  # uses s0 + 448 as a scalar


def _pick_device() -> str:
    dev = os.environ.get("TORCHINDUCTOR_REPRO_DEVICE")
    if dev:
        return dev
    if torch.cuda.is_available():
        return "cuda"
    for name in ("xpu", "npu"):
        backend = getattr(torch, name, None)
        if backend is not None and backend.is_available():
            return name
    return "cpu"


def main() -> None:
    device = _pick_device()
    if device == "cpu":
        print(
            "WARNING: no Triton-capable GPU found. This repro targets GPU; on CPU "
            "the fx_wrapper aborts on the forward's non-Triton kernel, not on the "
            "compound-symint backward input, so it will NOT show the target error."
        )

    torch.manual_seed(0)
    a = torch.randn(1, 8, 448, device=device, requires_grad=True)
    b = torch.randn(1, 8, 40, device=device, requires_grad=True)
    torch._dynamo.mark_dynamic(b, 2)  # dynamic last dim -> symbol s0

    torch._dynamo.reset()

    # The whole usage: flip on the FX wrapper and run an ordinary torch.compile.
    # Backward is triggered inside the context so it compiles through fx_wrapper too.
    with inductor_config.patch(
        fx_wrapper=True, size_asserts=False, alignment_asserts=False
    ):
        compiled = torch.compile(M(), dynamic=False)
        try:
            out = compiled(a, b)
            out.sum().backward()
        except Exception as e:  # noqa: BLE001 - classify the failure
            msg = str(e)
            hit = "Unable to extract buffer from node" in msg
            print(f"\nCOMPILE RAISED: {type(e).__name__}")
            print(f"reproduced the FXIR compound-symint bug: {hit}")
            for line in msg.splitlines():
                if "Unable to extract buffer" in line:
                    print("  ->", line.strip())
                    break
            raise SystemExit(1 if hit else 2)

    # Reached only when the fix is present: check numerics against eager.
    a_ref = a.detach().clone().requires_grad_()
    b_ref = b.detach().clone().requires_grad_()
    z = torch.cat([a_ref, b_ref], dim=-1)
    (z * z.shape[-1]).sum().backward()
    ok = torch.allclose(a.grad, a_ref.grad) and torch.allclose(b.grad, b_ref.grad)
    print("\nCOMPILE SUCCEEDED (fix present)")
    print(f"device: {device}   gradients match eager: {ok}")
    raise SystemExit(0 if ok else 3)


if __name__ == "__main__":
    main()

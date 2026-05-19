"""
Test for AllReduce + Residual + RMSNorm + Per-Token FP8 Quant fusion.

Tests both variants:
- kARResidualRMSNormPerTokenFP8PackedQuant: outputs quant_out only
- kARResidualRMSNormOutPerTokenFP8PackedQuant: outputs both norm_out and quant_out

Verification flow (identical for both variants):
1. Verify residual_out = residual_in + allreduce_in
2. Verify RMSNorm result (kernel output for _with_norm_out, reference for _without_norm_out)
3. Verify quantization (scale_out and quant_out) against reference
"""

import multiprocessing as mp
import tempfile

import numpy as np
import pytest
import torch
import torch.distributed as dist

import flashinfer.comm as comm
from flashinfer.comm.mnnvl import TorchDistBackend

FP8_E4M3_MAX = 448.0

# All (hidden_dim, token_num) combos to test.
# Per-token has no group_size parameter, so fewer combinations needed.
# Each case is run with both oneshot and twoshot (when token_num > world_size).
TEST_CASES = [
    # Basic cases
    (7168, 4),
    (7168, 1),
    (7168, 3),
    (7168, 64),
    (7168, 127),
    (7168, 512),
    # Various hidden_dims
    (768, 4),
    (768, 1),
    (768, 3),
    (640, 4),
    (640, 3),
    (640, 253),
    (384, 4),
    (384, 1),
    (256, 4),
    (256, 1),
    # Larger hidden_dims
    (4096, 4),
    (4096, 3),
    (8192, 4),
    (8192, 64),
    # Small token counts
    (7168, 2),
    (7168, 8),
    # Edge cases
    (128, 1),
    (128, 4),
    (256, 1),
]


def ref_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Reference RMSNorm implementation.

    Computes: out = x / sqrt(mean(x^2) + eps) * weight
    """
    token_num, hidden_dim = x.shape
    rms = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x.float() * rms * weight.float()).to(x.dtype)


def ref_per_token_quant_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference per-token FP8 quantization with UE8M0 scale.

    1. Per-token absmax
    2. UE8M0 scale = 2^ceil(log2(max(absmax / 448, 1e-10)))
    3. Quantize: clamp(x / scale, -448, 448) -> fp8_e4m3
    4. Store exponent (8-bit) in scale_out
    """
    token_num, hidden_dim = x.shape

    absmax = x.float().abs().amax(dim=-1)
    y_s = torch.exp2(torch.ceil(torch.log2((absmax / FP8_E4M3_MAX).clamp(min=1e-10))))

    q = (x.float() / y_s.unsqueeze(-1)).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    quant_out = q.to(torch.float8_e4m3fn)

    exponents = ((y_s.cpu().numpy().view(np.uint32) >> 23) & 0xFF).astype(np.uint8)
    scale_out = torch.from_numpy(exponents.astype(np.int32)).to(x.device)

    return quant_out, scale_out


def _run_single_case(
    workspace,
    world_size: int,
    rank: int,
    dtype: torch.dtype,
    hidden_dim: int,
    token_num: int,
    use_oneshot: bool,
    output_norm_out: bool,
    device: torch.device,
):
    """Run a single test case.

    Args:
        output_norm_out: if True, test kARResidualRMSNormOutPerTokenFP8PackedQuant (outputs norm_out)
                        if False, test kARResidualRMSNormPerTokenFP8PackedQuant (no norm_out)

    Verification (identical for both variants):
    1. Verify residual_out = residual_in + allreduce_in
    2. Verify RMSNorm result (kernel output if output_norm_out else reference)
    3. Verify quantization: scale_out and quant_out match reference
    """
    if not workspace[0].is_buffer_size_sufficient(
        world_size, token_num, hidden_dim, dtype
    ):
        workspace[0].destroy()
        workspace[0] = comm.create_allreduce_fusion_workspace(
            backend="trtllm",
            world_size=world_size,
            rank=rank,
            max_token_num=token_num,
            hidden_dim=hidden_dim,
            dtype=dtype,
            comm_backend=TorchDistBackend(),
        )

    allreduce_in = torch.randn(token_num, hidden_dim, dtype=dtype, device=device) * 8
    residual_in = torch.randn(token_num, hidden_dim, dtype=dtype, device=device) * 8
    rms_gamma = torch.randn(hidden_dim, dtype=dtype, device=device)

    residual_out = torch.empty_like(allreduce_in)
    norm_out = torch.empty_like(allreduce_in) if output_norm_out else None
    quant_out = torch.empty(
        token_num, hidden_dim, dtype=torch.float8_e4m3fn, device=device
    )
    scale_out = torch.empty(token_num, dtype=torch.int32, device=device)
    scale_out.fill_(0x7F7F7F7F)

    pattern = (
        comm.AllReduceFusionPattern.kARResidualRMSNormOutPerTokenFP8PackedQuant
        if output_norm_out
        else comm.AllReduceFusionPattern.kARResidualRMSNormPerTokenFP8PackedQuant
    )

    comm.allreduce_fusion(
        input=allreduce_in,
        workspace=workspace[0],
        pattern=pattern,
        residual_in=residual_in,
        residual_out=residual_out,
        norm_out=norm_out,
        quant_out=quant_out,
        scale_out=scale_out,
        rms_gamma=rms_gamma,
        rms_eps=1e-5,
        fp32_acc=True,
        use_oneshot=use_oneshot,
    )
    torch.cuda.synchronize()

    # Step 1: Verify residual computation
    ref_residual_out = allreduce_in + residual_in
    torch.testing.assert_close(
        residual_out,
        ref_residual_out,
        atol=1e-2,
        rtol=1e-2,
        msg=f"[hidden={hidden_dim}, tokens={token_num}, oneshot={use_oneshot}] "
            f"Residual mismatch",
    )

    # Step 2: Verify RMSNorm
    if output_norm_out:
        ref_norm_out = ref_rms_norm(ref_residual_out, rms_gamma, eps=1e-5)
        torch.testing.assert_close(
            norm_out,
            ref_norm_out,
            atol=1e-2,
            rtol=1e-2,
            msg=f"[hidden={hidden_dim}, tokens={token_num}, oneshot={use_oneshot}] "
                f"RMSNorm mismatch",
        )
        ref_for_quant = ref_norm_out
    else:
        ref_for_quant = ref_rms_norm(ref_residual_out, rms_gamma, eps=1e-5)

    # Step 3: Verify quantization
    ref_quant, ref_scale = ref_per_token_quant_fp8(ref_for_quant)

    assert torch.equal(scale_out.cpu(), ref_scale), (
        f"[hidden={hidden_dim}, tokens={token_num}, oneshot={use_oneshot}] "
        f"Per-token scale mismatch: "
        f"{(scale_out.cpu() != ref_scale).sum().item()}/{token_num} differ"
    )

    assert torch.equal(quant_out.view(torch.uint8), ref_quant.view(torch.uint8)), (
        f"[hidden={hidden_dim}, tokens={token_num}, oneshot={use_oneshot}] "
        f"FP8 quant mismatch: "
        f"{(quant_out.view(torch.uint8) != ref_quant.view(torch.uint8)).sum().item()}"
        f"/{quant_out.numel()} differ"
    )


def _run_batch_worker(
    world_size: int,
    rank: int,
    dtype: torch.dtype,
    cases: list,
    store_path: str,
    output_norm_out: bool,
    gpu_offset: int = 0,
):
    """Worker that runs all test cases in a single process group."""
    device = torch.device(f"cuda:{rank + gpu_offset}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{store_path}",
        rank=rank,
        world_size=world_size,
    )
    group = dist.group.WORLD

    max_tokens = max(c[1] for c in cases)
    max_hidden = max(c[0] for c in cases)
    workspace = [
        comm.create_allreduce_fusion_workspace(
            backend="trtllm",
            world_size=world_size,
            rank=rank,
            max_token_num=max_tokens,
            hidden_dim=max_hidden,
            dtype=dtype,
            comm_backend=TorchDistBackend(),
        )
    ]

    try:
        for hidden_dim, token_num in cases:
            for use_oneshot in [True, False]:
                if not use_oneshot and token_num <= world_size:
                    continue

                np.random.seed(42)
                torch.manual_seed(42)
                torch.cuda.manual_seed_all(42)

                _run_single_case(
                    workspace,
                    world_size,
                    rank,
                    dtype,
                    hidden_dim,
                    token_num,
                    use_oneshot,
                    output_norm_out,
                    device,
                )
                dist.barrier(group=group)
    finally:
        workspace[0].destroy()
        dist.destroy_process_group(group=group)


def _multi_process_batch(
    world_size: int,
    dtype: torch.dtype,
    cases: list,
    output_norm_out: bool,
    gpu_offset: int = 0,
) -> None:
    mp.set_start_method("spawn", force=True)
    store_file = tempfile.mktemp(prefix="flashinfer_dist_store_")
    procs = []
    for i in range(world_size):
        proc = mp.Process(
            target=_run_batch_worker,
            args=(world_size, i, dtype, cases, store_file, output_norm_out, gpu_offset),
            name=f"Worker-{i}",
        )
        proc.start()
        procs.append(proc)
    for i, proc in enumerate(procs):
        proc.join()
        assert proc.exitcode == 0, f"Process {i} failed with exit code {proc.exitcode}"


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_allreduce_rmsnorm_per_token_fp8_quant_with_norm_out(world_size, dtype):
    """Test kARResidualRMSNormOutPerTokenFP8PackedQuant pattern (outputs norm_out + quant_out)."""
    available_gpus = torch.cuda.device_count()
    if world_size > available_gpus:
        pytest.skip(f"Need {world_size} GPUs, have {available_gpus}")

    _multi_process_batch(world_size, dtype, TEST_CASES, output_norm_out=True)


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_allreduce_rmsnorm_per_token_fp8_quant_without_norm_out(world_size, dtype):
    """Test kARResidualRMSNormPerTokenFP8PackedQuant pattern (outputs quant_out only)."""
    available_gpus = torch.cuda.device_count()
    if world_size > available_gpus:
        pytest.skip(f"Need {world_size} GPUs, have {available_gpus}")

    _multi_process_batch(world_size, dtype, TEST_CASES, output_norm_out=False)

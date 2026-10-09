# SageAttention on RTX 20xx / GTX 16xx (Turing, sm75) on Windows

SageAttention working on an RTX 2060 (Turing, compute capability 7.5) on Windows, with ComfyUI portable.

**TL;DR:** On Turing GPUs, SageAttention 2 runs Triton kernels. With recent `triton-windows` versions these kernels fail to compile on sm75. Pinning `triton-windows==3.2.0.post21` fixes it. Use the Triton mode, **not** the `_cuda` modes (see the warning below).

---

## Tested setup

| Component | Version |
|---|---|
| GPU | RTX 2060 6 GB (sm_75) |
| RAM / OS | 16 GB, Windows 10 |
| ComfyUI | portable, 0.39.0 |
| Python | 3.13 (portable `python_embeded`) |
| PyTorch | 2.14.0+cu130 |
| triton-windows | **3.2.0.post21** |
| SageAttention | `2.2.0+cu130torch2.10.0andhigher.post6` (wheel from woct0rdho's Windows fork) |

I only tested this exact combination. Other GPUs and other CUDA/torch versions are untested.

---

## The problem

On Turing GPUs (GTX 16xx / RTX 20xx) the SageAttention Windows wheels do not include a CUDA kernel for sm75. SageAttention falls back to its Triton kernels, which use `tl.dot` on int8 tensors.

With newer `triton-windows` versions, compiling that kernel fails:

```
'arith.extf' op operand #0 must be floating-point-like, but got 'tensor<...xi8...>'
Pipeline failed while executing [TritonGPUAccelerateMatmul on 'builtin.module' operation]
RuntimeError: PassManager::run failed
```

I isolated it without SageAttention, using a small Triton script (below), on the same machine and the same torch:

| triton-windows | fp16 `tl.dot` | int8 `tl.dot` |
|---|---|---|
| 3.2.0.post21 | OK | **OK** |
| 3.4.0.post21 | OK | FAIL |
| 3.8.0.post29 | OK | FAIL |

I did not test 3.3.x.

---

## Install (ComfyUI portable)

Close ComfyUI first. Run these from the folder that contains `python_embeded`.

1. Remove any old SageAttention:
```bat
   python_embeded\python.exe -m pip uninstall -y sageattention
```
2. Install the older Triton:
```bat
   python_embeded\python.exe -m pip install triton-windows==3.2.0.post21
```
3. Download a wheel from https://github.com/woct0rdho/SageAttention/releases.
   I used `sageattention-2.2.0+cu130torch2.10.0andhigher.post6-cp310-abi3-win_amd64.whl`.
   Pick the wheel that matches the CUDA **major** version of your torch (12 vs 13). I only tested cu130.
4. Install it **without dependencies**, so pip does not replace Triton:
```bat
   python_embeded\python.exe -m pip install --no-deps "PATH\TO\that.whl"
```
5. Test before starting ComfyUI:
```bat
   python_embeded\python.exe -c "import torch; from sageattention import sageattn; q=torch.randn(1,8,256,64,dtype=torch.float16,device='cuda'); print(sageattn(q,q,q,tensor_layout='HND',is_causal=False).shape)"
```
   Expected output: `torch.Size([1, 8, 256, 64])`.
   A warning `Failed to find Python libs` appeared for me and did not stop the test. The first run is slower because Triton compiles the kernel.

---

## Use in ComfyUI

- Start ComfyUI with `--use-sage-attention`, or
- use KJNodes' **PatchSageAttentionKJ** and set the mode to `sageattn_qk_int8_pv_fp16_triton`.

### Warning: do NOT use the `_cuda` modes on Turing

- `sageattn_qk_int8_pv_fp16_cuda` raises **no error** and looks much faster (about 1.0 s/it in my case), but the output is **pure noise**.
- The other `_cuda` modes I tried failed with `CUDA error: no kernel image is available for execution on the device`.
- `sageattn3` targets Blackwell (RTX 50) and does not apply to Turing.

Use `sageattn_qk_int8_pv_fp16_triton`, and always look at the actual output, not just the speed.

---

## Results (my numbers only)

Wan 2.1 1.3B, 33 frames, 480-640 resolution, 6 steps, one workflow:

| | Without Sage | With Sage (Triton mode) |
|---|---|---|
| Sampler | ~2.1 s/it | ~1.45 s/it |
| Whole prompt | ~28 s | ~24 s |

The first run after starting ComfyUI is much slower (Triton compiles and caches kernels). Measure from the second run. The gain on the whole prompt is smaller than on the sampler, because loading and VAE decode are not affected.

---

## Check Triton on its own (no SageAttention needed)

Save as `test_dot.py` and run `python_embeded\python.exe test_dot.py`.
You want `float16 OK` and `int8 OK`.

```python
import torch, triton, triton.language as tl

@triton.jit
def k(a, b, c, N: tl.constexpr):
    i = tl.arange(0, N)
    x = tl.load(a + i[:, None] * N + i[None, :])
    y = tl.load(b + i[:, None] * N + i[None, :])
    tl.store(c + i[:, None] * N + i[None, :], tl.dot(x, y))

for dt, od in ((torch.float16, torch.float32), (torch.int8, torch.int32)):
    a = torch.ones(32, 32, dtype=dt, device='cuda')
    b = torch.ones(32, 32, dtype=dt, device='cuda')
    c = torch.empty(32, 32, dtype=od, device='cuda')
    try:
        k[(1,)](a, b, c, N=32)
        print(dt, 'OK')
    except Exception as e:
        print(dt, 'FAIL', type(e).__name__)
```

---

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `PassManager::run failed` / `arith.extf ... xi8` | Triton is too new. Reinstall `triton-windows==3.2.0.post21`. |
| `No module named 'sageattention'` | The wheel went into a different Python. Use `python_embeded\python.exe -m pip ...`. |
| pip says the wheel is not supported | Wrong wheel for your torch / CUDA major version. |
| Black image or color noise | You selected a `_cuda` mode. Switch to `sageattn_qk_int8_pv_fp16_triton`. |
| It worked, then broke after an update | Triton was upgraded again. Run `pip list \| findstr triton` and re-pin 3.2.0.post21. |

To go back to the original state:

```bat
python_embeded\python.exe -m pip uninstall -y sageattention
python_embeded\python.exe -m pip install triton-windows
```

---

## Notes

- If you update ComfyUI or install custom nodes, check `pip list | findstr triton`; it can be upgraded and the error returns.
- Only 3.2.0.post21 worked for me; I did not test 3.3.x.
- Triton 3.2 is much older than torch 2.14. The SageAttention kernels worked, but I did not test `torch.compile`-based nodes with this Triton version.
- I did not test other GPUs.
- This is a user report, not an official fix. Not affiliated with the SageAttention authors.

## Credits

- [thu-ml/SageAttention](https://github.com/thu-ml/SageAttention) for SageAttention.
- [woct0rdho](https://github.com/woct0rdho) for the Windows wheels and `triton-windows`.

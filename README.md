# SageAttention-Compute7.5-Guide--RTX-20-
SageAttention working on RTX 2060 (Turing/sm75) on Windows
On Turing GPUs (GTX 16xx / RTX 20xx), SageAttention 2 runs Triton kernels (no CUDA kernel for sm75 is bundled in the Windows wheels).
With recent triton-windows (3.4.0 and 3.8.0 in my tests) these kernels fail to compile on sm75 with:
`'arith.extf' op operand #0 must be floating-point-like ... tensor<...xi8...>` / `RuntimeError: PassManager::run failed`.
It works with triton-windows 3.2.0.post21.

**My setup:** RTX 2060 6GB, 16GB RAM, Windows 10, ComfyUI portable (Python 3.13, torch 2.14.0+cu130).

**Steps (ComfyUI portable; run from the folder that contains python_embeded):**
1. Remove old versions:
   `python_embeded\python.exe -m pip uninstall -y sageattention`
2. Install the older Triton:
   `python_embeded\python.exe -m pip install triton-windows==3.2.0.post21`
3. Download a wheel from https://github.com/woct0rdho/SageAttention/releases
   (I used `sageattention-2.2.0+cu130torch2.10.0andhigher.post6-cp310-abi3-win_amd64.whl`;
   pick the one matching your torch's CUDA major version, 12 vs 13).
4. Install it without dependencies, so pip does not replace Triton:
   `python_embeded\python.exe -m pip install --no-deps "PATH\TO\that.whl"`
5. Test before starting ComfyUI:
   `python_embeded\python.exe -c "import torch; from sageattention import sageattn; q=torch.randn(1,8,256,64,dtype=torch.float16,device='cuda'); print(sageattn(q,q,q,tensor_layout='HND',is_causal=False).shape)"`
   It should print `torch.Size([1, 8, 256, 64])`. A warning "Failed to find Python libs" appeared for me and was harmless.
6. In ComfyUI use `--use-sage-attention`, or KJNodes' PatchSageAttentionKJ with mode
   `sageattn_qk_int8_pv_fp16_triton`.

**Result (Wan 2.1 1.3B, 33 frames, 480-640, 6 steps, one workflow, my numbers only):**
- sampler: ~2.1 s/it -> ~1.45 s/it
- whole prompt: ~28 s -> ~24 s

**Notes**
- If you update ComfyUI or install nodes, check `pip list | findstr triton`; it can get upgraded and the error returns.
- Only 3.2.0.post21 worked for me. I did not test 3.3.x.
- I did not test other GPUs.
- Thanks to woct0rdho for the Windows wheels and triton-windows.
- [INFO] Model WAN21 prepared for dynamic VRAM loading. 2705MB Staged. 613 patches attached. Force pre-loaded 180 weights: 543 KB.
 12%|██              | 1/8 [00:06<00:45,  6.52s/it,  Model Initialization complete!  ][PROBE] sageattn_qk_int8_pv_fp16_triton call #7000
[PROBE] sageattn call #2000
100%|███████████████████████████████████████████████████| 8/8 [00:13<00:00,  1.63s/it]

# Building on a 2019 MacBook Pro / AMD Radeon Pro 5500M (4GB)

Apple ships no native Vulkan driver. On this hardware class — an Intel Mac with a
discrete AMD GPU — the working path is ggml's **Vulkan** backend running over
**MoltenVK**, Apple's Vulkan-over-Metal translation layer. MoltenVK is faithful
enough to run ggml, but not faithful enough to give compute shaders every
guarantee a native Vulkan driver would, and this branch carries the fixes for the
gaps that matter on an RDNA1 card (and, defensively, on older Intel iGPUs).

Target: **MacBook Pro 16" 2019 · AMD Radeon Pro 5500M (4 GB, RDNA1 / Navi 14) ·
Intel UHD 630 (present, excluded) · macOS 14.x.**

## Requires MoltenVK >= 1.4.2

Up to 1.4.1, MoltenVK reported `subgroupSize 64` on AMD Macs while Metal actually
runs **32-wide SIMD groups**. Every subgroup reduction in every shader therefore
spanned the wrong lane count — the single root cause behind the matmul
mistranslation, the NaN in the hybrid-model shaders, and upstream issue 15846.
1.4.2 corrects the report (`subgroupSize 32`, min 32, max 32).

```sh
brew list --versions molten-vk    # must be >= 1.4.2
```

The branch keeps working on older drivers: the three subgroup workarounds are
gated on `driverVersion >= 10402` (MoltenVK encodes major*10000 + minor*100 +
patch), so 1.4.1 and below take the safe path automatically. But the flash
attention and quantized-KV gains below need 1.4.2.

## What this branch changes

Everything is in `src/ggml-vulkan/`: `ggml-vulkan.cpp` plus three `.comp`
shaders. Nothing changes behaviour on GPUs where subgroup arithmetic behaves as
specified (native Vulkan AMD/NVIDIA, Linux Mesa).

**Correctness** — each of these produced *wrong output, not a crash*:

- `vulkan: fix MoltenVK/Intel subgroup matmul correctness on RDNA1 and older
  Intel iGPUs` — matmul shaders assumed subgroup-uniform control flow MoltenVK
  does not guarantee. Upstream already disables the path on Apple for AMD; this
  extends it to Intel, so a mis-targeted iGPU fails visibly instead of quietly.
- `vulkan: fix NaN SSM_SCAN/GATED_DELTA_NET output on MoltenVK/RDNA1` — same root
  cause, different shaders: a shared-memory reduction indexed by
  `gl_SubgroupInvocationID` instead of the linear workgroup index. Without it,
  Mamba-2 and Gated DeltaNet models emit token salad on the GPU while staying
  coherent on the CPU.
- `vulkan: disable subgroup clustered ops on MoltenVK (AMD/Intel)` —
  `quantize_q8_1`'s subgroup variant reduces 8-lane blocks with
  `subgroupClusteredMax`/`subgroupClusteredAdd`, which MoltenVK does not map to
  the intended lanes. It corrupts the quantized activations both integer-dot
  matmul paths consume.
- `vulkan: do not use mul_mat_vecq for q8_0 on MoltenVK` — the multi-column
  `mul_mat_vecq` variant returns wrong results for q8_0 (n=2..9, err ~1.0) on this
  driver while `n=1` and the `mul_mmq` path are correct. The guard is
  unconditional because 1.4.2 does not fix it either.
- `vulkan: keep q8_0 off the flash-attention MMQ path on MoltenVK` — the same
  defect in a second place, and one this branch caused itself. Enabling integer
  dot makes `ggml_vk_fa_scalar_uses_mmq` accept q8_0, and its MMQ block loader is
  the only q8_0 path repacking through `pack32(i16vec2(...))` from a 16-bit view;
  every other type packs from `u16vec2`. Result: **every** graded q8_0 flash
  attention case failed (337/337, err 0.041–0.092) while q8_0 stayed correct in
  `GET_ROWS`, `CPY`, `MUL_MAT` and friends. Routing it back to the dequantize
  path restores 4757/4757. This is the shared root cause of both q8_0 defects.

**Performance** — both were software ceilings, not silicon limits:

- `vulkan: enable emulated integer dot for matmul on MoltenVK/AMD` — MoltenVK
  advertises `VK_KHR_shader_integer_dot_product` but reports every `*Accelerated`
  flag false, because Metal has no DP4a-style intrinsic. The SPIRV-Cross emulation
  still wins decisively for batched matmul: **+68% prefill** (pp512 35.0 → 58.7),
  decode unchanged. `mmvq_mode` defaults to the float path here, because the
  emulation loses on the memory-bound single-column path.
- `vulkan: run flash attention on the GPU under MoltenVK` — `FLASH_ATTN_EXT` was
  *unsupported* on this platform, so ggml silently scheduled every attention op on
  the **CPU** (`test-backend-ops support -o FLASH_ATTN_EXT` reported 0 of 5097
  cases supported). The scalar shader already had a subgroup-free path; one
  dependency (`subgroupAll` in the mask block-skip) sat outside it. Guarding that
  with a workgroup reduction makes FA run on the GPU, turning a **3.5x decode
  penalty into a 4.7% gain** — and making quantized KV usable at all.
- `vulkan: trust MoltenVK subgroups from 1.4.2 onwards` — gates the three subgroup
  workarounds above on `driverVersion >= 10402`, with env overrides in both
  directions. Worth ~1% on decode on top of the +5.8% the driver upgrade itself
  brings.

Flash attention stays on the subgroup-free path regardless of driver version: it
is still wrong with subgroups on 1.4.2 (897 of 4759 cases fail, err 0.04–0.72).

## Dependencies (Homebrew)

```sh
brew install cmake molten-vk vulkan-loader vulkan-headers spirv-headers shaderc glslang
```

`molten-vk` is the driver; `vulkan-loader` + `vulkan-headers` are what you build
against (never link `libMoltenVK.dylib` directly — go through the loader so the
driver stays swappable); `spirv-headers`/`shaderc`/`glslang` compile the `.comp`
shaders to SPIR-V at build time.

## Build

```sh
git clone --branch macbook-pro-2019-radeon-5500m-4gb https://github.com/maximosipov/ggml.git
cd ggml

P="$(brew --prefix)"
cmake -B build \
  -DGGML_VULKAN=1 \
  -DGGML_METAL=OFF \
  -DVulkan_INCLUDE_DIR="$P/opt/vulkan-headers/include" \
  -DVulkan_LIBRARY="$P/opt/vulkan-loader/lib/libvulkan.dylib" \
  -DVulkan_GLSLC_EXECUTABLE="$P/opt/shaderc/bin/glslc" \
  -DVulkan_GLSLANG_VALIDATOR_EXECUTABLE="$P/opt/glslang/bin/glslangValidator" \
  -DCMAKE_CXX_FLAGS="-I$P/opt/spirv-headers/include" \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j"$(sysctl -n hw.ncpu)"
```

`GGML_METAL=OFF` matters: on an Intel Mac with a discrete AMD GPU, ggml's Metal
backend re-reads model weights over PCIe every token instead of keeping them
resident — Vulkan/MoltenVK does not have this problem. Pass it explicitly;
ggml auto-enables Metal on macOS, and Metal then wins device 0 at runtime.

Building ggml as the top-level project (as above, not vendored inside llama.cpp)
auto-enables `GGML_BUILD_TESTS`, which produces `build/bin/test-backend-ops`.

## Point the Vulkan loader at MoltenVK, not a native driver

```sh
export VK_ICD_FILENAMES="$(brew --prefix)/opt/molten-vk/etc/vulkan/icd.d/MoltenVK_icd.json"
export DYLD_LIBRARY_PATH="$(brew --prefix)/opt/molten-vk/lib:$(brew --prefix)/opt/vulkan-loader/lib:${DYLD_LIBRARY_PATH:-}"
```

If your Mac has more than one GPU (e.g. an Intel iGPU alongside a discrete AMD
GPU), list devices and pick the right index:

```sh
./build/bin/test-backend-ops list  # or run any binary with GGML_VK_VISIBLE_DEVICES unset once to see enumeration in the log
export GGML_VK_VISIBLE_DEVICES=0   # restrict to the device you want
```

On this specific machine (Radeon Pro 5500M + Intel UHD 630), the UHD reports
32GiB of "VRAM" (it's shared system memory) against the Radeon's real 4GiB, so
anything that free-lists by VRAM size picks the UHD first — pin the index
explicitly rather than trusting auto-selection.

The capability banner on this card, on MoltenVK 1.4.2:

```
ggml_vulkan: Found 2 Vulkan devices:
ggml_vulkan: 0 = AMD Radeon Pro 5500M (MoltenVK) | uma: 0 | fp16: 1 | bf16: 0 | fp4: 0 | warp size: 32 | shared memory: 65536 | int dot: 0 | matrix cores: none
ggml_vulkan: 1 = Intel(R) UHD Graphics 630 (MoltenVK) | uma: 1 | fp16: 1 | bf16: 0 | fp4: 0 | warp size: 32 | shared memory: 65536 | int dot: 0 | matrix cores: none
```

Two fields in that line need reading carefully:

- **`warp size: 32`** is the check that you are on MoltenVK >= 1.4.2. On 1.4.1 and
  earlier this said `64`, which was the driver misreporting Metal's 32-wide SIMD
  group. If you see `64`, upgrade before trusting any output.
- **`int dot: 0` is expected even with the integer-dot fix active.** The banner
  applies the `integerDotProduct4x8BitPackedSignedAccelerated` gate itself, so it
  reports what the driver claims, not what this branch decided to do anyway. The
  path being live shows up as prefill throughput (~59 pp512 on a 4B Q4_K_M rather
  than ~35), not in the banner.

Device memory reads `4080 MB` for the Radeon against `32768 MB` for the UHD 630 —
which is the whole reason the pin above is load-bearing.

## Verify the fixes

```sh
./build/bin/test-backend-ops -o SSM_SCAN        -b Vulkan0
./build/bin/test-backend-ops -o GATED_DELTA_NET -b Vulkan0
./build/bin/test-backend-ops -o MUL_MAT         -b Vulkan0
./build/bin/test-backend-ops -o FLASH_ATTN_EXT  -b Vulkan0
```

The first two must report all cases passing (no NaN / no ERR≈1.0 failures) — that
is the exact regression test the shader fixes were validated against. `MUL_MAT`
covers the matmul, clustered-quantizer and integer-dot changes together.

`FLASH_ATTN_EXT` is the one to read carefully: before the fix it reported *0 cases
supported*, which is not the same as passing. Expect **4757/4757** now that q8_0
is routed off the MMQ path; if you see ~337 failures all with `type_K=q8_0`, you
are on a build without that fix.

For full backend coverage: `./build/bin/test-backend-ops -b Vulkan0` (no `-o`
filter) — expect a small number of intentional CPU fallbacks (e.g. `CUMSUM`),
not correctness failures.

## Diagnostic knobs

All opt-in; default behaviour is unchanged when unset.

| Variable | Effect |
|---|---|
| `GGML_VK_FORCE_MOLTENVK_WORKAROUNDS` | keep the subgroup workarounds on even on MoltenVK >= 1.4.2 |
| `GGML_VK_NO_MOLTENVK_WORKAROUNDS` | drop them on older drivers (expect wrong output below 1.4.2) |
| `GGML_VK_DISABLE_INTEGER_DOT_PRODUCT` | back out the integer-dot enablement (upstream knob) |
| `GGML_VK_FORCE_MMVQ` / `GGML_VK_DISABLE_MMVQ` | override the vector-matmul path selection |
| `GGML_VK_VISIBLE_DEVICES` | restrict device enumeration — load-bearing here, see above |

The llama.cpp fork's branch carries the same fixes plus a few more diagnostic
knobs (`GGML_VK_FORCE_ARCH`, `GGML_VK_RM_KQ`, `GGML_VK_RM_STDQ`,
`GGML_VK_FORCE_INTEGER_DOT`) used to isolate them.

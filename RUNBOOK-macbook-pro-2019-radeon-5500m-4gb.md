# Building on a 2019 MacBook Pro / AMD Radeon Pro 5500M (4GB)

This branch carries two correctness fixes for ggml's Vulkan backend running over
**MoltenVK** (Apple's Vulkan-over-Metal translation layer) on this GPU class
(AMD RDNA1, and separately older Intel iGPUs):

- `vulkan: fix MoltenVK/Intel subgroup matmul correctness on RDNA1 and older Intel
  iGPUs` — matmul shaders assumed subgroup-uniform control flow MoltenVK does not
  guarantee, producing wrong (not crashing, just *wrong*) output.
- `vulkan: fix NaN SSM_SCAN/GATED_DELTA_NET output on MoltenVK/RDNA1` — same root
  cause, different shaders: a shared-memory reduction indexed by
  `gl_SubgroupInvocationID` instead of the linear workgroup index.

Apple does not ship a native Vulkan driver; MoltenVK translates Vulkan calls to
Metal, and both fixes exist because that translation does not give shaders the
subgroup guarantees they'd get on a native Vulkan driver (AMD/NVIDIA/Linux Mesa).
Neither fix changes behavior on GPUs where subgroup arithmetic is used as-is.

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
resident — Vulkan/MoltenVK does not have this problem.

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

## Verify the fixes

```sh
./build/bin/test-backend-ops -o SSM_SCAN -b Vulkan0
./build/bin/test-backend-ops -o GATED_DELTA_NET -b Vulkan0
```

Both must report all cases passing (no NaN / no ERR≈1.0 failures). This is the
exact regression test the fixes on this branch were validated against.

For full backend coverage: `./build/bin/test-backend-ops -b Vulkan0` (no `-o`
filter) — expect a small number of intentional CPU fallbacks (e.g. `CUMSUM`),
not correctness failures.

# llama.cpp-S4000-MUSA-patch

让 llama.cpp 的 MUSA 后端在摩尔线程 **MTT S4000** 上能正常跑、算对的补丁分支。

Patches that make llama.cpp's MUSA backend run correctly on the Moore Threads MTT S4000 (mp_22). Upstream llama.cpp crashes in `test-backend-ops` on this card; this branch is upstream master plus small MUSA-only fixes.

## 现状（2026-11-21）

上游正式版 b12606（今天发布）在 S4000 上跑 `test-backend-ops`，矩阵乘一开始就崩。更早的版本不崩，但量化矩阵乘会算错。

本分支 = 上游主线 `44c20b1c` + 下面几处修复。所有改动都包在 `#if defined(GGML_USE_MUSA)` 里，NVIDIA / AMD 等其他后端的编译结果不变。

| `test-backend-ops test -b MUSA0` | 上游 b12606 | 上游 3cf0325（09-20） | 本分支 |
|---|---|---|---|
| MUL_MAT | 跑到 `q5_0, n=8` 崩：`too many resources requested for launch` | 1184 OK / **113 FAIL**（算错、NaN） | **2692 OK / 0 FAIL** |
| MUL_MAT_ID | 第一个用例就崩 | 530 OK / **397 FAIL** | 1065 OK / 2 FAIL（见"已知问题"） |
| SOFT_MAX | 崩：`musaFuncSetAttribute(...) invalid argument` | 通过 | 209 OK，之后崩（见"已知问题"） |
| 全部 160 个算子（逐个算子跑，每个限时 15 分钟） | — | — | 14493 OK / 2 FAIL，SOFT_MAX 崩，SOLVE_TRI 卡住 |

计数只算测试用例那一行，不算每个算子末尾的 `Backend MUSA0: OK` 汇总行。FLASH_ATTN_EXT 在这张卡上 5776 条全部是 not supported，一条都没实际执行。

## 修复

1. **量化矩阵乘的分块参数，host 和 device 选的不一样**（`ggml/src/ggml-cuda/mmq.cuh`）
   host 端按 ampere 那张表算 grid 和共享内存大小，device 端因为 MUSA 被条件编译排除在外，落到了 pascal_older 那张表上。两边对"每个线程块负责几行"理解不一致，结果是批量矩阵乘和大尺寸矩阵乘算错甚至出 NaN。修法：MUSA 的 device 端也用 ampere 表。
   另一种做法是让 host 也选 pascal_older，也试过：预填充更快一些，但 `q2_K` 有一条用例算错，所以没用。
2. **warp 宽度**（`ggml/src/ggml-cuda/common.cuh`、`ggml-cuda.cu`）
   S4000 报告 `warpSize = 128`，host 就按 128 发线程块；但所有核函数都是按 32 宽编译的（`__launch_bounds__`），启动时报 `too many resources requested for launch`。
   修法：新增常量 `GGML_CUDA_MUSA_WARP_SIZE`，设备端的 `ggml_cuda_get_physical_warp_size()` 和 host 端的 `warp_size` 都用它，两边永远一致。S5000 看起来本来就报 32，对它没有影响。上游原来就有一个 MUSA 一律设成 32 的特判，在给 S5000 修问题的 PR #29193 里被删掉了。
3. **动态共享内存上限**（`ggml/src/ggml-cuda/common.cuh`、`mmid.cu`）
   MUSA 编译器给**每个**核函数都放了至少 20 字节静态共享内存（代码里没写也有），驱动要求"静态 + 动态 ≤ 73728 字节"，所以按 73728 申请必然报 `invalid argument`。修法：新增 `ggml_cuda_musa_max_dynamic_smem()`，算出"73728 减去这个核函数自己的静态用量"。设置上限时用它，MoE 路由辅助核函数 `mm_ids_helper` 启动时算共享内存大小也用它。

## 已知问题（排查中）

- **MMQ 在批量 17 到 32 时部分列算错**：`q2_K` 选中宽度 32 的分块时，第 16 到 23 列算错；`q4_K` 在 `m=81920, k=256` 时也错。上游的 MUL_MAT 测试批量最大只测到 16，所以全量测试发现不了，是另外补测试用例才查出来的。
- **MUL_MAT_ID**：2 条 `dup=1` 用例算错（同一个 token 重复选中同一个专家，`q8_0` 和 `q4_0`，n=64），可能和上一条是同一个原因。
- **SOFT_MAX**：跑完一行 200001 个数的超宽用例后，下一个用例报 `operation aborted`。6050b467 上没有这个问题。
- **SOLVE_TRI**：走 muBLAS 批量三角求解那条路时，和前面的异步工作之间没有排好顺序。单独跑就对，连着跑就算错或卡住。
- `FLASH_ATTN_EXT` 在 S4000 上没开。
- 只在 S4000（mp_22）上测过，S80（mp_21）和 S5000（mp_31）没测。

## 编译

```sh
cmake -B build -DGGML_MUSA=ON -DMUSA_ARCHITECTURES=22
cmake --build build -j
```

测试环境：AutoDL 租的 1× MTT S4000 48 GB，驱动 5.2.0-server，MUSA SDK 5.2.0。

## 为什么上游没发现

上游的 CI 只在 S5000 的 SDK 容器里**编译** MUSA（`-DMUSA_ARCHITECTURES=31`），不在真卡上跑测试，S4000 这一代不在测试范围里。S5000 上 MMQ 也算错，上游的处理是在 `ggml_cuda_should_use_mmq()` 里直接对 S5000 关掉 MMQ，没有找根因。

## 速度

这些修复只保证算对，还没调速。目前在 S4000 上（4096×14336 的矩阵）：

- 单 token 解码：q8_0 只用到显存带宽的约 37%，IQ 系列只有 7–8%。
- 512 个 token 预填充：走 MMQ 的 q8_0、iq2_xxs、iq2_xs 要 58–63 ms；同一张卡上走 muBLAS 的 iq1_m 只要 2.24 ms。原因是 MMQ 在 MUSA 上用不上矩阵单元，而 `ggml_cuda_should_use_mmq()` 里"有 fp16 矩阵单元时，大批量改用 BLAS"这条规则只对 NVIDIA 生效。下一步会试着让 S4000 在大批量时也走 muBLAS。

---

# llama.cpp

![llama.cpp](https://raw.githubusercontent.com/ggml-org/llama-brand/refs/heads/master/logo/svg/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v1)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[v1.0.0](https://huggingface.co/blog/ggml-org/llama-cpp-v1) / [ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/pulls?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Asanmai%20OR%20author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Athelittlefireman%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3AmasterFoad%20OR%20author%3Aallozaur%20OR%20author%3AAndreas-Krebbel%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Achaxu01%20OR%20author%3Alhez%20OR%20author%3AAlekseiNikiforovIBM%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming%20OR%20author%3Aynankani)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Description

The main goal of `llama.cpp` is to enable LLM (and MLLM) inference with minimal setup and state-of-the-art performance on a wide range of hardware - locally and in the cloud.

The `llama.cpp` project is built on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Quick start

A few options to get `llama.cpp` installed on your machine:

```bash
# curl
curl -LsSf https://llama.app/install.sh | sh

# powershell
irm https://llama.app/install.ps1 | iex
```

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Llama Chat, the built-in chat app, against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Llama Chat, the built-in chat app, against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [Text-to-speech (TTS) models](docs/multimodal/tts.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain

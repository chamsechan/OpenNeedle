# 编译原生引擎

OpenNeedle 的原生计算核心使用 **C++17 + OpenMP**，通过 C ABI 供 Python 调用。
`pip install -e .` 或安装当前 wheel 会安装源码及 NumPy；PyTorch、safetensors 和 regex 是 `.[torch]` 可选依赖，`.[test]` 包含测试所需的 PyTorch 依赖。默认在首次调用原生后端时编译，
后续进程复用缓存。纯 PyTorch 后端和模型转换不需要编译本项目的 C++ 内核。

| 方式 | 适用场景 | 产物 |
|---|---|---|
| 首次推理自动编译 | 本地快速使用 | 哈希命名的缓存共享库 |
| `python -m needle2 build-native` | 提前检查环境、预热部署缓存 | 与自动编译相同的库 |
| CMake | 显式构建、CI、预编译部署 | `libneedle2_native.so` |

## 环境

以下命令从项目根目录运行。当前独立 CMake 构建面向 Linux，已在 Linux ARM64、
GCC 13.3、CMake 3.28 上验证。其他 Linux 架构包含标量路径，需在目标平台验证；
SDOT 需要 Linux ARM64 DotProd。Windows 和 macOS 不在当前 CMake 支持范围内。

Ubuntu / Debian 的构建依赖：

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake
```

自动编译只需要支持 C++17 和 OpenMP 的编译器，不需要 CMake。
CMake 构建要求 CMake ≥ 3.16；构建共享库本身不依赖 Python、PyTorch 或模型文件。
从 Python 使用时，仍需按[快速开始](../README.md#快速开始)安装项目依赖。

## 提前编译到运行时缓存

```bash
python -m needle2 build-native
```

命令构建或复用共享库，实际加载并输出 JSON：库的绝对路径、编译特性以及
当前 CPU 的 `sdot_available`。它不下载模型、不执行推理。

编译器默认 `c++`，缓存默认 `~/.cache/needle2`。可指定：

```bash
CXX=g++ NEEDLE2_NATIVE_CACHE="$PWD/build/cache" \
  python -m needle2 build-native
```

后续推理使用相同的 `CXX` 与 `NEEDLE2_NATIVE_CACHE` 即可复用该缓存。
缓存键包含 C++ 源码、编译命令、固定参数与机器架构。更换编译器版本但命令名未变时，
请使用新的缓存目录重新构建。环境变量应在启动 Python 进程前设置。

## CMake 构建

```bash
cmake -S . -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native --parallel 4
```

产物为 `build/native/libneedle2_native.so`。首次配置时可通过
`-DCMAKE_CXX_COMPILER=g++` 选择编译器；更换工具链时使用新的构建目录。
调试构建可选择 `-DCMAKE_BUILD_TYPE=RelWithDebInfo`，测速使用 Release。

让 Python 加载该产物：

```bash
export NEEDLE2_NATIVE_LIBRARY="$PWD/build/native/libneedle2_native.so"
python -m needle2 build-native
python -m needle2 run artifacts/official/needle2.cact \
  --tools examples/tools.json --prompt 'Turn on the kitchen light.' \
  --backend native --threads 4
```

设置 `NEEDLE2_NATIVE_LIBRARY` 后直接加载指定文件，跳过自动编译。
路径无效、依赖缺失或库不兼容时会报错。使用与 Python 包相同源码版本构建的
OpenNeedle 库，并保持目标 CPU 架构及系统运行库兼容。
执行 `unset NEEDLE2_NATIVE_LIBRARY` 后，在新 Python 进程中恢复自动缓存模式。

可选安装到指定前缀：

```bash
cmake --install build/native --prefix "$PWD/artifacts/native-install"
```

库安装目录遵循 `GNUInstallDirs`，以安装输出中的实际路径为准。
将该库与同版本 Python 包部署到兼容环境，并设置 `NEEDLE2_NATIVE_LIBRARY`，
运行时即可省去编译器；仍需共享库依赖的 C++、OpenMP 等系统运行库。
Linux 下可用 `ldd build/native/libneedle2_native.so` 检查动态依赖。

## 构建组织与验证

编译入口为 `needle2/csrc/cq.cpp`，它直接包含 `sdot.cpp` 与 `engine.cpp`，
因此 CMake 只将 `cq.cpp` 作为编译单元。三者共同生成一个共享库。
Python 层负责 CACT 加载、tokenizer 与 grammar，产物是计算库。

自动构建使用 `-O3 -DNDEBUG -std=c++17 -fPIC -shared -pthread -fopenmp`。
CMake 通过 [C++ 编译特性](https://cmake.org/cmake/help/latest/command/target_compile_features.html)
声明 C++17，并通过 [OpenMP 导入目标](https://cmake.org/cmake/help/latest/module/FindOpenMP.html)
配置编译和链接。默认不启用 `-ffast-math` 或全局 DotProd 指令选项；
SDOT 保留独立函数目标标记与运行时能力检测。

验证 CMake 库的 packed 计算和整网数值路径：

```bash
python -m pip install -e '.[test]'
NEEDLE2_NATIVE_LIBRARY="$PWD/build/native/libneedle2_native.so" \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m pytest tests/test_native.py -q
```

编译器缺失时检查 `CXX`；OpenMP 检测失败时安装所选编译器对应的 OpenMP 开发库。
共享库加载失败时检查 `ldd`、CPU 架构及 Python/库源码版本。
数值模式与平台能力说明见[原生引擎文档](native-engine.md)。

## macOS 自动编译

Python 的自动构建路径使用当前所选编译器的 SDK，不另外注入 Command Line Tools 的 libc++ 头文件。使用 Apple Clang 时先安装 `brew install libomp`，供编译器查找 OpenMP 头文件。

如果已安装的 PyTorch 包含 `torch/lib/libomp.dylib`，自动构建和 profiling 会链接这份运行库并记录其 rpath，使 PyTorch 与 OpenNeedle 共用 OpenMP。生成库中的 OpenMP 依赖会改写到实际安装位置，再进行 ad-hoc 签名，以兼容 wheel 中遗留的绝对 install ID；不会修改 PyTorch 自身的库。没有这份文件时使用环境或 Homebrew 的 libomp。不要通过允许重复 OpenMP 运行库的环境开关掩盖冲突。移动 Python 环境后应重新构建原生库。

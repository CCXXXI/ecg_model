# ECG Model

[![GitHub license](https://img.shields.io/github/license/ccxxxi/ecg_model)](LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/ccxxxi/ecg_model)](https://github.com/CCXXXI/ecg_model/commits)
[![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-%23FE5196?logo=conventionalcommits&logoColor=white)](https://conventionalcommits.org)
[![C++](https://img.shields.io/badge/C++-00599C?logo=cplusplus)](https://isocpp.org)
[![LibTorch](https://img.shields.io/badge/LibTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![CodeFactor](https://www.codefactor.io/repository/github/ccxxxi/ecg_model/badge)](https://www.codefactor.io/repository/github/ccxxxi/ecg_model)
[![check](https://github.com/CCXXXI/ecg_model/actions/workflows/check.yml/badge.svg)](https://github.com/CCXXXI/ecg_model/actions/workflows/check.yml)

## 依赖

### LibTorch

C++ 的依赖管理工具似乎都装不了 LibTorch，只能手动下载了。

```shell
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.13.1%2Bcpu.zip && \
unzip libtorch-cxx11-abi-shared-with-deps-1.13.1+cpu.zip && \
rm libtorch-cxx11-abi-shared-with-deps-1.13.1+cpu.zip
```

也可以换成 [这里](https://pytorch.org/get-started/locally/) 的其他版本。

### 其他

由 [vcpkg](https://github.com/microsoft/vcpkg) 管理。

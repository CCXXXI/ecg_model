# 基于 LibTorch 的模型

## 依赖

### LibTorch

C++ 的依赖管理工具似乎都装不了 LibTorch，只能手动下载了。

```shell
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.13.1%2Bcpu.zip && \
unzip libtorch-cxx11-abi-shared-with-deps-1.13.1+cpu.zip && \
rm libtorch-cxx11-abi-shared-with-deps-1.13.1+cpu.zip
```

也可以换成 [这里](https://pytorch.org/get-started/locally/) 的其他版本。

# DocAlignerDemo(2025.02.08)
A demo to run python docAligner on Android

# 原项目地址
https://github.com/DocsaidLab/DocAligner

# 原项目文档
https://docsaid.org/en/docs/docaligner/benchmark

# 探索思路1: onnx转换为 paddle lite格式并运行
## 1. 问题 opset_version版本过高

```bash
 onnx2paddle support convert onnx model opset_verison [7, 8, 9, 10, 11, 12, 13, 14, 15], opset_verison of your onnx model is 16.
```

### 解决办法：版本降级

```python
import onnx
from onnx import version_converter


def downgrade_opset_version(model_path, target_opset_version):
    model = onnx.load(model_path)
    converted_model = version_converter.convert_version(model, target_opset_version)
    onnx.save(converted_model, 'downgraded_model.onnx')


# 假设你的模型文件是 'your_model.onnx'
model_path = 'heatmap_reg/ckpt/fastvit_t8_h_e_bifpn_256_fp32.onnx'
# model_path = 'heatmap_reg/ckpt/lcnet100_h_e_bifpn_256_fp32.onnx'
downgrade_opset_version(model_path, 14)

```

## 2. 问题 x2paddle 1.6.0 => Einsum 操作不支持

使用x2paddle develop分支，2025-01-15更新,需要依赖最新的x2paddle代码(非relase版)

## 3. 问题：Op(Conv) 操作输入不匹配 -- 待解决

```bash
 ValueError: (InvalidArgument) The number of input's channels should be equal to filter's channels * groups for Op(Conv). But received: the input's channels is 1, the input's shape is [1, 1, 1, 1]; the filter's channels is 1, the filter's shape is [64, 1, 3, 3]; the groups is 64, the data_format is NCHW. The error may come from wrong data_format setting.
  [Hint: Expected input_channels == filter_dims[1] * groups, but received input_channels:1 != filter_dims[1] * groups:64.] (at /Users/paddle/xly/workspace/6c14b372-ef6b-44b4-a87e-de189b4a4f60/Paddle/paddle/phi/infermeta/binary.cc:563)
  [operator < depthwise_conv2d > error]
```

# 探索思路2: 尝试Android依赖openCv库实现python对应功能,并依赖onnx-runtime运行模型 -- 本项目采用的方案

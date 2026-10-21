# GNSS Signal Classifier

本仓库保存城市环境 GNSS 信号传播类型识别项目的研究代码。项目使用 RINEX 观测、广播星历、参考轨迹和建筑物三维模型生成逐卫星特征与射线追踪标签，再使用 MLP、Random Forest、BiLSTM、ConvLSTM 和带空间遮罩的 SMConvLSTM 进行分类。

> 这是研究代码归档。部分早期脚本仍包含本机绝对路径、不同版本的类别定义及实验性实现。原始 UBX/RINEX、个人轨迹及大部分训练 CSV 未上传。

## 1. 研究目标

城市峡谷中的建筑物会遮挡或改变卫星信号传播路径，形成 LOS、反射、绕射、多路径或纯 NLOS 信号。本项目用于：

1. 从原始 GNSS 数据生成逐历元、逐卫星的观测特征；
2. 利用建筑 KML、天空遮罩和射线追踪生成信号类型标签；
3. 训练机器学习和时序神经网络分类器；
4. 将分类结果提供给后续 GNSS 加权定位及 GNSS/INS 融合模块。

## 2. 总体数据流

```text
UBX/RINEX + broadcast ephemeris + reference trajectory
                         |
                         v
        DataProcess/RINEX_processing_main_f.m
                         |
                         v
 per-satellite features: geometry, C/N0, residual, Doppler...
                         |
 building KML + sky mask + reflection/diffraction ray tracing
                         |
                         v
         DataTag/zzy_skymask_generation_f.m
                         |
                         v
             labelled GNSS measurements
                         |
       MLP / RF / BiLSTM / ConvLSTM / SMConvLSTM
                         |
                         v
            predicted signal type/quality
                         |
                         v
             GNSS weighting or GNSS/INS
```

## 3. 信号类别

早期标签代码使用以下五类定义：

| 数值 | 名称 | 含义 |
|---:|---|---|
| 0 | NLOS | 直达信号被遮挡，未确认有效反射或绕射路径 |
| 1 | LOS | 直视信号 |
| 2 | NLOS_Refl | 以反射路径到达的 NLOS 信号 |
| 3 | NLOS_Diff | 以绕射路径到达的 NLOS 信号 |
| 4 | LOS_Diff | 存在直达分量，同时包含绕射影响 |

后期部分 Python 模型使用四类输出，并在某些训练程序中用 `Uncertainty` 而不是 `Type` 作为标签。复现实验前必须核对目标 CSV 的类别映射。

## 4. 数据字段

`TagDatabase/TagDatabase.csv` 的列如下：

| 字段 | 说明 |
|---|---|
| `GPS_Time` | GPS 周内秒或实验使用的 GPS 时间 |
| `PRN` | 卫星编号 |
| `Elevation`, `Azimuth` | 卫星仰角和方位角 |
| `CNR` | 载噪比/信号强度特征 |
| `Pr_Residual` | 伪距残差 |
| `LS_Latitude`, `LS_Longitude`, `LS_Height` | 最小二乘定位结果 |
| `GT_Latitude`, `GT_Longitude`, `GT_Height` | 参考系统位置 |
| `Doppler` | 多普勒观测 |
| `SaVX`, `SaVY`, `SaVZ` | 卫星 ECEF 速度 |
| `lambda` | 信号波长 |
| `corrected_Pr` | 应用卫星钟差、电离层、对流层等改正后的伪距 |
| `SaX`, `SaY`, `SaZ` | 卫星 ECEF 位置 |
| `Max_PRN`, `Max_Residual` | 用于差分/参考的卫星及其残差 |
| `Visibility` | 基于天空遮罩的 `Visible`/`Blocked` 判断 |
| `ReflDist`, `DiffDist` | 射线追踪得到的反射、绕射附加路径长度 |
| `D_Coeff` | 绕射系数 |
| `Type` | 射线追踪生成的信号类型标签 |

不同阶段的数据还可能包含 `Site`、`CNR_rate`、`Diff_DelPr_PrRate`、`Uncertainty`、`Predicted_Type` 和 `Predicted_Label`。

## 5. 根目录文件

### `README.md`

当前项目交接文档，记录研究目标、数据格式、各文件用途、运行顺序及已知问题。

### `.gitignore`

排除原始 RINEX/UBX、压缩包、生成结果、缓存、MATLAB 临时文件和大部分 CSV，避免上传大型数据或个人轨迹。已经存在于历史提交中的文件不会因为加入 `.gitignore` 而自动取消跟踪。

### `predict_gnss_quality.m`

MATLAB 到 Python 分类器的实验接口。从一个 `GNSS_data` 历元提取仰角、方位角和 CNR，使用给定均值/标准差归一化，再计划调用 `py.predict_gnss.predict_single_epoch` 返回 `pred_Type`。

当前仓库尚缺少对应的 `predict_gnss.py`；`pred_Uncertainty` 也只是 `pred_Type` 的复制，并非真实置信度。因此它代表 GNSS/INS 集成方向，尚不是完整可运行接口。

### `Gaussian_mixture_1.py`

使用 LOS、反射和绕射状态及附加路径长度构造伪距误差高斯混合模型，并绘制各分量和总概率密度。它与分类标签有关，但更接近 GNSS Error Map/测量噪声建模，目前作为交叉项目实验文件保留。

## 6. `DataProcess/`：GNSS 数据预处理

### `DataProcess/RINEX_processing_main_f.m`

原始 GNSS 处理主程序：

- 读取 RINEX observation 与 broadcast navigation；
- 计算卫星位置、速度、钟差和传播改正；
- 生成仰角、方位角、C/N0、Doppler 和改正伪距；
- 执行最小二乘定位并与参考轨迹对齐；
- 计算伪距残差或差分伪距误差；
- 输出供标签生成和模型训练使用的逐卫星 CSV。

文件中的输入路径和实验坐标需要按本机数据修改。

### `DataProcess/goGPS/`

第三方 goGPS MATLAB 库。主要子模块：

- `input_output/`：RINEX、RTCM、UBX、NMEA、SP3 等格式读写；
- `positioning/`：卫星位置、误差改正、最小二乘和 Kalman 定位；
- `utility/`：坐标、时间、统计和 GNSS 工具函数；
- `obj/`：goGPS 对象类；
- `settings/`：示例配置；
- `plot/`：定位和卫星结果绘图。

### `DataProcess/gtsam_toolbox-master/`

第三方 GTSAM MATLAB toolbox，提供因子图、导航与优化接口。它主要作为历史开发环境依赖保存；当前分类训练脚本不直接使用它作为模型框架。

### `DataProcess/fsaxen-ParforProgMon-f99c6f0/`

第三方 MATLAB 并行循环进度显示工具：

- `ParforProgressbar.m`：面向 `parfor` 的进度条类；
- `progressbar.m`：进度条实现；
- `testParforProgressbar.m`：示例/测试；
- `Readme.md`、`license.txt`：上游说明和许可证；
- `progress.png`：上游文档图片。

## 7. `DataTag/`：建筑环境与信号标签

### `DataTag/zzy_skymask_generation_f.m`

标签生成主程序。读取逐卫星 CSV 和建筑 KML，生成天空遮罩并调用反射/绕射射线追踪，最后添加：

- `Visibility`；
- `ReflDist`；
- `DiffDist`；
- `D_Coeff`；
- `Type`。

输出通常为 `<experiment>_tag.csv`，另可输出被过滤的记录。

### `DataTag/Attachment/`

标签生成所需的历史附件和算法依赖，主要包括建筑 KML 读取、sky-mask 生成与合并、LOS/NLOS 判断、反射/绕射射线追踪、UTD 绕射系数、GNSS RUMS 示例及 RINEX/定位辅助函数。

其内部还包含 goGPS、GTSAM、satnav3p01 等第三方库副本。保留它们是为了还原原实验环境，后续应考虑去重并改为依赖说明。

## 8. 数据、基线与可视化

### `TagDatabase/TagDatabase.csv`

早期项目阶段汇总的带标签 GNSS 数据库，包含几何、C/N0、伪距残差、参考位置、射线追踪结果和 `Type` 标签。该文件可能含实验地点和参考轨迹；仓库已设为 Private。若以后公开，应先匿名化或删除。

### `MLP/MLP_1.py`

最早期多层感知机基线。主要使用 `CNR` 和 `Elevation` 对 `Type` 分类，包含标准化、类别平衡、训练、验证和决策边界可视化。当前使用占位式 CSV 路径，代表早期模型探索。

### `Visualized_Path/to kml.f.py`

轨迹可视化工具。读取 CSV 中的 ECEF/定位结果，将坐标转换为经纬高并生成 Google Earth KML 路径。建议以后改名为 `export_trajectory_to_kml.py`。

## 9. `models/`：分类模型

### Random Forest

#### `models/RF/RF.py`

固定时间窗口 RF 基线。按站点、卫星和连续时间构建窗口，将窗口特征展平并以窗口多数类别作为标签。使用 `Elevation`、`CNR`、`CNR_rate`、`Diff_DelPr_PrRate`，输出分类报告、混淆矩阵和特征重要性。

#### `models/RF/RF_nogroup.py`

统计特征 RF 版本。对卫星分组提取均值、标准差等统计量，用于比较“统计特征”和“展开时间窗口”方法。

### BiLSTM

#### `models/BiLSTM/biLSTM.2.py`

较完整的 Python BiLSTM 训练入口：构建重叠时序、标准化输入、使用双向 LSTM/普通 LSTM/Dropout、使用 Focal Loss、输出分类报告/混淆矩阵/训练曲线/permutation importance，并保存 Keras 模型。它是与项目总结中 BiLSTM 分类结果最相关的候选程序。当前路径写死，建议改名为 `train_bilstm.py`。

#### `models/BiLSTM/biLSTM.2 - 副本.py`

较早的 Python BiLSTM 副本，与正式版本存在少量差异。保留用于追溯开发过程；后续建议移入 `archive/` 或删除。

#### `models/BiLSTM/BiLSTM.m`

MATLAB BiLSTM 原型。构建两层 BiLSTM，使用类别权重训练五分类模型，并保存 `trained_gnss_model.mat`。当前依赖未随文件提供的辅助函数，且 Focal Loss 注释与实际配置不完全一致。

### Base ConvLSTM

#### `models/Base_convLSTM/base_conv_lstm.py`

基础 ConvLSTM/ConvRNN 实现，定义 `ConvLSTMCell` 和 `ConvLSTM`，负责卷积循环状态、shape、dropout 和 state 管理。它是底层层实现，不是直接训练入口。

#### `models/Base_convLSTM/base_conv_lstm - 副本.py`

与 `base_conv_lstm.py` 内容完全相同的历史副本。后续建议删除。

#### `models/Base_convLSTM/conv_lstm2d.py`

在基础 `ConvLSTM` 上提供二维封装 `ConvLSTM2D`，使其以类似 Keras layer 的方式接收二维空间序列。

### Time-masked ConvLSTM

#### `models/Time_masked_convLSTM/self-made masked_conv_LSTM.py`

早期自制 Masked ConvLSTM，实现卷积循环单元和 mask 传播，避免将缺失卫星/缺失时刻的零填充值视为真实观测。

#### `models/Time_masked_convLSTM/masked_convLSTM2D_module.py`

另一阶段的 `MaskedConvLSTM2D` 模块，定义 `ConvLSTMCell`、`ConvLSTM` 和 `MaskedConvLSTM2D`。与模型根目录同名文件并非完全相同。

#### `models/masked_convLSTM2D_module.py`

后期复制到模型根目录的 Masked ConvLSTM 实现，供脚本直接导入。后续需通过实验记录确定两版中的权威版本。

### SMConvLSTM

#### `models/Rho_module.py`

后期模型核心自定义层，定义 `SMConvLSTMCell` 和 `SMConvLSTM`，同时处理时间、卫星/空间维度和观测可用性 mask。

#### `models/SMconLSTM_training.py`

SMConvLSTM 训练入口。将 CSV 整理为“样本 × 时间 × PRN × 特征”张量，以 mask 标记卫星可用性，使用 `Elevation`、`Azimuth` 和 `CNR` 训练四分类模型，并保存 HDF5 checkpoint。当前脚本使用 `Uncertainty` 作为标签，且路径写死。

#### `models/load_model_SMconvLSTM.py`

SMConvLSTM 批量推理和评估程序。加载 `.h5` 与自定义层，对新 CSV 预测，生成 `Predicted_Type` 和 `Predicted_Label`；存在真实标签时输出分类报告与混淆矩阵。

#### `models/gnss_masked_convlstm_model.h5`

已训练的 SMConvLSTM 模型，约 169 KB。仍需补充训练数据版本、类别映射、TensorFlow 版本、验证性能和随机种子。

#### `models/load_model.py`

较早模型的通用加载与评估脚本。使用 `Elevation`、`CNR`、`CNR_rate`、`Diff_DelPr_PrRate` 构建时序样本，执行预测并绘制混淆矩阵。文件名未说明具体模型，后续建议改名。

## 10. 推荐阅读与运行顺序

1. `DataProcess/RINEX_processing_main_f.m`：理解特征来源；
2. `DataTag/zzy_skymask_generation_f.m`：理解标签来源；
3. `TagDatabase/TagDatabase.csv`：检查字段和类别分布；
4. `MLP/MLP_1.py` 或 `models/RF/`：建立基线；
5. `models/BiLSTM/biLSTM.2.py`：主要时序分类；
6. `models/SMconLSTM_training.py` + `models/Rho_module.py`：带卫星 mask 的模型；
7. `models/load_model_SMconvLSTM.py`：新数据推理；
8. `predict_gnss_quality.m`：连接 GNSS/INS 的后续工作。

## 11. 依赖

MATLAB 侧主要依赖 MATLAB、Mapping Toolbox、Statistics and Machine Learning Toolbox，以及仓库中的 goGPS、GTSAM、satnav3p01 和射线追踪代码。

Python 侧主要依赖：

```text
numpy
pandas
scipy
matplotlib
scikit-learn
tensorflow
keras
```

具体版本尚未锁定。后续应从原环境导出 `requirements.txt` 或 `environment.yml`。

## 12. 已知问题与整理状态

- 多个脚本包含 `C:/Users/ZHANG/...` 绝对路径；
- `Type` 与 `Uncertainty` 的含义尚未统一；
- 四分类与五分类代码同时存在；
- `predict_gnss_quality.m` 缺少 `predict_gnss.py`；
- `pred_Uncertainty` 尚不是真实概率或置信度；
- 存在 `- 副本.py` 和多份 Masked ConvLSTM；
- 第三方 MATLAB 库在多个目录重复；
- 尚无统一训练配置、随机种子、依赖版本和自动测试；
- `Gaussian_mixture_1.py` 更适合迁移至 GNSS Error Map 仓库。

这些问题不影响研究过程归档，但在正式公开或复现实验前应逐项处理。

## 13. 数据与隐私

仓库为 Private。原始观测和参考轨迹可能暴露实验时间及精确位置，因此默认不上传 `.ubx`、RINEX、CPT/INS 原始记录、大型实验 CSV 及生成的预测/定位输出。如需共享，建议另建受控数据存储，只在仓库保留匿名化小样例和字段说明。

## 14. 第三方代码

`goGPS`、GTSAM、ParforProgMon、satnav3p01 及其他附件来自各自上游项目或研究代码。使用、再分发或公开发布时，应遵循相应目录中的许可证和引用要求。本仓库收录这些文件不表示重新授权。

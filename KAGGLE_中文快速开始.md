# Kaggle 网站训练：中文快速开始

你现在最需要看的不是整个工程，而是这三个文件：

1. `notebooks/kaggle_hf_yolo_trocr_regions.py`
   - 这是目前最重要的高分方向代码。
   - 它适配你上传的提交格式：`image, regions`。
   - `regions` 是 JSON，里面每个框包含 `bbox`、`type`、`text`。
   - 思路是：YOLO 检测区域 + TrOCR 识别区域文字。

2. `notebooks/kaggle_trocr_baseline.py`
   - 这是旧的简单 OCR baseline。
   - 它更适合“整张图片输出一段文字”的比赛，不是这个比赛的最佳方向。

3. `KAGGLE_中文快速开始.md`
   - 就是你现在看的这份说明。

其他文件是工程化版本，先不用管。


## 重要更新：完整数据在 Hugging Face

比赛 Kaggle Input 里可能只有 `sample_submission.csv`。完整数据在 Hugging Face：

```python
from datasets import load_dataset
ds = load_dataset("UkrainianCatholicUniversity/rukopys")
```

所以你现在应该优先运行：

```text
notebooks/kaggle_hf_yolo_trocr_regions.py
```

这个文件会自动从 Hugging Face 下载 `train`、`silver`、`test`。


### 如果卡在 silver 下载

新版 `notebooks/kaggle_hf_yolo_trocr_regions.py` 已经改成只加载 `train` 和 `test`。

请确认代码里是这样的：

```python
train_ds = load_dataset(DATASET_ID, split="train")
test_ds = load_dataset(DATASET_ID, split="test")
```

不要用：

```python
ds = load_dataset(DATASET_ID)
```

因为那样可能会解析/下载很大的 `silver` split，导致 Kaggle 看起来卡死。

## 一、在 Kaggle 网站上怎么操作

### 1. 打开比赛页面

进入：

```text
https://www.kaggle.com/competitions/handwritten-to-data
```

点击：

```text
Code -> New Notebook
```

### 2. 打开 GPU

在 Kaggle Notebook 右侧：

```text
Settings -> Accelerator -> GPU T4 x2 或 GPU P100
```

如果只有 CPU，也能检查数据，但训练会很慢。

### 3. 添加比赛数据

通常从比赛页面创建 Notebook 时，数据会自动挂载到：

```text
/kaggle/input/handwritten-to-data/
```

如果没有，右侧点：

```text
Add Input -> Competition Data -> handwritten-to-data
```

### 4. 复制高分方向代码

优先把这个文件里的代码复制到 Kaggle Notebook：

```text
notebooks/kaggle_hf_yolo_trocr_regions.py
```

建议分成几个代码格：

1. 安装依赖
2. 导入库和设置参数
3. 检查数据
4. 解析 `regions`
5. 训练 YOLO 检测框和类型
6. 训练 TrOCR 识别每个框里的文字
7. 对 test 图片检测 + OCR
8. 生成 `submission.csv`

如果你不想分格，也可以整份复制进去直接跑。

## 二、代码写在哪里了？

目前主要文件是：

```text
README.md                          # 工程版完整说明
KAGGLE_中文快速开始.md              # 中文快速说明
requirements.txt                   # 本地依赖
config/baseline.yaml               # 工程版配置
src/htd/data.py                    # 数据列名识别、图片路径解析
src/htd/regions.py                 # regions JSON 解析和 bbox 工具
src/htd/metrics.py                 # CER/WER 指标
scripts/00_inspect_data.py         # 本地检查数据
scripts/05_prepare_yolo_regions.py # 本地把 regions 转成 YOLO 数据集
notebooks/kaggle_trocr_baseline.py # 简单整图 OCR 版
notebooks/kaggle_hf_yolo_trocr_regions.py # 高分方向：Hugging Face 数据 + YOLO + TrOCR 区域版
```

你现在优先使用：

```text
notebooks/kaggle_hf_yolo_trocr_regions.py
```

## 三、你上传的 sample_submission 说明了什么？

你上传的 `sample_submission.csv` 显示提交格式是：

```text
image,regions
```

其中 `regions` 是一个 JSON list，例如：

```json
[
  {
    "bbox": [1213, 350, 2052, 530],
    "type": "handwritten",
    "text": "Магія голосу."
  },
  {
    "bbox": [395, 600, 900, 750],
    "type": "formula",
    "text": "E = mc^2"
  }
]
```

所以这个比赛不是简单的整图 OCR，而是：

```text
找出区域 bbox -> 判断区域类型 type -> 识别文字 text
```

## 四、第一版高分方向是什么？

我给你的新版是：

```text
YOLOv8 检测 regions + microsoft/trocr-base-handwritten 识别文字
```

第一版目标是先跑通：

```text
图片 -> bbox/type -> 裁剪区域 -> text -> regions JSON -> submission.csv
```

跑通之后，我们再根据分数做增强。

## 五、跑完第一步后，把什么发给我？

请先在 Kaggle Notebook 里运行“检查数据”那一段，然后把输出发给我。

我需要看：

- `train.csv` 的列名和前几行
- `test.csv` 的列名和前几行
- `sample_submission.csv` 的列名和前几行
- 图片目录长什么样
- `train.csv` 里是否也有 `regions` 列
- `regions` 里面的 `type` 总共有几类，比如 `handwritten`、`formula`

如果代码报错，也直接把完整报错复制给我。

## 六、重要提醒

只给 `sample_submission.csv` 还不能真正训练模型，因为它只告诉我们提交格式，不包含训练标注。

你现在还需要给我：

1. `train.csv`
2. `test.csv`
3. 或者至少运行 Notebook 前面的“检查数据”部分，把输出复制给我

不用上传全部图片到 GitHub。图片很大，而且 Kaggle 数据通常不适合公开提交到仓库。

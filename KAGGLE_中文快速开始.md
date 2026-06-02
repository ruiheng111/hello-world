# Kaggle 网站训练：中文快速开始

你现在最需要看的不是整个工程，而是这两个文件：

1. `notebooks/kaggle_trocr_baseline.py`
   - 这是给 Kaggle Notebook 用的代码。
   - 打开后，把里面的代码复制到 Kaggle Notebook 的代码格里运行。

2. `KAGGLE_中文快速开始.md`
   - 就是你现在看的这份说明。

其他文件是工程化版本，先不用管。

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

### 4. 复制代码

把这个文件里的代码复制到 Kaggle Notebook：

```text
notebooks/kaggle_trocr_baseline.py
```

建议分成几个代码格：

1. 安装依赖
2. 导入库和设置参数
3. 检查数据
4. 建训练/验证集
5. 训练模型
6. 预测 test
7. 生成 submission.csv

如果你不想分格，也可以整份复制进去直接跑。

## 二、代码写在哪里了？

目前我已经写了这些文件：

```text
README.md                         # 工程版完整说明，比较详细
KAGGLE_中文快速开始.md             # 中文快速说明
requirements.txt                  # 本地安装依赖
config/baseline.yaml              # 工程版配置
src/htd/data.py                   # 数据列名识别、图片路径解析
src/htd/metrics.py                # CER/WER 指标
scripts/00_inspect_data.py        # 本地检查数据
scripts/01_make_folds.py          # 本地生成验证集
scripts/02_train_trocr.py         # 本地/工程版训练 TrOCR
scripts/03_predict_trocr.py       # 本地/工程版预测
scripts/04_make_submission.py     # 本地/工程版生成提交
notebooks/kaggle_trocr_baseline.py # Kaggle Notebook 直接复制版
```

你现在优先使用：

```text
notebooks/kaggle_trocr_baseline.py
```

## 三、跑完第一步后，把什么发给我？

请先在 Kaggle Notebook 里运行“检查数据”那一段，然后把输出发给我。

我需要看：

- `train.csv` 的列名
- `test.csv` 的列名
- `sample_submission.csv` 的列名
- 图片目录长什么样
- 目标列到底是 `text`、`label`、`transcription` 还是别的

如果代码报错，也直接把完整报错复制给我。

## 四、第一版 baseline 是什么？

我给你的第一版是：

```text
microsoft/trocr-base-handwritten
```

这是一个手写文字识别模型。第一版目标是先跑通：

```text
图片 -> 文字预测 -> submission.csv -> 提交 Kaggle
```

跑通之后，我们再根据分数做增强。

## 五、重要提醒

这个比赛可能不是简单 OCR。如果目标是结构化字段、JSON、表格或多字段抽取，那么第一版 TrOCR 只能作为起点。

所以你第一步一定要把数据检查输出发给我，我才能把方案改成真正适配比赛的版本。

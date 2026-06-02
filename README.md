# Handwritten to Data Kaggle Starter

这是一个面向 Kaggle `handwritten-to-data` 比赛的最小可运行 starter。当前仓库没有比赛数据，因此代码采用“尽量自动识别列名 + 配置可覆盖”的方式。


## 当前提交格式：image + regions JSON

用户上传的 `sample_submission.csv` 显示提交列是：

```text
image,regions
```

`regions` 是 JSON list，每个元素包含：

```json
{"bbox": [x1, y1, x2, y2], "type": "handwritten", "text": "..."}
```

因此这个比赛不是简单的整图 OCR。优先使用：

```text
notebooks/kaggle_yolo_trocr_regions.py
```

这个 Kaggle Notebook 版代码采用两阶段方案：

1. YOLOv8 检测 region 的 `bbox` 和 `type`
2. TrOCR 对每个裁剪 region 识别 `text`

旧的 `notebooks/kaggle_trocr_baseline.py` 只适合整图 OCR baseline，不能作为当前格式的主要冲分方案。

目标不是一开始就写复杂方案，而是先建立一条可靠闭环：

1. 下载并检查数据格式
2. 建立本地验证集
3. 跑一个 OCR/HTR baseline
4. 生成 Kaggle submission
5. 根据错误分析继续增强

## 0. 你需要先准备什么

### Kaggle API

在 Kaggle 账号页面创建 API token，得到 `kaggle.json`。然后本地执行：

```bash
mkdir -p ~/.kaggle
cp /path/to/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

下载比赛数据：

```bash
kaggle competitions download -c handwritten-to-data -p data/raw
unzip data/raw/handwritten-to-data.zip -d data/raw
```

如果 Kaggle 下载得到多个 zip，也全部解压到 `data/raw/`。

### Python 环境

建议 Python 3.10+，GPU 环境优先。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## 1. 检查数据

先运行：

```bash
python scripts/00_inspect_data.py --data-dir data/raw
```

把输出发给我，尤其是：

- `train.csv` / `test.csv` / `sample_submission.csv` 是否存在
- 每个 CSV 的列名
- 图片目录结构
- `sample_submission.csv` 的列名和前几行

这些信息决定后续训练目标列、图片列和提交格式。

## 2. 配置路径和列名

默认配置在：

```text
config/baseline.yaml
```

如果自动识别失败，需要手动填写：

```yaml
columns:
  id: id
  image: image
  target: text
submission:
  id: id
  prediction: text
```

常见列名自动识别规则已经写在 `src/htd/data.py`。

## 3. 建立本地验证集

```bash
python scripts/01_make_folds.py --config config/baseline.yaml
```

会生成：

```text
data/processed/train_folds.csv
```

本地验证非常重要。不要只看 public leaderboard，否则容易过拟合 public LB。

## 4. 跑第一版 TrOCR baseline

先用较小训练量确认 pipeline 正常：

```bash
python scripts/02_train_trocr.py \
  --config config/baseline.yaml \
  --epochs 1 \
  --max-train-samples 128 \
  --max-val-samples 64
```

确认能跑通后，再逐步加大：

```bash
python scripts/02_train_trocr.py \
  --config config/baseline.yaml \
  --epochs 5 \
  --batch-size 4 \
  --grad-accum 4
```

默认模型是 `microsoft/trocr-base-handwritten`。如果 GPU 显存较小，可以在配置里改成更小模型或降低 batch size。

## 5. 生成预测和提交

```bash
python scripts/03_predict_trocr.py --config config/baseline.yaml
python scripts/04_make_submission.py --config config/baseline.yaml
```

输出文件：

```text
submissions/submission.csv
```

提交到 Kaggle：

```bash
kaggle competitions submit -c handwritten-to-data -f submissions/submission.csv -m "trocr baseline"
```

## 6. 第一轮之后怎么冲分

拿到第一版分数后，请告诉我：

1. 你的 public LB 分数
2. 当前第 10 名分数
3. 评价指标名字和方向，越大越好还是越小越好
4. `00_inspect_data.py` 的输出
5. 训练日志中的本地验证分数
6. 几个预测错误样例：图片、真实文本、预测文本

然后可以进入迭代：

- 更合理的图像预处理：裁剪、去噪、旋转矫正、对比度增强
- 更强模型：TrOCR/LLaVA-style OCR/Donut/PaddleOCR/CRNN-CTC 方案对比
- 按字段类型做后处理：日期、数字、姓名、地址等
- pseudo-labeling：用高置信度 test 预测继续训练
- ensemble：不同输入尺寸、不同 seed、不同模型融合
- public/private split 风险控制

## 我还缺少的信息

目前我缺少以下比赛关键信息：

- 训练 CSV 的真实列名
- 图片路径和文件命名方式
- 目标是整行文本、字段抽取，还是结构化 JSON/表格
- 评价指标
- 提交文件格式
- 你可用的 GPU/显存
- 当前榜单第 10 名分数和你的分数

你先完成 `第 1 步`，把输出贴给我，我就能继续把代码改成完全匹配这个比赛的版本。

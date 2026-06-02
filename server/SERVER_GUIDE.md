# 在实验室 4090 服务器上跑 Handwritten to Data

> **安全提醒**：不要把账号密码发到公开群/论坛。若已泄露，请联系助教重置密码。

## 服务器信息

| 项目 | 值 |
|------|-----|
| IP | `10.77.110.130` |
| 用户 | `ml-2-3` |
| 数据盘 | `/data1/ml-2-3` |
| 网络 | **无外网**，文件用 SFTP 上传 |
| GPU | 每组同时只能用 **1 张卡** |

---

## 一、整体流程

```
本地电脑（有网）                    SFTP 上传                 服务器（无网）
─────────────────                ──────────                ───────────────
1. 下载 RUKOPYS 数据集      →    data/rukopys/      →    训练读这里
2. 下载 pip wheels          →    wheels/            →    pip 离线安装
3. 下载 YOLO + TrOCR 权重   →    pretrained/        →    训练加载
4. 上传 code/               →    code/handwritten-to-data/
```

---

## 二、SSH 登录

**Windows（PowerShell / CMD）：**
```bash
ssh ml-2-3@10.77.110.130
```

**Mac / Linux：**
```bash
ssh ml-2-3@10.77.110.130
```

首次连接输入密码。登录后先确认 GPU：

```bash
nvidia-smi
```

应能看到 **4090**。记下 **CUDA 版本**（右上角，如 12.1），本地下载 PyTorch 时要匹配。

---

## 三、在服务器上建目录（登录后执行一次）

```bash
export ML_ROOT=/data1/ml-2-3
mkdir -p $ML_ROOT/{data,models,output,cache/{huggingface,torch,tmp,pip,conda},wheels,pretrained,code}
mkdir -p $ML_ROOT/code/handwritten-to-data
```

**以后所有东西都放在 `$ML_ROOT` 下，不要往 `~/` 里放大文件。**

---

## 四、本地准备并 SFTP 上传

### 4.1 下载数据集（本地有网）

```bash
pip install -U huggingface_hub
huggingface-cli download UkrainianCatholicUniversity/rukopys \
  --repo-type dataset \
  --local-dir ./rukopys
```

用 **FileZilla / WinSCP / MobaXterm** 把整个 `rukopys/` 上传到：

`/data1/ml-2-3/data/rukopys/`

目录里应有 `train/images/`、`train/metadata.jsonl`、`test/images/` 等。

### 4.2 下载 Python 包与预训练模型（本地有网）

在包含 `server/` 脚本的目录：

```bash
cd server
chmod +x download_on_local_machine.sh
./download_on_local_machine.sh ./offline_bundle
```

上传：

| 本地 | 服务器路径 |
|------|------------|
| `offline_bundle/wheels/*` | `/data1/ml-2-3/wheels/` |
| `offline_bundle/pretrained/*` | `/data1/ml-2-3/pretrained/` |
| `offline_bundle/code/*` | `/data1/ml-2-3/code/handwritten-to-data/` |

### 4.3 PyTorch（重要）

`wheels` 里的 torch 必须和服务器 **CUDA 版本一致**。在服务器执行 `nvidia-smi` 后，在本地：

```bash
# 示例：CUDA 12.1
pip download torch torchvision --index-url https://download.pytorch.org/whl/cu121 -d ./offline_bundle/wheels
```

再 SFTP 覆盖上传到 `/data1/ml-2-3/wheels/`。

---

## 五、服务器上安装环境（离线）

```bash
export ML_ROOT=/data1/ml-2-3
# 用 Miniconda 装到数据盘（若管理员已装好 conda，可跳过）
# wget 也需在本地下载安装包后 SFTP 上传

conda create -y --prefix $ML_ROOT/cache/conda/envs/htr python=3.10
source $ML_ROOT/cache/conda/envs/htr/bin/activate

pip install --no-index --find-links=$ML_ROOT/wheels -r $ML_ROOT/code/handwritten-to-data/requirements.txt
```

加载环境变量（每次训练前执行）：

```bash
source /data1/ml-2-3/code/handwritten-to-data/env.sh
```

`env.sh` 会把 HF/Torch 缓存指到数据盘，并设置 `CUDA_VISIBLE_DEVICES=0`。

---

## 六、开始训练

```bash
source /data1/ml-2-3/code/handwritten-to-data/env.sh
source /data1/ml-2-3/cache/conda/envs/htr/bin/activate
cd /data1/ml-2-3/code/handwritten-to-data

# 快速试跑（约几十分钟）
python train_scheme_a.py --quick

# 正式训练（数小时，占满一张 4090）
python train_scheme_a.py

# 只重新推理（已有权重时）
python train_scheme_a.py --infer-only
```

输出：

- 权重：`/data1/ml-2-3/output/scheme_a/det_best.pt`、`trocr_uk/`
- 提交文件：`/data1/ml-2-3/output/scheme_a/submission.csv`

用 SFTP 把 `submission.csv` **下载回本地**，再到 Kaggle 竞赛页提交。

---

## 七、长时间训练不断线（推荐）

SSH 断开会导致任务被杀，用 `screen` 或 `tmux`：

```bash
screen -S htr
source /data1/ml-2-3/code/handwritten-to-data/env.sh
source /data1/ml-2-3/cache/conda/envs/htr/bin/activate
python /data1/ml-2-3/code/handwritten-to-data/train_scheme_a.py
# 按 Ctrl+A 再按 D 脱离；重新连接后 screen -r htr
```

---

## 八、常见问题

| 现象 | 处理 |
|------|------|
| `找不到数据` | 确认 `/data1/ml-2-3/data/rukopys/train/metadata.jsonl` 存在 |
| `CUDA out of memory` | `train_scheme_a.py` 里把 `batch_rec` 改为 8 或 4 |
| `无法下载模型` | 必须设 `HF_HUB_OFFLINE=1` 且已上传 `pretrained/trocr-base-handwritten/` |
| `pip 装不上` | 检查 wheels 是否含对应平台的包（Linux x86_64） |
| 磁盘满 | `du -sh /data1/ml-2-3/*`，删掉多余的 `yolo_train` 重复实验目录 |

---

## 九、与助教的约定

- 每组 **单次只用一张卡** → 已用 `CUDA_VISIBLE_DEVICES=0`
- 数据、模型、缓存、conda 放 **数据盘** → 见 `env.sh`
- 无法访问外网 → 全部 SFTP + 离线 pip
- 有问题联系：**助教吴昊宸**

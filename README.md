# TT100K / CCTSDB SSD

基于 `torch` / `torchvision` 的 SSD300-VGG16 交通标志检测项目，支持 YOLO txt 标注格式的 TT100K 与 CCTSDB。

## 环境

```bash
pip install torch torchvision pyyaml pillow
```

如果机器不能联网下载 torchvision 的 VGG backbone 预训练权重，可以把配置里的 `model.pretrained_backbone` 改成 `false`。

## 数据格式

YOLO label 每行：

```text
class_id x_center y_center width height
```

坐标为归一化坐标，代码会自动转换为 torchvision detection 模型使用的像素 `xyxy`，并把类别 id 从 `0..N-1` 转成 `1..N`，`0` 保留给背景类。

支持以下布局：

```text
TT100K-2016/
  train.txt
  test.txt
  train/images/*.jpg
  train/labels/*.txt
  test/images/*.jpg
  test/labels/*.txt
```

```text
CCTSDB2021/
  images/train/*.jpg
  images/test/*.jpg
  labels/train/*.txt
  labels/test/*.txt
```

也支持 `train/images + train/labels` 这种目录形式。

## 训练

TT100K：

```bash
CUDA_VISIBLE_DEVICES=0 python train_ssd.py \
  --config configs/tt100k_ssd.yaml \
  --data-root /home/sutongtong/LanTu_team1/TT100K-2016 \
  --output-dir outputs/tt100k_ssd \
  --epochs 80 \
  --batch-size 16 \
  --lr 0.005 \
  --eval-map-every 10 \
  --quick-eval-samples 100
```

CCTSDB：

```bash
CUDA_VISIBLE_DEVICES=0 python train_ssd.py \
  --config configs/cctsdb_ssd.yaml \
  --data-root /home/sutongtong/LanTu_team1/advYOLO+AdaAD+CCTSDB/CCTSDB2021 \
  --output-dir outputs/cctsdb_ssd \
  --epochs 80 \
  --batch-size 16 \
  --lr 0.005 \
  --eval-map-every 10 \
  --quick-eval-samples 100
```

训练会保存：

- `last.pth`：每轮覆盖保存完整训练状态
- `best_loss.pth`：训练 loss 最低
- `best_map50.pth`：完整验证集 mAP50 最高
- `quick_eval.csv`：每轮快速 mAP50 / Recall
- `train_metrics.csv`：每轮 loss、lr、quick/full mAP50、Recall、跳过 batch 数

前台运行时如果安装了 `tqdm` 会显示进度条；无 `tqdm` 时仍会按 `--log-interval` 打印 batch 进度。默认每 10 个 batch 输出一次，可自行调整：

```bash
CUDA_VISIBLE_DEVICES=0 python train_ssd.py \
  --config configs/tt100k_ssd.yaml \
  --data-root /home/sutongtong/LanTu_team1/TT100K-2016 \
  --output-dir outputs/tt100k_ssd \
  --epochs 80 \
  --batch-size 16 \
  --lr 0.005 \
  --quick-eval-samples 100 \
  --eval-map-every 10 \
  --log-interval 10
```

## 继续训练与微调

恢复完整训练状态，包括 model / optimizer / scheduler / epoch：

```bash
CUDA_VISIBLE_DEVICES=0 python train_ssd.py \
  --config configs/tt100k_ssd.yaml \
  --data-root /home/sutongtong/LanTu_team1/TT100K-2016 \
  --output-dir outputs/tt100k_ssd \
  --resume outputs/tt100k_ssd/last.pth
```

只加载模型权重，重新创建 optimizer / scheduler，用于换学习率微调：

```bash
CUDA_VISIBLE_DEVICES=0 python train_ssd.py \
  --config configs/cctsdb_ssd.yaml \
  --data-root /home/sutongtong/LanTu_team1/advYOLO+AdaAD+CCTSDB/CCTSDB2021 \
  --output-dir outputs/cctsdb_ssd_finetune \
  --finetune-from outputs/cctsdb_ssd/best_map50.pth \
  --lr 0.001
```

## 后台训练

TT100K：

```bash
bash scripts/train_tt100k_ssd_bg.sh
```

CCTSDB：

```bash
bash scripts/train_cctsdb_ssd_bg.sh
```

可用环境变量覆盖默认值：

```bash
GPU_ID=1 DATA_ROOT=/path/to/CCTSDB2021 EXP_NAME=cctsdb_ssd_v2 bash scripts/train_cctsdb_ssd_bg.sh
```

脚本会写入 `outputs/<EXP_NAME>/train.pid` 和 `logs/<EXP_NAME>_时间.log`。

## 评估

TT100K clean + adv：

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_ssd.py \
  --config configs/tt100k_ssd.yaml \
  --checkpoint outputs/tt100k_ssd/best_map50.pth \
  --data-root /home/sutongtong/LanTu_team1/TT100K-2016 \
  --adv-root /home/sutongtong/LanTu_team1/yolov9_adv_TT100K/images \
  --source-model YOLOv9 \
  --target-detector SSD \
  --batch-size 16 \
  --score-threshold 0.3 \
  --max-samples 1000 \
  --output-csv outputs/tt100k_ssd_adv_metrics.csv
```

CCTSDB clean + adv：

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_ssd.py \
  --config configs/cctsdb_ssd.yaml \
  --checkpoint outputs/cctsdb_ssd/best_map50.pth \
  --data-root /home/sutongtong/LanTu_team1/advYOLO+AdaAD+CCTSDB/CCTSDB2021 \
  --adv-root /home/sutongtong/LanTu_team1/yolov9_adv_CCTSDB/adv_images \
  --adv-suffix _adv \
  --source-model YOLOv9 \
  --target-detector SSD \
  --batch-size 16 \
  --score-threshold 0.3 \
  --output-csv outputs/cctsdb_ssd_adv_metrics.csv
```

`ASR = max(0, clean_recall - adv_recall) / clean_recall`。

## 单图预测

```bash
CUDA_VISIBLE_DEVICES=0 python predict_ssd.py \
  --checkpoint outputs/cctsdb_ssd/best_map50.pth \
  --config configs/cctsdb_ssd.yaml \
  --image /home/sutongtong/LanTu_team1/yolov9_adv_CCTSDB/adv_images/18993_adv.jpg \
  --output /home/sutongtong/LanTu_team1/yolov9_adv_CCTSDB/ssd_vis_18993_adv.jpg \
  --score-threshold 0.3
```

脚本会打印 `bbox、label id、类别名、score`，并保存画框后的图片。

## Git 注意

`.gitignore` 已排除 `outputs/`、`logs/`、`*.pth`、`*.pt`、`datasets/`、`data/`，不要把数据集和权重提交到仓库。

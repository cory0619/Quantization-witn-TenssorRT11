'''
FP32 학습/평가 - dataset_pcb multi-label 분할 (pcb / terminal / hole, 한 픽셀이 여러 클래스에 동시에 속함)   (2026-10-17, ConvNeXt 백본)

- 모델 : model.unet_convnext.UNet_ConvNeXt(n_channels=IN_CHANNELS, n_classes=N_CLASSES, multilabel=True, encoder_name=ENCODER_NAME)
         = ImageNet-22k 사전학습 ConvNeXt-Base 백본 (참조 코드 train_test_forgeai.py 의 "tu-convnext_base" 와 같은 timm 모델, encoder 로만 사용)
           + R5 식 경량 디코더 (nearest x2 + 덧셈 skip + 3x3 conv, full-res 가지 포함). 입력 H, W 는 32 의 배수.
         R5 (37k 파라미터, 처음부터 학습) 결과: pcb 0.945 / terminal 0.877 / hole 0.596 IoU 에서 정체 -> 사전학습 백본으로 보완
         사전학습 가중치: PRETRAINED_PATH (참조 코드의 weights/model.safetensors). ImageNet mean/std 정규화는 모델 안에서 함
         입력 (B,3,H,W) float 0~1 (컬러 BGR, OpenCV 순서), 출력 logits (B,N_CLASSES,H,W). 채널 c = 클래스 c 의 독립 이진 logit (배경 채널 없음)
         학습률: 백본 LR_ENCODER (사전학습이라 작게), 디코더 LR_DECODER. GPU 메모리: 1088x1344 batch 2 에서 약 9.9 GB (AMP),
         batch 4 는 --grad_ckpt (백본 gradient checkpointing) 로 약 11 GB. 체크포인트 .pt 는 약 355 MB
- 라벨 : images_dataset.dataset_experiment(classes=CLASSES, multilabel=True) 가 labelme json (polygon / circle / rectangle) 을
         (N_CLASSES,H,W) float {0,1} 로 만듦. 겹치는 영역은 해당 채널이 모두 1 (예: pcb 위의 hole -> pcb=1, hole=1)
- 손실 : BCEWithLogits(채널별 독립 sigmoid) + 클래스별 soft Dice 평균. 이유는 combined_loss 참고.
         softmax + CrossEntropy 는 픽셀마다 클래스 하나를 강제하므로 겹치는 라벨에는 쓸 수 없음
- AMP  : CUDA 면 항상 fp16 autocast + GradScaler (가중치는 fp32, 저장되는 .pt 도 fp32). 손실 계산은 fp32.
         export / TensorRT 변환과는 무관 (정밀도는 ModelOpt / 엔진 빌드 단계에서 결정)
- 예측 : 채널별 sigmoid > THRESH. 지표: 클래스별 IoU / Precision / Recall / F1 (pixel-level, per-image 평균), 클래스 평균 mIoU
- 배포 : 모델 deploy=True 면 채널별 마스크를 비트 패킹한 uint8 (B,1,H,W). model.bitmask.unpack_bits 로 복원

사용:
    python train_test_fp32_pcb.py train --num_epochs 500 --batch_size 2
    python train_test_fp32_pcb.py test  --model_path output\\fp32_model_<ts>\\origin_model_best.pt
    옵션: --dataset_dir dataset_pcb  --img_size 832 1024  (H W, 32 의 배수)  --class_weights 1 1 4  (pcb terminal hole 손실 비중)
          --pretrained_path <safetensors>  --no_pretrained (백본 무작위 초기화)  --grad_ckpt (메모리 절약, batch 4 용)
'''

import os
import cudnn_fix  # torch 보다 먼저: PATH 의 시스템 cuDNN 9.27 과 torch 번들 cuDNN 충돌 회피 (cudnn_fix.py 참고)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import warnings
warnings.simplefilter("ignore", category=Warning)
import argparse
import time

import albumentations as A
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from images_dataset import dataset_experiment
from model.unet_convnext import UNet_ConvNeXt, load_pretrained_encoder, ensure_pretrained, PRETRAINED_PATH_DEFAULT

# ======================================================
# 데이터셋 / 클래스 설정
# ======================================================
DATASET_DIR = "dataset_pcb"
IN_CHANNELS = 3                             # 입력 채널. PCB 이미지는 컬러(BGR). 패터닝 데이터셋은 1 (grayscale)
CLASSES = ["pcb", "terminal", "hole"]       # 출력 채널 순서 = 이 순서. 배경 채널 없음 (모든 채널 0 = 배경)
N_CLASSES = len(CLASSES)                    # 모델 출력 채널 = 3
THRESH = 0.5                                # 채널별 sigmoid > THRESH 를 해당 클래스로 판정
IMG_SIZE = (1088, 1344)                     # (H, W). 원본 1104x1358 에 가장 가까운 32 의 배수 (ConvNeXt 1/32 특징 때문, 각 1.4% 축소). hole 반지름 약 24px
FILL_VALUE = 0                              # Affine/CoarseDropout 채움값. PCB 이미지는 어두움(가장자리 평균 약 25). 패터닝 데이터셋은 255 였음
PALETTE = np.array([[255, 128, 0],          # pcb      : 파랑 (BGR)
                    [0, 200, 0],            # terminal : 초록
                    [0, 0, 255]], np.uint8) # hole     : 빨강
ALPHAS = [0.45, 0.45, 0.9]                  # 클래스별 overlay 불투명도. hole 은 pcb(파랑) 위에 겹치므로 거의 불투명하게 + 윤곽선
CONTOUR_THICKNESS = 2                       # 클래스별 윤곽선 두께 (0 이면 안 그림). 겹치는 영역의 경계를 보이게 함
CLASS_WEIGHTS = [1.0, 1.0, 1.0]             # 손실의 클래스별 가중치 (CLASSES 순서). [1,1,4] 면 hole 손실이 다른 클래스의 4배 비중.
                                            # 합으로 정규화해 쓰므로 비율만 의미 있음 ([1,1,1] 과 [2,2,2] 는 같음). --class_weights 로 변경

# ======================================================
# 모델 / 학습 설정 (ConvNeXt 백본)
# ======================================================
ENCODER_NAME = "convnext_base"              # timm 이름. 참조 코드의 "tu-convnext_base" 와 같은 백본. 가벼운 대안: convnext_tiny (사전학습 파일 별도)
PRETRAINED_PATH = PRETRAINED_PATH_DEFAULT   # timm/convnext_base.fb_in22k_ft_in1k safetensors (참조 코드 weights/ 폴더). --pretrained_path 로 변경
DROP_PATH_RATE = 0.1                        # 백본 stochastic depth (학습 시만). 학습 22장이라 과적합 억제용
LR_ENCODER = 1e-4                           # 사전학습 백본 학습률 (크면 사전학습 특징이 망가짐)
LR_DECODER = 1e-3                           # 디코더 (무작위 초기화) 학습률
SIZE_MULTIPLE = UNet_ConvNeXt.SIZE_MULTIPLE # 입력 H, W 조건 (32)


def build_model(device, pretrained_path=None, grad_ckpt=False):
    """모델 생성 (+ 사전학습 백본 로드). pretrained_path=None 이면 백본 무작위 초기화 (test / resume 시 체크포인트가 덮어씀)"""
    model = UNet_ConvNeXt(n_channels=IN_CHANNELS, n_classes=N_CLASSES, multilabel=True,
                          encoder_name=ENCODER_NAME, drop_path_rate=DROP_PATH_RATE)
    if pretrained_path:
        load_pretrained_encoder(model, pretrained_path, bgr_input=(IN_CHANNELS == 3))
    if grad_ckpt:
        model.encoder.set_grad_checkpointing(True)   # 백본 활성화를 backward 때 다시 계산: 메모리 약 절반, 시간 약 1.25배
    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_all = sum(p.numel() for p in model.parameters())
    print(f"[model] UNet_ConvNeXt({ENCODER_NAME}) params {n_all:,} (encoder {n_enc:,} + decoder {n_all - n_enc:,}) "
          f"| pretrained {'none' if not pretrained_path else pretrained_path} | grad_ckpt {grad_ckpt}")
    return model.to(device)


def main(args):
    img_size = tuple(args.img_size)
    plot_dir = r"loss_plot"
    output_dir = os.path.join("output", f"fp32_model_{time.strftime('%Y%m%d_%H%M%S')}")
    inference_dir = r"inference"
    result_dir = r"result"
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(inference_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    train_losses, test_losses = [], []
    best_loss, best_epoch = np.inf, -1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # AMP (mixed precision): CUDA 에서는 forward 를 fp16 autocast 로, 가중치/optimizer 는 fp32 유지. GradScaler 로 fp16 기울기 underflow 방지.
    # 손실의 BCEWithLogits / sum 은 autocast 가 fp32 로 승격하고, Dice 는 명시적으로 fp32 로 계산 (combined_loss 참고).
    amp_enabled = device.type == "cuda"
    torch.backends.cudnn.benchmark = True   # 입력 크기가 고정이라 conv 알고리즘 자동 선택이 이득
    set_class_weights(args.class_weights, device)
    print(f"[config] multi-label classes {CLASSES} | class weights {args.class_weights} (normalized {[round(float(w), 3) for w in _class_weights]}) "
          f"| thresh {THRESH} | input ({IN_CHANNELS},{img_size[0]},{img_size[1]}) | dataset {args.dataset_dir} "
          f"| device {device} | AMP {'fp16' if amp_enabled else 'off'} | encoder {ENCODER_NAME} lr {LR_ENCODER} / decoder lr {LR_DECODER}")

    # ======================================================
    # train
    # ======================================================
    if args.command == 'train':
        os.makedirs(output_dir, exist_ok=True)
        model_path_best = os.path.join(output_dir, "origin_model_best.pt")
        print(f"[train] best model will be saved to: {model_path_best}")

        # albumentations 2.x API. (H,W,K) 마스크도 nearest 보간 + fill_mask=0(배경) 으로 같이 변환됨
        augmentation = A.Compose([
            A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR),
            A.Affine(translate_percent=(-0.03, 0.03), scale=(0.9, 1.1),
                     border_mode=cv2.BORDER_CONSTANT, fill=FILL_VALUE, fill_mask=0),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            A.GaussNoise(std_range=(0.0, 0.05), p=0.5),
            A.CoarseDropout(num_holes_range=(8, 8), hole_height_range=(16, 16), hole_width_range=(16, 16),
                            fill=FILL_VALUE, p=0.2),   # 마스크는 그대로 (가려진 곳도 정답 유지)
        ], p=1.0)

        train_dataset = dataset_experiment(os.path.join(args.dataset_dir, "train"), augmentation,
                                           classes=CLASSES, size=img_size, multilabel=True, channels=IN_CHANNELS)
        test_dataset = dataset_experiment(os.path.join(args.dataset_dir, "test"), None,
                                          classes=CLASSES, size=img_size, multilabel=True, channels=IN_CHANNELS)
        print(f"[data] train {len(train_dataset)} / test {len(test_dataset)} images")
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        if args.resume:
            if not (args.model_path and os.path.exists(args.model_path)):
                raise FileNotFoundError(f"--resume requires --model_path to an existing .pt (got: {args.model_path})")
            model = build_model(device, pretrained_path=None, grad_ckpt=args.grad_ckpt)   # 체크포인트가 백본까지 담고 있음
            print(f"[resume] Loading: {args.model_path}")
            model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True), strict=True)
        else:
            pretrained = None if args.no_pretrained else ensure_pretrained(args.pretrained_path, auto_download=args.download_weights)
            if pretrained is None and not args.no_pretrained:
                raise SystemExit(f"[pretrained] 백본 가중치를 준비하지 못했습니다: {args.pretrained_path}\n"
                                 f"  --pretrained_path 로 timm/convnext_base.fb_in22k_ft_in1k model.safetensors 를 지정하거나, "
                                 f"--download_weights True 로 내려받거나, --no_pretrained 로 무작위 초기화 학습")
            model = build_model(device, pretrained_path=pretrained, grad_ckpt=args.grad_ckpt)

        # 사전학습 백본은 작은 학습률, 무작위 초기화된 디코더는 큰 학습률
        optimizer = torch.optim.Adam([{"params": model.encoder_parameters(), "lr": LR_ENCODER},
                                      {"params": model.decoder_parameters(), "lr": LR_DECODER}])
        scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

        for epoch in range(args.num_epochs):
            print(f"\nEpoch: {epoch}/{args.num_epochs - 1}")

            # 1) Train (AMP: autocast forward + GradScaler backward/step)
            model.train()
            train_loss = 0.0
            for images, labels, _ in tqdm(train_loader, desc="Training"):
                images, labels = images.to(device), labels.to(device)       # labels (B,N_CLASSES,H,W) float {0,1}
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                    outputs = model(images)                                  # (B,N_CLASSES,H,W) logits (fp16)
                    loss = combined_loss(outputs, labels)                    # 손실은 fp32
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item()
            train_loss /= max(len(train_loader), 1)
            mem = f" | GPU peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB" if device.type == "cuda" else ""
            print(f"\tTrain loss: {train_loss}{mem}")

            # 2) Test
            model.eval()
            test_loss = 0.0
            with torch.no_grad(), torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                for images, labels, image_paths in tqdm(test_loader, desc="Testing"):
                    images, labels = images.to(device), labels.to(device)
                    # print('image shape:', images.shape)
                    outputs = model(images)
                    test_loss += combined_loss(outputs, labels).item()
            test_loss /= max(len(test_loader), 1)
            print(f"\tTest loss: {test_loss}")

            # 3) 시각화 (test_loader 마지막 이미지): 왼쪽 정답, 오른쪽 예측
            try:
                save_epoch_visual(image_paths[0], labels[0], outputs[0], os.path.join(result_dir, f"epoch_{epoch}.png"))
            except Exception as e:
                print(f"[warn] visualization failed: {e}")

            # 4) best 저장
            if test_loss < best_loss:
                best_loss, best_epoch = test_loss, epoch
                torch.save(model.state_dict(), model_path_best)
                print(f"[save] New best model saved with loss: {best_loss:.4f}")

            train_losses.append(train_loss)
            test_losses.append(test_loss)
            plot_loss_curve(train_losses, test_losses, plot_dir)

        # 5) 학습 종료 summary: best 모델을 다시 로드해 test set 평가
        if os.path.exists(model_path_best):
            model.load_state_dict(torch.load(model_path_best, map_location=device, weights_only=True))
            metrics = compute_segmentation_metrics(model, test_loader, device, amp_enabled=amp_enabled)
            header = f"[Summary] best model: {model_path_best} (epoch {best_epoch}, best test loss {best_loss:.4f})"
            print_metrics_summary(metrics, header, save_path=os.path.join(output_dir, "summary.txt"))
        else:
            print("[Summary] no best model was saved; skipping evaluation.")

    # ======================================================
    # test
    # ======================================================
    if args.command == 'test':
        model_path = args.model_path
        if not (model_path and os.path.exists(model_path)):
            raise FileNotFoundError(f"test requires --model_path to an existing .pt (got: {model_path})")
        print(f"[test] loading model: {model_path}")
        model = build_model(device, pretrained_path=None)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True), strict=True)
        model.eval()

        test_dataset = dataset_experiment(os.path.join(args.dataset_dir, "test"), None,
                                          classes=CLASSES, size=img_size, multilabel=True, channels=IN_CHANNELS)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)   # 추론 batch 4 (fp16): 1088x1344 에서 약 2.6 GB

        with torch.no_grad(), torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            t_total = time.perf_counter()
            for images, labels, paths in test_loader:
                print('image shape:', images.shape)
                images = images.to(device)
                t0 = time.perf_counter()
                preds = (torch.sigmoid(model(images).float()) > THRESH).cpu().numpy().astype(np.uint8)   # (B,K,H,W)
                print(f"inference time ({len(paths)} imgs): {time.perf_counter() - t0:.4f}s")
                labels = labels.numpy().astype(np.uint8)

                for pred, gt, path in zip(preds, labels, paths):
                    gray = read_for_visual(path)
                    h, w = gray.shape[:2]
                    pred_full = resize_masks(pred, (h, w))
                    gt_full = resize_masks(gt, (h, w))
                    name = os.path.splitext(os.path.basename(path))[0] + ".png"

                    # 예측 overlay
                    cv2.imwrite(os.path.join(inference_dir, name), overlay_masks(gray, pred_full))
                    # 정답 overlay 위에, 어느 한 클래스라도 예측이 다른 픽셀을 흰색으로
                    diff = overlay_masks(gray, gt_full)
                    diff[(pred_full != gt_full).any(axis=0)] = (255, 255, 255)
                    cv2.imwrite(os.path.join(inference_dir, "diff_" + name), diff)
            print(f"total time: {time.perf_counter() - t_total:.3f}s")

        metrics = compute_segmentation_metrics(model, test_loader, device, amp_enabled=amp_enabled)
        print_metrics_summary(metrics, f"[Summary] test model: {model_path}")


# ======================================================
# 시각화
# ======================================================
def resize_masks(masks, hw):
    """(K,h,w) -> (K,H,W) nearest"""
    return np.stack([cv2.resize(m, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST) for m in masks])


def read_for_visual(path):
    """시각화용 원본: IN_CHANNELS 에 맞춰 컬러(BGR) 또는 grayscale"""
    return cv2.imread(path, cv2.IMREAD_COLOR if IN_CHANNELS == 3 else cv2.IMREAD_GRAYSCALE)


def overlay_masks(image, masks, alphas=ALPHAS, contour=CONTOUR_THICKNESS):
    """원본(BGR 또는 grayscale) 위에 클래스별 마스크 (K,H,W) 를 PALETTE 색으로 차례로 덧칠하고 윤곽선을 그림.
    multi-label 이라 영역이 겹치므로, 뒤 클래스(hole)는 불투명도를 높여 앞 클래스(pcb) 색에 묻히지 않게 함"""
    h, w = image.shape[:2]
    if masks.shape[1:] != (h, w):
        masks = resize_masks(masks.astype(np.uint8), (h, w))
    img = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for c in range(masks.shape[0]):
        fg = masks[c] > 0
        if fg.any():
            a = alphas[c] if c < len(alphas) else alphas[-1]
            color = np.empty_like(img); color[:] = PALETTE[c]
            img[fg] = cv2.addWeighted(img, 1 - a, color, a, 0)[fg]
    if contour > 0:
        for c in range(masks.shape[0]):
            cnts, _ = cv2.findContours((masks[c] > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cnts, -1, tuple(int(v) for v in PALETTE[c]), contour)
    return img


def save_epoch_visual(image_path, label, logits, save_path):
    """label (K,H,W) float, logits (K,H,W) -> [정답 | 예측] 나란히 저장 (학습 입력 크기 기준)"""
    gt = (label > 0.5).cpu().numpy().astype(np.uint8)
    pred = (torch.sigmoid(logits.float()) > THRESH).cpu().numpy().astype(np.uint8)
    img = cv2.resize(read_for_visual(image_path), (gt.shape[2], gt.shape[1]), interpolation=cv2.INTER_LINEAR)
    left, right = overlay_masks(img, gt), overlay_masks(img, pred)
    cv2.putText(left, "GT", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(right, "pred", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    for i, name in enumerate(CLASSES):   # 범례
        y = 46 + 22 * i
        cv2.rectangle(right, (8, y - 14), (24, y + 2), tuple(int(c) for c in PALETTE[i]), -1)
        cv2.putText(right, name, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.imwrite(save_path, np.hstack([left, right]))


# ======================================================
# 지표 (multi-label: 클래스마다 독립 이진 지표)
# ======================================================
def _ratio(num, den, both_empty):
    """den == 0 이면: 예측도 정답도 비어 있으면 1.0(완벽), 한쪽만 비어 있으면 0.0"""
    if den == 0:
        return 1.0 if both_empty else 0.0
    return num / den


def _binary_metrics(tp, fp, fn):
    e = (tp + fp + fn) == 0
    return {"iou": _ratio(tp, tp + fp + fn, e), "precision": _ratio(tp, tp + fp, e),
            "recall": _ratio(tp, tp + fn, e), "f1": _ratio(2 * tp, 2 * tp + fp + fn, e), "tp": tp, "fp": fp, "fn": fn}


def compute_segmentation_metrics(model, loader, device, class_names=CLASSES, thresh=THRESH, amp_enabled=False):
    """
    정답: loader 가 주는 labels (K,H,W) {0,1}, 예측: sigmoid(model(images)) > thresh (채널별)
    반환: 클래스별 pixel-level(전체 픽셀 TP/FP/FN 합산) 과 per-image 평균 지표, 클래스 평균 mIoU
    """
    model.eval()
    k = len(class_names)
    tot = np.zeros((k, 3), dtype=np.int64)          # tp, fp, fn
    per_img = {c: {"iou": [], "precision": [], "recall": [], "f1": []} for c in range(k)}
    n = 0
    with torch.no_grad(), torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
        for images, labels, _ in loader:
            preds = torch.sigmoid(model(images.to(device)).float()) > thresh
            gts = labels.to(device) > 0.5
            for p, g in zip(preds, gts):                                   # (K,H,W)
                tp = (p & g).flatten(1).sum(1); fp = (p & ~g).flatten(1).sum(1); fn = (~p & g).flatten(1).sum(1)
                for c in range(k):
                    m = _binary_metrics(int(tp[c]), int(fp[c]), int(fn[c]))
                    tot[c] += (m["tp"], m["fp"], m["fn"])
                    for key in per_img[c]:
                        per_img[c][key].append(m[key])
                n += 1
    px = [_binary_metrics(*map(int, tot[c])) for c in range(k)]
    return {
        "n_images": n, "thresh": thresh, "class_names": list(class_names),
        "pixel_level": {class_names[c]: px[c] for c in range(k)},
        "per_image_mean": {class_names[c]: {key: float(np.mean(v)) if v else float("nan") for key, v in per_img[c].items()} for c in range(k)},
        "miou": float(np.mean([px[c]["iou"] for c in range(k)])),
        "miou_per_image": float(np.mean([np.mean(per_img[c]["iou"]) for c in range(k)])) if n else float("nan"),
    }


def print_metrics_summary(metrics, header, save_path=None):
    names = metrics["class_names"]
    lines = [
        "=" * 112,
        header,
        f"GT: labelme json multi-label masks | pred: sigmoid > {metrics['thresh']} per class | classes: {names} | test images: {metrics['n_images']}",
        f"{'class':12s}{'IoU':>8s}{'Precision':>11s}{'Recall':>8s}{'F1':>8s}   |{'IoU':>8s}{'Precision':>11s}{'Recall':>8s}{'F1':>8s}   (TP / FP / FN)",
        f"{'':12s}{'--------- pixel-level ---------':^35s}   |{'------ per-image mean ------':^35s}",
    ]
    for c in names:
        px, pm = metrics["pixel_level"][c], metrics["per_image_mean"][c]
        lines.append(f"{c:12s}{px['iou']:8.4f}{px['precision']:11.4f}{px['recall']:8.4f}{px['f1']:8.4f}   |"
                     f"{pm['iou']:8.4f}{pm['precision']:11.4f}{pm['recall']:8.4f}{pm['f1']:8.4f}   "
                     f"({px['tp']} / {px['fp']} / {px['fn']})")
    lines.append(f"mIoU (class mean, pixel-level) {metrics['miou']:.4f} | mIoU (class mean, per-image) {metrics['miou_per_image']:.4f}")
    lines.append("=" * 112)
    text = "\n".join(lines)
    print("\n" + text)
    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"[Summary] saved to {save_path}")


# ======================================================
# 손실
# ======================================================
class MultiLabelDiceLoss(torch.nn.Module):
    """채널(클래스)마다 sigmoid 확률 vs 정답의 soft Dice 를 batch 전체 픽셀로 계산한 뒤 클래스 가중 평균"""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets, weights=None):
        probs = torch.sigmoid(logits.float())                  # (B,K,H,W). AMP 에서도 fp32 로 계산 (큰 픽셀 합의 정밀도)
        targets = targets.float()
        dims = (0, 2, 3)
        inter = (probs * targets).sum(dims)
        dice = (2.0 * inter + self.smooth) / (probs.sum(dims) + targets.sum(dims) + self.smooth)   # (K,)
        return 1.0 - _weighted_class_mean(dice, weights)


_dice_loss = MultiLabelDiceLoss()
_class_weights = None   # (K,) tensor, 합이 K 가 되도록 정규화된 클래스 가중치. set_class_weights() 로 설정


def set_class_weights(weights, device):
    """CLASSES 순서의 가중치 리스트 -> 합이 K 인 tensor. [1,1,1] -> [1,1,1], [1,1,4] -> [0.5,0.5,2.0] (hole 이 다른 클래스의 4배)"""
    global _class_weights
    w = torch.as_tensor(list(weights), dtype=torch.float32)
    if w.numel() != N_CLASSES or (w <= 0).any():
        raise ValueError(f"class_weights must be {N_CLASSES} positive numbers for {CLASSES}, got {list(weights)}")
    _class_weights = (w * N_CLASSES / w.sum()).to(device)


def _weighted_class_mean(per_class, weights):
    """(K,) 클래스별 값의 가중 평균. weights 가 None 이거나 모두 같으면 단순 평균과 동일"""
    if weights is None:
        return per_class.mean()
    return (per_class * weights.to(per_class.device)).sum() / weights.sum()


def combined_loss(outputs, labels):
    """
    outputs: logits (B,K,H,W),  labels: (B,K,H,W) float {0,1}  (채널마다 독립, 겹침 허용)
    클래스 가중치(_class_weights): BCE 는 클래스별 평균을 낸 뒤, Dice 는 클래스별 값을 가중 평균. 가중치는 합으로 정규화되어
    [1,1,4] 면 전체 손실에서 hole 이 4/6, pcb 와 terminal 이 1/6 씩을 차지 (기본 [1,1,1] 은 1/3 씩 = 가중치 없는 것과 동일).

    BCE + Dice 를 쓰는 이유:
    - 한 픽셀이 여러 클래스에 속하므로 클래스마다 독립적인 이진 문제로 봐야 함 -> 채널별 sigmoid + BCE.
      softmax + CrossEntropy 는 확률 합이 1 이라 픽셀마다 클래스 하나를 강제해서 pcb 위의 hole 같은 겹침을 표현할 수 없음.
    - BCE 는 모든 픽셀에 고른 기울기를 주지만 픽셀 수에 비례해 큰 클래스(pcb 27%)가 작은 클래스(hole 0.5%)를 지배함.
      Dice 는 클래스별 영역 겹침을 직접 최적화해서 작은 클래스가 묻히지 않게 함 -> 둘을 더해 보완.
    - MSE 는 분류용이 아님: 확률이 포화되면 기울기가 사라지고 양/음 픽셀 불균형에 더 취약함.
    """
    bce_per_class = F.binary_cross_entropy_with_logits(outputs.float(), labels, reduction="none").mean(dim=(0, 2, 3))  # (K,), fp32
    bce = _weighted_class_mean(bce_per_class, _class_weights)
    return bce + _dice_loss(outputs, labels, _class_weights)


# ======================================================
# 기타
# ======================================================
def plot_loss_curve(train_losses, test_losses, plot_dir):
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.yscale("log")
    plt.grid()
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(plot_dir, "pcb_loss_epoch.png"))
    plt.close()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def argparse_args():
    parser = argparse.ArgumentParser(description="dataset_pcb multi-label 분할 학습/평가 (ConvNeXt 백본 UNet)")
    parser.add_argument('command', help="'train' or 'test'")
    parser.add_argument('--num_epochs', default=500, type=int)
    parser.add_argument('--batch_size', default=2, type=int,
                        help=f"기본 2 (1088x1344 AMP 약 9.9 GB). 4 는 --grad_ckpt 와 함께 (약 11 GB)")
    parser.add_argument('--resume', default=False, type=str2bool)
    parser.add_argument('--model_path', default=None, type=str,
                        help="test 시 로드할 .pt 경로 / --resume 시 이어서 학습할 .pt 경로")
    parser.add_argument('--dataset_dir', default=DATASET_DIR, type=str, help="train/, test/ 폴더가 있는 경로")
    parser.add_argument('--img_size', default=list(IMG_SIZE), type=int, nargs=2, metavar=("H", "W"),
                        help=f"모델 입력 크기 ({SIZE_MULTIPLE} 의 배수). 기본 {IMG_SIZE}")
    parser.add_argument('--class_weights', default=list(CLASS_WEIGHTS), type=float, nargs=N_CLASSES, metavar="W",
                        help=f"손실의 클래스별 가중치, {CLASSES} 순서. 비율만 의미 있음. 기본 {CLASS_WEIGHTS}. 예: --class_weights 1 1 4")
    parser.add_argument('--pretrained_path', default=PRETRAINED_PATH, type=str,
                        help=f"(train) 백본 사전학습 safetensors. 기본 {PRETRAINED_PATH}")
    parser.add_argument('--no_pretrained', action='store_true', help="(train) 백본을 사전학습 없이 무작위 초기화")
    parser.add_argument('--download_weights', default=False, type=str2bool,
                        help="(train) 사전학습 파일이 없으면 HuggingFace 에서 weights/ 로 내려받음 (약 338 MB)")
    parser.add_argument('--grad_ckpt', action='store_true',
                        help="(train) 백본 gradient checkpointing: 메모리 약 절반 (batch 4 에서 18.5 -> 11 GB), 시간 약 1.25배")
    args = parser.parse_args()
    if any(s % SIZE_MULTIPLE for s in args.img_size):
        parser.error(f"--img_size must be multiples of {SIZE_MULTIPLE} (got {args.img_size})")
    if any(w <= 0 for w in args.class_weights):
        parser.error(f"--class_weights must be positive (got {args.class_weights})")
    return args


if __name__ == '__main__':
    main(argparse_args())

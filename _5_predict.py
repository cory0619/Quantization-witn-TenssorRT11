r"""
_5_predict.py : dense 모델 (PyTorch FP32 체크포인트) 과 TensorRT 엔진 (INT8 / FP32) 으로 폴더의 이미지를 추론
                -> 모델별 폴더에 예측 반투명 오버레이 + 이미지별 IoU / Precision / Recall / F1 (라벨 json 이 있을 때만)
                -> 둘 다 돌리면 dense vs 엔진 비교 (클래스별 지표 차이, 픽셀 일치율, 속도)                           (R6 UNet_ConvNeXt)

- --model dense : CKPT_PATH 의 state_dict 를 UNet_ConvNeXt(deploy=True) 에 올려 PyTorch 로 추론 (기본 TF32 off = 순수 FP32, _1 의 torch deploy 와 같은 값)
  --model trt   : _4 가 만든 엔진 (--engine, 기본 INT8 엔진)
  --model both  : (기본) 둘 다 돌리고 비교
  두 경로 모두 같은 deploy 인터페이스 (uint8 BGR (B,3,H,W) 입력 -> uint8 비트 패킹 (B,1,H,W) 출력) 라서
  같은 이미지·같은 GT·같은 지표 코드로 계산한 값을 그대로 비교할 수 있다.
- 입력: 폴더의 이미지 (png / jpg / jpeg / bmp, 대소문자 무관). 라벨은 같은 이름의 labelme .json.
  json 이 하나도 없는 폴더면 지표 계산은 건너뛰고 추론 + 오버레이만 저장한다 (두 모델의 픽셀 일치율은 라벨 없이도 계산).
  일부만 있으면 있는 이미지만 지표를 계산한다.
- 전처리 / 지표는 학습 스크립트·_4 와 동일: IMG_SIZE 로 INTER_LINEAR resize 한 uint8 BGR 을 모델에 넣고, /255·ImageNet 정규화·
  sigmoid > THRESH·비트 패킹은 모델/엔진 안. 지표는 모델 해상도(IMG_SIZE) 마스크로 계산 (GT 는 images_dataset 과 같은 방식으로 json 에서 그려 nearest resize).
  IoU/Precision/Recall/F1 = 이미지 1장의 클래스별 TP/FP/FN, mean = 클래스 평균.
- 오버레이는 원본 해상도: 예측 마스크를 nearest 로 원본 크기에 맞춘 뒤 eval_common.overlay_masks (학습 스크립트와 같은 PALETTE 반투명 채움 + 윤곽선) 로 그림.
  --draw-gt 면 GT 윤곽선(흰색, 얇게) 도 같이 그린다.
- GPU 메모리: 모델별로 로드 전 / 로드 후 / 추론 후의 프로세스 전용 GPU 메모리를 Windows 성능 카운터로 읽는다 (_4 와 같은 측정, 작업관리자 값).
  torch 의 max_memory_allocated 는 torch 할당기 카운터라 TensorRT/cupy 메모리를 못 보므로 쓰지 않는다 (dense 는 참고로만 적음).
  both 면 두 모델이 한 프로세스에서 차례로 돌아 CUDA 컨텍스트가 먼저 돈 모델 쪽에 잡히므로, 비교는 '이 모델 몫' (로드 전 대비 증가분) 으로 한다.
- 출력 (--out, 기본 MODEL_DIR/predict_<폴더이름>/):
    dense_fp32/                 dense 모델 결과   ┐ 각 폴더 안:
    trt_<엔진이름>/              엔진 결과         ┘   overlay/<이름>.png  예측 반투명 오버레이 (pcb 파랑 / terminal 초록 / hole 빨강)
                                                      mask/<이름>.png     모델 출력 그대로 (uint8 비트 패킹: bit0 pcb, bit1 terminal, bit2 hole -> eval_common.unpack_masks)
                                                      metrics.csv         이미지 × 클래스 행 (pred_px 는 항상, iou/precision/recall/f1/tp/fp/fn 은 라벨이 있을 때만)
                                                      summary.txt         전체 요약 (라벨이 있으면 pixel-level 합산 + 이미지별 평균)
    compare.csv / compare.txt   (both 일 때) 이미지·클래스별 dense / 엔진 IoU·F1 과 차이, 픽셀 일치율, 클래스별 합산 지표, 속도

파이프라인:  _1 -> _2 -> _3 -> _4 (엔진)  ->  _5_predict.py (체크포인트 / 엔진 사용)

사용:
    python _5_predict.py                                         # dense + INT8 엔진, dataset_pcb/test, 비교표
    python _5_predict.py --model trt --dir D:\images\unlabeled    # 엔진만, 라벨 없는 폴더: 추론 + 오버레이만
    python _5_predict.py --model dense --batch 2                  # dense 만 (GPU 메모리가 모자라면 batch 를 줄임)
    python _5_predict.py --engine output\...\r6_fp32.engine --draw-gt --ext jpg
"""
import cudnn_fix  # noqa: F401  (images_dataset / dense 경로가 torch 를 불러오므로 먼저. cudnn_fix.py 참고)
import argparse
import csv
import os
import time

import cv2
import numpy as np

from eval_common import (MODEL_DIR, CKPT_PATH, ENGINE_INT8, TEST_DIR, IMG_SIZE, CLASSES, N_CLASSES, IN_CHANNELS, THRESH,
                         load_images, unpack_masks, metrics, agreement, iou_summary, print_accuracy_table, overlay_masks, GpuMonitor)
from images_dataset import dataset_experiment

GT_CONTOUR_COLOR, GT_CONTOUR_THICKNESS = (255, 255, 255), 1
IMREAD_FLAG = cv2.IMREAD_COLOR if IN_CHANNELS == 3 else cv2.IMREAD_GRAYSCALE
LINE = "-" * 118


# ======================================================
# GT 마스크 (labelme json) : images_dataset.get_mask 와 같은 그리기 (circle / rectangle / polygon)
# ======================================================
def _label_reader():
    """dataset_experiment 는 폴더의 모든 json 이 있어야 생성되므로, 마스크 그리기 부분만 쓰기 위해 인스턴스를 직접 구성"""
    ds = dataset_experiment.__new__(dataset_experiment)
    ds.classes, ds.multilabel = list(CLASSES), True
    ds.class_to_idx = {c: i for i, c in enumerate(CLASSES)}
    ds.strict_labels, ds._warned = False, set()
    return ds


def load_gt_small(label_path, image_hw, reader, size=IMG_SIZE):
    """json -> 모델 해상도 bool (K,h,w). json 의 imageHeight/Width 가 이미지와 다르면 None (지표 생략)"""
    m = reader.get_mask(label_path)                                   # (H,W,K) uint8, json 의 imageHeight/Width
    if m.shape[:2] != tuple(image_hw):
        print(f"[warn] json imageHeight/Width {m.shape[:2]} != image {tuple(image_hw)} -> 지표 생략: {label_path}")
        return None
    small = cv2.resize(m, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)   # Dataset 의 A.Resize 와 같이 nearest
    return small.transpose(2, 0, 1) > 0


def draw_gt_contours(img, label_path, reader):
    """원본 해상도 GT 윤곽선 (흰색) 을 오버레이 위에 그림"""
    gt_full = reader.get_mask(label_path).transpose(2, 0, 1) > 0
    for c in range(gt_full.shape[0]):
        cnts, _ = cv2.findContours(gt_full[c].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, GT_CONTOUR_COLOR, GT_CONTOUR_THICKNESS)
    return img


# ======================================================
# 모델 실행기 : dense (PyTorch) / TensorRT 엔진.  둘 다 uint8 (N,3,h,w) -> uint8 (N,1,h,w) 비트 패킹
# ======================================================
def _encoder_name():
    """학습 스크립트의 ENCODER_NAME (체크포인트와 같은 백본). import 실패 시 convnext_base"""
    try:
        import train_test_fp32_pcb as T
        return T.ENCODER_NAME
    except Exception as e:
        print(f"[warn] train_test_fp32_pcb import 실패 ({type(e).__name__}: {e}) -> encoder convnext_base 로 가정")
        return "convnext_base"


class DenseRunner:
    """PyTorch FP32 체크포인트를 deploy 모드로 실행 (_1 의 export 모델과 동일 구성)"""

    def __init__(self, ckpt_path, tf32=False):
        import torch
        from model.unet_convnext import UNet_ConvNeXt
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.allow_tf32 = tf32                        # 기본 off: 순수 FP32 (엔진과 비교하는 기준값)
        torch.backends.cuda.matmul.allow_tf32 = tf32
        encoder = _encoder_name()
        self.model = UNet_ConvNeXt(n_channels=IN_CHANNELS, n_classes=N_CLASSES, multilabel=True,
                                   deploy=True, deploy_mask=True, deploy_thresh=THRESH, encoder_name=encoder, drop_path_rate=0.0)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.eval().to(self.device)
        n_param = sum(p.numel() for p in self.model.parameters())
        self.desc = (f"{os.path.basename(ckpt_path)} UNet_ConvNeXt({encoder}) params {n_param:,}, "
                     f"torch {torch.__version__} {self.device}, TF32 {'on' if tf32 else 'off'}")

    def run_all(self, images, batch):
        outs = []
        with self.torch.no_grad():
            for i in range(0, len(images), batch):
                x = self.torch.from_numpy(images[i:i + batch]).to(self.device)
                outs.append(self.model(x).cpu().numpy())              # .cpu() 가 GPU 동기화
        return np.concatenate(outs)

    def fit_batch(self, images, batch):
        """첫 실행 (warm-up, 시간 측정 제외). batch 가 GPU 메모리를 넘으면 1 로 줄여 반환"""
        try:
            self.run_all(images[:batch], batch)
            return batch
        except self.torch.cuda.OutOfMemoryError:
            self.torch.cuda.empty_cache()
            print(f"[warn] dense batch {batch} 는 GPU 메모리 부족 -> batch 1 로 실행")
            self.run_all(images[:1], 1)
            return 1

    def peak_mib(self):
        return self.torch.cuda.max_memory_allocated() / 2 ** 20 if self.device.type == "cuda" else 0.0

    def close(self):
        del self.model
        if self.device.type == "cuda":
            self.torch.cuda.empty_cache()


def _mib(v):
    return f"{v:.0f} MiB" if v is not None else "n/a"


def _share(mem):
    """이 모델 몫 = 추론 후 최대 - 이 모델 로드 전 (프로세스 전용 GPU 메모리 증가분)"""
    return f"+{mem['peak'] - mem['before']:.0f} MiB" if mem["peak"] is not None and mem["before"] is not None else "n/a"


def cuda_base_mib():
    """CUDA 컨텍스트를 먼저 만들고 (torch 1원소 텐서) 그 시점의 프로세스 GPU 메모리를 잰다 (_4 의 base_mib 와 같은 기준).
    컨텍스트가 없으면 성능 카운터에 이 프로세스 항목이 없어 '로드 전' 이 n/a 가 되고, 첫 모델 몫에 컨텍스트가 섞인다"""
    import torch
    if torch.cuda.is_available():
        torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
    return GpuMonitor.process_mib()


def mem_line(mem):
    if mem["peak"] is None:
        return f"gpu mem  : 측정 불가 (Windows GPU 성능 카운터 없음). {mem['note']}"
    return (f"gpu mem  : 프로세스 전용 GPU 메모리 로드 후 {_mib(mem['load'])} / 추론 후 최대 {_mib(mem['peak'])} "
            f"(로드 전 {_mib(mem['before'])} -> 이 모델 몫 {_share(mem)}). {mem['note']}")


def run_dense(images, args):
    print(f"\n[dense] {args.ckpt} 로드 중 ...")
    before = GpuMonitor.process_mib()                                 # 이 모델 로드 전 (_4 와 같은 프로세스 전용 GPU 메모리 측정)
    r = DenseRunner(args.ckpt, tf32=args.tf32)
    load = GpuMonitor.process_mib()
    batch = r.fit_batch(images, args.batch)
    t0 = time.perf_counter()
    packed = r.run_all(images, batch)                                 # uint8 (N,1,h,w)
    ms = (time.perf_counter() - t0) * 1e3 / len(images)
    mem = {"before": before, "load": load, "peak": GpuMonitor.process_mib(), "note": f"torch allocator peak {r.peak_mib():.0f} MiB (참고)"}
    r.close()
    print(f"[dense] {r.desc}: batch {batch}, {ms:.2f} ms/img (복사 포함, {len(images)} 장)")
    print("[dense] " + mem_line(mem))
    return {"key": "dense", "name": "dense_fp32", "desc": r.desc, "packed": packed, "pred": unpack_masks(packed), "ms": ms, "batch": batch,
            "mem": mem, "model_line": f"model    : {args.ckpt}  (PyTorch deploy 모드, TF32 {'on' if args.tf32 else 'off'})"}


def run_trt(images, args):
    from model.trt_runner import TrtRunner
    before = GpuMonitor.process_mib()
    r = TrtRunner(args.engine)
    load = GpuMonitor.process_mib()
    batch = min(args.batch, r.batch_max)
    if batch != args.batch:
        print(f"[warn] --batch {args.batch} > 엔진 profile max {r.batch_max} -> {batch} 로 실행")
    r.infer(images[:batch])                                           # 버퍼 할당 + 첫 실행 (시간 측정 제외)
    t0 = time.perf_counter()
    packed = r.run_all(images, batch)                                 # uint8 (N,1,h,w)
    ms = (time.perf_counter() - t0) * 1e3 / len(images)
    mem = {"before": before, "load": load, "peak": GpuMonitor.process_mib(), "note": f"엔진 activation {r.activation_mib():.0f} MiB (max batch)"}
    r.close()
    stem = os.path.splitext(os.path.basename(args.engine))[0]
    desc = f"{os.path.basename(args.engine)} TensorRT profile {r.profile}"
    print(f"\n[trt] {desc}: batch {batch}, {ms:.2f} ms/img (복사 포함, {len(images)} 장)")
    print("[trt] " + mem_line(mem))
    return {"key": "trt", "name": f"trt_{stem}", "desc": desc, "packed": packed, "pred": unpack_masks(packed), "ms": ms, "batch": batch,
            "mem": mem, "model_line": f"engine   : {args.engine}  (profile {r.profile})"}


# ======================================================
# 모델 하나의 결과 저장 : overlay/ mask/ metrics.csv summary.txt
# ======================================================
def per_class_row(m, c):
    px = m["px"][c]
    return [f"{px['iou']:.4f}", f"{px['precision']:.4f}", f"{px['recall']:.4f}", f"{px['f1']:.4f}", px["tp"], px["fp"], px["fn"]]


def write_outputs(res, out_dir, paths, labels, gts_small, reader, args):
    """반환: (라벨 이미지 합산 metrics 또는 None, 이미지별 metrics 리스트 (라벨 없는 이미지는 None))"""
    pred, packed = res["pred"], res["packed"]
    os.makedirs(os.path.join(out_dir, "overlay"), exist_ok=True)
    if not args.no_mask:
        os.makedirs(os.path.join(out_dir, "mask"), exist_ok=True)
    rows, per_img, labeled_idx = [], [None] * len(paths), []
    print(f"\n[{res['name']}] {res['desc']}")
    print(LINE)
    print(f"{'image':22s}" + "".join(f"{c:>9s} IoU  P     R     F1   " for c in CLASSES) + " mean IoU")
    print(LINE)
    for i, p in enumerate(paths):
        name = os.path.splitext(os.path.basename(p))[0]
        orig = cv2.imread(p, IMREAD_FLAG)
        if orig is None:
            print(f"[warn] 원본을 다시 읽지 못함, 오버레이 생략: {p}")
            continue
        m = None
        if gts_small[i] is not None:
            m = metrics(pred[i:i + 1], gts_small[i][None])            # 이 이미지 1장의 클래스별 TP/FP/FN
            per_img[i] = m
            labeled_idx.append(i)

        # overlay (원본 해상도) + 마스크 저장
        img = overlay_masks(orig, pred[i].astype(np.uint8))
        if args.draw_gt and gts_small[i] is not None:
            img = draw_gt_contours(img, labels[i], reader)
        cv2.imwrite(os.path.join(out_dir, "overlay", f"{name}.{args.ext}"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 95] if args.ext == "jpg" else [])
        if not args.no_mask:
            cv2.imwrite(os.path.join(out_dir, "mask", f"{name}.png"), packed[i, 0])

        # csv 행 + 콘솔
        line = f"{name[:21]:22s}"
        for k, c in enumerate(CLASSES):
            pred_px = int(pred[i, k].sum())
            if m is not None:
                px = m["px"][c]
                rows.append([name, "yes", c, pred_px, px["tp"] + px["fn"]] + per_class_row(m, c))
                line += f"{px['iou']:9.4f}{px['precision']:6.3f}{px['recall']:6.3f}{px['f1']:6.3f}   "
            else:
                rows.append([name, "no", c, pred_px, "", "", "", "", "", "", "", ""])
                line += f"{'-':>9s}{'':18s}   "
        if m is not None:
            rows.append([name, "yes", "mean", "", "", f"{m['miou_px']:.4f}",
                         f"{np.mean([m['px'][c]['precision'] for c in CLASSES]):.4f}",
                         f"{np.mean([m['px'][c]['recall'] for c in CLASSES]):.4f}",
                         f"{np.mean([m['px'][c]['f1'] for c in CLASSES]):.4f}", "", "", ""])
            line += f" {m['miou_px']:.4f}"
        print(line)
    print(LINE)

    with open(os.path.join(out_dir, "metrics.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["image", "labeled", "class", "pred_px", "gt_px", "iou", "precision", "recall", "f1", "tp", "fp", "fn"])
        w.writerows(rows)

    agg = metrics(pred[labeled_idx], np.stack([gts_small[i] for i in labeled_idx])) if labeled_idx else None
    lines = [res["model_line"], f"dir      : {args.dir}  (이미지 {len(paths)} 장, 라벨 {len(labeled_idx)} 장)",
             f"input    : uint8 BGR {IMG_SIZE} (모델 해상도), 임계값 sigmoid > {THRESH} 는 모델/엔진 안",
             f"speed    : {res['ms']:.2f} ms/img (batch {res['batch']}, 복사 포함)",
             mem_line(res["mem"]),
             f"outputs  : {out_dir}  (overlay/ {args.ext}, " + ("mask/ png, " if not args.no_mask else "") + "metrics.csv)"]
    if agg is not None:
        lines += ["", f"[pixel-level, {len(labeled_idx)} 장 합산]  {iou_summary(agg)}",
                  f"[per-image mean]  " + " / ".join(f"{c} {agg['per'][c]['iou']:.4f}" for c in CLASSES) + f" (mIoU {agg['miou_per']:.4f})", "",
                  f"{'class':10s}{'IoU':>8s}{'Precision':>11s}{'Recall':>8s}{'F1':>8s}   | {'IoU(img)':>9s}{'P(img)':>8s}{'R(img)':>8s}{'F1(img)':>8s}"]
        for c in CLASSES:
            px, pm = agg["px"][c], agg["per"][c]
            lines.append(f"{c:10s}{px['iou']:8.4f}{px['precision']:11.4f}{px['recall']:8.4f}{px['f1']:8.4f}   | "
                         f"{pm['iou']:9.4f}{pm['precision']:8.4f}{pm['recall']:8.4f}{pm['f1']:8.4f}")
    else:
        lines += ["", "라벨 json 이 없어 지표는 계산하지 않음 (추론 + 오버레이만)"]
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[:6]))
    return agg, per_img


# ======================================================
# dense vs 엔진 비교 : compare.csv / compare.txt
# ======================================================
def write_compare(rd, rt, agg_d, agg_t, per_d, per_t, out_root, paths, args):
    pred_d, pred_t = rd["pred"], rt["pred"]
    agr = agreement(rd["packed"], rt["packed"])
    n_label = sum(m is not None for m in per_d)
    lines = [f"dir       : {args.dir}  (이미지 {len(paths)} 장, 라벨 {n_label} 장)",
             f"dense     : {rd['desc']}  ->  {rd['ms']:.2f} ms/img (batch {rd['batch']})",
             f"trt       : {rt['desc']}  ->  {rt['ms']:.2f} ms/img (batch {rt['batch']})",
             f"speed     : 엔진이 dense 의 {rd['ms'] / rt['ms']:.2f} 배",
             f"gpu mem   : dense 추론 후 최대 {_mib(rd['mem']['peak'])} (이 모델 몫 {_share(rd['mem'])})  |  "
             f"trt 추론 후 최대 {_mib(rt['mem']['peak'])} (이 모델 몫 {_share(rt['mem'])})   "
             f"[프로세스 전용 GPU 메모리. 한 프로세스에서 차례로 돌아 CUDA 컨텍스트는 먼저 돈 모델에 잡힘 -> '몫' 으로 비교]",
             f"agreement : 두 모델 출력이 같은 픽셀 비율 (전체 {len(paths)} 장, 모델 해상도)  all {agr['all'] * 100:.3f}%  |  "
             + "  |  ".join(f"{c} {agr[c] * 100:.3f}%" for c in CLASSES)]

    # 이미지별
    rows, img_lines = [], [f"{'image':22s}{'mIoU dense':>12s}{'mIoU trt':>10s}{'diff':>9s}{'agree(all)':>12s}"]
    for i, p in enumerate(paths):
        name = os.path.splitext(os.path.basename(p))[0]
        md, mt = per_d[i], per_t[i]
        ag_all = float((rd["packed"][i] == rt["packed"][i]).mean())
        for k, c in enumerate(CLASSES):
            ag = float((pred_d[i, k] == pred_t[i, k]).mean())
            if md is not None and mt is not None:
                a, b = md["px"][c], mt["px"][c]
                rows.append([name, c, f"{a['iou']:.4f}", f"{b['iou']:.4f}", f"{b['iou'] - a['iou']:+.4f}",
                             f"{a['f1']:.4f}", f"{b['f1']:.4f}", f"{ag:.5f}"])
            else:
                rows.append([name, c, "", "", "", "", "", f"{ag:.5f}"])
        if md is not None and mt is not None:
            rows.append([name, "mean", f"{md['miou_px']:.4f}", f"{mt['miou_px']:.4f}", f"{mt['miou_px'] - md['miou_px']:+.4f}", "", "", f"{ag_all:.5f}"])
            img_lines.append(f"{name[:21]:22s}{md['miou_px']:12.4f}{mt['miou_px']:10.4f}{mt['miou_px'] - md['miou_px']:+9.4f}{ag_all * 100:11.3f}%")
        else:
            img_lines.append(f"{name[:21]:22s}{'-':>12s}{'-':>10s}{'-':>9s}{ag_all * 100:11.3f}%")
    with open(os.path.join(out_root, "compare.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["image", "class", "iou_dense", "iou_trt", "iou_diff", "f1_dense", "f1_trt", "agreement"])
        w.writerows(rows)

    # 클래스별 합산
    if agg_d is not None and agg_t is not None:
        lines += ["", f"[pixel-level, {n_label} 장 합산]   dense {iou_summary(agg_d)}", f"{'':28s}trt   {iou_summary(agg_t)}", "",
                  f"{'class':10s}{'IoU dense':>10s}{'IoU trt':>9s}{'diff':>8s} | {'P dense':>8s}{'P trt':>7s} | "
                  f"{'R dense':>8s}{'R trt':>7s} | {'F1 dense':>9s}{'F1 trt':>7s} | {'agree':>8s}"]
        for c in CLASSES:
            a, b = agg_d["px"][c], agg_t["px"][c]
            lines.append(f"{c:10s}{a['iou']:10.4f}{b['iou']:9.4f}{b['iou'] - a['iou']:+8.4f} | {a['precision']:8.4f}{b['precision']:7.4f} | "
                         f"{a['recall']:8.4f}{b['recall']:7.4f} | {a['f1']:9.4f}{b['f1']:7.4f} | {agr[c] * 100:7.3f}%")
        lines.append(f"{'mean':10s}{agg_d['miou_px']:10.4f}{agg_t['miou_px']:9.4f}{agg_t['miou_px'] - agg_d['miou_px']:+8.4f}")
    else:
        lines += ["", "라벨 json 이 없어 지표 비교는 생략 (픽셀 일치율만)"]
    lines += ["", "[per image]"] + img_lines
    with open(os.path.join(out_root, "compare.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n[compare] dense vs trt")
    print("\n".join(lines))


# ======================================================
def main():
    ap = argparse.ArgumentParser(description="dense (PyTorch) / TensorRT 엔진으로 폴더 추론: 모델별 오버레이 + 이미지별 지표 (라벨 있을 때), 비교표")
    ap.add_argument("--model", default="both", choices=["dense", "trt", "both"], help="dense = PyTorch 체크포인트, trt = 엔진, both = 둘 다 + 비교 (기본)")
    ap.add_argument("--dir", default=TEST_DIR, help=f"이미지 폴더 (기본 {TEST_DIR})")
    ap.add_argument("--ckpt", default=CKPT_PATH, help=f"dense 체크포인트 (기본 {CKPT_PATH})")
    ap.add_argument("--engine", default=ENGINE_INT8, help=f"TensorRT 엔진 (기본 {ENGINE_INT8})")
    ap.add_argument("--out", default=None, help="출력 폴더 (기본 MODEL_DIR/predict_<폴더>/, 그 아래 dense_fp32/ 와 trt_<엔진>/)")
    ap.add_argument("--batch", type=int, default=4, help="batch (엔진은 profile max 이하, dense 는 메모리 부족 시 1 로 자동 축소. 기본 4)")
    ap.add_argument("--ext", default="png", choices=["png", "jpg"], help="오버레이 저장 형식 (기본 png)")
    ap.add_argument("--draw-gt", action="store_true", help="GT 윤곽선(흰색) 도 오버레이에 그림 (라벨 있는 이미지만)")
    ap.add_argument("--no-mask", action="store_true", help="비트 패킹 마스크 png 저장 생략")
    ap.add_argument("--tf32", action="store_true", help="dense 추론에 TF32 허용 (기본 off = 순수 FP32)")
    args = ap.parse_args()

    want_dense, want_trt = args.model in ("dense", "both"), args.model in ("trt", "both")
    if not os.path.isdir(args.dir):
        raise SystemExit(f"폴더가 없음: {args.dir}")
    if want_dense and not os.path.exists(args.ckpt):
        raise SystemExit(f"체크포인트가 없음: {args.ckpt} (eval_common.MODEL_DIR / --ckpt 확인)")
    if want_trt and not os.path.exists(args.engine):
        raise SystemExit(f"엔진이 없음: {args.engine} (_4_onnx2trt.py 를 먼저 실행)")
    out_root = args.out or os.path.join(MODEL_DIR, f"predict_{os.path.basename(os.path.normpath(args.dir))}")
    os.makedirs(out_root, exist_ok=True)

    # ---------- 이미지 + 라벨 (두 모델이 공유) ----------
    images, paths = load_images([args.dir])                           # uint8 (N,3,h,w) BGR, 모델 해상도
    labels = [os.path.splitext(p)[0] + ".json" for p in paths]
    has_label = [os.path.exists(l) for l in labels]
    n_label = sum(has_label)
    print(f"[data] {args.dir}: 이미지 {len(paths)} 장, 라벨 json {n_label} 장 -> "
          + ("지표 계산 + 오버레이" if n_label else "라벨 없음: 지표 계산 생략, 추론 + 오버레이만"))
    reader = _label_reader() if n_label else None
    gts_small = [None] * len(paths)
    for i, p in enumerate(paths):
        if has_label[i]:
            orig = cv2.imread(p, IMREAD_FLAG)
            if orig is not None:
                gts_small[i] = load_gt_small(labels[i], orig.shape[:2], reader)

    # ---------- 추론 (모델별) + 저장 ----------
    print(f"[gpu] CUDA 컨텍스트 후 프로세스 전용 GPU 메모리 {_mib(cuda_base_mib())}  (모델 몫 = 추론 후 최대 - 그 모델 로드 전)")
    results, aggs, pers = {}, {}, {}
    if want_dense:
        results["dense"] = run_dense(images, args)
    if want_trt:
        results["trt"] = run_trt(images, args)
    for key, res in results.items():
        aggs[key], pers[key] = write_outputs(res, os.path.join(out_root, res["name"]), paths, labels, gts_small, reader, args)

    # ---------- 요약 / 비교 ----------
    if n_label:
        rows = [(("dense fp32" if k == "dense" else os.path.splitext(os.path.basename(args.engine))[0]), ("torch" if k == "dense" else "TRT"), aggs[k])
                for k in results if aggs[k] is not None]
        print_accuracy_table(rows, sum(m is not None for m in gts_small))
    if want_dense and want_trt:
        write_compare(results["dense"], results["trt"], aggs["dense"], aggs["trt"], pers["dense"], pers["trt"], out_root, paths, args)
    print(f"\n[OK] -> {out_root}  (" + ", ".join(f"{r['name']}/ {r['ms']:.2f} ms/img" for r in results.values()) + ")")


if __name__ == "__main__":
    main()

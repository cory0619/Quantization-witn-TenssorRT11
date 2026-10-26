r"""
eval_common.py : _1_model2onnx.py / _2_onnx_int8_ptq.py / _3_int8_layer_check.py / _4_onnx2trt.py 가 같이 쓰는 설정 + 헬퍼
                 (2026-10-17, R6 UNet_ConvNeXt / dataset_pcb multi-label 용으로 갱신)

- 설정: 체크포인트 / ONNX / 엔진 / 리포트 경로, 입력 크기, 클래스. 학습 스크립트(train_test_fp32_pcb.py) 와 같아야 하며 _1 이 대조한다.
- 데이터: 이미지 로드(학습과 같은 전처리: 컬러 BGR, IMG_SIZE 로 INTER_LINEAR resize, uint8 그대로. /255 와 ImageNet 정규화는 모델 안),
          test set (images_dataset.dataset_experiment 로 labelme json -> (K,H,W) 마스크, 학습 평가와 동일 경로)
- 배포 출력: 모델이 내는 uint8 비트 패킹 마스크 (B,1,H,W) (bit c = 클래스 c, sigmoid > THRESH) -> (B,K,H,W) bool 복원
- 지표: 클래스별 IoU / Precision / Recall / F1 — pixel-level(전체 픽셀 TP/FP/FN 합산) 과 per-image 평균, 클래스 평균 mIoU
        (학습 스크립트 compute_segmentation_metrics 와 같은 정의라 summary.txt 와 바로 비교 가능)
- 측정: 추론 시간 통계, GPU 메모리(프로세스 전용 메모리 = 작업관리자 값, NVML 장치 전체), GPU 사용률 샘플링
- 시각화: PALETTE / overlay_masks (학습 스크립트와 같은 색·불투명도·윤곽선. torch 없이 _5 / server_inference / client 가 공유)
- 리포트: 결과를 json 에 섹션별로 저장 (save_report) -> report_html.py 가 네 항목(속도 / GPU 메모리 / INT8·FP16·FP32 위치 / 정확도) html 생성
"""
import glob
import os
import subprocess
import threading
import time

import cv2
import numpy as np

# ======================================================
# 설정 (경로 / 모델 인터페이스). 학습 스크립트와 같은 값이어야 함
# ======================================================
MODEL_DIR    = r".\output\fp32_model_20261017_221814"           # 학습 결과 폴더 (R6 UNet_ConvNeXt, dataset_pcb)
CKPT_PATH    = os.path.join(MODEL_DIR, "origin_model_best.pt")   # 학습 스크립트가 저장한 state_dict (약 355 MB)
ONNX_FP32    = os.path.join(MODEL_DIR, "r6_fp32.onnx")           # _1 결과: deploy 그래프 (uint8 BGR 입력 -> uint8 비트 패킹 마스크)
ONNX_INT8    = os.path.join(MODEL_DIR, "r6_int8_qdq.onnx")       # _2 결과: ModelOpt Q/DQ 삽입 + 나머지 fp16
ENGINE_FP32  = os.path.join(MODEL_DIR, "r6_fp32.engine")         # _4 결과 (비교용 FP32 엔진)
ENGINE_INT8  = os.path.join(MODEL_DIR, "r6_int8.engine")         # _4 결과 (배포용 INT8 엔진)
TIMING_CACHE = os.path.join(MODEL_DIR, "r6_trt_timing.cache")    # _4 TensorRT tactic timing cache (재빌드 시간 단축)
REPORT_JSON  = os.path.join(MODEL_DIR, "compare_fp32_vs_int8.json")   # _2/_3/_4 결과 (save_report 가 섹션별로 기록)
REPORT_HTML  = os.path.join(MODEL_DIR, "compare_fp32_vs_int8.html")   # report_html.py 가 json 으로 생성 (속도 / GPU 메모리 / 정밀도 위치 / 정확도)

DATASET_DIR = r".\dataset_pcb"
TEST_DIR    = os.path.join(DATASET_DIR, "test")                   # 평가용 (라벨 json 필요)
CALIB_DIRS  = [os.path.join(DATASET_DIR, "train")]                # _2 calibration 용 (라벨 불필요). test 는 넣지 말 것

IN_CHANNELS = 3                                  # 컬러 BGR (OpenCV 순서, NCHW)
CLASSES     = ["pcb", "terminal", "hole"]        # 출력 비트/채널 순서 = 이 순서 (bit0 pcb, bit1 terminal, bit2 hole)
N_CLASSES   = len(CLASSES)
THRESH      = 0.5                                # sigmoid > THRESH. deploy 모델 안에 상수로 들어감
IMG_SIZE    = (1088, 1344)                       # (H, W). 32 의 배수. 학습 IMG_SIZE 와 동일
HEIGHT, WIDTH = IMG_SIZE
INPUT_NAME, OUTPUT_NAME = "input", "output"
OPSET = 19                                       # ModelOpt INT8 양자화의 최소 opset


# ======================================================
# 데이터
# ======================================================
def load_images(dirs, max_images=None, size=IMG_SIZE, channels=IN_CHANNELS):
    """
    학습(images_dataset) 과 같은 전처리로 이미지만 로드 -> uint8 (N,C,H,W). C=3 이면 BGR.
    모델(deploy) 이 /255 와 ImageNet 정규화를 안에서 하므로 여기서는 원본 픽셀 값을 그대로 둔다.
    """
    paths = []
    for d in dirs:
        for ext in ("*.png", "*.jpeg", "*.jpg", "*.bmp", "*.PNG", "*.JPEG", "*.JPG", "*.BMP"):
            paths += glob.glob(os.path.join(d, ext))
    paths = sorted(set(os.path.normpath(p) for p in paths))
    if max_images:
        paths = paths[:max_images]
    imgs = []
    for p in paths:
        im = cv2.imread(p, cv2.IMREAD_COLOR if channels == 3 else cv2.IMREAD_GRAYSCALE)
        if im is None:
            print(f"[warn] cannot read {p}")
            continue
        im = cv2.resize(im, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
        imgs.append(im if im.ndim == 3 else im[:, :, None])
    if not imgs:
        raise RuntimeError(f"no images found in {dirs}")
    return np.ascontiguousarray(np.stack(imgs).transpose(0, 3, 1, 2)), paths


def load_test_set(test_dir=TEST_DIR, size=IMG_SIZE):
    """
    학습 평가와 똑같은 Dataset 경로로 test set 로드.
    반환: images uint8 (N,C,H,W) BGR, gts bool (N,K,H,W) (클래스 순서 = CLASSES), paths
    Dataset 은 float(0~1) 이미지를 주므로 x255 반올림으로 uint8 원본 픽셀을 복원 (정확히 역변환됨)
    """
    from images_dataset import dataset_experiment   # torch / albumentations 는 필요할 때만 import
    ds = dataset_experiment(test_dir, classes=CLASSES, size=size, multilabel=True, channels=IN_CHANNELS)
    images, gts, paths = [], [], []
    for i in range(len(ds)):
        img, mask, path = ds[i]                                         # (C,H,W) float 0~1, (K,H,W) float {0,1}
        images.append(np.rint(img.numpy() * 255.0).astype(np.uint8))
        gts.append(mask.numpy() > 0.5)
        paths.append(path)
    return np.ascontiguousarray(np.stack(images)), np.stack(gts), paths


def unpack_masks(packed, n_classes=N_CLASSES):
    """deploy 출력 uint8 (N,1,H,W) (bit c = 클래스 c) -> bool (N,K,H,W). model.bitmask.unpack_bits 의 numpy 판"""
    packed = np.asarray(packed)
    if packed.ndim != 4 or packed.shape[1] != 1:
        raise ValueError(f"packed mask must be (N,1,H,W), got {packed.shape}")
    shifts = np.arange(n_classes, dtype=np.uint8).reshape(1, n_classes, 1, 1)
    return ((packed.astype(np.uint8) >> shifts) & 1).astype(bool)


# ======================================================
# 지표 (학습 스크립트 compute_segmentation_metrics 와 같은 정의)
# ======================================================
def _ratio(num, den, both_empty):
    if den == 0:
        return 1.0 if both_empty else 0.0
    return num / den


def _binary_metrics(tp, fp, fn):
    e = (tp + fp + fn) == 0
    return {"iou": _ratio(tp, tp + fp + fn, e), "precision": _ratio(tp, tp + fp, e),
            "recall": _ratio(tp, tp + fn, e), "f1": _ratio(2 * tp, 2 * tp + fp + fn, e), "tp": tp, "fp": fp, "fn": fn}


def metrics(pred, gt, class_names=CLASSES):
    """
    pred, gt: bool (N,K,H,W) -> {"n_images", "classes", "px": {cls: {...}}, "per": {cls: {...}}, "miou_px", "miou_per"}
    px = 전체 픽셀 TP/FP/FN 합산, per = 이미지별 지표의 평균
    """
    pred, gt = np.asarray(pred, bool), np.asarray(gt, bool)
    if pred.shape != gt.shape:
        raise ValueError(f"pred {pred.shape} != gt {gt.shape}")
    k = pred.shape[1]
    tot = np.zeros((k, 3), dtype=np.int64)
    per = {c: {"iou": [], "precision": [], "recall": [], "f1": []} for c in range(k)}
    for p, g in zip(pred, gt):
        tp = (p & g).reshape(k, -1).sum(1); fp = (p & ~g).reshape(k, -1).sum(1); fn = (~p & g).reshape(k, -1).sum(1)
        for c in range(k):
            m = _binary_metrics(int(tp[c]), int(fp[c]), int(fn[c]))
            tot[c] += (m["tp"], m["fp"], m["fn"])
            for key in per[c]:
                per[c][key].append(m[key])
    px = [_binary_metrics(*map(int, tot[c])) for c in range(k)]
    return {
        "n_images": int(pred.shape[0]), "classes": list(class_names),
        "px": {class_names[c]: px[c] for c in range(k)},
        "per": {class_names[c]: {key: float(np.mean(v)) for key, v in per[c].items()} for c in range(k)},
        "miou_px": float(np.mean([px[c]["iou"] for c in range(k)])),
        "miou_per": float(np.mean([np.mean(per[c]["iou"]) for c in range(k)])),
    }


def agreement(packed_a, packed_b, class_names=CLASSES):
    """두 deploy 출력(uint8 비트 패킹) 의 픽셀 일치율. "all" = 모든 클래스 비트가 같은 픽셀 비율, 클래스별 = 그 비트가 같은 비율"""
    a, b = unpack_masks(packed_a, len(class_names)), unpack_masks(packed_b, len(class_names))
    out = {"all": float((np.asarray(packed_a) == np.asarray(packed_b)).mean())}
    for c, name in enumerate(class_names):
        out[name] = float((a[:, c] == b[:, c]).mean())
    return out


def iou_summary(m):
    """'pcb 0.9452 / terminal 0.8769 / hole 0.5958 (mIoU 0.8060)' 식 한 줄 (pixel-level)"""
    return " / ".join(f"{c} {m['px'][c]['iou']:.4f}" for c in m["classes"]) + f" (mIoU {m['miou_px']:.4f})"


def print_accuracy_table(rows, n_images):
    """rows: [(name, note, metrics dict), ...]. 학습 스크립트 summary.txt 와 같은 배치 (pixel-level | per-image 평균)"""
    print("=" * 118)
    print(f"[accuracy] test images: {n_images} | GT: labelme json multi-label masks | pred: sigmoid > {THRESH} per class (모델 안 임계값) | classes: {CLASSES}")
    print(f"{'model':14s}{'run':7s}{'class':10s}{'IoU':>8s}{'Precision':>11s}{'Recall':>8s}{'F1':>8s}   |{'IoU':>8s}{'Precision':>11s}{'Recall':>8s}{'F1':>8s}   (TP / FP / FN)")
    print(f"{'':31s}{'--------- pixel-level ---------':^35s}   |{'------ per-image mean ------':^35s}")
    for name, note, m in rows:
        for c in m["classes"]:
            px, pm = m["px"][c], m["per"][c]
            print(f"{name[:13]:14s}{note[:6]:7s}{c:10s}"
                  f"{px['iou']:8.4f}{px['precision']:11.4f}{px['recall']:8.4f}{px['f1']:8.4f}   |"
                  f"{pm['iou']:8.4f}{pm['precision']:11.4f}{pm['recall']:8.4f}{pm['f1']:8.4f}   ({px['tp']} / {px['fp']} / {px['fn']})")
        print(f"{'':31s}mIoU (class mean) pixel-level {m['miou_px']:.4f} | per-image {m['miou_per']:.4f}")
    print("=" * 118)


# ======================================================
# 시간 / GPU 측정
# ======================================================
def benchmark(fn, warmup=10, iters=50):
    """fn() 은 동기화까지 끝내고 돌아와야 함. 반환: ms 통계 dict"""
    for _ in range(warmup):
        fn()
    t = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        t.append((time.perf_counter() - t0) * 1e3)
    t = np.array(t)
    return {"mean": float(t.mean()), "median": float(np.median(t)), "std": float(t.std()),
            "min": float(t.min()), "max": float(t.max()), "iters": iters}


class GpuMonitor:
    """
    - device_used_mib(): NVML 장치 전체 사용량 (다른 프로세스 포함)
    - process_mib(): 이 프로세스의 GPU 전용 메모리 (Windows 성능 카운터 'GPU Process Memory\\Dedicated Usage',
                     작업관리자 '전용 GPU 메모리' 와 같은 값). Windows 가 아니거나 실패하면 None
    - sample_util(): with 블록 동안 NVML GPU 사용률(%) 을 주기적으로 샘플링
    """

    def __init__(self, index=0):
        self.ok = False
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nv = pynvml
            self.h = pynvml.nvmlDeviceGetHandleByIndex(index)
            self.ok = True
        except Exception as e:
            print(f"[warn] NVML unavailable: {e}")

    def info(self):
        if not self.ok:
            return {}
        name = self.nv.nvmlDeviceGetName(self.h)
        name = name.decode() if isinstance(name, bytes) else name
        drv = self.nv.nvmlSystemGetDriverVersion()
        drv = drv.decode() if isinstance(drv, bytes) else drv
        total = self.nv.nvmlDeviceGetMemoryInfo(self.h).total / 2 ** 20
        return {"gpu": name, "driver": drv, "total_mib": total}

    def device_used_mib(self):
        if not self.ok:
            return None
        return self.nv.nvmlDeviceGetMemoryInfo(self.h).used / 2 ** 20

    @staticmethod
    def process_mib(pid=None):
        if os.name != "nt":
            return None
        pid = pid or os.getpid()
        try:
            r = subprocess.run(["typeperf", rf"\GPU Process Memory(pid_{pid}_*)\Dedicated Usage", "-sc", "1"],
                               capture_output=True, timeout=30)
            lines = [ln for ln in r.stdout.decode("utf-8", errors="ignore").splitlines() if ln.startswith('"')]
            if len(lines) < 2:
                return None
            vals = [v.strip('"') for v in lines[1].split(",")[1:]]
            return sum(float(v) for v in vals if v) / 2 ** 20
        except Exception:
            return None

    class _Sampler:
        def __init__(self, mon, interval):
            self.mon, self.interval, self.samples, self._stop = mon, interval, [], threading.Event()

        def _run(self):
            while not self._stop.is_set():
                try:
                    self.samples.append(self.mon.nv.nvmlDeviceGetUtilizationRates(self.mon.h).gpu)
                except Exception:
                    pass
                time.sleep(self.interval)

        def __enter__(self):
            if self.mon.ok:
                self.t = threading.Thread(target=self._run, daemon=True)
                self.t.start()
            return self

        def __exit__(self, *a):
            self._stop.set()
            if self.mon.ok:
                self.t.join()

        @property
        def mean(self):
            return float(np.mean(self.samples)) if self.samples else float("nan")

        @property
        def max(self):
            return float(np.max(self.samples)) if self.samples else float("nan")

    def sample_util(self, interval=0.02):
        return GpuMonitor._Sampler(self, interval)


def fmt_mib(v):
    return "n/a" if v is None else f"{v:.0f} MiB"


def fmt_bytes(path):
    n = os.path.getsize(path)
    return f"{n / 2**20:.1f} MB" if n >= 2**20 else f"{n / 1e3:.0f} KB"


# ======================================================
# 시각화 (train_test_fp32_pcb.py 의 PALETTE / overlay_masks 와 같은 규칙. torch 가 필요 없는 쪽에서 쓴다)
# ======================================================
PALETTE = np.array([[255, 128, 0],          # pcb      : 파랑 (BGR)
                    [0, 200, 0],            # terminal : 초록
                    [0, 0, 255]], np.uint8) # hole     : 빨강
ALPHAS = [0.45, 0.45, 0.9]                  # 클래스별 overlay 불투명도. hole 은 pcb 위에 겹치므로 거의 불투명하게
CONTOUR_THICKNESS = 2                       # 클래스별 윤곽선 두께 (0 이면 안 그림)


def resize_masks(masks, hw):
    """(K,h,w) -> (K,H,W) nearest"""
    return np.stack([cv2.resize(np.ascontiguousarray(m), (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST) for m in masks])


def overlay_masks(image, masks, alphas=ALPHAS, contour=CONTOUR_THICKNESS):
    """원본(BGR 또는 grayscale) 위에 클래스별 마스크 (K,H,W) {0,1} 을 PALETTE 색으로 반투명 덧칠 + 윤곽선. 크기가 다르면 nearest 로 맞춤"""
    h, w = image.shape[:2]
    masks = np.asarray(masks).astype(np.uint8)
    if masks.shape[1:] != (h, w):
        masks = resize_masks(masks, (h, w))
    img = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for c in range(masks.shape[0]):
        fg = masks[c] > 0
        if fg.any():
            a = alphas[c] if c < len(alphas) else alphas[-1]
            color = np.empty_like(img)
            color[:] = PALETTE[c % len(PALETTE)]
            img[fg] = cv2.addWeighted(img, 1 - a, color, a, 0)[fg]
    if contour > 0:
        for c in range(masks.shape[0]):
            cnts, _ = cv2.findContours(masks[c], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cnts, -1, tuple(int(v) for v in PALETTE[c % len(PALETTE)]), contour)
    return img


# ======================================================
# 리포트: json 에 섹션별 저장 -> report_html.py 가 html 생성
# ======================================================
def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and o != o:          # nan -> null
        return None
    return o


def save_report(section, data, path=REPORT_JSON, html_path=REPORT_HTML):
    """
    path 의 json 에서 data[section] 을 교체/추가 (갱신 시각 포함) 하고 html 리포트를 다시 만든다.
    섹션: env, ort (_2) / precision_onnx, precision_engine (_3) / trt (_4). 어느 스크립트를 먼저 돌려도 되고, 없는 섹션은 html 에 자리만 표시.
    """
    import json
    doc = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                doc = json.load(f)
        except Exception as e:
            print(f"[report] {path} 읽기 실패 ({type(e).__name__}: {e}) -> 새로 만듦")
    data = _jsonable(data)
    data["updated"] = time.strftime("%Y-%m-%d %H:%M")
    doc[section] = data
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    print(f"[report] section '{section}' -> {path}")
    try:
        from report_html import write_report_html
        write_report_html(path, html_path)
    except Exception as e:                        # 렌더링 오류로 긴 실행이 끝에서 죽지 않게. json 은 이미 저장됨
        import traceback
        print(f"[report] html 생성 실패: {type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}  -> python report_html.py 로 재시도")

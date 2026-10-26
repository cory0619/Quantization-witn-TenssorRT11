r"""
server_inference.py : INT8 TensorRT 엔진 웹 서비스 (Flask + 공유메모리, batch 추론)   (R6 UNet_ConvNeXt, dataset_pcb)

- 구조는 참조 서버(weld 검사 server_ls.py) 와 같다: 클라이언트가 이미지를 공유메모리에 넣고 JSON 으로 POST /predict,
  서버는 추론 전용 스레드 1개에서 처리하고 요청은 직렬(threaded=False). 공유메모리는 예외 여부와 무관하게 close.
- 모델은 torch 가 아니라 _4_onnx2trt.py 가 만든 TensorRT 엔진 (model/trt_runner.TrtRunner). /255·ImageNet 정규화·임계값(sigmoid > 0.5)·
  비트 패킹이 모두 엔진 안이라 서버는 resize 와 윤곽선 추출만 한다. torch / onnxruntime 이 없는 환경에서도 돈다 (tensorrt, cupy, flask, cv2).
- 클라이언트는 이미지 B장 (기본 4) 을 (B,H,W,C) uint8 로 공유메모리 하나에 넣는다 (한 요청의 이미지는 같은 크기).
  서버는 BGR 3채널로 맞추고 엔진 입력 크기 (1088x1344) 로 resize 해 배치로 넣은 뒤, 클래스별 마스크를 원본 크기로 되돌려
  polygon(윤곽선) 좌표를 응답한다. 요청에 mask_shm_name 이 있으면 원본 해상도 비트 패킹 마스크 (B,H,W) uint8 도 그 공유메모리에 써 준다.
  B 가 엔진 profile max (8) 보다 크면 안에서 나눠 돈다.

요청 JSON (POST /predict):
  {"product_id": "...", "shm_name": "...", "batch": 4, "width": 1358, "height": 1104, "image_channel": 3 (1/3/4),
   "names": ["0001.JPG", ...] (선택), "rgba_order": "BGRA" (channel 4 일 때), "mask_shm_name": "..." (선택, 클라이언트가 B*H*W 바이트로 생성)}
응답 JSON:
  {"product_id", "status": "OK", "message", "batch", "model_size": [1088, 1344], "infer_ms", "total_ms",
   "images": [{"index": 0, "name": "0001.JPG",
               "classes": {"pcb": {"pixels": N, "polygons": [[x0,y0,x1,y1,...], ...]}, "terminal": {...}, "hole": {...}}}, ...]}
  polygon 좌표는 클라이언트가 보낸 (width, height) 기준, pixels 는 그 해상도의 마스크 픽셀 수. MIN_CONTOUR_AREA 미만 조각은 제외.
GET /health: {"status": "ok", "engine", "profile", "model_size", "classes"}

사용:
    python server_inference.py                                     # eval_common.ENGINE_INT8, 127.0.0.1:5004
    python server_inference.py --engine output\...\r6_fp32.engine --port 5004 --save-dir server_result   # 오버레이도 서버에서 저장
    python client.py                                               # 다른 터미널에서 dataset_pcb/test 를 4장씩 전송
"""
import warnings
warnings.simplefilter("ignore", category=Warning)

import argparse
import multiprocessing.shared_memory as shared_mem
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, request, jsonify

from eval_common import ENGINE_INT8, CLASSES, overlay_masks
from model.bitmask import unpack_bits, pack_bits
from model.trt_runner import TrtRunner

app = Flask(__name__)

N_CLASSES = len(CLASSES)
MIN_CONTOUR_AREA = 10          # px^2 (클라이언트 해상도). 이보다 작은 조각은 polygon 응답에서 제외 (노이즈)
WARMUP_BATCH = 4               # 기동 시 warm-up batch (엔진 profile max 이하로 잘림)

# ---- 결과 이미지 저장 워커 (참조 서버와 같은 대기열 제한: 인코딩이 추론보다 느려도 메모리가 쌓이지 않게)
SAVE_WORKERS, SAVE_MAX_PENDING, SAVE_BLOCK_WHEN_FULL = 2, 4, True
_image_saver = ThreadPoolExecutor(max_workers=SAVE_WORKERS, thread_name_prefix="img_saver")
_save_slots = threading.BoundedSemaphore(SAVE_MAX_PENDING)

# ---- 추론 전용 스레드 1개: TrtRunner(실행 컨텍스트, 스트림, 버퍼) 는 이 스레드에서만 만들고 쓴다
_infer_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="infer")

STATE = {"runner": None, "engine": None, "save_dir": None}


# ======================================================
# 공유메모리
# ======================================================
def read_batch_from_shm(shm_name, batch, width, height, channel, rgba_order="BGRA"):
    """공유메모리의 (B,H,W,C) uint8 -> (B,H,W,3) BGR 독립 사본. 뷰는 이 함수 안에서만 쓰고 항상 close (Windows 는 unlink no-op)"""
    expected = batch * height * width * channel
    shm = shared_mem.SharedMemory(name=shm_name)
    try:
        if shm.size < expected:
            raise ValueError(f"Shared memory '{shm_name}' too small: {shm.size} bytes < {expected} "
                             f"(batch={batch}, width={width}, height={height}, channel={channel})")
        view = np.ndarray((batch, height, width, channel), dtype=np.uint8, buffer=shm.buf)
        if channel == 3:
            out = view.copy()
        elif channel == 1:
            out = np.stack([cv2.cvtColor(v, cv2.COLOR_GRAY2BGR) for v in view])      # 모델은 컬러로 학습됨: grayscale 은 정확도 보장 안 됨
        elif channel == 4:
            order = str(rgba_order).upper()
            if order == "BGRA":
                out = np.stack([cv2.cvtColor(v, cv2.COLOR_BGRA2BGR) for v in view])
            elif order == "RGBA":
                out = np.stack([cv2.cvtColor(v, cv2.COLOR_RGBA2BGR) for v in view])
            elif order in ("ABGR", "ARGB"):
                idx = [1, 2, 3] if order == "ABGR" else [3, 2, 1]
                out = np.ascontiguousarray(view[..., idx])
            else:
                raise ValueError(f"Unsupported rgba_order: {rgba_order}")
        else:
            raise ValueError(f"Unsupported image_channel: {channel}")
        del view
        return out
    finally:
        shm.close()
        shm.unlink()      # POSIX 호환용. Windows 에서는 no-op (클라이언트가 close 하면 해제)


def write_masks_to_shm(shm_name, packed):
    """원본 해상도 비트 패킹 마스크 (B,H,W) uint8 을 클라이언트가 만든 공유메모리에 기록 (클라이언트가 읽은 뒤 close/unlink)"""
    shm = shared_mem.SharedMemory(name=shm_name)
    try:
        if shm.size < packed.nbytes:
            raise ValueError(f"mask shared memory '{shm_name}' too small: {shm.size} < {packed.nbytes}")
        view = np.ndarray(packed.shape, dtype=np.uint8, buffer=shm.buf)
        view[...] = packed
        del view
    finally:
        shm.close()


# ======================================================
# 추론 (추론 스레드)
# ======================================================
def _infer_impl(bgr):
    """(B,H,W,3) uint8 BGR -> 엔진 입력 크기로 resize -> (B,1,h,w) uint8 비트 패킹 (엔진 안에서 /255·정규화·임계값·패킹)"""
    r = STATE["runner"]
    h, w = r.in_hw
    x = np.empty((len(bgr), 3, h, w), np.uint8)
    for i, im in enumerate(bgr):
        if im.shape[:2] != (h, w):
            im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)        # 학습 전처리와 동일
        x[i] = im.transpose(2, 0, 1)
    return r.run_all(x, r.batch_max)


def predict_batch(bgr):
    return _infer_executor.submit(_infer_impl, bgr).result()


def mask_polygons(mask_u8):
    """(H,W) {0,1} -> [[x0,y0,x1,y1,...], ...] (RETR_EXTERNAL, 면적 MIN_CONTOUR_AREA 이상)"""
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c.reshape(-1, 2).flatten().tolist() for c in cnts if cv2.contourArea(c) >= MIN_CONTOUR_AREA]


# ======================================================
# 결과 이미지 저장 (백그라운드, 대기열 제한)
# ======================================================
def _write_png(path, img):
    try:
        ok, buf = cv2.imencode(".png", img)
        if ok:
            buf.tofile(path)                       # 한글 경로 호환
        else:
            print(f"[ERROR] PNG encode failed: {path}")
    except Exception as e:
        print(f"[ERROR] image save failed ({path}): {e}")
    finally:
        _save_slots.release()


def save_image_async(path, img):
    if not _save_slots.acquire(blocking=SAVE_BLOCK_WHEN_FULL):
        print(f"[WARN] image save queue full ({SAVE_MAX_PENDING}), skipped: {path}")
        return False
    try:
        _image_saver.submit(_write_png, path, img)
    except Exception:
        _save_slots.release()
        raise
    return True


# ======================================================
# Flask endpoints
# ======================================================
@app.route("/health", methods=["GET"])
def health():
    r = STATE["runner"]
    return jsonify({"status": "ok", "engine": os.path.basename(STATE["engine"]), "profile": r.profile,
                    "model_size": list(r.in_hw), "classes": list(CLASSES)})


@app.route("/predict", methods=["POST"])
def predict():
    t_req = time.perf_counter()
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
    try:
        shm_name = data["shm_name"]
        batch = int(data.get("batch", 1))
        width, height = int(data["width"]), int(data["height"])
        channel = int(data.get("image_channel", 3))
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"bad request: {e}"}), 400
    if batch < 1:
        return jsonify({"error": "batch must be >= 1"}), 400
    product_id = data.get("product_id") or datetime.now().strftime("%Y%m%d_%H%M%S")
    names = list(data.get("names") or [])
    names += [f"img{i}" for i in range(len(names), batch)]
    mask_shm = data.get("mask_shm_name")
    print(f"[REQ] {product_id}: batch {batch}, {width}x{height}x{channel}, shm {shm_name}" + (f", mask -> {mask_shm}" if mask_shm else ""))

    try:
        bgr = read_batch_from_shm(shm_name, batch, width, height, channel, data.get("rgba_order", "BGRA"))
    except FileNotFoundError:
        return jsonify({"error": f"Shared memory '{shm_name}' not found"}), 500
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    t0 = time.perf_counter()
    packed = predict_batch(bgr)                                     # (B,1,h,w) uint8
    infer_ms = (time.perf_counter() - t0) * 1e3
    small = unpack_bits(packed, N_CLASSES)                          # (B,K,h,w) {0,1}

    images_out, full_packed = [], []
    for i in range(batch):
        masks = np.stack([cv2.resize(small[i, k], (width, height), interpolation=cv2.INTER_NEAREST)
                          for k in range(N_CLASSES)])              # (K,H,W) 원본 해상도
        classes = {c: {"pixels": int(masks[k].sum()), "polygons": mask_polygons(masks[k])} for k, c in enumerate(CLASSES)}
        images_out.append({"index": i, "name": names[i], "classes": classes})
        if mask_shm:
            full_packed.append(pack_bits(masks)[0])                # (H,W) uint8
        if STATE["save_dir"]:
            stem = os.path.splitext(os.path.basename(str(names[i])))[0] or f"img{i}"
            save_image_async(os.path.join(STATE["save_dir"], f"{product_id}_{stem}_overlay.png"), overlay_masks(bgr[i], masks))
    if mask_shm:
        try:
            write_masks_to_shm(mask_shm, np.stack(full_packed))
        except (FileNotFoundError, ValueError) as e:
            return jsonify({"error": f"mask shared memory: {e}"}), 400

    total_ms = (time.perf_counter() - t_req) * 1e3
    print(f"[RES] {product_id}: infer {infer_ms:.1f} ms ({infer_ms / batch:.1f} ms/img), total {total_ms:.1f} ms | "
          + ", ".join(f"{im['name']}: " + "/".join(str(im['classes'][c]['pixels']) for c in CLASSES) for im in images_out))
    return jsonify({"product_id": product_id, "status": "OK", "message": "AI INFERENCE COMPLETED", "batch": batch,
                    "model_size": list(STATE["runner"].in_hw), "classes": list(CLASSES),
                    "infer_ms": round(infer_ms, 2), "total_ms": round(total_ms, 2), "images": images_out})


# ======================================================
def main():
    ap = argparse.ArgumentParser(description="INT8 TensorRT 엔진 웹 서비스 (공유메모리 + Flask, batch 추론)")
    ap.add_argument("--engine", default=ENGINE_INT8, help=f"TensorRT 엔진 (기본 {ENGINE_INT8})")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5004)
    ap.add_argument("--save-dir", default=None, help="지정하면 요청마다 오버레이 png 를 <save-dir>/<날짜>/ 에 저장 (백그라운드)")
    args = ap.parse_args()
    if not os.path.exists(args.engine):
        raise SystemExit(f"엔진이 없음: {args.engine} (_4_onnx2trt.py 를 먼저 실행)")
    sys.stdout.reconfigure(line_buffering=True)                      # 로그를 파일로 돌려도 요청마다 바로 기록되게
    STATE["engine"] = args.engine
    if args.save_dir:
        STATE["save_dir"] = os.path.join(args.save_dir, datetime.now().strftime("%Y%m%d"))
        os.makedirs(STATE["save_dir"], exist_ok=True)

    def _load():                                                     # 엔진 로드 + warm-up 도 추론 스레드에서
        r = TrtRunner(args.engine)
        for _ in range(3):                                           # 첫 요청의 지연(버퍼 할당, 커널 로드)을 기동 시에 흡수
            r.warmup(WARMUP_BATCH)
        return r
    t0 = time.perf_counter()
    STATE["runner"] = _infer_executor.submit(_load).result()
    r = STATE["runner"]
    print(f"[MODEL] {args.engine}: profile {r.profile}, input uint8 (B,{','.join(str(v) for v in r.in_chw)}) BGR, "
          f"output uint8 bit-packed ({', '.join(f'bit{i} {c}' for i, c in enumerate(CLASSES))}), "
          f"activation {r.activation_mib():.0f} MiB, load+warm-up {time.perf_counter() - t0:.1f}s")
    print(f"[INFO] http://{args.host}:{args.port}/predict  (GET /health)" + (f", overlay -> {STATE['save_dir']}" if STATE["save_dir"] else ""))
    try:
        app.run(host=args.host, port=args.port, debug=False, threaded=False)
    finally:
        _image_saver.shutdown(wait=True)
        _infer_executor.submit(r.close).result()
        _infer_executor.shutdown(wait=True)


if __name__ == "__main__":
    main()

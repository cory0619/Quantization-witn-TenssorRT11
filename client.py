r"""
client.py : server_inference.py 클라이언트. 폴더의 이미지를 BATCH(기본 4) 장씩 공유메모리로 보내고 polygon 응답을 받아 오버레이 / JSON 저장

- 한 요청 = 이미지 B장을 (B,H,W,3) uint8 BGR 로 공유메모리 하나에 쓰고 JSON POST. 한 배치의 이미지는 같은 크기여야 하므로
  크기가 다른 이미지는 배치 첫 장 크기로 resize 해 보내고, 응답 마스크를 원본 크기로 되돌린다.
- 응답의 polygon 을 클래스별 마스크로 다시 채운 뒤 eval_common.overlay_masks (학습 / _5 와 같은 색) 로 오버레이를 그린다.
  --mask-shm 을 주면 서버가 원본 해상도 비트 패킹 마스크를 두 번째 공유메모리로 돌려주며, 그 경우 polygon 대신 그 마스크로 그린다 (픽셀 단위로 정확).
- 저장: <out>/<날짜>/<이름>_overlay.png, <이름>.json (서버 응답의 해당 이미지 항목)

사용:
    python client.py                                             # dataset_pcb/test, 4장씩, http://127.0.0.1:5004/predict
    python client.py --dir D:\images --batch 4 --mask-shm --out client_result
"""
import argparse
import json
import multiprocessing.shared_memory as shared_mem
import os
import time
from datetime import datetime

import cv2
import numpy as np
import requests

from eval_common import TEST_DIR, CLASSES, overlay_masks
from model.bitmask import unpack_bits

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def imread_unicode(path, flag=cv2.IMREAD_COLOR):
    try:
        with open(path, "rb") as f:
            return cv2.imdecode(np.frombuffer(f.read(), np.uint8), flag)
    except Exception as e:
        print(f"[ERROR] Failed to read {path}: {e}")
        return None


def imwrite_unicode(path, img):
    ok, buf = cv2.imencode(os.path.splitext(path)[1], img)
    if ok:
        buf.tofile(path)
    return ok


def polygons_to_masks(classes_json, hw):
    """응답의 클래스별 polygon -> (K,H,W) uint8 {0,1}"""
    masks = np.zeros((len(CLASSES),) + tuple(hw), np.uint8)
    for k, c in enumerate(CLASSES):
        for poly in classes_json.get(c, {}).get("polygons", []):
            pts = np.array(poly, np.float64).reshape(-1, 2).round().astype(np.int32).reshape(-1, 1, 2)
            if len(pts) >= 3:
                cv2.fillPoly(masks[k], [pts], 1)
    return masks


def send_batch(url, images, names, shm_name, mask_shm_name=None, timeout=120):
    """images: 같은 크기의 (H,W,3) uint8 BGR 리스트. 반환: (응답 json, 서버 마스크 (B,K,H,W) 또는 None, 왕복 ms)"""
    b, (h, w) = len(images), images[0].shape[:2]
    shm = shared_mem.SharedMemory(name=shm_name, create=True, size=b * h * w * 3)
    mshm = None
    try:
        view = np.ndarray((b, h, w, 3), dtype=np.uint8, buffer=shm.buf)
        for i, im in enumerate(images):
            view[i] = im
        del view
        data = {"product_id": datetime.now().strftime("%Y%m%d_%H%M%S_%f"), "shm_name": shm_name, "batch": b,
                "width": w, "height": h, "image_channel": 3, "names": names}
        if mask_shm_name:
            mshm = shared_mem.SharedMemory(name=mask_shm_name, create=True, size=b * h * w)
            data["mask_shm_name"] = mask_shm_name
        t0 = time.perf_counter()
        resp = requests.post(url, json=data, timeout=timeout)       # 서버가 읽을 때까지 공유메모리를 열어 둬야 함 (finally 에서 close)
        rt_ms = (time.perf_counter() - t0) * 1e3
        if resp.status_code != 200:
            print(f"[ERROR] HTTP {resp.status_code}: {resp.text[:300]}")
            return None, None, rt_ms
        result = resp.json()
        masks = None
        if mshm is not None:
            packed = np.ndarray((b, h, w), dtype=np.uint8, buffer=mshm.buf).copy()
            masks = unpack_bits(packed[:, None], len(CLASSES))       # (B,K,H,W)
        return result, masks, rt_ms
    finally:
        shm.close()
        shm.unlink()                                                 # Windows 에서는 no-op (모든 핸들이 닫히면 해제)
        if mshm is not None:
            mshm.close()
            mshm.unlink()


def main():
    ap = argparse.ArgumentParser(description="server_inference.py 에 이미지를 batch 로 보내고 결과 저장")
    ap.add_argument("--dir", default=TEST_DIR, help=f"이미지 폴더 (기본 {TEST_DIR})")
    ap.add_argument("--url", default="http://127.0.0.1:5004/predict")
    ap.add_argument("--batch", type=int, default=4, help="한 요청에 보낼 장수 (기본 4)")
    ap.add_argument("--out", default="client_result", help="결과 폴더 (기본 client_result/<날짜>/)")
    ap.add_argument("--mask-shm", action="store_true", help="서버에서 비트 패킹 마스크도 공유메모리로 받음 (polygon 대신 그 마스크로 오버레이)")
    ap.add_argument("--shm-name", default="pcb_insp_shm", help="공유메모리 이름 접두어")
    args = ap.parse_args()

    paths = sorted(os.path.join(args.dir, f) for f in os.listdir(args.dir) if f.lower().endswith(IMAGE_EXTS))
    if not paths:
        raise SystemExit(f"이미지가 없음: {args.dir}")
    try:
        h = requests.get(args.url.rsplit("/", 1)[0] + "/health", timeout=5).json()
        print(f"[server] {h.get('engine')} profile {h.get('profile')}, model_size {h.get('model_size')}, classes {h.get('classes')}")
    except Exception as e:
        print(f"[WARN] /health 실패 ({e}). 서버가 떠 있는지 확인: python server_inference.py")
    out_dir = os.path.join(args.out, datetime.now().strftime("%Y%m%d"))
    os.makedirs(out_dir, exist_ok=True)

    t_all = time.perf_counter()
    n_done, sum_rt, sum_infer = 0, 0.0, 0.0
    for bi, start in enumerate(range(0, len(paths), args.batch)):
        group = paths[start:start + args.batch]
        originals, sent, names = [], [], []
        for p in group:
            im = imread_unicode(p)
            if im is None:
                continue
            originals.append(im)
            names.append(os.path.basename(p))
        if not originals:
            continue
        hw = originals[0].shape[:2]
        for im in originals:                                         # 배치 안 크기 통일 (첫 장 기준)
            if im.shape[:2] != hw:
                print(f"[WARN] 크기 {im.shape[:2]} != {hw}: 배치 첫 장 크기로 resize 해서 전송")
                im = cv2.resize(im, (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR)
            sent.append(np.ascontiguousarray(im))

        result, masks, rt_ms = send_batch(args.url, sent, names, f"{args.shm_name}_{bi}",
                                          f"{args.shm_name}_mask_{bi}" if args.mask_shm else None)
        if result is None:
            continue
        infer_ms = float(result.get("infer_ms", 0))
        print(f"[batch {bi}] {len(sent)} 장: 왕복 {rt_ms:.1f} ms ({rt_ms / len(sent):.1f} ms/img), 서버 추론 {infer_ms:.1f} ms ({infer_ms / len(sent):.1f} ms/img)")
        for i, (p, orig) in enumerate(zip(group, originals)):
            info = result["images"][i]
            m = masks[i] if masks is not None else polygons_to_masks(info["classes"], hw)   # (K,H,W) 전송 크기 기준
            overlay = overlay_masks(orig, m)                         # 원본 크기가 다르면 overlay_masks 가 nearest 로 맞춤
            stem = os.path.splitext(os.path.basename(p))[0]
            imwrite_unicode(os.path.join(out_dir, f"{stem}_overlay.png"), overlay)
            with open(os.path.join(out_dir, f"{stem}.json"), "w", encoding="utf-8") as f:
                json.dump(info, f, ensure_ascii=False)
            print("   " + f"{stem:16s}" + "  ".join(f"{c} {info['classes'][c]['pixels']:>7d}px/{len(info['classes'][c]['polygons'])}poly" for c in CLASSES))
        n_done += len(sent)
        sum_rt += rt_ms
        sum_infer += infer_ms
    if n_done:
        print(f"\n[done] {n_done} 장, 왕복 평균 {sum_rt / n_done:.1f} ms/img, 서버 추론 평균 {sum_infer / n_done:.1f} ms/img, "
              f"전체 {time.perf_counter() - t_all:.2f} s -> {out_dir}")


if __name__ == "__main__":
    main()

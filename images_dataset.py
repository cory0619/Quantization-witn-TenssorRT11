"""
images_dataset.py : labelme json -> segmentation mask Dataset  (2026-10-16 다중 클래스 / multi-label 지원)

세 가지 모드 (classes, multilabel 인자):
- 이진 (classes=None, 기존 동작): 모든 shape 를 전경으로. mask (1,H,W) float {0,1}  -> sigmoid + BCE/Dice
- multi-label (classes=[...], multilabel=True): 클래스마다 독립 채널. mask (K,H,W) float {0,1}  -> sigmoid + BCE/Dice
  한 픽셀이 여러 클래스에 동시에 속할 수 있음 (예: pcb 위의 hole 은 pcb=1, hole=1). dataset_pcb 가 이 경우.
- 배타적 다중 클래스 (classes=[...], multilabel=False): mask (H,W) int64, 배경 0, classes 순서대로 1..K  -> softmax + CrossEntropy
  classes 순서대로 그리므로 큰 영역을 앞에, 작은 영역을 뒤에 적을 것 (나중에 그린 것이 덮어씀). 겹침이 있는 데이터에는 쓰지 말 것.

공통:
- classes 에 없는 라벨은 경고 후 건너뜀 (strict_labels=True 면 예외)
- shape_type: polygon, rectangle(2점), circle(labelme: [중심, 원주 위 한 점]) 지원. 그 외(point, line 등)는 경고 후 건너뜀.
  * 예전 코드는 circle 도 fillPoly 로 그려서 선 하나만 남았음 (dataset_pcb 의 hole 라벨이 전부 유실되던 원인)
- 이미지: png / jpg / jpeg / bmp (대소문자 무관). 라벨은 같은 이름의 .json (UTF-8 BOM 허용)
- size=(H, W): 모델 입력 크기. R5 모델은 H, W 가 4 의 배수여야 함. 마스크 resize 는 albumentations 가 nearest 로 처리
- channels: 1 이면 grayscale (1,H,W), 3 이면 컬러 BGR (3,H,W). OpenCV 순서(BGR) 그대로 쓰며 배포 시 전처리도 같은 순서여야 함
"""
import json
import os
import warnings

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


class dataset_experiment(Dataset):
    def __init__(self, dataset_dir, augmentation=None, classes=None, size=(256, 256), multilabel=False, strict_labels=False, channels=1):
        if channels not in (1, 3):
            raise ValueError(f"channels must be 1 (grayscale) or 3 (BGR), got {channels}")
        self.channels = channels
        if not os.path.isdir(dataset_dir):
            raise FileNotFoundError(f"dataset dir not found: {dataset_dir}")
        files = sorted(os.listdir(dataset_dir))
        self.image_paths = [os.path.join(dataset_dir, f) for f in files if f.lower().endswith(IMAGE_EXTS)]
        if not self.image_paths:
            raise FileNotFoundError(f"no images {IMAGE_EXTS} in {dataset_dir}")
        self.label_paths = [os.path.splitext(p)[0] + ".json" for p in self.image_paths]
        missing = [p for p in self.label_paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"{len(missing)} label json missing, e.g. {missing[:3]}")

        self.augmentation = augmentation
        self.size = (int(size[0]), int(size[1]))
        self.base_transform = A.Compose([A.Resize(self.size[0], self.size[1], interpolation=cv2.INTER_LINEAR)])

        self.classes = list(classes) if classes else None
        self.multilabel = bool(multilabel) and self.classes is not None
        if self.classes:
            self.class_to_idx = {c: i for i, c in enumerate(self.classes)}          # multilabel: 채널 번호
            self.class_names = list(self.classes) if self.multilabel else ["background"] + self.classes
            self.num_classes = len(self.classes) if self.multilabel else len(self.classes) + 1   # 모델 n_classes
        else:
            self.class_to_idx = None
            self.class_names = ["foreground"]
            self.num_classes = 1
        self.strict_labels = strict_labels
        self._warned = set()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE if self.channels == 1 else cv2.IMREAD_COLOR)   # (H,W) 또는 (H,W,3) BGR
        if image is None:
            raise RuntimeError(f"Failed to load image: {image_path}")
        mask = self.get_mask(label_path)                        # (H,W) uint8  또는 multilabel (H,W,K) uint8
        if mask.shape[:2] != image.shape[:2]:
            raise RuntimeError(f"json imageHeight/Width {mask.shape[:2]} != image {image.shape[:2]}: {label_path}")

        transform = self.augmentation if self.augmentation else self.base_transform
        out = transform(image=image, mask=mask)                 # albumentations: (H,W,K) 마스크도 nearest 로 같이 변환
        image, mask = out["image"], out["mask"]
        if image.shape[:2] != self.size:                        # augmentation 에 Resize 가 없으면 여기서 맞춤
            out = self.base_transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]

        image = np.ascontiguousarray(image)
        if image.ndim == 2:
            image = torch.from_numpy(image).float().unsqueeze(0) / 255.0            # (1,H,W) 0~1
        else:
            image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0        # (3,H,W) 0~1, BGR
        mask = np.ascontiguousarray(mask)
        if self.multilabel:
            mask = (torch.from_numpy(mask).permute(2, 0, 1) > 0).float()                      # (K,H,W) {0,1}
        elif self.classes:
            mask = torch.from_numpy(mask.astype(np.int64))                                    # (H,W) long
        else:
            mask = (torch.from_numpy(mask).unsqueeze(0) > 0).float()                          # (1,H,W) {0,1}
        return image, mask, image_path

    # ------------------------------------------------------------------
    def get_mask(self, label_path):
        with open(label_path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        h, w = int(data["imageHeight"]), int(data["imageWidth"])
        shapes = list(data.get("shapes", []))

        if self.multilabel:
            mask = np.zeros((h, w, len(self.classes)), dtype=np.uint8)
            for shape in shapes:
                label = shape.get("label")
                if label not in self.class_to_idx:
                    self._warn(f"unknown label '{label}' skipped ({os.path.basename(label_path)})")
                    continue
                ch = np.ascontiguousarray(mask[..., self.class_to_idx[label]])
                self._draw(ch, shape, 1, label_path)
                mask[..., self.class_to_idx[label]] = ch
            return mask

        mask = np.zeros((h, w), dtype=np.uint8)
        if self.classes:
            # 클래스 인덱스 순서로 그려서 뒤 클래스(작은 영역)가 앞 클래스(큰 영역)를 덮어쓰게 함
            shapes.sort(key=lambda s: self.class_to_idx.get(s.get("label"), -1))
        for shape in shapes:
            label = shape.get("label")
            if self.classes:
                if label not in self.class_to_idx:
                    self._warn(f"unknown label '{label}' skipped ({os.path.basename(label_path)})")
                    continue
                value = self.class_to_idx[label] + 1
            else:
                value = 1
            self._draw(mask, shape, value, label_path)
        return mask

    def _draw(self, mask, shape, value, label_path):
        pts = np.asarray(shape.get("points", []), dtype=np.float32)
        st = shape.get("shape_type", "polygon")
        if st == "circle" and len(pts) == 2:
            (cx, cy), (px, py) = pts
            r = int(round(float(np.hypot(px - cx, py - cy))))
            cv2.circle(mask, (int(round(cx)), int(round(cy))), max(r, 1), int(value), thickness=-1)
        elif st == "rectangle" and len(pts) == 2:
            (x1, y1), (x2, y2) = pts.round().astype(np.int32)
            cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), int(value), thickness=-1)
        elif st in ("polygon", "rectangle") and len(pts) >= 3:
            cv2.fillPoly(mask, [pts.round().astype(np.int32)], int(value))
        else:
            self._warn(f"shape_type '{st}' with {len(pts)} points skipped ({os.path.basename(label_path)})")

    def _warn(self, msg):
        if self.strict_labels:
            raise ValueError(msg)
        if msg not in self._warned:
            self._warned.add(msg)
            warnings.warn(msg)
            print(f"[images_dataset] warn: {msg}")

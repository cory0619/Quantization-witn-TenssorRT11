r"""
model/bitmask.py : 배포 모델 출력 (비트 패킹 uint8 마스크) 의 공통 포맷

deploy=True 인 모델 (unet_convnext.UNet_ConvNeXt, multilabel) 은 클래스별 0/1 마스크를
한 픽셀 1 byte 로 합쳐 내보낸다:  값 = sum(mask_c << c)  ->  bit0 = 클래스 0, bit1 = 클래스 1, ...  (dataset_pcb: bit0 pcb, bit1 terminal, bit2 hole)
예) 5 = 0b101 -> pcb 이면서 hole, 3 = 0b011 -> pcb 이면서 terminal, 0 = 배경.
float32 3채널(12 byte/px) 대신 1 byte/px 라 서버 복사량이 1/12 이고 임계값 판정이 모델/엔진 안에서 끝난다.

unpack_bits 는 그 역변환. torch 텐서 / numpy 배열 모두 받는다. eval_common.unpack_masks 는 같은 연산의 numpy(bool) 판.
"""
import sys

import numpy as np


def unpack_bits(packed, n_classes):
    """(…,1,H,W) uint8 비트 패킹 -> (…,K,H,W) {0,1} uint8. bit c = 클래스 c"""
    torch = sys.modules.get("torch")             # 이미 import 된 경우에만 torch 텐서 검사 (서버처럼 torch 를 안 쓰는 쪽에서 불러오지 않음)
    if torch is not None and isinstance(packed, torch.Tensor):
        shifts = torch.arange(n_classes, device=packed.device, dtype=torch.int32).view(*([1] * (packed.dim() - 3)), n_classes, 1, 1)
        return ((packed.to(torch.int32) >> shifts) & 1).to(torch.uint8)
    packed = np.asarray(packed)
    shifts = np.arange(n_classes, dtype=np.uint8).reshape(*([1] * (packed.ndim - 3)), n_classes, 1, 1)
    return ((packed.astype(np.uint8) >> shifts) & 1).astype(np.uint8)


def pack_bits(masks):
    """(…,K,H,W) {0,1} -> (…,1,H,W) uint8 비트 패킹 (unpack_bits 의 역). numpy 전용 (서버에서 원본 해상도 마스크를 되돌려 줄 때)"""
    masks = np.asarray(masks)
    k = masks.shape[-3]
    if k > 8:
        raise ValueError("uint8 비트 패킹은 클래스 8개까지")
    weights = (1 << np.arange(k, dtype=np.uint8)).reshape(*([1] * (masks.ndim - 3)), k, 1, 1)
    return ((masks > 0).astype(np.uint8) * weights).sum(axis=-3, keepdims=True).astype(np.uint8)

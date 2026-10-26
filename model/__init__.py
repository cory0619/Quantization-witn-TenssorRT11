r"""
model : 추론 모델 패키지

    unet_convnext.py  R6 UNet_ConvNeXt (ConvNeXt-Base 백본 + 경량 디코더). 학습(train_test_fp32_pcb.py) 과 export(_1) 가 사용
    bitmask.py        deploy 출력 비트 패킹 마스크 포맷 (unpack_bits / pack_bits). 두 모델과 서버가 공유
    trt_runner.py     TensorRT 엔진 실행기 TrtRunner (_4, _5, server_inference.py 가 공유)
    docs/             모델 구조 설명 html

import 는 지연(lazy) 이다: `from model import TrtRunner` 는 torch/timm 을 불러오지 않고, `from model import UNet_ConvNeXt` 때만 불러온다.
자체 테스트는 패키지 상대 import 때문에 `python -m model.unet_convnext` 로 실행한다.
"""
import importlib

_LAZY = {
    "UNet_ConvNeXt": ".unet_convnext", "load_pretrained_encoder": ".unet_convnext",
    "ensure_pretrained": ".unet_convnext", "PRETRAINED_PATH_DEFAULT": ".unet_convnext",
    "unpack_bits": ".bitmask", "pack_bits": ".bitmask",
    "TrtRunner": ".trt_runner",
}
__all__ = list(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module 'model' has no attribute '{name}'")

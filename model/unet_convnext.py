"""
model/unet_convnext.py : ConvNeXt 백본 + UNet 식 경량 디코더 (R6, 2026-10-17)

왜: R5 (37k 파라미터, 처음부터 학습) 로는 dataset_pcb (학습 22장) 에서 hole IoU 0.60, terminal 0.88 에서 멈춤.
    참조 코드 E:\\forgeAI_practice\\ForgeAIServer_v1.0.1\\train_test_forgeai.py 는 smp.DeepLabV3Plus(encoder_name="tu-convnext_base") 로
    ImageNet-22k 사전학습 ConvNeXt-Base 를 encoder(특징 추출기)로만 쓴다. 여기서는 그 백본만 차용하고 디코더는 R5 식으로 둔다.

구조 (입력 H, W 는 32 의 배수):
    입력 (B,C,H,W) 0~1 (deploy 면 uint8) -> ImageNet mean/std 정규화 (모델 안, BGR 순서 버퍼)
    ├ 경량 full-res 가지 (R5 와 동일):  inc  3x3 conv -> s1 (B,16,H,W)
    │                                   down1 3x3 s2 conv -> s2 (B,32,H/2,W/2)
    └ ConvNeXt 백본 (timm features_only, "tu-convnext_base" 의 실체):  f4 (B,128,H/4), f8 (B,256,H/8), f16 (B,512,H/16), f32 (B,1024,H/32)
    디코더: Up = nearest x2 -> 1x1 reduce -> (+ 1x1 lateral(skip)) -> 3x3 conv+BN+ReLU   (concat 대신 덧셈, R5 와 같은 이유: INT8 엔진의 requantize/transpose 절약)
        up1 f32+f16 -> 256 @1/16,  up2 +f8 -> 128 @1/8,  up3 +f4 -> 64 @1/4,  up4 +s2 -> 32 @1/2,  up5 +s1 -> 16 @1/1
    outc 1x1 -> logits (B,n_classes,H,W)
    * DeepLabV3+ 와 달리 ASPP 가 없고, 1/4 에서 x4 bilinear 로 끝내지 않고 full-res skip(s1, s2) 으로 경계를 복원한다 (hole 지름 ~50px 경계).
    * ConvNeXt 는 stride-4 patchify stem 이라 1/2, 1/1 특징이 없다. 그래서 inc/down1 가지를 따로 둔다.

사전학습: load_pretrained_encoder(model, path). 참조 코드와 같은 safetensors (timm/convnext_base.fb_in22k_ft_in1k) 를 key 변환
    ("stages.0." -> "stages_0.") 해서 encoder 에만 로드. head.* (ImageNet 분류기) 는 버림. stem conv 는 RGB 로 학습되어 있으므로
    - n_channels=3: 데이터셋이 BGR 이라 stem 가중치의 입력 채널 순서를 [2,1,0] 으로 바꿔 로드 (런타임 채널 flip 불필요)
    - n_channels=1: RGB 3 채널 가중치를 합산 (참조 코드와 동일)

학습/배포 인터페이스는 R5 와 같다: deploy=False -> logits, deploy=True -> uint8 입력 / uint8 마스크(비트 패킹) 출력. unpack_bits 로 복원.
파라미터: ConvNeXt-Base 백본 87.6M + 디코더 약 2M. 배포 비용은 R5 의 수백 배이므로 속도가 필요하면 encoder_name="convnext_tiny" 등으로 바꿀 것
(사전학습 파일은 그 모델 것으로 따로 받아야 함).
자체 테스트: python -m model.unet_convnext  (패키지 상대 import 라 파일 직접 실행은 안 됨)
"""
import math
import os
import re

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .bitmask import unpack_bits  # noqa: F401  (deploy 출력 복원. 모델 공통 포맷이라 bitmask.py 로 분리)

# 참조 코드가 쓰는 사전학습 파일 (timm/convnext_base.fb_in22k_ft_in1k, 338 MB). 없으면 ensure_pretrained() 로 내려받음
PRETRAINED_PATH_DEFAULT = r"E:\forgeAI_practice\ForgeAIServer_v1.0.1\weights\model.safetensors"
HF_REPO_ID = "timm/convnext_base.fb_in22k_ft_in1k"
HF_FILENAME = "model.safetensors"

# ImageNet 정규화 상수 (RGB 순서). 백본이 이 분포로 학습되었으므로 입력도 같은 분포로 맞춰야 사전학습 특징이 그대로 쓰인다
IMAGENET_MEAN_RGB = (0.485, 0.456, 0.406)
IMAGENET_STD_RGB = (0.229, 0.224, 0.225)


class ConvBlock(nn.Module):
    """dense 3x3 conv + BN + ReLU (stride 2 면 다운샘플 겸용). R5 와 동일"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Up(nn.Module):
    """nearest 2x -> 1x1 conv(reduce) 로 out 채널 -> skip 은 1x1 conv(lateral) 로 out 채널 (채널이 같으면 생략) -> 덧셈 -> 3x3 ConvBlock"""
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.lateral = nn.Conv2d(skip_channels, out_channels, kernel_size=1, bias=False) if skip_channels != out_channels else nn.Identity()
        self.conv = ConvBlock(out_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(self.reduce(x) + self.lateral(skip))


class UNet_ConvNeXt(nn.Module):
    SIZE_MULTIPLE = 32   # 입력 H, W 조건 (ConvNeXt 1/32 특징을 nearest x2 로 다섯 번 올려 skip 과 크기를 맞추려면)

    def __init__(self, n_channels=3, n_classes=1, multilabel=False,
                 deploy=False, deploy_mask=True, deploy_thresh=0.5,
                 encoder_name="convnext_base", drop_path_rate=0.1, decoder_channels=(256, 128, 64, 32, 16)):
        """
        n_channels   : 1 (grayscale) 또는 3 (컬러, 데이터셋과 같은 BGR 순서)
        n_classes    : 출력 채널 수. multilabel=True 면 클래스 수 (채널마다 sigmoid), False 면 1=이진 / K+1=배타적(softmax)
        deploy       : True 면 uint8 입력 (0~255) -> deploy_mask 에 따라 uint8 마스크 (비트 패킹 / class map) 또는 float logits. R5 와 동일
        encoder_name : timm 모델 이름. 참조 코드의 "tu-convnext_base" = timm "convnext_base" (smp 의 tu- 접두어는 timm universal encoder)
        drop_path_rate : 백본 stochastic depth (학습 시만). 학습 이미지가 적을 때 과적합 억제. eval/export 에서는 identity
        decoder_channels : (1/16, 1/8, 1/4, 1/2, 1/1) 디코더 폭
        """
        super().__init__()
        if n_channels not in (1, 3):
            raise ValueError(f"n_channels must be 1 (grayscale) or 3 (BGR), got {n_channels}")
        if multilabel and n_classes > 8:
            raise ValueError("multilabel deploy 는 uint8 비트 패킹이라 n_classes <= 8")
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.multilabel = multilabel
        self.deploy = deploy
        self.deploy_mask = deploy_mask
        self.encoder_name = encoder_name
        self.logit_thresh = math.log(deploy_thresh / (1.0 - deploy_thresh))   # sigmoid(l) > p  <=>  l > log(p/(1-p))
        self.register_buffer("bit_weights", (2.0 ** torch.arange(n_classes)).view(1, n_classes, 1, 1), persistent=False)

        # 입력 정규화 버퍼 (데이터셋 채널 순서 = BGR 이므로 RGB 상수를 뒤집어 저장. 1채널은 RGB 평균)
        if n_channels == 3:
            mean, std = IMAGENET_MEAN_RGB[::-1], IMAGENET_STD_RGB[::-1]
        else:
            mean, std = (sum(IMAGENET_MEAN_RGB) / 3.0,), (sum(IMAGENET_STD_RGB) / 3.0,)
        self.register_buffer("in_mean", torch.tensor(mean).view(1, n_channels, 1, 1), persistent=False)
        self.register_buffer("in_inv_std", (1.0 / torch.tensor(std)).view(1, n_channels, 1, 1), persistent=False)

        # ConvNeXt 백본: 분류 head 없이 4 단계 특징맵만 (timm features_only). 참조 코드의 smp TimmUniversalEncoder 가 내부에서 만드는 것과 같은 객체
        self.encoder = timm.create_model(encoder_name, pretrained=False, features_only=True, out_indices=(0, 1, 2, 3),
                                         in_chans=n_channels, drop_path_rate=drop_path_rate)
        enc_ch = list(self.encoder.feature_info.channels())        # convnext_base: [128, 256, 512, 1024]
        reductions = list(self.encoder.feature_info.reduction())   # [4, 8, 16, 32]
        if reductions != [4, 8, 16, 32]:
            raise ValueError(f"{encoder_name}: feature strides {reductions} != [4, 8, 16, 32]")
        self.encoder_channels = enc_ch

        d16, d8, d4, d2, d1 = decoder_channels
        self.inc = ConvBlock(n_channels, d1)            # full-res 가지 (ConvNeXt 에 없는 1/1, 1/2 특징)
        self.down1 = ConvBlock(d1, d2, stride=2)
        self.up1 = Up(enc_ch[3], enc_ch[2], d16)        # 1/32 -> 1/16
        self.up2 = Up(d16, enc_ch[1], d8)               # 1/16 -> 1/8
        self.up3 = Up(d8, enc_ch[0], d4)                # 1/8  -> 1/4
        self.up4 = Up(d4, d2, d2)                       # 1/4  -> 1/2  (+ down1)
        self.up5 = Up(d2, d1, d1)                       # 1/2  -> 1/1  (+ inc)
        self.outc = nn.Conv2d(d1, n_classes, kernel_size=1)

    # 학습률을 다르게 줄 때 쓰는 파라미터 그룹
    def encoder_parameters(self):
        return self.encoder.parameters()

    def decoder_parameters(self):
        enc = set(id(p) for p in self.encoder.parameters())
        return [p for p in self.parameters() if id(p) not in enc]

    def forward(self, x):
        if x.shape[-2] % self.SIZE_MULTIPLE or x.shape[-1] % self.SIZE_MULTIPLE:
            raise ValueError(f"input H, W must be multiples of {self.SIZE_MULTIPLE}, got {tuple(x.shape[-2:])}")
        if self.deploy:
            x = x.float() * (1.0 / 255.0)               # uint8 원본 픽셀 -> 0~1
        x = (x - self.in_mean) * self.in_inv_std        # ImageNet 정규화 (전처리를 모델 안으로)
        s1 = self.inc(x)                                # (B,16,H,W)
        s2 = self.down1(s1)                             # (B,32,H/2,W/2)
        f4, f8, f16, f32 = self.encoder(x)              # (B,128,H/4) (B,256,H/8) (B,512,H/16) (B,1024,H/32)
        y = self.up1(f32, f16)                          # (B,256,H/16)
        y = self.up2(y, f8)                             # (B,128,H/8)
        y = self.up3(y, f4)                             # (B,64,H/4)
        y = self.up4(y, s2)                             # (B,32,H/2)
        y = self.up5(y, s1)                             # (B,16,H)
        logits = self.outc(y)                           # (B,n_classes,H,W)
        if self.deploy and self.deploy_mask:
            if self.n_classes == 1:
                return (logits > self.logit_thresh).to(torch.uint8)
            if self.multilabel:
                bits = (logits > self.logit_thresh).to(logits.dtype)
                return (bits * self.bit_weights).sum(dim=1, keepdim=True).to(torch.uint8)   # 비트 패킹 (B,1,H,W)
            return logits.argmax(dim=1, keepdim=True).to(torch.uint8)
        return logits


# ----------------------------------------------------------------------------------------------------------------------
# 사전학습 백본 로딩 (참조 코드 load_pretrained_encoder / pretrained_weights.py 의 역할)
# ----------------------------------------------------------------------------------------------------------------------
def ensure_pretrained(path=PRETRAINED_PATH_DEFAULT, auto_download=False, download_dir="weights"):
    """path 의 safetensors 가 읽히면 그 경로를, 없고 auto_download 면 HF 에서 download_dir 로 내려받아 그 경로를 돌려준다. 실패 시 None"""
    def ok(p):
        try:
            from safetensors import safe_open
            with safe_open(p, framework="pt") as f:
                return len(list(f.keys())) > 0
        except Exception:
            return False
    if path and os.path.exists(path) and ok(path):
        return path
    local = os.path.join(download_dir, HF_FILENAME)
    if os.path.exists(local) and ok(local):
        return local
    if not auto_download:
        print(f"[pretrained] not found: {path} (auto_download=False) -> encoder 는 무작위 초기화")
        return None
    try:
        from huggingface_hub import hf_hub_download
        print(f"[pretrained] downloading {HF_REPO_ID}/{HF_FILENAME} (~338 MB) -> {download_dir}")
        got = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME, local_dir=download_dir)
        return got if ok(got) else None
    except Exception as e:
        print(f"[pretrained] download failed: {type(e).__name__}: {e}")
        return None


def load_pretrained_encoder(model, path=PRETRAINED_PATH_DEFAULT, bgr_input=True, verbose=True):
    """
    timm ConvNeXt safetensors (분류 모델 전체) -> model.encoder (features_only) 에 로드.
    - key: "stages.0.blocks..." -> "stages_0.blocks..." (timm FeatureListNet 이 최상위 Sequential 을 '_' 로 평탄화). "head.*" 는 버림
    - stem conv (out,3,4,4) 는 RGB 로 학습됨: n_channels=3 이고 bgr_input 이면 입력 채널을 [2,1,0] 으로 재배열 (데이터셋이 OpenCV BGR),
      n_channels=1 이면 3채널 합산 (참조 코드와 동일)
    반환: (matched, total_encoder_tensors)
    """
    from safetensors.torch import load_file
    pre = load_file(path)
    enc_sd = model.encoder.state_dict()
    remapped, stem_note = {}, ""
    for k, v in pre.items():
        if k.startswith("head."):
            continue
        nk = re.sub(r"^(stages|stem)\.(\d+)\.", r"\1_\2.", k)
        if nk not in enc_sd:
            continue
        t = enc_sd[nk].shape
        is_stem_conv = nk == "stem_0.weight" and v.ndim == 4 and v.shape[1] == 3
        if is_stem_conv and t[1] == 3 and bgr_input:
            remapped[nk] = v[:, [2, 1, 0]].contiguous()          # RGB -> BGR
            stem_note = f"stem {k}: RGB -> BGR 입력 채널 재배열"
        elif is_stem_conv and t[1] == 1 and v.shape[0] == t[0] and v.shape[2:] == t[2:]:
            remapped[nk] = v.sum(dim=1, keepdim=True)             # RGB -> gray
            stem_note = f"stem {k}: RGB 3채널 합산 -> 1채널"
        elif v.shape == t:
            remapped[nk] = v
    missing = [k for k in enc_sd if k not in remapped]
    enc_sd.update(remapped)
    model.encoder.load_state_dict(enc_sd, strict=True)
    if verbose:
        print(f"[pretrained] {os.path.basename(path)}: matched {len(remapped)}/{len(enc_sd)} encoder tensors"
              f" ({sum(1 for k in pre if k.startswith('head.'))} head tensors skipped){'; ' + stem_note if stem_note else ''}")
        if missing:
            print(f"[pretrained] *** {len(missing)} encoder tensors NOT loaded (random init): {missing[:8]}")
    if not remapped:
        raise RuntimeError("pretrained: 0 tensors matched (key 변환 확인 필요)")
    return len(remapped), len(enc_sd)


if __name__ == "__main__":
    import time
    m = UNet_ConvNeXt(3, 3, multilabel=True).eval()
    n = lambda mod: sum(p.numel() for p in mod.parameters())
    print(f"params total {n(m):,} | encoder {n(m.encoder):,} | decoder {n(m) - n(m.encoder):,} | encoder channels {m.encoder_channels}")
    p = ensure_pretrained()
    if p:
        load_pretrained_encoder(m, p)
    with torch.no_grad():
        for h, w in [(256, 320), (1088, 1344)]:
            x = torch.rand(1, 3, h, w)
            t0 = time.perf_counter(); y = m(x)
            print(f"train-mode in (1,3,{h},{w}) -> out {tuple(y.shape)} {y.dtype}  ({time.perf_counter() - t0:.2f}s cpu)")
        u8 = torch.randint(0, 256, (1, 3, 256, 320), dtype=torch.uint8)
        ref = (torch.sigmoid(m(u8.float() / 255.0)) > 0.5).to(torch.uint8)
        m.deploy = True
        y = m(u8)
        print(f"deploy in uint8 (1,3,256,320) -> out {y.dtype} {tuple(y.shape)}, values {sorted(y.unique().tolist())}, "
              f"unpack mismatch vs per-channel threshold: {(unpack_bits(y, 3) != ref).sum().item()} px")
        try:
            m(torch.randint(0, 256, (1, 3, 256, 300), dtype=torch.uint8))   # 300 은 32 의 배수가 아님
        except ValueError as e:
            print("expected error:", e)

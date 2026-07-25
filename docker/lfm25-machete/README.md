# LFM2.5 + Machete int4 RTN — tái lập `ductransa01/vllm-lfm25-machete:r2v18`

Nhánh này tái lập chính xác image dùng cho Viettel AI Race vòng 2
(model `LiquidAI/LFM2.5-1.2B-Instruct`, 1 slice MiG H200 1g.18gb / 3 vCPU / 8GB RAM).

Base: `vllm/vllm-openai:v0.22.1` (đúng bản BTC phát) — xác minh trong image:
`importlib.metadata.version("vllm") == "0.22.1"`.

## Chuỗi build

```
vllm/vllm-openai:v0.22.1
  + machete_rtn.py  +  hook `import machete_rtn` nối vào vllm/__init__.py
  = ductransa01/vllm-lfm25-machete:r2                (docker/lfm25-machete/Dockerfile.base-r2)
  + machete_rtn.py v2 (thêm lm_head, tie-safe)
  = ductransa01/vllm-lfm25-machete:r2v18             (docker/lfm25-machete/Dockerfile.r2v18)
```

## `machete_rtn.py` làm gì

Lượng tử hoá int4 RTN **online, trong bộ nhớ** (luật vòng thi chỉ cho phép online quantization,
trọng số do BTC mount ở `/model` và không được sửa). Nó monkeypatch:

- `UnquantizedLinearMethod.process_weights_after_loading` → đóng gói weight sang buffer machete
  (`ops.machete_prepack_B`), rồi giải phóng weight bf16;
- `UnquantizedLinearMethod.apply` → `ops.machete_mm`;
- `UnquantizedEmbeddingMethod.apply` → int4 cho phép nhân logits của `lm_head`, **giữ**
  `layer.weight` ở bf16 để phép tra cứu embedding (tied) không vỡ.

Chọn layer qua `VLLM_MACHETE_RTN` (khớp chuỗi con trên `layer.prefix`), `VLLM_MACHETE_BITS`,
`VLLM_MACHETE_GS`.

## Đo được (25/07/2026, chạy trực tiếp image trên node)

`VLLM_MACHETE_RTN=feed_forward.w,self_attn.,conv.in_proj,conv.out_proj,lm_head`

| Cấu hình | Số layer machete bắt được |
|---|---|
| **có** `--quantization=fp8` | **21** — 10 `conv.in_proj`, 10 `conv.out_proj`, 1 `lm_head` |
| **không** `--quantization=fp8` | **65** — thêm 16 `feed_forward.w13`, 16 `feed_forward.w2`, 6 `self_attn.qkv_proj`, 6 `self_attn.out_proj` |

Vì `machete_rtn` chỉ hook `UnquantizedLinearMethod`, còn `--quantization=fp8` gán
`Fp8LinearMethod` cho các linear layer. Trên `v0.22.1`, `ShortConv` **không** truyền `quant_config`
xuống `in_proj`/`out_proj` (sửa ở PR #48917 của upstream, merged 2026-07-21, **không** có trong
`v0.22.1`) nên hai lớp đó còn Unquantized và machete bắt được; `lm_head` cũng vậy qua
`UnquantizedEmbeddingMethod`.

Hệ quả: dùng kèm `--quantization=fp8` thì MLP (~75% lưu lượng weight) chạy **fp8, không phải int4**.
Byte/step: 1079 MB (kèm fp8) so với 627 MB (không kèm).

## Giấy phép

Tệp bắt nguồn từ vLLM giữ nguyên giấy phép Apache-2.0 của dự án.

## Khác biệt so với image đã ship

Nhánh này **không** giống image `:r2v18` từng byte. Ba thay đổi, đều theo review, đều không đổi
hành vi khi `VLLM_MACHETE_RTN` được set (tức là mọi lần chạy thi):

1. `vllm/__init__.py` — bọc `import machete_rtn` trong điều kiện `VLLM_MACHETE_RTN` được set.
   Image ship bản không điều kiện, nên `import vllm` từ wheel bình thường (không có
   `machete_rtn.py`) sẽ in `machete_rtn import err ...` ra stdout.
2. `machete_rtn.py` — `except` khi khởi tạo giờ `raise` lại sau khi log. Bản ship chỉ in rồi chạy
   tiếp với method chưa vá, nghĩa là một lần chạy khai là int4 có thể âm thầm chạy bf16/fp8 mà
   benchmark vẫn ra số bình thường.
3. `machete_rtn.py` — thêm báo cáo độ phủ theo từng target ở lần forward đầu tiên, và cảnh báo to
   khi một target khớp 0 layer.

Điểm 3 sinh ra từ một lỗi có thật: cấu hình thi chạy
`VLLM_MACHETE_RTN=feed_forward.w,self_attn.,conv.in_proj,conv.out_proj,lm_head` **kèm**
`--quantization=fp8` suốt nhiều tuần, tin rằng mình đang chạy "int4-everything". Thực tế
`feed_forward.w` và `self_attn.` khớp 0 layer (đã bị `Fp8LinearMethod` chiếm) nên MLP — 75% lưu
lượng weight — chạy fp8. Không có lỗi nào được ném ra, không có cảnh báo nào, số benchmark trông
hoàn toàn hợp lý. Với bản vá này log sẽ in thẳng:

```
[machete_rtn] coverage: {'feed_forward.w': 0, 'self_attn.': 0, 'conv.in_proj': 10, 'conv.out_proj': 10, 'lm_head': 0}
[machete_rtn] WARNING: targets matched 0 layers: ['feed_forward.w', 'self_attn.'] ...
```

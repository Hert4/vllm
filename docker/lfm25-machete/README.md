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

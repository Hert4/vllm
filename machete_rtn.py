"""In-memory Machete int4 RTN weight-only quant. v2 = + lm_head (tied-safe, lazy).

Linear layers (MLP/attn/conv): hook UnquantizedLinearMethod (as v1).
lm_head: hook UnquantizedEmbeddingMethod.apply — quantize the LOGITS matmul weight to int4
lazily on first call, keep layer.weight bf16 so the tied embedding gather (.embedding) is intact.
Target lm_head by adding 'lm_head' to VLLM_MACHETE_RTN.
"""
import os
_T = [t for t in os.environ.get("VLLM_MACHETE_RTN", "").split(",") if t]
print(f"[machete_rtn] imported v2, TARGETS={_T}", flush=True)
if _T:
    try:
        import torch
        import vllm._custom_ops as ops
        from vllm.scalar_type import scalar_types
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod
        from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod
        from vllm.model_executor.layers.quantization.utils.quant_utils import pack_quantized_values_into_int32
        from vllm.model_executor.layers.quantization.utils.marlin_utils_test import gptq_quantize_weights
        WT = scalar_types.uint8b128 if os.environ.get("VLLM_MACHETE_BITS", "4") == "8" else scalar_types.uint4b8
        GS = int(os.environ.get("VLLM_MACHETE_GS", "128"))

        def _is_t(layer):
            p = getattr(layer, "prefix", "") or ""
            return any(t in p for t in _T)

        def _mk_machete(w):
            # w: [out, in] bf16 -> machete int4 buffers (wq_m, scales, gs)
            wk = w.t().contiguous()
            _wref, q_w, s, _g, _p = gptq_quantize_weights(wk, WT, GS, False)
            wq_packed = pack_quantized_values_into_int32(q_w, WT, packed_dim=0)
            wq_m = ops.machete_prepack_B(wq_packed.t().contiguous().t(),
                                         a_type=w.dtype, b_type=WT, group_scales_type=w.dtype)
            return (wq_m, s.contiguous(), GS)

        def _machete_mm(m, x, bias=None):
            wq_m, s, gs = m
            x2d = x.reshape(-1, x.shape[-1])
            out = ops.machete_mm(a=x2d, b_q=wq_m, b_type=WT, b_group_zeros=None,
                                 b_group_scales=s, b_group_size=gs)
            if bias is not None:
                out.add_(bias)
            return out.reshape(x.shape[:-1] + (out.shape[-1],))

        # ---- Linear (MLP/attn/conv): eager, free bf16 weight (v1 behavior) ----
        _op = UnquantizedLinearMethod.process_weights_after_loading
        _oa = UnquantizedLinearMethod.apply

        def _lin_process(self, layer):
            _op(self, layer)
            if not _is_t(layer):
                return
            w = layer.weight.data
            layer._machete = _mk_machete(w)
            layer.weight = torch.nn.Parameter(torch.empty(0, device=w.device, dtype=w.dtype),
                                              requires_grad=False)
            print(f"[machete_rtn] quantized {getattr(layer,'prefix','?')}", flush=True)

        def _lin_apply(self, layer, x, bias=None):
            m = getattr(layer, "_machete", None)
            if m is None:
                return _oa(self, layer, x, bias)
            return _machete_mm(m, x, bias)

        UnquantizedLinearMethod.process_weights_after_loading = _lin_process
        UnquantizedLinearMethod.apply = _lin_apply

        # ---- lm_head (Embedding.apply = logits matmul): lazy, KEEP bf16 weight (tie-safe) ----
        # NOTE: VocabParallelEmbedding has NO self.prefix, and .apply() is ONLY ever called for
        # the lm_head logits matmul (embed_tokens lookup uses .embedding()). So gate on a global
        # flag, not layer.prefix.
        _LM = any("lm_head" in t for t in _T)
        _ea = UnquantizedEmbeddingMethod.apply

        def _emb_apply(self, layer, x, bias=None):
            if not _LM:
                return _ea(self, layer, x, bias)
            m = getattr(layer, "_machete_lm", None)
            if m is None:
                m = _mk_machete(layer.weight.data)  # keep layer.weight for .embedding() gather
                layer._machete_lm = m
                print(f"[machete_rtn] quantized(lm) {getattr(layer,'prefix','?')} "
                      f"shape={tuple(layer.weight.shape)}", flush=True)
            return _machete_mm(m, x, bias)

        UnquantizedEmbeddingMethod.apply = _emb_apply

        print(f"[machete_rtn] PATCH v2 APPLIED targets={_T} GS={GS} bits={'8' if WT==scalar_types.uint8b128 else '4'}", flush=True)
    except Exception as e:
        import traceback
        print("[machete_rtn] FAIL:", e)
        traceback.print_exc()

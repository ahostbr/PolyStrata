"""Write a deterministic MiMo GGUF and NumPy reference logits (no downloaded weights).

python tools/make_tiny_mimo.py --gguf-py PATH/TO/llama.cpp/gguf-py --out build-mimo/tiny-mimo
Requires NumPy and the pinned gguf-py checkout used by the build. Exercises a dense
layer, SWA with sinks, full GQA, unequal K/V widths, biased sigmoid top-k routing,
Q8_0 experts, and metadata for three skipped MTP blocks. --gsq uses synthetic
Q2_K/Q3_K/MXFP4 experts, including both directions of mixed gate/up formats,
and Q8_0 dense matrices, matching the target formats. NumPy dequantizes these
blocks for its reference. Not a quality benchmark.
"""
import argparse
from pathlib import Path
import sys
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf-py", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gsq", action="store_true", help="exercise GSQ-RCO mixed tensor types")
    args = parser.parse_args()
    sys.path.insert(0, str(args.gguf_py))
    import gguf

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(args.out) + ".gguf", "mimo2")
    integers = dict(block_count=4 if args.gsq else 7, nextn_predict_layers=0 if args.gsq else 3, context_length=8192,
                    embedding_length=256, feed_forward_length=256, vocab_size=64,
                    expert_count=4, expert_used_count=2, expert_feed_forward_length=256,
                    expert_gating_func=2, **{"attention.head_count": 4,
                    "attention.key_length": 192, "attention.value_length": 128,
                    "attention.sliding_window": 128, "rope.dimension_count": 64})
    for key, val in integers.items():
        writer.add_uint32("mimo2." + key, val)
    for key, val in {"rope.freq_base": 1e7, "rope.freq_base_swa": 1e4,
                     "attention.layer_norm_rms_epsilon": 1e-6,
                     "attention.value_scale": .707}.items():
        writer.add_float32("mimo2." + key, val)
    writer.add_array("mimo2.attention.head_count_kv", [1, 2, 1, 2, 2, 2, 2])
    writer.add_array("mimo2.attention.sliding_window_pattern", [0, 1, 0, 1, 1, 1, 1])
    rng = np.random.default_rng(2606)
    weights = {}

    def add(name, shape, norm=False, quant=False, scale=.025):
        x = (rng.standard_normal(shape) * scale + (1 if norm else 0)).astype(np.float32)
        mixed = None
        if args.gsq:
            if "_exps.weight" in name:
                il = int(name.split(".")[1])
                kind = "gate" if "gate" in name else "up" if "up" in name else "down"
                mixed = {1: {"gate": "Q2_K", "up": "Q3_K", "down": "Q2_K"},
                         2: {"gate": "Q3_K", "up": "Q2_K", "down": "Q3_K"},
                         3: {"gate": "Q3_K", "up": "Q3_K", "down": "MXFP4"}}[il][kind]
            elif len(shape) == 2 and "ffn_gate_inp" not in name:
                quant = True
        if mixed:
            qt = getattr(gguf.GGMLQuantizationType, mixed)
            block, size = gguf.GGML_QUANT_SIZES[qt]
            raw = rng.integers(0, 256, (int(np.prod(shape)) // block, size), dtype=np.uint8)
            offset = size - 4 if mixed == "Q2_K" else size - 2
            if mixed == "MXFP4":
                raw[:, 0] = 120  # finite E8M0 scale, representable exactly in the Q5_0 load conversion
            else:
                raw[:, offset:offset+2] = np.array([1.0], dtype=np.float16).view(np.uint8)
                if mixed == "Q2_K":
                    # Q2_K keeps d/dmin after its scales and 2-bit codes.
                    raw[:, offset:offset+4] = np.array([1.0, 1.0], dtype=np.float16).view(np.uint8)
            q = raw.reshape(*shape[:-1], shape[-1] // block * size)
            decoded = gguf.dequantize(q, qt)
            if mixed != "MXFP4":
                raw[:, offset:offset+2] = np.array([scale / decoded.std()], dtype=np.float16).view(np.uint8)
                if mixed == "Q2_K":
                    raw[:, offset+2:offset+4] = raw[:, offset:offset+2]
            writer.add_tensor(name, q, raw_dtype=qt)
            x = gguf.dequantize(q, qt)
        elif quant:
            q = gguf.quantize(x, gguf.GGMLQuantizationType.Q8_0)
            writer.add_tensor(name, q, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
            x = gguf.dequantize(q, gguf.GGMLQuantizationType.Q8_0)
        else:
            writer.add_tensor(name, x)
        weights[name] = x

    add("token_embd.weight", (64, 256), scale=.25)
    add("output_norm.weight", (256,), norm=True)
    add("output.weight", (64, 256))
    for il, kv in enumerate([1, 2, 1, 2]):
        pre = f"blk.{il}."
        add(pre + "attn_norm.weight", (256,), norm=True)
        add(pre + "ffn_norm.weight", (256,), norm=True)
        add(pre + "attn_qkv.weight", (4 * 192 + kv * 320, 256))
        add(pre + "attn_output.weight", (256, 4 * 128))
        if il % 2 == 1:
            add(pre + "attn_sinks.weight", (4,), scale=1)
        if il == 0:
            for kind in ["gate", "up", "down"]:
                add(pre + f"ffn_{kind}.weight", (256, 256))
        else:
            add(pre + "ffn_gate_inp.weight", (4, 256))
            add(pre + "exp_probs_b.bias", (4,), scale=.12)
            for kind in ["gate", "up", "down"]:
                add(pre + f"ffn_{kind}_exps.weight", (4, 256, 256), quant=True)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    def norm(x, w):
        return x / np.sqrt(np.mean(x*x, axis=-1, keepdims=True) + 1e-6) * w

    def silu(x):
        return x / (1 + np.exp(-x))

    def rope(x, base):
        x = x.copy()
        angle = np.arange(len(x))[:, None, None] * base ** (-np.arange(32)[None, None, :] / 32)
        a, b = x[..., :32].copy(), x[..., 32:64].copy()
        x[..., :32] = a*np.cos(angle) - b*np.sin(angle)
        x[..., 32:64] = a*np.sin(angle) + b*np.cos(angle)
        return x

    tokens = np.array([(i * 7 + 3) % 64 for i in range(137)], dtype=np.int32)
    x = weights["token_embd.weight"][tokens].copy()
    for il, kv in enumerate([1, 2, 1, 2]):
        pre = f"blk.{il}."
        w = lambda name: weights[pre + name]
        fused = norm(x, w("attn_norm.weight")) @ w("attn_qkv.weight").T
        q, k, v = np.split(fused, [4*192, (4+kv)*192], axis=1)
        base = 1e4 if il % 2 == 1 else 1e7
        q = rope(q.reshape(-1, 4, 192), base)
        k = rope(k.reshape(-1, kv, 192), base).astype(np.float16).astype(np.float32)
        v = v.reshape(-1, kv, 128).astype(np.float16).astype(np.float32)
        att = np.empty((len(tokens), 4, 128), np.float32)
        for t in range(len(tokens)):
            first = max(0, t-127) if il % 2 == 1 else 0
            for h in range(4):
                kh = h // (4 // kv)
                score = k[first:t+1, kh] @ q[t, h] / np.sqrt(192)
                sink = float(w("attn_sinks.weight")[h]) if il % 2 == 1 else -np.inf
                mx = max(float(score.max()), sink)
                probs = np.exp(score - mx)
                probs /= probs.sum() + np.exp(sink - mx)
                att[t, h] = probs @ v[first:t+1, kh]
        x += (att.reshape(-1, 512) @ w("attn_output.weight").T) * np.float32(.707)
        xn = norm(x, w("ffn_norm.weight"))
        if il == 0:
            x += (silu(xn @ w("ffn_gate.weight").T) * (xn @ w("ffn_up.weight").T)) @ w("ffn_down.weight").T
        else:
            probs = 1 / (1 + np.exp(-(xn @ w("ffn_gate_inp.weight").T)))
            ids = np.argsort(-(probs + w("exp_probs_b.bias")), axis=1, kind="stable")[:, :2]
            routing = np.take_along_axis(probs, ids, axis=1)
            routing /= routing.sum(axis=1, keepdims=True).clip(6.103515625e-5)
            for t in range(len(tokens)):
                for j, e in enumerate(ids[t]):
                    hidden = silu(w("ffn_gate_exps.weight")[e] @ xn[t]) * (w("ffn_up_exps.weight")[e] @ xn[t])
                    x[t] += routing[t, j] * (w("ffn_down_exps.weight")[e] @ hidden)
    logits = norm(x, weights["output_norm.weight"]) @ weights["output.weight"].T
    tokens.astype("<i4").tofile(str(args.out) + ".tokens")
    logits.astype("<f4").tofile(str(args.out) + ".logits")
    print(f"Wrote {args.out}.gguf and NumPy logits for {len(tokens)} tokens")


if __name__ == "__main__":
    main()

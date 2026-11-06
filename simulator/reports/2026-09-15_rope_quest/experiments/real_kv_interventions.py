"""Matched, local interventions on archived post-RoPE keys. No model/service edits.

Every length uses the same window starts. Local dephasing preserves the first
key's phase and the pre-RoPE content at the inspected layer. It is not a forward
pass of a model trained or run without RoPE. Common rotations also rotate Q and
must preserve every logit. Outputs retain capture/layer/seed for clustered CIs.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import platform
import time
import numpy as np
import torch


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def rotate(x, angle):
    """Qwen split-half pairing: (j,j+64), not adjacent storage coordinates."""
    c, s = angle.cos(), angle.sin()
    a, b = x[..., :64], x[..., 64:]
    return torch.cat((a * c - b * s, b * c + a * s), dim=-1)


def metrics(x, q, original_logits=None):
    lo, hi = x.amin(1), x.amax(1)
    center, radius = (hi + lo) / 2, (hi - lo) / 2
    qp = q.mean(2)
    cp = torch.einsum("qhd,shd->sqh", qp, center) / math.sqrt(128)
    wp = torch.einsum("qhd,shd->sqh", qp.abs(), radius) / math.sqrt(128)
    lp = torch.einsum("qhd,slhd->sqhl", qp, x) / math.sqrt(128)
    tp = lp.amax(-1)
    cf = torch.einsum("qhgd,shd->sqhg", q, center) / math.sqrt(128)
    wf = torch.einsum("qhgd,shd->sqhg", q.abs(), radius) / math.sqrt(128)
    lf = torch.einsum("qhgd,slhd->sqhgl", q, x) / math.sqrt(128)
    tf = lf.amax(-1)
    # The bound is for the maximum within each head, then averaged over heads.
    # It is not the maximum of head-averaged logits.
    assert (cp + wp - tp).min().item() >= -0.0005
    assert (cf + wf - tf).min().item() >= -0.0005
    pair = ((hi - lo)[..., :64] + (hi - lo)[..., 64:]) / 2
    band = pair.reshape(-1, 4, 8, 8).mean((0, 1, 3)).tolist()
    row = dict(range_sum=(hi-lo).sum(-1).mean().item(),
               pooled_center=cp.mean().item(), pooled_width=wp.mean().item(),
               pooled_bound=(cp+wp).mean().item(), pooled_true_max=tp.mean().item(),
               pooled_gap=(cp+wp-tp).mean().item(),
               perhead_width=wf.mean().item(), perhead_bound=(cf+wf).mean().item(),
               perhead_true_max=tf.mean().item(), perhead_gap=(cf+wf-tf).mean().item())
    row.update({f"band_{j*8}_{j*8+7}": v for j, v in enumerate(band)})
    row["logit_invariance_max_abs_error"] = (
        (lf-original_logits["logits"]).abs().max().item() if original_logits is not None else 0.0)
    row["pooled_bound_mean_abs_change_vs_original"] = ((cp+wp-original_logits["bound"]).abs().mean().item() if original_logits is not None else 0.0)
    return row, {"logits": lf, "bound": cp+wp}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--spans", type=int, default=64)
    ap.add_argument("--seeds", default="11,29,47")
    ap.add_argument("--layers", default="0,12,24,36")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    base = Path("/workspace/scoring_case_study")
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    lengths = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    layers = [int(x) for x in args.layers.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    config_path = Path("/models/Qwen3-VL-30B-A3B-Instruct/config.json")
    config = json.loads(config_path.read_text())
    text_config = config["text_config"]
    assert text_config["rope_theta"] == 5_000_000
    theta = 5_000_000.0 ** (-torch.arange(64, device=args.device, dtype=torch.float32)/64)
    rel = torch.arange(max(lengths), device=args.device, dtype=torch.float32)
    base_angle = rel[None, :, None, None] * theta
    modes = [("original", None), ("collapse_all", -1.0),
             ("phase_speed_025", -0.75), ("phase_speed_4", 3.0),
             ("collapse_high_0_7", (0, 8)), ("collapse_mid_8_23", (8, 24)),
             ("collapse_low_24_63", (24, 64)),
             ("common_shift_137", 137), ("common_shift_997", 997),
             ("common_shift_4093", 4093)]
    captures = []
    for ds in ["discord2048", "discord10"]:
        meta = [json.loads(l) for l in Path(f"/workspace/sparse3way/offline_{ds}.jsonl").read_text().splitlines() if l]
        for i, item in enumerate(meta, 1):
            p = base/f"dumps_{ds}"/f"req_{i:03d}"
            if not (p/"token_ids.json").exists(): continue
            ids = json.loads((p/"token_ids.json").read_text())
            assert not ({151655,151656} & set(ids)), "Non-text positions need different handling"
            captures.append(dict(path=p, capture=f"{ds}/req_{i:03d}",
                prompt_hash=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                task=item.get("task"), site=item.get("site"), step=item.get("step"),
                n_tokens=len(ids), token_file_sha256=digest(p/"token_ids.json")))
    if args.limit: captures = captures[:args.limit]
    manifest = dict(protocol=__doc__, script_sha256=digest(__file__), config_sha256=digest(config_path),
        torch=torch.__version__, numpy=np.__version__, python=platform.python_version(),
        device=torch.cuda.get_device_name(0) if args.device=="cuda" else "cpu",
        text_config=text_config, args=vars(args), lengths=lengths, captures=[], elapsed_seconds=None)
    begin=time.time(); writer=None
    with (output/"paired_metrics.csv").open("w", newline="") as stream:
        for capture in captures:
            p=capture["path"]
            cm={k:v for k,v in capture.items() if k!="path"}; cm["layers"]=[]
            for layer in layers:
                kp=p/f"k_layer_{layer:02d}.pt"; qp=p/f"q_layer_{layer:02d}.bin"
                k=torch.load(kp, map_location="cpu", weights_only=True).to(device=args.device,dtype=torch.float32)
                assert k.shape==(capture["n_tokens"],4,128)
                qall=np.fromfile(qp,dtype=np.float16).reshape(-1,32,128)
                calls=sorted({0,min(4,len(qall)-1),len(qall)-1})
                q=torch.from_numpy(qall[calls].copy()).to(device=args.device,dtype=torch.float32).reshape(-1,4,8,128)
                cm["layers"].append(dict(layer=layer,key_sha256=digest(kp),query_sha256=digest(qp),
                    query_calls=calls,total_query_calls=len(qall)))
                for seed in seeds:
                    rng=np.random.default_rng(seed)
                    starts=rng.integers(0,len(k)-max(lengths)+1,size=args.spans)
                    indices=torch.as_tensor(starts[:,None]+np.arange(max(lengths))[None,:],device=args.device)
                    original=k[indices]
                    baseline={}
                    for mode,value in modes:
                        qm=q
                        if mode=="original": x=original
                        elif mode.startswith("common_shift"):
                            angle=value*theta
                            x=rotate(original,angle); qm=rotate(q,angle)
                        elif isinstance(value,tuple):
                            mask=torch.zeros(64,device=args.device);mask[value[0]:value[1]]=1
                            x=rotate(original,-base_angle*mask)
                        else: x=rotate(original,value*base_angle)
                        for length in lengths:
                            ref=baseline.get(length) if mode.startswith("common_shift") else None
                            measured,logits=metrics(x[:,:length],qm,ref)
                            if mode=="original": baseline[length]=logits
                            if ref is not None:
                                assert measured["logit_invariance_max_abs_error"]<0.002
                            row=dict(capture=capture["capture"],prompt_hash=capture["prompt_hash"],
                                     task=capture["task"],site=capture["site"],layer=layer,
                                     seed=seed,length=length,mode=mode,spans=args.spans,query_count=len(calls),**measured)
                            if writer is None:
                                writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader()
                            writer.writerow(row)
                        del x
                    del original,baseline
                del k,q
                stream.flush()
                print(capture["capture"],"layer",layer,"elapsed",round(time.time()-begin,1),flush=True)
            manifest["captures"].append(cm)
            (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    manifest["elapsed_seconds"]=time.time()-begin
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print("COMPLETE",round(manifest["elapsed_seconds"],1),flush=True)


if __name__=="__main__":
    with torch.inference_mode(): main()

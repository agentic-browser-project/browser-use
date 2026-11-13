"""Paired all-layer teacher forcing and free sparse action replay.

The numerical decoder is FlashAttention BF16. BrowSA's unchanged CUDA kernels
provide mean-query pooling and distinct-page admission. Prefix K/V is shared,
but every variant maintains its own generated K/V at all 64 layers.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from flash_attn import flash_attn_with_kvcache
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3 import modeling_qwen3

from runtime import ROOT, PREVIOUS, CONFIGS, extension


class ReplayAttention:
    def __init__(self, record, max_tokens):
        self.record = record
        self.plen = record['prompt_tokens']
        self.capacity = max_tokens
        self.batch = 41
        self.step = torch.zeros((), device='cuda', dtype=torch.int64)
        self.prefix = []
        self.history = []
        self.layouts = {16: [], 32: []}
        self.summaries = {}
        self.metadata = {}
        self.qmean = extension('qmean')
        self.selector = extension('select')
        self.diagnostics = []
        layoutdir = ROOT / 'layouts' / record['capture']
        for cap in (16, 32):
            self.summaries[cap] = torch.load(layoutdir / f'cap{cap}_summaries.pt', map_location='cuda', weights_only=True)
        for layer in range(64):
            data = torch.load(ROOT / 'captures' / record['capture'] / f'layer_{layer:02d}.pt', weights_only=True)
            self.prefix.append((data['k'][:self.plen].cuda().contiguous(), data['v'][:self.plen].cuda().contiguous()))
            self.history.append((torch.zeros((41, 8, max_tokens, 128), dtype=torch.bfloat16, device='cuda'),
                                 torch.zeros((41, 8, max_tokens, 128), dtype=torch.bfloat16, device='cuda')))
            for cap in (16, 32):
                self.layouts[cap].append(torch.load(layoutdir / f'cap{cap}_layer{layer:02d}.pt', map_location='cuda', weights_only=True))

    def configure(self, width):
        self.width = width
        self.offset = torch.arange(width, device='cuda')
        self.position = torch.zeros((1, width), dtype=torch.long, device='cuda')
        self.input = torch.zeros((41, width), dtype=torch.long, device='cuda')
        for cap in (16, 32):
            spans = np.array(self.record['geometry'][str(cap)]['spans'])
            lengths = spans[:, 1] - spans[:, 0] + 1
            ps = 2*cap
            pages = math.ceil((self.plen + self.capacity)/ps)
            max_selected = min(pages, max(math.floor(pages*.125), math.ceil(self.capacity/ps)+6))
            count = 20*width
            def tile(x):
                return x[:, None].expand(-1, width, *x.shape[1:]).reshape(count, *x.shape[1:]).contiguous()
            factors = torch.tensor(np.array([(a/lengths)**alpha for a, alpha in CONFIGS]), dtype=torch.float32, device='cuda')
            self.metadata[cap] = dict(
                lens=torch.tensor(lengths, dtype=torch.float32, device='cuda')[None].repeat(count, 1),
                ends=torch.tensor(lengths.cumsum(), dtype=torch.int32, device='cuda')[None].repeat(count, 1),
                factor=tile(factors), aif=torch.full((count,), self.plen, dtype=torch.int32, device='cuda'),
                phys=torch.arange(pages, dtype=torch.int32, device='cuda')[None].repeat(count, 1),
                max_selected=max_selected,
                page_offset=torch.arange(ps, device='cuda'),
                head=torch.arange(8, device='cuda')[None,:,None],
                row=torch.arange(20, device='cuda')[:,None].expand(-1,width).reshape(count,1,1),
                first=[tile(d['first']) for d in self.layouts[cap]],
                last=[tile(d['last']) for d in self.layouts[cap]],
            )

    def __call__(self, module, query, key, value, mask, **kwargs):
        layer = module.layer_idx
        pk, pv = self.prefix[layer]
        hk, hv = self.history[layer]
        positions = self.step + self.offset
        hk.index_copy_(2, positions, key)
        hv.index_copy_(2, positions, value)
        seq = (self.plen + positions + 1).to(torch.int32)
        results = []
        for start, cap in ((0, 16), (20, 32)):
            meta = self.metadata[cap]
            ps = 2*cap
            count = 20*self.width
            queries = query[start:start+20].permute(0,2,1,3).reshape(count,64,128).contiguous()
            qg, qa = self.qmean.qmean(queries, 8, 8)
            center, width = (self.summaries[cap][key][layer] for key in ('center', 'width'))
            c = torch.bmm(qg.transpose(0,1), center.transpose(1,2)).transpose(0,1).contiguous()
            w = torch.bmm(qa.transpose(0,1), width.transpose(1,2)).transpose(0,1).contiguous()
            seqlens = seq[None].repeat(20,1).reshape(count).contiguous()
            budget = torch.floor(((seqlens+ps-1)//ps).float()*.125)
            local, n_pages = self.selector.select(c,w,meta['lens'],meta['factor'],budget,
                meta['first'][layer],meta['last'][layer],seqlens,meta['aif'],meta['phys'],
                ps,256,False,meta['ends'],2*ps,ps,distinct_page_budget=True)
            if not torch.cuda.is_current_stream_capturing():
                assert int(n_pages.max())<=meta['max_selected']
            local = local.reshape(count,8,-1)[:,:,:meta['max_selected']]
            physical = (local.long()[...,None]*ps + meta['page_offset']).flatten(2).clamp(min=0)
            layout = self.layouts[cap][layer]['permutation']
            original = layout[meta['row'],meta['head'],physical.clamp(max=self.plen-1)]
            original = torch.where(physical<self.plen,original,physical).long()
            generated = (original-self.plen).clamp(min=0,max=self.capacity-1)
            hrow = meta['row']+start
            prefix_index = original.clamp(min=0,max=self.plen-1)
            kk = torch.where((original<self.plen)[...,None], pk[prefix_index,meta['head']],
                             hk[hrow,meta['head'],generated])
            vv = torch.where((original<self.plen)[...,None], pv[prefix_index,meta['head']],
                             hv[hrow,meta['head'],generated])
            # Sorted selected pages include the current final page. Therefore
            # every invalid causal slot is at the end of the gathered cache.
            kept = n_pages.reshape(count,8)*ps - ((ps-seqlens%ps)%ps)[:,None]
            out = flash_attn_with_kvcache(queries.reshape(count*8,1,8,128),
                kk.reshape(count*8,-1,1,128),vv.reshape(count*8,-1,1,128),
                cache_seqlens=kept.reshape(-1).int(),causal=False)
            results.append(out.reshape(20,self.width,64,128))
        dense_k = torch.cat((pk[None],hk[40:41].transpose(1,2)),dim=1)
        dense_v = torch.cat((pv[None],hv[40:41].transpose(1,2)),dim=1)
        dense = flash_attn_with_kvcache(query[40:41].transpose(1,2),dense_k,dense_v,
            cache_seqlens=seq[-1:].contiguous(),causal=True)
        results.append(dense)
        return torch.cat(results), None


def make_graph(model, attention, width):
    attention.configure(width)
    attention.step.zero_()
    attention.position.copy_(attention.plen+attention.offset[None])
    attention.input.fill_(151645)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            logits = model(input_ids=attention.input,position_ids=attention.position,
                           attention_mask={'full_attention':None},use_cache=False).logits
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        logits = model(input_ids=attention.input,position_ids=attention.position,
                       attention_mask={'full_attention':None},use_cache=False).logits
    torch.cuda.synchronize()
    return graph, logits


def action_info(text):
    decoder = json.JSONDecoder()
    candidates = []
    for index, char in enumerate(text):
        if char != '{':
            continue
        try:
            obj, _ = decoder.raw_decode(text[index:])
            if isinstance(obj,dict) and isinstance(obj.get('action'),list):
                candidates.append(obj['action'])
        except (ValueError, TypeError):
            continue
    if not candidates:
        return {'parsed':False,'actions':None,'types':None,'indices':None}
    actions = candidates[-1]
    types = [name for a in actions if isinstance(a,dict) for name in a]
    indices = []
    def visit(obj):
        if isinstance(obj,dict):
            for k,v in obj.items():
                if k in ('index','element_index'):
                    indices.append(v)
                visit(v)
        elif isinstance(obj,list):
            for v in obj: visit(v)
    visit(actions)
    return {'parsed':True,'actions':actions,'types':types,'indices':indices}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--teacher-width',type=int,default=4)
    parser.add_argument('--max-new-tokens',type=int,default=1024)
    parser.add_argument('--pilot',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    records=json.loads((ROOT/'geometry.json').read_text())
    # Identical input prompts are replayed once. Fixed-QKV diagnostics retain all
    # historical continuations, and separately deduplicate when aggregating.
    records=list({r['prompt_hash']:r for r in reversed(records)}.values())[::-1]
    if args.limit: records=records[:args.limit]
    outdir=ROOT/('replay_pilot' if args.pilot else 'replay')
    outdir.mkdir(exist_ok=True)
    tokenizer=AutoTokenizer.from_pretrained('/models/Qwen3-32B')
    begin=time.monotonic()
    model=AutoModelForCausalLM.from_pretrained('/models/Qwen3-32B',dtype=torch.bfloat16,
        device_map={'':'cuda'},attn_implementation='sdpa',low_cpu_mem_usage=True).eval()
    print('Loaded model',time.monotonic()-begin,flush=True)
    eos=model.generation_config.eos_token_id
    eos=[eos] if isinstance(eos,int) else eos
    for record in records:
        destination=outdir/(record['capture'].replace('/','_')+'.json')
        if destination.exists(): continue
        while not (ROOT/'layouts'/record['capture']/'all_layers.json').exists(): time.sleep(10)
        generation=json.loads((PREVIOUS/'qwen32'/record['capture']/'generation.json').read_text())
        outputs=generation['output_ids']
        start=time.monotonic()
        capacity=max(args.max_new_tokens,len(outputs))+args.teacher_width
        attention=ReplayAttention(record,capacity)
        modeling_qwen3.ALL_ATTENTION_FUNCTIONS['sdpa']=attention
        print('Ready',record['capture'],'memory',torch.cuda.memory_allocated()/2**30,flush=True)
        teacher_path=destination.with_suffix('.teacher.json')
        graph,logits=make_graph(model,attention,args.teacher_width)
        teacher=json.loads(teacher_path.read_text()) if teacher_path.exists() else []
        calls=list(range(len(outputs)-1))
        if args.pilot: calls=calls[:8]
        if teacher:
            assert len(teacher)==len(calls)
            calls=[]
        saved=torch.load(ROOT/'captures'/record['capture']/'dense_logits.pt',weights_only=True)
        for offset in range(0,len(calls),args.teacher_width):
            positions=calls[offset:offset+args.teacher_width]
            n=len(positions)
            tokens=outputs[offset:offset+n]+[eos[0]]*(args.teacher_width-n)
            attention.step.fill_(offset)
            attention.position.copy_(attention.plen+attention.step+attention.offset[None])
            attention.input.copy_(torch.tensor(tokens,device='cuda')[None].expand(41,-1))
            graph.replay()
            lp=logits[:,:n].float().log_softmax(-1)
            dense=lp[40]
            kl=(dense.exp()[None]*(dense[None]-lp)).sum(-1)
            labels=torch.tensor(outputs[offset+1:offset+n+1],device='cuda')
            nll=-lp.gather(-1,labels[None,:,None].expand(41,-1,1)).squeeze(-1)
            agreement=(logits[:,:n].argmax(-1)==logits[40,:n].argmax(-1)[None])
            gold=saved[offset+1:offset+n+1].cuda().float().log_softmax(-1)
            validation=(gold.exp()*(gold-dense)).sum(-1)
            for j,call in enumerate(positions):
                teacher.append(dict(call=call,action=call in record['action_calls'],
                    kl=kl[:,j].cpu().tolist(),nll=nll[:,j].cpu().tolist(),
                    greedy_agreement=agreement[:,j].cpu().tolist(),dense_reference_kl=float(validation[j])))
            if offset%128==0:
                print('Teacher',record['capture'],offset,'dense_KL',float(validation.mean()),
                      'seconds',round(time.monotonic()-start,1),flush=True)
        teacher_path.write_text(json.dumps(teacher)+'\n')
        del graph,logits,saved
        gc.collect();torch.cuda.empty_cache()
        graph,logits=make_graph(model,attention,1)
        first=torch.load(ROOT/'captures'/record['capture']/'dense_logits.pt',weights_only=True)[0].argmax().item()
        trajectories=[[first] for _ in range(41)]
        done=np.zeros(41,dtype=bool)
        current=torch.full((41,1),first,device='cuda',dtype=torch.long)
        maximum=min(args.max_new_tokens,16) if args.pilot else args.max_new_tokens
        for step in range(maximum-1):
            attention.step.fill_(step)
            attention.position.fill_(attention.plen+step)
            attention.input.copy_(current)
            graph.replay()
            current=logits.argmax(-1)
            ids=current[:,0].cpu().tolist()
            for i,token in enumerate(ids):
                if not done[i]:
                    trajectories[i].append(token)
                    done[i]=token in eos
            if step%128==0:
                print('Free',record['capture'],step,'done',int(done.sum()),'seconds',round(time.monotonic()-start,1),flush=True)
            if done.all():break
        variants=[dict(cap=cap,A=a,alpha=b) for cap in (16,32) for a,b in CONFIGS]+[dict(cap=0,A=0,alpha=0)]
        for i,item in enumerate(variants):
            text=tokenizer.decode(trajectories[i],skip_special_tokens=True)
            item.update(tokens=trajectories[i],text=text,finished=bool(done[i]),**action_info(text))
        result=dict(complete=True,capture=record['capture'],task=record['task'],prompt_hash=record['prompt_hash'],
            variants=variants,teacher=teacher,historical_action=action_info(tokenizer.decode(outputs,skip_special_tokens=True)),
            elapsed_seconds=time.monotonic()-start,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            torch=torch.__version__,max_new_tokens=args.max_new_tokens,
            decoder='BF16 FlashAttention; canonical BrowSA mean-query and distinct-page CUDA selectors',
            generation='greedy, shared dense prefix; independent sparse generated states')
        temporary=destination.with_suffix('.tmp')
        temporary.write_text(json.dumps(result,ensure_ascii=False)+'\n');temporary.replace(destination)
        print('COMPLETE',record['capture'],round(time.monotonic()-start,1),flush=True)
        modeling_qwen3.ALL_ATTENTION_FUNCTIONS['sdpa']=None
        del attention,graph,logits,current,result,teacher,trajectories
        gc.collect();torch.cuda.empty_cache()
    print('ALL COMPLETE',time.monotonic()-begin,flush=True)


if __name__=='__main__':
    with torch.inference_mode():main()

"""Page32 replay preserving all legacy model-row positions and score shapes."""
from functools import lru_cache
import hashlib
import math
from pathlib import Path

import numpy as np
import torch
from flash_attn import flash_attn_with_kvcache
import runtime


@lru_cache(None)
def gather_extension():
    from torch.utils.cpp_extension import load
    source=Path(__file__).with_name('fused_kv_gather.cu')
    tag=hashlib.sha256(source.read_bytes()).hexdigest()[:12]
    return load(name=f'browsa_kv_gather_{tag}',sources=[str(source)],
                extra_cuda_cflags=['-O3'],verbose=False)


def reference_gather(pages,permutation,pk,pv,hk,hv,page_size):
    """Original Torch indexing expression, without the fused copy kernel."""
    count=len(pages);plen=len(pk);capacity=hk.shape[2]
    row=torch.arange(count,device=pk.device)[:,None,None]
    head=torch.arange(8,device=pk.device)[None,:,None]
    physical=(pages.long()[...,None]*page_size+torch.arange(page_size,device=pk.device)).flatten(2).clamp(min=0)
    original=permutation[row,head,physical.clamp(max=plen-1)]
    original=torch.where(physical<plen,original,physical).long()
    generated=(original-plen).clamp(min=0,max=capacity-1)
    prefix=original.clamp(min=0,max=plen-1)
    kk=torch.where((original<plen)[...,None],pk[prefix,head],hk[row,head,generated])
    vv=torch.where((original<plen)[...,None],pv[prefix,head],hv[row,head,generated])
    return kk,vv


class Page32Attention:
    def __init__(self,root,record,configs,capacity,optimized=True):
        self.plen=record['prompt_tokens'];self.capacity=capacity
        self.count=len(configs);self.batch=41;self.optimized=optimized
        assert self.count==29
        self.step=torch.zeros((),device='cuda',dtype=torch.long)
        self.offset=torch.arange(1,device='cuda')
        self.prefix=[];self.history=[];self.layouts=[];self.dense_cache=[]
        self.qmean=runtime.extension('qmean');self.selector=runtime.extension('select')
        if optimized:self.gather=gather_extension()
        directory=root/'layouts'/record['capture']
        self.summaries=torch.load(directory/'cap16_summaries.pt',map_location='cuda',weights_only=True)
        for layer in range(64):
            data=torch.load(Path(record['source_root'])/'captures'/record['capture']/f'layer_{layer:02d}.pt',weights_only=True)
            pk,pv=data['k'][:self.plen].cuda().contiguous(),data['v'][:self.plen].cuda().contiguous()
            hk=torch.zeros((self.count+1,8,capacity,128),device='cuda',dtype=torch.bfloat16)
            hv=torch.zeros_like(hk)
            if optimized:
                dk=torch.cat((pk[None],hk[-1:].transpose(1,2)),dim=1)
                dv=torch.cat((pv[None],hv[-1:].transpose(1,2)),dim=1)
                self.dense_cache.append((dk,dv))
                pk,pv=dk[0,:self.plen],dv[0,:self.plen]
            self.prefix.append((pk,pv));self.history.append((hk,hv))
            self.layouts.append(torch.load(directory/f'cap16_layer{layer:02d}.pt',map_location='cuda',weights_only=True))
        spans=np.asarray(record['geometry']['16']['spans'])
        lengths=spans[:,1]-spans[:,0]+1
        pages=math.ceil((self.plen+capacity)/32)
        self.max_selected=min(pages,max(math.floor(pages*.125),math.ceil(capacity/32)+6))
        self.meta=dict(
            lens=torch.as_tensor(lengths,device='cuda',dtype=torch.float32)[None].repeat(self.count,1),
            ends=torch.as_tensor(lengths.cumsum(),device='cuda',dtype=torch.int32)[None].repeat(self.count,1),
            factor=torch.tensor(np.asarray([(a/lengths)**alpha for a,alpha in configs]),device='cuda',dtype=torch.float32),
            aif=torch.full((self.count,),self.plen,device='cuda',dtype=torch.int32),
            phys=torch.arange(pages,device='cuda',dtype=torch.int32)[None].repeat(self.count,1))

    def __call__(self,module,query,key,value,mask,**kwargs):
        layer=module.layer_idx
        pk,pv=self.prefix[layer];hk,hv=self.history[layer]
        positions=self.step+self.offset
        # Rows 0:20 keep the old twenty-setting grid. Rows 20:29 hold the
        # nine missing settings. Row 40 remains the original dense row.
        active_key=torch.cat((key[:29],key[40:41]))
        active_value=torch.cat((value[:29],value[40:41]))
        hk.index_copy_(2,positions,active_key);hv.index_copy_(2,positions,active_value)
        seq=(self.plen+positions+1).to(torch.int32)
        outputs=[];layout=self.layouts[layer]
        for start,end in ((0,20),(20,29)):
            n=end-start
            queries=query[start:end].permute(0,2,1,3).reshape(n,64,128).contiguous()
            qg,qa=self.qmean.qmean(queries,8,8)
            center,width=(self.summaries[name][layer] for name in ('center','width'))
            c=torch.bmm(qg.transpose(0,1),center.transpose(1,2)).transpose(0,1).contiguous()
            w=torch.bmm(qa.transpose(0,1),width.transpose(1,2)).transpose(0,1).contiguous()
            seqlens=seq.repeat(n).contiguous()
            budget=torch.floor(((seqlens+31)//32).float()*.125)
            meta={name:value[start:end] for name,value in self.meta.items()}
            local,n_pages=self.selector.select(c,w,meta['lens'],meta['factor'],budget,
                layout['first'][start:end],layout['last'][start:end],seqlens,meta['aif'],meta['phys'],
                32,256,False,meta['ends'],64,32,distinct_page_budget=True)
            if not torch.cuda.is_current_stream_capturing():
                assert int(n_pages.max())<=self.max_selected
            local=local.reshape(n,8,-1)[:,:,:self.max_selected]
            if self.optimized:
                kk,vv=self.gather.gather(local,layout['permutation'][start:end],pk,pv,hk,hv,start,32)
            else:
                kk,vv=reference_gather(local,layout['permutation'][start:end],pk,pv,hk[start:end],hv[start:end],32)
            kept=n_pages.reshape(n,8)*32-((32-seqlens%32)%32)[:,None]
            sparse=flash_attn_with_kvcache(queries.reshape(n*8,1,8,128),
                kk.reshape(n*8,-1,1,128),vv.reshape(n*8,-1,1,128),
                cache_seqlens=kept.reshape(-1).int(),causal=False).reshape(n,1,64,128)
            outputs.append(sparse)
        if self.optimized:
            dk,dv=self.dense_cache[layer]
            dk.index_copy_(1,self.plen+positions,key[40:41].transpose(1,2))
            dv.index_copy_(1,self.plen+positions,value[40:41].transpose(1,2))
        else:
            dk=torch.cat((pk[None],hk[-1:].transpose(1,2)),dim=1)
            dv=torch.cat((pv[None],hv[-1:].transpose(1,2)),dim=1)
        dense=flash_attn_with_kvcache(query[40:41].transpose(1,2),dk,dv,
            cache_seqlens=seq[-1:].contiguous(),causal=True)
        outputs.extend((torch.zeros((11,1,64,128),device=query.device,dtype=query.dtype),dense))
        return torch.cat(outputs),None


class GroupAttention:
    def __init__(self,children):
        self.children=children;self.batch=children[0].batch

    def __call__(self,module,query,key,value,mask,**kwargs):
        outputs=[]
        for i,child in enumerate(self.children):
            part=slice(i*self.batch,(i+1)*self.batch)
            outputs.append(child(module,query[part],key[part],value[part],mask,**kwargs)[0])
        return torch.cat(outputs),None


def make_graph(model,children):
    batch=children[0].batch
    inputs=torch.full((batch*len(children),1),151645,device='cuda',dtype=torch.long)
    positions=torch.cat([(child.plen+child.offset[None]).repeat(batch,1) for child in children])
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            logits=model(input_ids=inputs,position_ids=positions,attention_mask={'full_attention':None},use_cache=False).logits
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        logits=model(input_ids=inputs,position_ids=positions,attention_mask={'full_attention':None},use_cache=False).logits
    torch.cuda.synchronize()
    return graph,logits,inputs,positions

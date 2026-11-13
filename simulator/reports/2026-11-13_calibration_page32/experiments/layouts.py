"""Packing preparation, imported with explicitly configured globals."""
def prepare_layout(record):
    directory=ROOT/'layouts'/record['capture'];done=directory/'all_layers.json'
    if done.exists():
        saved=json.loads(done.read_text())
        assert saved['prompt_hash']==record['prompt_hash'] and saved['settings_sha256']==sha(ROOT/'settings.json')
        return
    directory.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    spans=np.asarray(record['geometry']['16']['spans']);lengths=spans[:,1]-spans[:,0]+1
    offsets=np.r_[0,lengths.cumsum()]
    lo,hi=runtime.middle_chunk_bounds(lengths,32,64,False)
    summaries=[]
    for layer in range(64):
        data=torch.load(Path(record['source_root'])/'captures'/record['capture']/f'layer_{layer:02d}.pt',weights_only=True)
        keys=data['k'][:record['prompt_tokens']].cuda();summaries.append(runtime.envelopes(keys,spans));del data,keys
    mx=torch.stack([r[0] for r in summaries]);mn=torch.stack([r[1] for r in summaries])
    torch.save(dict(center=torch.stack([r[2] for r in summaries]).cpu(),
                    width=torch.stack([r[3] for r in summaries]).cpu()),directory/'cap16_summaries.pt')
    results=[[] for _ in range(64)]
    for anchor,alpha in INTERNAL:
        factor=torch.as_tensor((anchor/lengths[lo:hi])**alpha,device='cuda',dtype=torch.float32)
        first,last,perm,events,stats=bounded_layout(runtime.extension('packing'),runtime.extension('linkage'),
            mx[:,lo:hi],mn[:,lo:hi],factor,lengths,lo,hi,32)
        full=torch.arange(record['prompt_tokens'],device='cuda',dtype=torch.int32)[None,None].repeat(64,8,1)
        full[:,:,offsets[lo]:offsets[hi]]=perm+int(offsets[lo])
        first,last,full=first.cpu(),last.cpu(),full.cpu()
        for layer in range(64):
            results[layer].append((first[layer].clone(),last[layer].clone(),full[layer].clone()))
    for layer in range(64):
        data=dict(first=torch.stack([r[0] for r in results[layer]]),
                  last=torch.stack([r[1] for r in results[layer]]),
                  permutation=torch.stack([r[2] for r in results[layer]]))
        path=directory/f'cap16_layer{layer:02d}.pt';temp=path.with_suffix('.tmp')
        torch.save(data,temp);temp.replace(path)
    atomic_json(done,dict(complete=True,prompt_hash=record['prompt_hash'],settings_sha256=sha(ROOT/'settings.json'),
        page_size=32,maximum_chunk_size=16,configurations=INTERNAL,elapsed_seconds=time.monotonic()-start))
    print('LAYOUT',record['task'],round(time.monotonic()-start,1),flush=True)
    del summaries,mx,mn,results,data,first,last,full,perm
    gc.collect();torch.cuda.empty_cache()


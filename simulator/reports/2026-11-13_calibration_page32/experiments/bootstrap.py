"""Load the model while CPU-side experiment preparation finishes."""
import importlib.util
from pathlib import Path
import time
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer

root=Path(__file__).resolve().parents[1]
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
tokenizer=AutoTokenizer.from_pretrained('/models/Qwen3-32B')
model=AutoModelForCausalLM.from_pretrained('/models/Qwen3-32B',dtype=torch.bfloat16,
    device_map={'':'cuda'},attn_implementation='sdpa',low_cpu_mem_usage=True).eval()
print('MODEL LOADED; waiting for prepared experiment',flush=True)
deadline=time.monotonic()+1800
while not (root/'ready.json').exists():
    if time.monotonic()>deadline:raise RuntimeError('Preparation did not finish within 30 minutes')
    time.sleep(3)
spec=importlib.util.spec_from_file_location('page32_driver',root/'experiments/run_replay.py')
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
with torch.inference_mode():driver.main(model,tokenizer)

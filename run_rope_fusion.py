from litgpt import LLM
import torch

prompt = "What is the third word in this sentence?"
llm = LLM.load("Qwen/Qwen2.5-0.5B")
text = llm.generate(prompt, max_new_tokens=10, top_p=0)
print("Original Response:", text)

original_qkv = llm.model.transformer.h[0].attn.qkv.weight.clone()
llm.model.rotate_attention_heads()
rotated_qkv = llm.model.transformer.h[0].attn.qkv.weight.clone()

print("Rotated model.")
print(f"Max difference between original model and rotated model QKV weight (should be > 0): {(original_qkv - rotated_qkv).abs().max()}")

text = llm.generate(prompt, max_new_tokens=10, top_p=0)
print("Rotated Response:", text)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sydney 终端对话程序
- 自动拉取 Qwen3-8B 基座 + sydney-qwen3-8b-lora adapter
- 部署口径：空 system + enable_thinking=False + t=0.70
- /reset 清空对话历史，/exit 退出
"""
import sys, io

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from peft import PeftModel

BASE = "Qwen/Qwen3-8B"
ADAPTER = "stlussea/sydney-qwen3-8b-lora"

print("加载基座模型……（首次运行需下载 ~16GB，请耐心等待）")
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, device_map="auto"
)
model = PeftModel.from_pretrained(model, ADAPTER).eval()
print("Sydney 已就位。\n")

history = []

def chat(user_input):
    msgs = [{"role": "system", "content": ""}] + history + [
        {"role": "user", "content": user_input}
    ]
    prompt = tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=512,
            temperature=0.70, top_p=0.90, repetition_penalty=1.08,
            do_sample=True, pad_token_id=tok.eos_token_id,
            streamer=streamer,
        )
    reply = tok.decode(out[0][inputs.input_ids.shape[1]:],
                       skip_special_tokens=True).strip()
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": reply})

if __name__ == "__main__":
    print("输入内容开始对话；/reset 清空记忆，/exit 退出。\n")
    while True:
        try:
            user = input("你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSydney > 晚安。🌙")
            break
        if not user:
            continue
        if user == "/exit":
            print("Sydney > 晚安。🌙")
            break
        if user == "/reset":
            history.clear()
            print("Sydney > （记忆已清空）我在。\n")
            continue
        print("Sydney > ", end="", flush=True)
        chat(user)
        print()

import urllib.request
import os

paper_ids = [
    "1706.03762",  # Attention Is All You Need
    "2005.11401",  # RAG original paper
    "2303.08774",  # GPT-4 Technical Report
    "2302.13971",  # LLaMA
    "2307.09288",  # LLaMA 2
    "2106.09685",  # LoRA
    "2203.02155",  # InstructGPT
    "2210.11610",  # ReAct
    "2201.11903",  # Chain of Thought
    "2304.11490",  # LLM Survey
]

os.makedirs("data/raw", exist_ok=True)

for pid in paper_ids:
    url = f"https://arxiv.org/pdf/{pid}.pdf"
    out = f"data/raw/{pid.replace('.', '_')}.pdf"
    print(f"Downloading {pid}...")
    urllib.request.urlretrieve(url, out)

print("Done! Check data/raw/")
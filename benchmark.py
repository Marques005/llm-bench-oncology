"""
P1 · LLM Bench Oncologia
Compara LLMs a responder a perguntas biomédicas (PubMedQA), com e sem o resumo
do artigo como contexto (um "mini-RAG").

- API (Groq, formato OpenAI): openai/gpt-oss-20b e openai/gpt-oss-120b (modelos de raciocínio)
- Local (transformers, no teu PC): Qwen/Qwen2.5-0.5B-Instruct

Saídas: results.csv (todas as respostas), summary.csv (métricas), accuracy.png (gráfico)
"""
import os, time, re
import pandas as pd
import torch, transformers
from datasets import load_dataset
from openai import OpenAI
from transformers import pipeline
from dotenv import load_dotenv

load_dotenv()                                 # lê GROQ_API_KEY do ficheiro .env
transformers.logging.set_verbosity_error()    # esconde avisos repetidos

# ---------------- Configuração ----------------
N = 100  # testa com 5; depois muda para 100
API_MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]   # via Groq
LOCAL_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"                    # via transformers, no teu PC
SYSTEM = "You are a biomedical expert. Reply with exactly one word: yes, no, or maybe."

# ---------------- Dados e clientes ----------------
ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train").shuffle(seed=42).select(range(N))
client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.environ["GROQ_API_KEY"])
local = pipeline("text-generation", model=LOCAL_MODEL, device=0 if torch.cuda.is_available() else -1)


def make_prompt(q, ctx=None):
    return (f"Context: {ctx}\n\n" if ctx else "") + f"Question: {q}\nAnswer with only one word: yes, no or maybe."


def parse(text):
    m = re.search(r"\b(yes|no|maybe)\b", (text or "").lower())
    return m.group(1) if m else "invalid"


def ask(model, prompt, retries=5):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    t0 = time.time()

    # Modelo local (transformers)
    if model == LOCAL_MODEL:
        out = local(msgs, max_new_tokens=10, do_sample=False)
        return out[0]["generated_text"][-1]["content"], time.time() - t0

    # Modelos pela API (Groq)
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=msgs,
                max_tokens=512,          # espaço para o raciocínio + a resposta
                temperature=0,
                reasoning_effort="low",  # raciocínio curto: mais rápido e gasta menos quota
            )
            text = r.choices[0].message.content or ""
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)  # por segurança
            return text, time.time() - t0
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)  # limite de pedidos: espera 1, 2, 4, 8 s e tenta de novo
            t0 = time.time()


# ---------------- Benchmark ----------------
rows = []
for i, ex in enumerate(ds):
    ctx = " ".join(ex["context"]["contexts"])
    for model in API_MODELS + [LOCAL_MODEL]:
        for mode, c in [("sem_contexto", None), ("com_contexto", ctx)]:
            try:
                text, secs = ask(model, make_prompt(ex["question"], c))
                pred = parse(text)
            except Exception as e:
                text, secs, pred = f"ERRO: {e}", None, "erro"
            rows.append(dict(
                id=ex["pubid"], model=model.split("/")[-1], mode=mode,
                gold=ex["final_decision"], pred=pred, correct=pred == ex["final_decision"],
                latency_s=secs, raw=text,
            ))
    print(f"{i+1}/{N} feito")

# ---------------- Resultados ----------------
df = pd.DataFrame(rows)
df.to_csv("results.csv", index=False)

summary = df.groupby(["model", "mode"]).agg(
    accuracy=("correct", "mean"),
    latency_s=("latency_s", "mean"),
    invalid=("pred", lambda p: (p == "invalid").sum()),
    erros=("pred", lambda p: (p == "erro").sum()),
).round(3)
print(summary)
summary.to_csv("summary.csv")

ax = summary["accuracy"].unstack().plot(
    kind="bar", figsize=(9, 5), rot=0,
    title=f"PubMedQA (n={N}): accuracy com vs sem contexto",
)
ax.set_ylabel("accuracy")
ax.set_ylim(0, 1)
ax.figure.tight_layout()
ax.figure.savefig("accuracy.png", dpi=150)
import time
import numpy as np
import zarr
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

# Config
MODEL_NAME = "Qwen/Qwen3-Embedding-4B"
OUTPUT_FILE = "embeddings.zarr"
BATCH_SIZE = 128

def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    elif m:
        return f"{m}m {s:02d}s"
    else:
        return f"{s}s"


# Load dataset and collect unique questions by their question ID
print("[INFO] Loading dataset...", flush=True)
dataset = load_dataset("quora", split="train")

# Each row has questions: {"id": [id1, id2], "text": [text1, text2]}
id_to_text: dict[int, str] = {}
for pair in dataset["questions"]:
    for qid, text in zip(pair["id"], pair["text"]):
        if qid not in id_to_text:
            id_to_text[qid] = text

# Sort by question ID for deterministic ordering
sorted_ids = sorted(id_to_text.keys())
sorted_texts = [id_to_text[qid] for qid in sorted_ids]
N = len(sorted_ids)
print(f"[INFO] Unique questions: {N}", flush=True)

# Load model
print(f"[INFO] Loading model: {MODEL_NAME} (device_map=auto, Flash Attention 2)", flush=True)
model = SentenceTransformer(
    MODEL_NAME,
    model_kwargs={
        "attn_implementation": "flash_attention_2",
        "device_map": "auto",
    },
)
dim = model.get_sentence_embedding_dimension()
print(f"[INFO] Embedding dimension: {dim}", flush=True)

# Open zarr store — one file holds IDs, texts, and embeddings
store = zarr.open(OUTPUT_FILE, mode="w")

# ids array: maps position → question id
ids_arr = store.zeros("ids", shape=(N,), dtype="int64", chunks=(BATCH_SIZE,))
ids_arr[:] = np.array(sorted_ids, dtype=np.int64)

# texts array: variable-length UTF-8 strings, same ordering as ids
# zarr v3 uses dtype="str" for variable-length UTF-8 strings natively
texts_arr = store.open_array(
    "texts",
    mode="w",
    shape=(N,),
    dtype="str",
    chunks=(BATCH_SIZE,),
)
texts_arr[:] = sorted_texts

# embeddings array: shape (N, dim), chunked so each row is one question embedding
emb_arr = store.zeros(
    "embeddings",
    shape=(N, dim),
    dtype="float32",
    chunks=(BATCH_SIZE, dim),
)

# Encode in batches and write directly into the zarr array
print(f"[INFO] Starting embedding of {N} questions in batches of {BATCH_SIZE}...", flush=True)
embed_start = time.time()
last_log_time = embed_start

for i in range(0, N, BATCH_SIZE):
    batch_texts = sorted_texts[i : i + BATCH_SIZE]
    embs = model.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False, prompt_name="query")
    emb_arr[i : i + len(batch_texts)] = embs

    now = time.time()
    elapsed = now - embed_start
    done = i + len(batch_texts)
    pct = done / N * 100
    rate = done / elapsed if elapsed > 0 else 0
    eta = (N - done) / rate if rate > 0 else 0

    # Log every batch OR at least every 30 seconds
    if (now - last_log_time) >= 30 or i == 0:
        print(
            f"[PROGRESS] {done}/{N} questions ({pct:.1f}%) | "
            f"Elapsed: {format_duration(elapsed)} | "
            f"ETA: {format_duration(eta)} | "
            f"Speed: {rate:.1f} q/s",
            flush=True,
        )
        last_log_time = now

total_time = time.time() - embed_start
print(f"[DONE] Embedding complete in {format_duration(total_time)} ({N/total_time:.1f} q/s avg)", flush=True)

print(f"Saved {N} embeddings to {OUTPUT_FILE}")
print(f"  store['ids']        shape: {store['ids'].shape}")
print(f"  store['texts']      shape: {store['texts'].shape}")
print(f"  store['embeddings'] shape: {store['embeddings'].shape}")
print()
print("Example lookup — question ID → text + embedding:")
example_id = sorted_ids[0]
pos = 0  # position in the sorted arrays
# To look up by arbitrary question ID at query time:
#   pos = int(np.searchsorted(store["ids"], question_id))
#   text = store["texts"][pos]
#   emb  = store["embeddings"][pos]
print(f"  question id : {store['ids'][pos]}")
print(f"  text        : {store['texts'][pos]!r}")
print(f"  embedding[:5]: {store['embeddings'][pos, :5]}")

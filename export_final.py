import torch
from sentence_transformers import SentenceTransformer
from pathlib import Path

print(">> 1. Loading the SentenceTransformer wrapper...")
model_name = "./qwen3-embedding-0.6b-local"
sbert_model = SentenceTransformer(model_name)

print(">> 2. Extracting the PURE computational engine (AutoModel)...")
raw_engine = sbert_model[0].auto_model

# =========================================================================
#   THE DEFINITIVE FIX: Directly targeting the 'use_cache' error.
#   This feature conflicts with the ONNX exporter. We must disable it.
# =========================================================================
print(">> 3. Disabling the conflicting 'use_cache' feature in the model's configuration...")
raw_engine.config.use_cache = False

print(">> 4. Creating example tensors for the engine...")
dummy_input_text = ["This is a dummy input"]
features = sbert_model.tokenizer(dummy_input_text, padding=True, truncation=True, return_tensors="pt")

engine_inputs = {
    'input_ids': features['input_ids'],
    'attention_mask': features['attention_mask'],
}
if 'token_type_ids' in features:
    engine_inputs['token_type_ids'] = features['token_type_ids']

output_dir = Path("./qwen_onnx_final")
output_dir.mkdir(exist_ok=True)
output_path = output_dir / "model.onnx"

print(f">> 5. Exporting the pure engine to ONNX...")

torch.onnx.export(
    raw_engine,
    args=tuple(engine_inputs.values()),
    f=str(output_path),
    opset_version=14, # A stable and compatible version
    input_names=list(engine_inputs.keys()),
    output_names=["last_hidden_state"],
    dynamic_axes={key: {0: "batch_size", 1: "sequence_length"} for key in engine_inputs},
)

print(f"\n✅✅✅ SUCCESS! The pure engine was exported to: {output_path} ✅✅✅")
print(">> Stage 1 is finally complete. My sincere apologies for this difficult process.")
print(">> Next, we will use the 'iree-compile' command on this ONNX file.")

"""Offline, revision-pinned Qwen report extraction with strict evidence checks."""
import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd

from .contracts import ID, TARGETS, sha256, write_json
from .labels import validate_extraction


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True, help="Offline official model snapshot")
    p.add_argument("--model-revision", required=True, help="Immutable Hugging Face commit, never main")
    p.add_argument("--model-manifest", type=Path, required=True, help="filename -> SHA256 for staged model files")
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--prompt", type=Path, default=Path("experiments/p13_extraction_prompt.txt"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=0, help="Benchmark subset; never accepted as complete labels")
    args = p.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.model_revision):
        raise ValueError("Require a pinned model commit")
    model_files = json.loads(args.model_manifest.read_text())
    actual_files = {str(p.relative_to(args.model)) for p in args.model.rglob("*") if p.is_file() and
                    p.suffix in {".json", ".safetensors", ".model", ".txt", ".jinja"}}
    if not actual_files <= set(model_files):
        raise ValueError("Every model/tokenizer/config file must be pinned in the manifest")
    for name, expected in model_files.items():
        path = (args.model / name).resolve()
        if not path.is_relative_to(args.model.resolve()) or sha256(path) != expected:
            raise ValueError("Model snapshot hash mismatch")
    if "config.json" not in model_files or not any(n.endswith(".safetensors") for n in model_files):
        raise ValueError("Model manifest must include configuration and weights")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                              bnb_4bit_compute_dtype=torch.float16), device_map="auto")
    model.eval()
    if (model.config.model_type, model.config.num_hidden_layers, model.config.hidden_size) != ("qwen3", 40, 5120):
        raise ValueError("Staged snapshot is not Qwen3-14B")
    metadata = pd.read_csv(args.train)
    prompt = args.prompt.read_text() + "\nExact finding keys: " + json.dumps(TARGETS)
    contract = {"model": "Qwen/Qwen3-14B", "model_revision": args.model_revision,
                "model_manifest_sha256": sha256(args.model_manifest), "prompt_sha256": sha256(args.prompt),
                "train_sha256": sha256(args.train), "limit": args.limit}
    contract_path = args.output.with_suffix(".contract.json")
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("Refusing mixed extractor provenance")
    write_json(contract_path, contract)
    existing = {}
    if args.output.exists():
        for line in args.output.read_text().splitlines():
            row = json.loads(line)
            if row[ID] in existing:
                raise ValueError("Duplicate extraction in resume file")
            existing[row[ID]] = row
    start = time.time()
    data = metadata.iloc[:args.limit] if args.limit else metadata
    with args.output.open("a") as f:
        for _, row in data.iterrows():
            if row[ID] in existing:
                validate_extraction(existing[row[ID]], row.Report)
                continue
            messages = [{"role": "system", "content": prompt}, {"role": "user", "content": row.Report}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            tokens = tokenizer(text, return_tensors="pt").to(model.device)
            if tokens.input_ids.shape[1] > 24000:
                raise ValueError("Report exceeds input budget; truncation is forbidden")
            with torch.inference_mode():
                output = model.generate(**tokens, max_new_tokens=4096, do_sample=False)
            answer = tokenizer.decode(output[0, tokens.input_ids.shape[1]:], skip_special_tokens=True).strip()
            answer = re.sub(r"^```(?:json)?\s*|\s*```$", "", answer)
            record = json.loads(answer)
            record[ID] = row[ID]
            validate_extraction(record, row.Report)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            existing[row[ID]] = record
            print(json.dumps({"extracted": len(existing), "seconds": time.time() - start}), flush=True)
    write_json(args.output.with_suffix(".manifest.json"), {**contract,
        "extractions_sha256": sha256(args.output), "rows": len(existing), "elapsed_seconds": time.time() - start,
        "complete": args.limit == 0 and set(existing) == set(metadata[ID])})


if __name__ == "__main__":
    main()

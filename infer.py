import os
import json
import argparse
import threading
import torch
from pathlib import Path
from time import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from typing import List, Dict, Any, Optional
from tqdm.auto import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM

from src.utils import read_json_file, build_messages
from src.llm_services import parse_json_from_string, parse_llm_response
from src.schema import Nl2CypherSample
from src.logger_config import setup_logger

RESULTS_DIR = "results"
LOG_DIR = "logging_data/qwen3"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
log_path = os.path.join(LOG_DIR, f"log_infer_{int(time())}.txt")
logger = setup_logger(__name__, log_file=log_path)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Load Qwen model and benchmark on text2cypher problems with CKPT support"
    )
    parser.add_argument(
        "--benchmark",
        default="Cypherbench",
        choices=["Cypherbench", "Mind_the_query", "Neo4j_Text2Cypher"],
        help="Benchmark name",
    )
    parser.add_argument(
        "--db",
        default=None,
        help='Database name of each benchmark. If omitted or set to "full", use all data.',
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen3-0.6B", help="Model name or path"
    )
    parser.add_argument(
        "--ckpt_path", type=str, default=None, help="Path to LoRA checkpoint or fine-tuned checkpoint"
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda", "auto"],
        help="Device to run the model on",
    )
    parser.add_argument(
        "--max-length", type=int, default=1024, help="Max generation length"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size for generation"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.5, help="Temperature for sampling"
    )
    parser.add_argument(
        "--top-p", type=float, default=0.95, help="Top-p for sampling"
    )
    parser.add_argument(
        "--top-k", type=int, default=0, help="Top-k for sampling"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of test samples"
    )
    return parser.parse_args()


def init_model(model_name_or_path, ckpt_path=None, device=None):
    if device is None or device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    is_peft = False
    if ckpt_path:
        if os.path.exists(os.path.join(ckpt_path, "adapter_config.json")):
            is_peft = True
            print("This is LoRA finetune")
        elif os.path.exists(os.path.join(ckpt_path, "config.json")):
            # If it's a full HF checkpoint we should just load directly from it
            model_name_or_path = ckpt_path
            print("This is a full finetune")

    logger.info(f"Loading tokenizer from {model_name_or_path}")
    print(f"Loading tokenizer from {model_name_or_path}")
    # Force left-padding for decoder-only architectures
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="left", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    logger.info(f"Loading model from {model_name_or_path} on {device}")
    print(f"Loading model from {model_name_or_path} on {device}")
    dtype = torch.bfloat16 if torch.cuda.is_available() and device != "cpu" else torch.float16
    
    device_map = "auto" if device == "cuda" else { "": device }
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True
    )
    
    if ckpt_path and is_peft:
        print(f"Loading PEFT checkpoint weights from {ckpt_path}")
        logger.info(f"Loading PEFT checkpoint weights from {ckpt_path}")
        try:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, ckpt_path)
            model = model.merge_and_unload()
            print("Successfully loaded and merged LoRA weights.")
        except Exception as peft_e:
            print(f"Failed to load as PEFT model: {peft_e}.")
            
    # For full checkpoint manually passed without config.json (rare, just fallback)
    elif ckpt_path and model_name_or_path != ckpt_path:
        print(f"Attempting manual weight loading from {ckpt_path}")
        import safetensors.torch
        if os.path.exists(os.path.join(ckpt_path, "model.safetensors")):
            state_dict = safetensors.torch.load_file(os.path.join(ckpt_path, "model.safetensors"))
            model.load_state_dict(state_dict)
            print("Loaded full model weights from safetensors.")
        elif os.path.exists(os.path.join(ckpt_path, "pytorch_model.bin")):
            model.load_state_dict(torch.load(os.path.join(ckpt_path, "pytorch_model.bin"), map_location="cpu"))
            print("Loaded full model weights from pytorch_model.bin.")
        else:
            print(f"Warning: neither adapter_config, config.json, model.safetensors, nor pytorch_model.bin found in {ckpt_path}.")
            
    model.eval()
    return tokenizer, model


def generate_response_batch(tokenizer, model, batch_messages, max_length=1024, temperature=0.5, top_p=0.95, top_k=0):
    texts = [
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        for messages in batch_messages
    ]
    
    tokenizer.padding_side = "left"
    inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
    
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        
    generated_ids = outputs[:, inputs["input_ids"].shape[-1]:]
    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return responses


def is_full_db(db: Optional[str]) -> bool:
    return db is None or str(db).strip().lower() in {"full", "all", ""}


def load_schema_for_graph(benchmark: str, graph_name: str) -> Optional[str]:
    if benchmark == "Cypherbench":
        schema_path = Path("benchmarks") / benchmark / "graphs" / "schemas" / f"{graph_name}_schema.json"
    elif benchmark == "Mind_the_query":
        schema_path = Path("benchmarks") / benchmark / "graphs" / "schemas" / f"{graph_name}.json"
    else:
        return None

    if not schema_path.exists():
        logger.warning(f"Schema file not found for graph={graph_name}: {schema_path}")
        return None

    schema = read_json_file(schema_path)
    return json.dumps(schema, indent=4, ensure_ascii=False)


def load_schema_and_subset_test_data(benchmark, db=None, limit=None):
    raw_test_data = read_json_file(Path("benchmarks") / benchmark / "test.json")

    use_all = is_full_db(db)
    subset_test_data = []

    for item in raw_test_data:
        try:
            sample = Nl2CypherSample(**item)
            if use_all or sample.graph == db:
                subset_test_data.append(sample)
        except Exception as e:
            logger.warning(f"Skipping invalid item: {e}")

    if limit is not None:
        subset_test_data = subset_test_data[:limit]

    shared_schema_str = None
    schema_map = {}

    if benchmark in ["Cypherbench", "Mind_the_query"]:
        if use_all:
            unique_graphs = sorted(
                {sample.graph for sample in subset_test_data if sample.graph}
            )
            for graph_name in unique_graphs:
                schema_map[graph_name] = load_schema_for_graph(benchmark, graph_name)
        else:
            shared_schema_str = load_schema_for_graph(benchmark, db)

    return subset_test_data, shared_schema_str, schema_map


def get_question_and_schema(
    sample: Nl2CypherSample,
    benchmark: str,
    shared_schema_str: Optional[str] = None,
    schema_map: Optional[Dict[str, Optional[str]]] = None,
):
    question = sample.nl_question

    if benchmark in ["Cypherbench", "Mind_the_query"]:
        if shared_schema_str is not None:
            schema_str = shared_schema_str
        else:
            schema_str = (schema_map or {}).get(sample.graph)
    else:
        schema_str = sample.schema

    return question, schema_str


def run_batch_inference(
    test_data,
    benchmark,
    db,
    tokenizer,
    model,
    max_length,
    batch_size=1,
    temperature=0.5,
    top_p=0.95,
    top_k=0,
    shared_schema_str=None,
    schema_map=None,
):
    results = []
    errors = []
    run_name = db if not is_full_db(db) else "full"

    progress_bar = tqdm(total=len(test_data), desc=f"Running {benchmark}/{run_name}")

    for i in range(0, len(test_data), batch_size):
        batch_samples = test_data[i : i + batch_size]
        
        batch_messages = []
        batch_questions = []
        for sample in batch_samples:
            question, schema_str = get_question_and_schema(
                sample, benchmark, shared_schema_str, schema_map
            )
            messages = build_messages(question, schema_str)
            batch_messages.append(messages)
            batch_questions.append(question)
            
        try:
            raw_responses = generate_response_batch(
                tokenizer, model, batch_messages, max_length=max_length,
                temperature=temperature, top_p=top_p, top_k=top_k
            )
            
            for sample, question, raw_response in zip(batch_samples, batch_questions, raw_responses):
                qid = sample.qid if sample.qid is not None else sample.instance_id
                if qid is None:
                    qid = "unknown"
                
                logger.info("-" * 80)
                logger.info(f"Processing item: {qid}")
                logger.info(f"Graph: {sample.graph}")
                logger.info(f"Question: {question}")
                
                try:
                    parsed = parse_llm_response(raw_response)
                    parsed_json = parse_json_from_string(parsed["final_answer"])
            
                    if not parsed_json or "cypher" not in parsed_json:
                        raise ValueError("Failed to parse JSON or missing 'cypher' key")
            
                    cypher = parsed_json["cypher"]
                    sample.pred_cypher = cypher
            
                    logger.info(f"Generated cypher for item {qid}: {cypher}")
            
                    results.append({
                        "qid": qid,
                        "graph": sample.graph,
                        "question": question,
                        "raw_response": raw_response,
                        "think": parsed.get("think", ""),
                        "final_answer": parsed.get("final_answer", ""),
                        "cypher": cypher,
                        "success": True,
                        "error": None,
                        "sample": sample.model_dump(mode="json"),
                    })
                except Exception as e:
                    logger.error(f"Error parsing item {qid}: {e}", exc_info=True)
                    sample.pred_cypher = None
                    errors.append({"qid": qid, "error": str(e)})
                    results.append({
                        "qid": qid,
                        "graph": sample.graph,
                        "question": question,
                        "raw_response": raw_response,
                        "think": "",
                        "final_answer": "",
                        "cypher": None,
                        "success": False,
                        "error": str(e),
                        "sample": sample.model_dump(mode="json"),
                    })
                    
        except Exception as batch_e:
            logger.error(f"Error generating batch starting at index {i}: {batch_e}", exc_info=True)
            for sample in batch_samples:
                qid = sample.qid if sample.qid is not None else sample.instance_id
                errors.append({"qid": qid, "error": str(batch_e)})
                
        progress_bar.update(len(batch_samples))
        
    logger.info(f"Finished. Success: {len(results) - len(errors)}, Failed: {len(errors)}")
    return results, errors


def main():
    args = parse_args()

    subset_test_data, schema_str, schema_map = load_schema_and_subset_test_data(
        args.benchmark,
        args.db,
        args.limit,
    )

    db_name = args.db if not is_full_db(args.db) else "full"

    tokenizer, model = init_model(args.model, args.ckpt_path, device=args.device)

    print(f"Running benchmark={args.benchmark}, db={db_name}, samples={len(subset_test_data)}")
    print(f"Using batch_size={args.batch_size}")

    results, errors = run_batch_inference(
        test_data=subset_test_data,
        benchmark=args.benchmark,
        db=args.db,
        tokenizer=tokenizer,
        model=model,
        max_length=args.max_length,
        batch_size=args.batch_size,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        shared_schema_str=schema_str,
        schema_map=schema_map,
    )

    output = []
    for r in results:
        output.append({
            "question": r["question"],
            "graph": r["graph"],
            "gold_cypher": r["sample"].get("gold_cypher"),
            "pred_cypher": r["cypher"]
        })

    os.makedirs(Path(RESULTS_DIR) / args.benchmark, exist_ok=True)
    
    # Differentiate output file based on whether a custom ckpt was used
    # output_path = Path(RESULTS_DIR) / args.benchmark / f"{db_name}_cyphers_result_{args.model.split('/')[-1]}.json"
    # output_path = Path(RESULTS_DIR) / args.benchmark / f"{db_name}_cyphers_result_{args.model.split('/')[-1]}_distill_fkl.json"
    # output_path = Path(RESULTS_DIR) / args.benchmark / f"{db_name}_cyphers_result_{args.model.split('/')[-1]}_distill_rkl.json"
    # output_path = Path(RESULTS_DIR) / args.benchmark / f"{db_name}_cyphers_result_{args.model.split('/')[-1]}_distill_sfkl.json"
    # output_path = Path(RESULTS_DIR) / args.benchmark / f"{db_name}_cyphers_result_{args.model.split('/')[-1]}_distill_csd.json"
    # output_path = Path(RESULTS_DIR) / args.benchmark / f"{db_name}_cyphers_result_{args.model.split('/')[-1]}_distill_fdd_sfkl.json"   #default kd_ratio 0.5
    # output_path = Path(RESULTS_DIR) / args.benchmark / f"{db_name}_cyphers_result_{args.model.split('/')[-1]}_distill_fdd_srkl.json" 
    output_path = Path(RESULTS_DIR) / args.benchmark / f"{db_name}_cyphers_result_{args.model.split('/')[-1]}_distill_fdd_srkl_updated_1.json" 
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

    print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    main()

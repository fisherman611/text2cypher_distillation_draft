import json
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Format text2cypher data to jsonl")
    parser.add_argument("--base-dir", default="", help="Base directory")
    parser.add_argument("--benchmark", default="Cypherbench", help="Benchmark name (e.g. Cypherbench)")
    parser.add_argument("--split", default="train", help="Dataset split (e.g. train or test)")
    args = parser.parse_args()

    base_dir = args.base_dir
    if args.benchmark == "Mind_the_query" and args.split == "train":
        train_path = os.path.join(base_dir, "benchmarks", args.benchmark, "train_val.json")
    else:
        train_path = os.path.join(base_dir, "benchmarks", args.benchmark, f"{args.split}.json")
        
    out_path = os.path.join(base_dir, "benchmarks", args.benchmark, f"{args.split}.jsonl")
    sys_prompt_path = os.path.join(base_dir, "prompts", "generator", "system_prompt.txt")
    usr_prompt_path = os.path.join(base_dir, "prompts", "generator", "user_prompt.txt")
    schema_dir = os.path.join(base_dir, "benchmarks", args.benchmark, "graphs", "schemas")

    with open(sys_prompt_path, 'r', encoding='utf-8') as f:
        sys_prompt = f.read()

    with open(usr_prompt_path, 'r', encoding='utf-8') as f:
        usr_prompt_template = f.read()

    # Preload schemas
    schemas = {}
    if os.path.exists(schema_dir):
        for filename in os.listdir(schema_dir):
            if filename.endswith("_schema.json"):
                graph_name = filename.replace("_schema.json", "")
                with open(os.path.join(schema_dir, filename), 'r', encoding='utf-8') as f:
                    schemas[graph_name] = f.read()

    with open(train_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open(out_path, 'w', encoding='utf-8') as out:
        for item in data:
            graph = item['graph']
            schema_json_str = schemas.get(graph, "")
            
            if args.benchmark == "Mind_the_query":
                question = item.get('question', "")
                gold_cypher = item.get("gold_cypher", "")
            else:
                question = item.get('nl_question', "")
                gold_cypher = item.get("gold_cypher", "")

            user_prompt = usr_prompt_template.replace("{question}", question).replace("{schema}", schema_json_str)
            response_obj = {"cypher": gold_cypher}
            response_str = json.dumps(response_obj)
            
            jsonl_obj = {
                "system_prompt": sys_prompt,
                "user_prompt": user_prompt,
                "response": response_str
            }
            out.write(json.dumps(jsonl_obj) + "\n")

    print(f"Done formatting {args.split}.json to {args.split}.jsonl for benchmark {args.benchmark}")

if __name__ == "__main__":
    main()
import json
import argparse
import os


def extract_cypher(record):
    parsed = record.get("parsed_text")

    if isinstance(parsed, dict):
        cypher = parsed.get("cypher")
        if isinstance(cypher, str):
            return cypher

    return parsed


def load_jsonl(file_path):
    data = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                data.append({})
                continue

            try:
                obj = json.loads(line)
                data.append(obj if isinstance(obj, dict) else {})
            except json.JSONDecodeError:
                data.append({})

    return data


def load_graph(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Graph file must be a list")

    return data


def merge_files(gt_file, pred_file, graph_file, output_file):
    gt_data = load_jsonl(gt_file)
    pred_data = load_jsonl(pred_file)
    graph_data = load_graph(graph_file)

    if not (len(gt_data) == len(pred_data) == len(graph_data)):
        raise ValueError("All input files must have same number of samples")

    results = []

    for gt, pred, graph_item in zip(gt_data, pred_data, graph_data):
        gold_cypher = extract_cypher(gt)
        pred_cypher = extract_cypher(pred)

        graph_value = ""
        if isinstance(graph_item, dict):
            graph_value = graph_item.get("graph", "")
        else:
            graph_value = graph_item  # fallback if not dict

        results.append({
            "gold_cypher": gold_cypher,
            "pred_cypher": pred_cypher,
            "graph": graph_value
        })

    # create output dir
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(results)} samples to {output_file}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-g", "--ground_truth", required=True)
    parser.add_argument("-p", "--prediction", required=True)
    parser.add_argument("-m", "--graph", required=True)
    parser.add_argument("-o", "--output", required=True)

    args = parser.parse_args()

    merge_files(
        gt_file=args.ground_truth,
        pred_file=args.prediction,
        graph_file=args.graph,
        output_file=args.output
    )


if __name__ == "__main__":
    main()
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
import types
import importlib.machinery

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_attention_alignment import build_one_sample

# The finetune module imports DeepSpeed at top level. Unit tests for helper
# functions do not need DeepSpeed, so provide a tiny import stub when the local
# environment does not have it installed.
if "deepspeed" not in sys.modules:
    deepspeed_stub = types.ModuleType("deepspeed")
    deepspeed_stub.__spec__ = importlib.machinery.ModuleSpec("deepspeed", None)
    deepspeed_stub.initialize = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("deepspeed.initialize is not available in this helper test stub")
    )
    deepspeed_stub.init_distributed = lambda *args, **kwargs: None
    deepspeed_stub.DeepSpeedEngine = object
    sys.modules["deepspeed"] = deepspeed_stub

if "accelerate" not in sys.modules:
    accelerate_stub = types.ModuleType("accelerate")
    accelerate_stub.__spec__ = importlib.machinery.ModuleSpec("accelerate", None)
    accelerate_stub.load_checkpoint_and_dispatch = lambda *args, **kwargs: None
    accelerate_stub.init_empty_weights = lambda *args, **kwargs: None
    sys.modules["accelerate"] = accelerate_stub

if "peft" not in sys.modules:
    peft_stub = types.ModuleType("peft")
    peft_stub.__spec__ = importlib.machinery.ModuleSpec("peft", None)
    peft_stub.get_peft_model = lambda *args, **kwargs: None
    peft_stub.LoraConfig = object
    peft_stub.TaskType = types.SimpleNamespace(CAUSAL_LM="CAUSAL_LM")
    peft_stub.PeftModel = object
    peft_stub.PeftMixedModel = object
    sys.modules["peft"] = peft_stub

if "rouge_metric" not in sys.modules:
    rouge_stub = types.ModuleType("rouge_metric")
    rouge_stub.__spec__ = importlib.machinery.ModuleSpec("rouge_metric", None)
    rouge_stub.compute_metrics = lambda *args, **kwargs: {}
    sys.modules["rouge_metric"] = rouge_stub

if "ed_eval" not in sys.modules:
    ed_stub = types.ModuleType("ed_eval")
    ed_stub.__spec__ = importlib.machinery.ModuleSpec("ed_eval", None)
    ed_stub.ed_evaluate = lambda *args, **kwargs: {}
    sys.modules["ed_eval"] = ed_stub

import updated_attn_loss_finetune as attn_ft


def assert_close(name, value, target=0.0, tol=1e-6):
    diff = abs(float(value) - float(target))
    print(f"{name}: value={float(value):.10f}, target={float(target):.10f}, diff={diff:.10f}")
    if diff > tol:
        raise AssertionError(f"{name} diff {diff} > {tol}")


def test_resolve_layer_index():
    print("\n[1] resolve_layer_index")
    print("resolve_layer_index(3, 28)  =", attn_ft.resolve_layer_index(3, 28))
    print("resolve_layer_index(-1, 28) =", attn_ft.resolve_layer_index(-1, 28))
    print("resolve_layer_index(-2, 28) =", attn_ft.resolve_layer_index(-2, 28))
    assert attn_ft.resolve_layer_index(3, 28) == 3
    assert attn_ft.resolve_layer_index(-1, 28) == 27
    assert attn_ft.resolve_layer_index(-2, 28) == 26


def test_get_module_by_path():
    print("\n[2] get_module_by_path")

    class Leaf:
        pass

    root = SimpleNamespace(a=SimpleNamespace(b=SimpleNamespace(c=Leaf())))
    found = attn_ft.get_module_by_path(root, "a.b.c")
    missing = attn_ft.get_module_by_path(root, "a.x.c")
    print("found type:", type(found).__name__)
    print("missing:", missing)
    assert isinstance(found, Leaf)
    assert missing is None


def test_rotate_half_and_rope():
    print("\n[3] rotate_half + apply_rope")
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    rotated = attn_ft.rotate_half(x)
    print("x:          ", x.tolist())
    print("rotate_half:", rotated.tolist())
    assert torch.equal(rotated, torch.tensor([[-3.0, -4.0, 1.0, 2.0]]))

    q = torch.ones(1, 1, 2, 4)
    k = torch.ones(1, 1, 2, 4) * 2
    cos = torch.ones(1, 2, 4)
    sin = torch.zeros(1, 2, 4)
    q_rope, k_rope = attn_ft.apply_rope(q, k, cos, sin)
    print("RoPE with cos=1/sin=0 keeps tensors unchanged.")
    assert torch.equal(q_rope, q)
    assert torch.equal(k_rope, k)


def test_repeat_kv():
    print("\n[4] repeat_kv")
    kv = torch.arange(1 * 2 * 3 * 1).view(1, 2, 3, 1)
    repeated = attn_ft.repeat_kv(kv, 2)
    print("input shape:   ", tuple(kv.shape))
    print("repeated shape:", tuple(repeated.shape))
    print("repeated heads:", repeated[0, :, :, 0].tolist())
    assert repeated.shape == (1, 4, 3, 1)
    assert torch.equal(repeated[:, 0], kv[:, 0])
    assert torch.equal(repeated[:, 1], kv[:, 0])
    assert torch.equal(repeated[:, 2], kv[:, 1])
    assert torch.equal(repeated[:, 3], kv[:, 1])


def test_find_between_and_span_mask():
    print("\n[5] find_between + build_token_span_mask")
    text = "QUESTION:\nWho are players?\n\nSCHEMA:\n{\"label\":\"Player\"}\n\nGenerate a Cypher query"
    query = attn_ft.find_between(text, "QUESTION:\n", "\n\nSCHEMA:")
    schema = attn_ft.find_between(text, "SCHEMA:\n", "\n\nGenerate a Cypher query")
    print("query: ", repr(query))
    print("schema:", repr(schema))
    assert query == "Who are players?"
    assert schema == '{"label":"Player"}'

    offsets = [(0, 8), (10, 13), (14, 17), (18, 26), (28, 35)]
    mask = attn_ft.build_token_span_mask(offsets, 10, 26, max_length=8, device=torch.device("cpu"))
    print("offset mask true idx:", torch.where(mask)[0].tolist())
    assert torch.where(mask)[0].tolist() == [1, 2, 3]


def test_pair_masks():
    print("\n[6] build_pair_mask + build_cypher_prefix_pair_mask")
    row_mask = torch.tensor([[False, False, True, True]])
    col_mask = torch.tensor([[True, True, False, False]])
    pair = attn_ft.build_pair_mask(row_mask, col_mask)
    print("pair shape:", tuple(pair.shape))
    print("true pairs:", torch.where(pair[0, 0]))
    assert pair.shape == (1, 1, 4, 4)
    assert pair[0, 0, 2, 0]
    assert pair[0, 0, 2, 1]
    assert not pair[0, 0, 2, 2]

    cypher_mask = torch.tensor([[False, True, True, True]])
    prefix = attn_ft.build_cypher_prefix_pair_mask(cypher_mask)
    true_pairs = list(zip(torch.where(prefix[0, 0])[0].tolist(), torch.where(prefix[0, 0])[1].tolist()))
    print("prefix true pairs:", true_pairs)
    assert true_pairs == [(2, 1), (3, 1), (3, 2)]


def test_masked_attention_distribution_loss():
    print("\n[7] masked_attention_distribution_loss")
    args = SimpleNamespace(attention_eps=1e-8, attention_loss_type="kl")

    student = torch.tensor([[[[0.7, 0.3, 0.0], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]]]])
    teacher = student.clone()
    row_mask = torch.tensor([[False, True, True]])
    col_mask = torch.tensor([[True, True, False]])
    pair_mask = attn_ft.build_pair_mask(row_mask, col_mask)

    for loss_type in ["kl", "js", "mse", "raw_mse", "mass_mse", "cka"]:
        args.attention_loss_type = loss_type
        loss = attn_ft.masked_attention_distribution_loss(student, teacher, pair_mask, args)
        assert_close(f"{loss_type} identical attention loss", loss.item(), 0.0, tol=1e-5)

    args.attention_loss_type = "kl"
    teacher2 = torch.tensor([[[[0.7, 0.3, 0.0], [0.8, 0.1, 0.1], [0.9, 0.05, 0.05]]]])
    loss = attn_ft.masked_attention_distribution_loss(student, teacher2, pair_mask, args)
    print("kl different attention loss:", float(loss))
    assert float(loss) > 0

    args.attention_loss_type = "mass_mse"
    loss = attn_ft.masked_attention_distribution_loss(student, teacher2, pair_mask, args)
    print("mass_mse different attention loss:", float(loss))
    assert float(loss) > 0

    args.attention_loss_type = "cka"
    scaled_teacher = student * 2.0
    loss = attn_ft.masked_attention_distribution_loss(student, scaled_teacher, pair_mask, args)
    assert_close("cka scaled-identical attention loss", loss.item(), 0.0, tol=1e-5)

    loss = attn_ft.masked_attention_distribution_loss(student, teacher2, pair_mask, args)
    print("cka different attention loss with two rows (degenerate/neutral):", float(loss))
    assert float(loss) >= 0

    # CKA needs enough valid rows to be informative. With only two rows,
    # centered linear CKA is often degenerate, so test a larger non-collinear map.
    student_big = torch.tensor([[[
        [0.40, 0.30, 0.20, 0.10],
        [0.10, 0.60, 0.20, 0.10],
        [0.25, 0.15, 0.50, 0.10],
        [0.10, 0.20, 0.20, 0.50],
    ]]])
    teacher_big = torch.tensor([[[
        [0.10, 0.20, 0.60, 0.10],
        [0.50, 0.20, 0.20, 0.10],
        [0.10, 0.60, 0.20, 0.10],
        [0.30, 0.10, 0.10, 0.50],
    ]]])
    big_mask = torch.ones(1, 1, 4, 4, dtype=torch.bool)
    loss = attn_ft.masked_attention_distribution_loss(student_big, teacher_big, big_mask, args)
    print("cka different attention loss with four rows:", float(loss))
    assert float(loss) > 0


def test_branch_attention_loss_type():
    print("\n[8] get_branch_attention_loss_type")
    args = SimpleNamespace(
        attention_loss_type="cka",
        query_attention_loss_type=None,
        schema_attention_loss_type="raw_mse",
        cypher_attention_loss_type=None,
    )
    print("query loss type: ", attn_ft.get_branch_attention_loss_type(args, "query"))
    print("schema loss type:", attn_ft.get_branch_attention_loss_type(args, "schema"))
    print("cypher loss type:", attn_ft.get_branch_attention_loss_type(args, "cypher"))
    assert attn_ft.get_branch_attention_loss_type(args, "query") == "cka"
    assert attn_ft.get_branch_attention_loss_type(args, "schema") == "raw_mse"
    assert attn_ft.get_branch_attention_loss_type(args, "cypher") == "mass_mse"

    args.cypher_attention_loss_type = "cka"
    print("explicit cypher loss type:", attn_ft.get_branch_attention_loss_type(args, "cypher"))
    assert attn_ft.get_branch_attention_loss_type(args, "cypher") == "cka"


def load_sample(args, device):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="right")
    tokenizer.eos_token_id = 151645
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

    with open(args.data_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx == args.sample_index:
                raw_item = json.loads(line)
                break
        else:
            raise IndexError(f"sample-index {args.sample_index} is out of range")

    sample = build_one_sample(raw_item, tokenizer, args.max_length, args.max_prompt_length, device)
    model_batch = {
        "input_ids": sample["input_ids"],
        "attention_mask": sample["attention_mask"],
    }
    no_model_batch = {
        "label": sample["labels"],
    }
    return tokenizer, sample, model_batch, no_model_batch


def test_region_masks(args):
    print("\n[9] get_attention_region_masks fallback on one real sample")
    device = torch.device("cpu")
    tokenizer, sample, model_batch, no_model_batch = load_sample(args, device)

    query_mask, schema_mask, cypher_mask = attn_ft.get_attention_region_masks(
        SimpleNamespace(),
        tokenizer,
        model_batch,
        no_model_batch,
    )

    print("query token count: ", int(query_mask.sum()))
    print("schema token count:", int(schema_mask.sum()))
    print("cypher row count:  ", int(cypher_mask.sum()))
    print("query idx range:   ", torch.where(query_mask[0])[0][0].item(), "->", torch.where(query_mask[0])[0][-1].item())
    print("schema idx range:  ", torch.where(schema_mask[0])[0][0].item(), "->", torch.where(schema_mask[0])[0][-1].item())
    print("cypher idx range:  ", torch.where(cypher_mask[0])[0][0].item(), "->", torch.where(cypher_mask[0])[0][-1].item())

    assert int(query_mask.sum()) > 0
    assert int(schema_mask.sum()) > 0
    assert int(cypher_mask.sum()) > 0


def test_cypher_mask_from_labels(args):
    print("\n[10] cypher_mask is taken from label != -100")
    device = torch.device("cpu")
    tokenizer, sample, model_batch, no_model_batch = load_sample(args, device)

    labels = no_model_batch["label"][0]
    cypher_mask = (labels != -100) & model_batch["attention_mask"][0].bool()
    cypher_positions = torch.where(cypher_mask)[0]
    cypher_label_ids = labels[cypher_mask].tolist()
    decoded_from_labels = tokenizer.decode(cypher_label_ids, skip_special_tokens=False)

    print(f"raw response:          {sample['response']}")
    print(f"decoded label region:  {decoded_from_labels}")
    print(f"cypher row count:      {int(cypher_mask.sum().item())}")
    print(f"cypher row idx range:  {int(cypher_positions[0])} -> {int(cypher_positions[-1])}")
    print("\nfirst cypher rows")
    print("idx | input token -> next-token label")
    print("-" * 72)

    input_ids = model_batch["input_ids"][0]
    for pos in cypher_positions[:12].tolist():
        input_text = tokenizer.decode([int(input_ids[pos].item())], skip_special_tokens=False)
        label_text = tokenizer.decode([int(labels[pos].item())], skip_special_tokens=False)
        print(f"{pos:>3} | {repr(input_text):<18} -> {repr(label_text)}")

    assert decoded_from_labels.startswith(sample["response"])
    assert decoded_from_labels.endswith("<|im_end|>")


def test_model_attention_hooks(args):
    print("\n[11] get_transformer_layers + register_attention_hooks + reconstruct_attention")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, sample, model_batch, no_model_batch = load_sample(args, device)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        attn_implementation="eager",
    ).to(device)
    model.eval()

    layers, layer_path = attn_ft.get_transformer_layers(model)
    layer_idx = attn_ft.resolve_layer_index(args.layer_idx, len(layers))
    print("layer path:", layer_path)
    print("layer idx: ", layer_idx)

    captured = {}
    handles = attn_ft.register_attention_hooks(
        model,
        selected_layers=[layer_idx],
        captured=captured,
        detach=True,
        require_training=False,
    )

    with torch.no_grad():
        outputs = model(
            **model_batch,
            use_cache=False,
            output_attentions=True,
            return_dict=True,
        )

    for handle in handles:
        handle.remove()

    hook_attn = captured[layer_idx]
    output_attn = outputs.attentions[layer_idx]
    print("hook attn shape:  ", tuple(hook_attn.shape))
    print("output attn shape:", tuple(output_attn.shape))
    max_abs = (hook_attn - output_attn).abs().max().item()
    mean_abs = (hook_attn - output_attn).abs().mean().item()
    print(f"hook vs output_attentions max_abs={max_abs:.10f}, mean_abs={mean_abs:.10f}")
    assert max_abs < args.attn_tolerance

    cache_args = SimpleNamespace(
        attention_eps=1e-8,
        attention_loss_type="kl",
        w_attention_loss=1.0,
        w_query_attention_loss=1.0,
        w_schema_attention_loss=1.0,
        w_cypher_attention_loss=0.5,
        use_query_attention_loss=True,
        use_schema_attention_loss=True,
        use_cypher_attention_loss=True,
    )
    loss_same = attn_ft.compute_overall_attention_loss(
        cache_args,
        tokenizer,
        model_batch,
        no_model_batch,
        {layer_idx: hook_attn},
        {layer_idx: hook_attn},
        [layer_idx],
        [layer_idx],
    )
    assert_close("overall attention loss with identical caches", loss_same.item(), 0.0, tol=1e-5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", default="benchmarks/Cypherbench/dev.jsonl")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--model-path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--max-length", type=int, default=892)
    parser.add_argument("--max-prompt-length", type=int, default=797)
    parser.add_argument("--with-model", action="store_true")
    parser.add_argument("--layer-idx", type=int, default=-1)
    parser.add_argument("--attn-tolerance", type=float, default=1e-5)
    args = parser.parse_args()

    test_resolve_layer_index()
    test_get_module_by_path()
    test_rotate_half_and_rope()
    test_repeat_kv()
    test_find_between_and_span_mask()
    test_pair_masks()
    test_masked_attention_distribution_loss()
    test_branch_attention_loss_type()
    test_region_masks(args)
    test_cypher_mask_from_labels(args)

    if args.with_model:
        test_model_attention_hooks(args)
    else:
        print("\n[11] model hook test skipped. Add --with-model to compare hook attention with output_attentions=True.")

    print("\nAll requested tests finished.")


if __name__ == "__main__":
    main()

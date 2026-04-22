import json
import sys
import types
import unittest

# Lightweight stubs to import lm_datasets without full training dependencies.
if "deepspeed" not in sys.modules:
    sys.modules["deepspeed"] = types.ModuleType("deepspeed")
if "accelerate" not in sys.modules:
    accelerate_mod = types.ModuleType("accelerate")
    accelerate_mod.load_checkpoint_and_dispatch = lambda *args, **kwargs: None
    accelerate_mod.init_empty_weights = lambda *args, **kwargs: None
    sys.modules["accelerate"] = accelerate_mod
if "peft" not in sys.modules:
    peft_mod = types.ModuleType("peft")
    peft_mod.get_peft_model = lambda *args, **kwargs: None
    peft_mod.LoraConfig = object
    peft_mod.TaskType = object
    peft_mod.PeftModel = object
    sys.modules["peft"] = peft_mod
if "transformers" not in sys.modules:
    tr_mod = types.ModuleType("transformers")
    tr_mod.AutoModelForCausalLM = object
    tr_mod.AutoTokenizer = object
    tr_mod.AutoConfig = object
    sys.modules["transformers"] = tr_mod

from data_utils.lm_datasets import (
    extract_text2cypher_span_items,
    extract_text2cypher_span_offsets,
)


class TestText2CypherSpans(unittest.TestCase):
    def setUp(self):
        self.cypher = (
            "MATCH (n:Person)<-[r:hasBoardMember]-(m0:Company)"
            "-[r1:basedIn]->(m1:Country {name: 'Russia'}) "
            "WITH count(DISTINCT m0) AS num, n "
            "RETURN n.name, num"
        )
        self.response = json.dumps({"cypher": self.cypher}, ensure_ascii=False)
        self.full_text = (
            "QUESTION: ai la board member cua company tai Nga?\n"
            "OUTPUT:\n"
            + self.response
        )

    def test_clause_pattern_expression_and_alias_present(self):
        items = extract_text2cypher_span_items(self.cypher)
        by_type = {}
        for item in items:
            by_type.setdefault(item["type"], set()).add(item["text"])

        self.assertIn("MATCH", by_type.get("clause", set()))
        self.assertIn("WITH", by_type.get("clause", set()))
        self.assertIn("RETURN", by_type.get("clause", set()))

        self.assertIn("(n:Person)", by_type.get("pattern", set()))
        self.assertIn("<-[r:hasBoardMember]-", by_type.get("pattern", set()))
        self.assertIn("(m1:Country {name: 'Russia'})", by_type.get("pattern", set()))

        self.assertIn("count(DISTINCT m0) AS num", by_type.get("expression", set()))
        self.assertIn("n.name", by_type.get("expression", set()))

        self.assertIn("n", by_type.get("variable_alias", set()))
        self.assertIn("m0", by_type.get("variable_alias", set()))
        self.assertIn("AS num", by_type.get("variable_alias", set()))

    def test_offsets_can_map_back_to_original_text(self):
        offsets = extract_text2cypher_span_offsets(self.full_text, self.response)
        spans = [self.full_text[s:e] for s, e in offsets]

        self.assertIn("MATCH", spans)
        self.assertIn("(n:Person)", spans)
        self.assertIn("count(DISTINCT m0) AS num", spans)
        self.assertIn("AS num", spans)


if __name__ == "__main__":
    unittest.main()

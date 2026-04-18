# Updated Fine-tuning Scripts

These scripts mirror the existing `scripts/qwen/*` launch style, but call
`updated_finetune.py` and expose the new loss switches added in `arguments.py`.

Method folders:

- `qwen/csd`: CSD logit distillation.
- `qwen/distillm`: adaptive SRKL DistillM-style training.
- `qwen/fdd`: updated representation-heavy distillation using span representation, span relation, and generated-query relation losses.
- `qwen/kd`: standard FKL/RKL knowledge distillation.
- `qwen/sfkl`: skewed forward KL distillation.
- `qwen/sft`: supervised fine-tuning without a teacher.

Recommended starting points:

- `qwen/updated_loss/train_0.6B_4B_all_losses.sh`: logit KD + query attention + Cypher-prefix attention + schema-used attention + span representation + span relational + generated-query relational.
- `qwen/updated_loss/train_0.6B_4B_attention.sh`: logit KD + attention losses only.
- `qwen/updated_loss/train_0.6B_4B_span_rep_rel.sh`: logit KD + representation losses + schema-used attention.

`--use-schema-attention-loss` uses the schema terms referenced by the gold
Cypher as its key/value side, so it distills Cypher-to-used-schema grounding.
If those masks cannot be built for a batch, training falls back to the full
schema span. `--use-cypher-attention-loss` remains Cypher-prefix attention.

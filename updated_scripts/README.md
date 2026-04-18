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

- `qwen/updated_loss/train_0.6B_4B_all_losses.sh`: logit KD + query/cypher attention + span representation + span relational + generated-query relational.
- `qwen/updated_loss/train_0.6B_4B_attention.sh`: logit KD + attention losses only.
- `qwen/updated_loss/train_0.6B_4B_span_rep_rel.sh`: logit KD + representation losses only.

Schema attention is intentionally left off in these templates because the
current dataset collate does not provide `schema_mask`. Enable it only after
adding `schema_mask` to `no_model_batch`.

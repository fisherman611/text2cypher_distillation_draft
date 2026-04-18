import random
import torch
import os
import json
import pickle
import re
import numpy as np
from torch.utils.data import Dataset
from .distributed_indexed import DistributedMMapIndexedDataset

from torch.distributed import get_rank, get_world_size, barrier
from utils import print_rank
from utils import save_rank


class LMTrainDataset(Dataset):
    def __init__(self, args, tokenizer, path, split, num, ratio, rng_sample: random.Random):
        self.args = args
        self.tokenizer = tokenizer
        self.split = split
        self.pad_id = self.tokenizer.eos_token_id
        self.ratio = ratio
        self.max_length = args.max_length
        self.max_prompt_length = args.max_prompt_length
        self.rng_sample = rng_sample
        self.lm_ctx = DistributedMMapIndexedDataset(path, f"{split}", get_rank(), get_world_size())
        self.t_lm_ctx = None

        if os.path.exists(os.path.join(path, f"teacher_train_0.bin")) and split == "train":
            self.t_lm_ctx = DistributedMMapIndexedDataset(path, f"teacher_train", get_rank(), get_world_size())

        if os.path.exists(os.path.join(path, f"{split}.jsonl")):
            with open(os.path.join(path, f"{split}.jsonl")) as f:
                self.raw = [json.loads(line) for line in f.readlines()]
                self.answers = [x["response"] if isinstance(x["response"], list) else [x["response"]] for x in self.raw]
                self.full_texts = [x["prompt"] + x["response"] for x in self.raw]
                self.offset_mapping = [tokenizer(text, return_offsets_mapping=True, truncation=True, 
                                                 max_length=self.max_length, padding="max_length",
                                                 add_special_tokens=False, return_tensors="pt")["offset_mapping"]
                                       for text in self.full_texts]
                
                self.get_span_offsets()
                self.get_query_offsets()
                self.get_schema_offsets()
        
        print_rank(len(self.lm_ctx))
        if num == -1:
            self.num = len(self.lm_ctx)
        else:
            self.num = num

        print_rank(f"Num LM instances: {len(self.lm_ctx)}")

    def _event_span_offsets(self, response_json, full_text, response_start):
        values_to_find = []
        for event in response_json.get("events", []):
            values_to_find.append(event[0])
            values_to_find.append(event[1])

            if len(event) > 3:
                for arg in event[2]:
                    values_to_find.append(arg[0])
                    values_to_find.append(arg[1])

                values_to_find.append(event[3])
            else:
                values_to_find.append(event[2])

        result_tuples = []
        search_start_idx = response_start
        for val in values_to_find:
            search_str = f"{val}"
            char_start = full_text.find(search_str, search_start_idx)

            if char_start != -1:
                char_end = char_start + len(search_str)
                result_tuples.append((char_start, char_end))
                search_start_idx = char_end

        return result_tuples

    def _cypher_span_offsets(self, response_str, response_start):
        span_offsets = []

        def add_matches(pattern, group=1, flags=0):
            for match in re.finditer(pattern, response_str, flags):
                start, end = match.span(group)
                if end > start:
                    span_offsets.append((response_start + start, response_start + end))

        add_matches(r":\s*([A-Za-z_][A-Za-z0-9_]*)")
        add_matches(r"\.\s*([A-Za-z_][A-Za-z0-9_]*)")
        add_matches(r"[\{,]\s*([A-Za-z_][A-Za-z0-9_]*)\s*:")
        add_matches(r"'((?:\\.|[^'\\])*)'")
        add_matches(r"(?<![A-Za-z0-9_])(-?\d+(?:\.\d+)?)(?![A-Za-z0-9_])")
        add_matches(
            r"\b(MATCH|OPTIONAL\s+MATCH|WHERE|WITH|DISTINCT|RETURN|ORDER\s+BY|"
            r"LIMIT|SKIP|ASC|DESC|COUNT|SUM|AVG|MIN|MAX|CASE|WHEN|THEN|ELSE|END)\b",
            flags=re.IGNORECASE,
        )

        return sorted(set(span_offsets), key=lambda x: (x[0], x[1]))

    def get_span_offsets(self):
        self.span_offsets = []
        for item, full_text in zip(self.raw, self.full_texts):
            response_str = item["response"]
            response_start = len(item["prompt"])

            try:
                response_json = json.loads(response_str)
            except json.JSONDecodeError:
                response_json = {}

            if isinstance(response_json, dict) and "events" in response_json:
                self.span_offsets.append(
                    self._event_span_offsets(response_json, full_text, response_start)
                )
            else:
                self.span_offsets.append(self._cypher_span_offsets(response_str, response_start))

    def get_query_offsets(self):
        self.query_offsets = []
        pattern = re.compile(r"QUESTION:\s*\n(.*?)\n\s*\nSCHEMA:", re.DOTALL)
        for item in self.raw:
            prompt = item["prompt"]
            match = pattern.search(prompt)
            if match is None:
                self.query_offsets.append(None)
            else:
                self.query_offsets.append(match.span(1))

    def _extract_response_cypher(self, response):
        if not isinstance(response, str):
            return ""

        try:
            parsed = json.loads(response)
            if isinstance(parsed, dict):
                cypher = parsed.get("cypher") or parsed.get("query") or parsed.get("pred_cypher")
                if isinstance(cypher, str):
                    return cypher
        except json.JSONDecodeError:
            pass

        return response

    def _extract_cypher_schema_terms(self, cypher):
        if not cypher:
            return set()

        terms = set()
        for match in re.finditer(r":\s*([A-Za-z_][A-Za-z0-9_]*)", cypher):
            terms.add(match.group(1))
        for match in re.finditer(r"\.\s*([A-Za-z_][A-Za-z0-9_]*)", cypher):
            terms.add(match.group(1))
        for match in re.finditer(r"[\{,]\s*([A-Za-z_][A-Za-z0-9_]*)\s*:", cypher):
            terms.add(match.group(1))
        return terms

    def _find_schema_term_offsets(self, schema_text, schema_start, terms):
        offsets = []
        for term in sorted(terms, key=len, reverse=True):
            pattern = re.compile(rf'"{re.escape(term)}"')
            for match in pattern.finditer(schema_text):
                offsets.append((schema_start + match.start(), schema_start + match.end()))
        return sorted(set(offsets), key=lambda x: (x[0], x[1]))

    def get_schema_offsets(self):
        self.schema_offsets = []
        self.cypher_schema_offsets = []
        schema_pattern = re.compile(
            r"SCHEMA:\s*\n(?P<schema>.*?)(?:\n\s*\nGenerate a Cypher query|\Z)",
            re.DOTALL,
        )

        for item in self.raw:
            prompt = item["prompt"]
            match = schema_pattern.search(prompt)
            if match is None:
                self.schema_offsets.append(None)
                self.cypher_schema_offsets.append([])
                continue

            schema_start, schema_end = match.span("schema")
            schema_text = match.group("schema")
            self.schema_offsets.append((schema_start, schema_end))

            cypher = self._extract_response_cypher(item.get("response", ""))
            used_terms = self._extract_cypher_schema_terms(cypher)
            self.cypher_schema_offsets.append(
                self._find_schema_term_offsets(schema_text, schema_start, used_terms)
            )

    def __len__(self):
        return self.num
   
    def __getitem__(self, index):
        return self._get_lm(index)
    
    def _get_lm(self, index):
        data = self.lm_ctx[index]
        input_ids = data.astype(int)

        t_input_ids = None
        if self.t_lm_ctx is not None:
            t_data = self.t_lm_ctx[index]
            t_input_ids = t_data.astype(int)

        return {
            "input_ids": input_ids,
            "t_input_ids": t_input_ids,
            "span_offsets": self.span_offsets[index],
            "offset_mapping": self.offset_mapping[index],
            "query_offsets": self.query_offsets[index],
            "schema_offsets": self.schema_offsets[index],
            "cypher_schema_offsets": self.cypher_schema_offsets[index],
        }

    def _process_lm(self, i, samp, model_data, no_model_data, gen_data):
        input_ids = samp["input_ids"]
        source_len = 1
        
        prompt = None
        if self.args.model_type in ["qwen"] and 4294967295 in input_ids:
            source_len = np.where(input_ids==4294967295)[0][0]
            prompt = input_ids[:source_len]
            input_ids = np.concatenate([input_ids[:source_len], input_ids[source_len+1:]], axis=0)
        elif 65535 in input_ids:
            source_len = np.where(input_ids==65535)[0][0]
            prompt = input_ids[:source_len]
            input_ids = np.concatenate([input_ids[:source_len], input_ids[source_len+1:]], axis=0)
        
        input_ids = input_ids[:self.max_length]
        input_len = len(input_ids)
        model_data["input_ids"][i][:input_len-1] = torch.tensor(input_ids[:-1], dtype=torch.long)
        model_data["attention_mask"][i][:input_len-1] = 1.0
        if self.args.model_type in ["gpt2"]:
            model_data["position_ids"][i][:input_len-1] = torch.arange(0, input_len-1, dtype=torch.long)
        no_model_data["label"][i][:input_len-1] = torch.tensor(input_ids[1:], dtype=torch.long)
        no_model_data["label"][i][:source_len-1] = -100
        if "loss_mask" in no_model_data:
            no_model_data["loss_mask"][i][:input_len-1] = 1.0
            no_model_data["loss_mask"][i][:source_len-1] = 0
        
        if prompt is not None and gen_data is not None:
            gen_data["input_ids"][i][-len(prompt):] = torch.tensor(prompt, dtype=torch.long)
            gen_data["attention_mask"][i][-len(prompt):] = 1.0

    def move_to_device(self, model_data, no_model_data, gen_data, device):
        for k in model_data:
            model_data[k] = model_data[k].to(device)

        for k in no_model_data:
            if isinstance(no_model_data[k], torch.Tensor):
                no_model_data[k] = no_model_data[k].to(device)

        if gen_data is not None:
            for k in gen_data:
                gen_data[k] = gen_data[k].to(device)

        return model_data, no_model_data, gen_data

    def _build_char_span_mask(self, samples, span_key):
        mask = torch.zeros(len(samples), self.max_length, dtype=torch.bool)
        for i, sample in enumerate(samples):
            span = sample.get(span_key)
            if span is None:
                continue
            span_start, span_end = span
            offsets = sample["offset_mapping"][0, :self.max_length]
            token_starts = offsets[:, 0]
            token_ends = offsets[:, 1]
            token_mask = (token_starts < span_end) & (token_ends > span_start)
            token_mask = token_mask & (token_ends > token_starts)
            mask[i, :token_mask.size(0)] = token_mask
        return mask

    def _build_char_spans_mask(self, samples, span_key):
        mask = torch.zeros(len(samples), self.max_length, dtype=torch.bool)
        for i, sample in enumerate(samples):
            spans = sample.get(span_key) or []
            if not spans:
                continue
            offsets = sample["offset_mapping"][0, :self.max_length]
            token_starts = offsets[:, 0]
            token_ends = offsets[:, 1]
            sample_mask = torch.zeros_like(token_starts, dtype=torch.bool)
            for span_start, span_end in spans:
                token_mask = (token_starts < span_end) & (token_ends > span_start)
                token_mask = token_mask & (token_ends > token_starts)
                sample_mask |= token_mask
            mask[i, :sample_mask.size(0)] = sample_mask
        return mask

    def collate(self, samples):
        bs = len(samples)

        max_length = self.max_length
        
        model_data = {
            "input_ids": torch.ones(bs, max_length, dtype=torch.long) * self.pad_id,
            "attention_mask": torch.zeros(bs, max_length),
        }
        
        if self.args.model_type in ["gpt2"]:
            model_data["position_ids"] = torch.zeros(bs, max_length, dtype=torch.long)
            
        no_model_data = {
            "label": torch.ones(bs, max_length, dtype=torch.long) * -100,
            "loss_mask": torch.zeros(bs, max_length),
            "span_offsets": [sample["span_offsets"] for sample in samples],
            "offset_mapping": torch.concat([sample["offset_mapping"] for sample in samples]),
            "query_mask": self._build_char_span_mask(samples, "query_offsets"),
            "schema_mask": self._build_char_span_mask(samples, "schema_offsets"),
            "cypher_schema_mask": self._build_char_spans_mask(samples, "cypher_schema_offsets"),
        }
        
        gen_data = {
            "input_ids": torch.ones(bs, self.max_prompt_length, dtype=torch.long) * self.pad_id,
            "attention_mask": torch.zeros(bs, self.max_prompt_length, dtype=torch.long),
        }

        for i, samp in enumerate(samples):
            self._process_lm(i, samp, model_data, no_model_data, gen_data)

        t_model_data, t_no_model_data = None, None
        if samples[0]["t_input_ids"] is not None:
            t_model_data = {
                "input_ids": torch.ones(bs, self.args.t_max_length, dtype=torch.long) * self.pad_id,
                "attention_mask": torch.zeros(bs, self.args.t_max_length),
            }
            
            if self.args.model_type in ["gpt2"]:
                t_model_data["position_ids"] = torch.zeros(bs, self.args.t_max_length, dtype=torch.long)
                
            t_no_model_data = {
                "label": torch.ones(bs, self.args.t_max_length, dtype=torch.long) * -100,
            }

            for i, samp in enumerate(samples):
                self._process_lm(i, {"input_ids": samp["t_input_ids"]}, t_model_data, t_no_model_data, None)
        
        return model_data, no_model_data, gen_data, t_model_data, t_no_model_data


class LMEvalDataset(Dataset):
    def __init__(self, args, tokenizer, path, split, rng_sample: random.Random):
        self.args = args
        self.tokenizer = tokenizer
        self.split = split
        self.pad_id = self.tokenizer.eos_token_id
        self.max_length = args.max_length
        self.max_prompt_length = args.max_prompt_length
        self.rng_sample = rng_sample
        self.lm_ctx = DistributedMMapIndexedDataset(path, f"{split}", 0, 1)

        if os.path.exists(os.path.join(path, f"{split}.jsonl")):
            with open(os.path.join(path, f"{split}.jsonl")) as f:
                self.raw = [json.loads(line) for line in f.readlines()]
                self.answers = [x["response"] if isinstance(x["response"], list) else [x["response"]] for x in self.raw]
        
        self.num = len(self.lm_ctx)

        print(f"Num LM instances: {len(self.lm_ctx)}")

    def __len__(self):
        return self.num
   
    def __getitem__(self, index):
        return self._get_lm(index)
    
    def _get_lm(self, index):
        data = self.lm_ctx[index]
        input_ids = data.astype(int)
        return {
            "input_ids": input_ids
        }

    def _process_lm(self, i, samp, model_data, no_model_data, gen_data):
        input_ids = samp["input_ids"]
        source_len = 1
        
        prompt = None
        if self.args.model_type in ["qwen"] and 4294967295 in input_ids:
            source_len = np.where(input_ids==4294967295)[0][0]
            prompt = input_ids[:source_len]
            input_ids = np.concatenate([input_ids[:source_len], input_ids[source_len+1:]], axis=0)
        elif 65535 in input_ids:
            source_len = np.where(input_ids==65535)[0][0]
            prompt = input_ids[:source_len]
            input_ids = np.concatenate([input_ids[:source_len], input_ids[source_len+1:]], axis=0)
        
        input_ids = input_ids[:self.max_length]
        input_len = len(input_ids)
        model_data["input_ids"][i][:input_len-1] = torch.tensor(input_ids[:-1], dtype=torch.long)
        model_data["attention_mask"][i][:input_len-1] = 1.0
        if self.args.model_type in ["gpt2"]:
            model_data["position_ids"][i][:input_len-1] = torch.arange(0, input_len-1, dtype=torch.long)
        no_model_data["label"][i][:input_len-1] = torch.tensor(input_ids[1:], dtype=torch.long)
        no_model_data["label"][i][:source_len-1] = -100
        no_model_data["loss_mask"][i][:input_len-1] = 1.0
        no_model_data["loss_mask"][i][:source_len-1] = 0
        
        if prompt is not None:
            gen_data["input_ids"][i][-len(prompt):] = torch.tensor(prompt, dtype=torch.long)
            gen_data["attention_mask"][i][-len(prompt):] = 1.0

    def move_to_device(self, model_data, no_model_data, gen_data, device):
        for k in model_data:
            model_data[k] = model_data[k].to(device)

        for k in no_model_data:
            no_model_data[k] = no_model_data[k].to(device)

        for k in gen_data:
            gen_data[k] = gen_data[k].to(device)

        return model_data, no_model_data, gen_data

    def collate(self, samples):
        bs = len(samples)

        max_length = self.max_length
        
        model_data = {
            "input_ids": torch.ones(bs, max_length, dtype=torch.long) * self.pad_id,
            "attention_mask": torch.zeros(bs, max_length),
        }
        
        if self.args.model_type in ["gpt2"]:
            model_data["position_ids"] = torch.zeros(bs, max_length, dtype=torch.long)
            
        no_model_data = {
            "label": torch.ones(bs, max_length, dtype=torch.long) * -100,
            "loss_mask": torch.zeros(bs, max_length)
        }
        
        gen_data = {
            "input_ids": torch.ones(bs, self.max_prompt_length, dtype=torch.long) * self.pad_id,
            "attention_mask": torch.zeros(bs, self.max_prompt_length, dtype=torch.long),
        }

        for i, samp in enumerate(samples):
            self._process_lm(i, samp, model_data, no_model_data, gen_data)
        
        return model_data, no_model_data, gen_data

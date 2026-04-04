import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
with open(r"results/Cypherbench/calculated_scores_Qwen3_4B_Instruct_2507/company_cyphers_result.json", "r", encoding="utf-8") as f:
    company = json.load(f)

with open(r"results/Cypherbench/calculated_scores_Qwen3_4B_Instruct_2507/fictional_character_cyphers_result.json", "r", encoding="utf-8") as f:
    fictional_character = json.load(f)
    
with open(r"results/Cypherbench/calculated_scores_Qwen3_4B_Instruct_2507/flight_accident_cyphers_result.json", "r", encoding="utf-8") as f:
    flight_accident = json.load(f)
    
with open(r"results/Cypherbench/calculated_scores_Qwen3_4B_Instruct_2507/geography_cyphers_result.json", "r", encoding="utf-8") as f:
    geography = json.load(f)
    
with open(r"results/Cypherbench/calculated_scores_Qwen3_4B_Instruct_2507/movie_cyphers_result.json", "r", encoding="utf-8") as f:
    movie = json.load(f)
    
with open(r"results/Cypherbench/calculated_scores_Qwen3_4B_Instruct_2507/nba_cyphers_result.json", "r", encoding="utf-8") as f:
    nba = json.load(f)
    
with open(r"results/Cypherbench/calculated_scores_Qwen3_4B_Instruct_2507/politics_cyphers_result.json", "r", encoding="utf-8") as f:
    politics = json.load(f)

test = company + fictional_character + flight_accident + geography + movie + nba + politics
with open(r"results/Cypherbench/calculated_scores_Qwen3_4B_Instruct_2507/test_result.json", "w", encoding="utf-8") as f:
    json.dump(test, f, ensure_ascii=False, indent=2)
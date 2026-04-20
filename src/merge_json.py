import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
with open(r"results\Cypherbench\calculated_scores_Qwen3_0.6B_4B_fkl_attn_loss_2/company_cyphers_result.json", "r", encoding="utf-8") as f:
    company = json.load(f)

with open(r"results\Cypherbench\calculated_scores_Qwen3_0.6B_4B_fkl_attn_loss_2/fictional_character_cyphers_result.json", "r", encoding="utf-8") as f:
    fictional_character = json.load(f)
    
with open(r"results\Cypherbench\calculated_scores_Qwen3_0.6B_4B_fkl_attn_loss_2/flight_accident_cyphers_result.json", "r", encoding="utf-8") as f:
    flight_accident = json.load(f)
    
with open(r"results\Cypherbench\calculated_scores_Qwen3_0.6B_4B_fkl_attn_loss_2/geography_cyphers_result.json", "r", encoding="utf-8") as f:
    geography = json.load(f)
    
with open(r"results\Cypherbench\calculated_scores_Qwen3_0.6B_4B_fkl_attn_loss_2/movie_cyphers_result.json", "r", encoding="utf-8") as f:
    movie = json.load(f)
    
with open(r"results\Cypherbench\calculated_scores_Qwen3_0.6B_4B_fkl_attn_loss_2/nba_cyphers_result.json", "r", encoding="utf-8") as f:
    nba = json.load(f)
    
with open(r"results\Cypherbench\calculated_scores_Qwen3_0.6B_4B_fkl_attn_loss_2/politics_cyphers_result.json", "r", encoding="utf-8") as f:
    politics = json.load(f)

test = company + fictional_character + flight_accident + geography + movie + nba + politics

with open(r"results\Cypherbench\calculated_scores_Qwen3_0.6B_4B_fkl_attn_loss_2/test_result.json", "w", encoding="utf-8") as f:
    json.dump(test, f, ensure_ascii=False, indent=2)

# with open(r"results/Mind_the_query/calculated_scores_Qwen3_0.6B_4B_csd/bloom50_cyphers_result.json", "r", encoding="utf-8") as f:
#     bloom50 = json.load(f)

# with open(r"results/Mind_the_query/calculated_scores_Qwen3_0.6B_4B_csd/healthcare_cyphers_result.json", "r", encoding="utf-8") as f:
#     healthcare = json.load(f)

# with open(r"results/Mind_the_query/calculated_scores_Qwen3_0.6B_4B_csd/wwc_cyphers_result.json", "r", encoding="utf-8") as f:
#     wwc = json.load(f)

# test = bloom50 + healthcare + wwc
# with open(r"results/Mind_the_query/calculated_scores_Qwen3_0.6B_4B_csd/test_result.json", "w", encoding="utf-8") as f:
#     json.dump(test, f, ensure_ascii=False, indent=2)
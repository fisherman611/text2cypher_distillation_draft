# **Text2Cypher Distillation**

## **Project Structure**

```
text2cypher-distillation/
├── benchmarks/              # Benchmark data after unzipping
├── configs/                 # Training and DeepSpeed configs
├── data_utils/              # Dataset and loader scripts
├── distillm/                # Distillation loss functions (KL, JSD, TVD, etc.)
├── prompts/                 # Prompt templates
├── scripts/                 # Training scripts (SFT, etc.)
├── src/                     # Core evaluation and metric logic
├── benchmarks.zip           # Benchmark datasets (Gold/Input files)
├── finetune.py              # Main training/fine-tuning script
├── main.py                  # Evaluation script
├── install.sh               # Dependency installation script
└── requirements.txt         # Package requirements
```

## **Setup**

### **Clone the Repository**

```bash
git clone https://github.com/fisherman611/text2cypher_distillation_draft
cd text2cypher_distillation_draft
```

### **Setup Environment**

1. Create a Conda environment:
   ```bash
   conda create -n text2cypher python=3.11 -y
   conda activate text2cypher
   ```

2. Install dependencies:
   ```bash
   bash install.sh
   ```

### **Prepare Benchmarks**

1. Unzip the provided benchmark datasets:
   ```bash
   unzip benchmarks.zip
   ```

2. Format and split the data (Example for Cypherbench):
   ```bash
   python format_data.py --benchmark Cypherbench --split train
   python format_data.py --benchmark Cypherbench --split test
   python split_data.py --benchmark Cypherbench
   ```

3. Process the data for training (Generate binary files):
   ```bash
   python process_data.py \
       --data-dir benchmarks/Cypherbench \
       --processed-data-dir processed_data/benchmarks/Cypherbench \
       --model-path Qwen/Qwen3-0.6B \
       --model-type qwen \
       --data-process-workers 8 \
       --max-prompt-length 797 --dev-num 1
   ```

### **Environment Configuration**

Set up your Hugging Face tokens in a `.env` file or export them:
```bash
HF_READ_TOKEN=<your_hf_read_token>
HF_WRITE_TOKEN=<your_hf_write_token>
```

## **Usage**

### **Fine-tuning (SFT)**

To start the fine-tuning process for Qwen3-0.6B on Cypherbench:

```bash
bash scripts/qwen/sft/sft_qwen3_0.6B.sh
```

For the 4B variant:
```bash
bash scripts/qwen/sft/sft_qwen3_4B.sh
```

The training process uses DeepSpeed for memory efficiency and `torchrun` for distributed execution across your available GPUs.

## **Evaluation**

- **`src/evaluator/evaluate.py`**: Main evaluation pipeline
- **`src/metrics/execution_accuracy.py`**: Measures if executed query returns correct results
- **`src/metrics/provenance_subgraph_jaccard_similarity.py`**: Computes graph-based similarity between gold and predicted queries
- **`src/metrics/executable.py`**: Measures if the query is executed or not

## **License**

See [LICENSE](LICENSE) for more details.


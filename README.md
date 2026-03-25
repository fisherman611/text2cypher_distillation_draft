# **Text2Cypher Distillation**

## **Project Structure**

```
text2cypher-distillation/
├── benchmarks/
│   ├── Cypherbench
│   ├── Mind_the_query
│   └── Neo4j_Text2Cypher
├── prompts/
├── src/
│   ├── baseline/
│   ├── evaluator/
│   ├── metrics/
├── main.py
└── requirements.txt
```

## **Setup**

### **Clone the Repository**

```bash
git clone https://github.com/yourusername/text2cypher_distillation_draft
cd text2cypher_distillation_draft
```

### **Download Benchmarks**

The benchmark datasets need to be downloaded and extracted to the `benchmarks/` folder:
1. Download the data from: https://drive.google.com/file/d/14ABwTkrWdu6xirargTlP2KLme7GnitWF/view
2. Extract the contents to the `benchmarks/` folder:
   ```bash
   # After downloading the file
   unzip benchmarks.zip
   ```

### **Install Dependencies**

1. Create a Python virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up .env file
   ```bash
   HF_READ_TOKEN=<your_hf_read_token>
   HF_WRITE_TOKEN=<your_hf_write_token>
   ```

## **Usage**

### **Running the main script**

The main script (`main.py`) runs the Qwen model on selected benchmarks and evaluates the generated Cypher queries.
```bash
python main.py --benchmark "Cypherbench" --db "nba" --model "Qwen/Qwen3-0.6B" --device "cuda" --max-workers 10
```

### **Command-Line Arguments Explanation**

```
--benchmark {Cypherbench,Mind_the_query,Neo4j_Text2Cypher}
    The benchmark dataset to use (default: Cypherbench)

--db DB
    Database/domain name within the benchmark (required)
    Cypherbench: 
        nba
        flight_accident
        fictional_character
        company
        geography
        movie
        politics
    
    Mind_the_query:
        bloom50
        healthcare
        wwc

    Neo4j_Text2Cypher:
        bluesky
        buzzoverflow
        companies
        neoflix
        fincen
        gameofthrones
        grandstack
        movies
        network
        northwind
        offshoreleaks
        recommendations
        stackoverflow2
        twitch
        twitter

--model MODEL
    Hugging Face model name or path (default: Qwen/Qwen3-0.6B, Qwen/Qwen3-4B, Qwen/Qwen3-8B)

--device {cpu,cuda}
    Device to run the model on (default: cuda if available, else cpu)

--max-length MAX_LENGTH
    Maximum generation length for the model (default: 8192)

--max-workers MAX_WORKERS
    Number of worker threads for parallel processing (default: 4)

--limit LIMIT
    Limit the number of test samples to process (default: None, process all)
```

## **Evaluation**

- **`src/evaluator/evaluate.py`**: Main evaluation pipeline
- **`src/metrics/execution_accuracy.py`**: Measures if executed query returns correct results
- **`src/metrics/provenance_subgraph_jaccard_similarity.py`**: Computes graph-based similarity between gold and predicted queries
- **`src/metrics/executable.py`**: Measures if the query is executed or not

## **License**

See [LICENSE](LICENSE) for more details.


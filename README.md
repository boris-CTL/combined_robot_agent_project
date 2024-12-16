## Installation of LLF-Bench's environment
Please follow the instruction in https://github.com/microsoft/LLF-Bench

## Run step-by-step unrolling by using LLMs
First, set up OpenAI key by modifying 
```python
config['api_key'] = "your api key here"
``` 
in `run_MetaWorld_reach_v2_gpt35.py`.

Second, simply run
```bash
python3 run_MetaWorld_reach_v2_gpt35.py
```

## To test WrappedMetaWorldEnv
Under the llm-prl/meta-world/ directory, simply run
```bash
python3 test_WrappedMetaWorldEnv.py
```

from datasets import load_dataset
ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
print(ds.column_names)

ds = load_dataset("princeton-nlp/SWE-bench_oracle", split="test")
print(ds.column_names)
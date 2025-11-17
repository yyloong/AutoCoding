from datasets import load_dataset

# Load the main SWE-bench dataset
swebench = load_dataset('princeton-nlp/SWE-bench', split='test')

# Load other variants as needed
swebench_lite = load_dataset('princeton-nlp/SWE-bench_Lite', split='test')
swebench_verified = load_dataset('princeton-nlp/SWE-bench_Verified', split='test')
swebench_multimodal_dev = load_dataset('princeton-nlp/SWE-bench_Multimodal', split='dev')
swebench_multimodal_test = load_dataset('princeton-nlp/SWE-bench_Multimodal', split='test')
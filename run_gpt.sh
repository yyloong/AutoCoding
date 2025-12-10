rm -rf memory
export OPENAI_API_KEY="Your_OpenAI_API_Key_Here"
export SERPER_KEY_ID="Your_Serper_API_Key_Here"
export JINA_API_KEYS="Your_Jina_API_Keys_Here"
export KAGGLE_API_TOKEN="Your_Kaggle_API_Token_Here"
python -m ms_agent.cli.cli run --config projects/kaggle --trust_remote_code true --openai_api_key ${OPENAI_API_KEY} --query "你需要基于 PyTorch 从零实现一个参数量约为 30M 的 Causal Transformer 模型（即 Mini-GPT），并使用当前目录下的 `data.zip` 进行训练。最终能使得你训练的模型能输出相对合理的文本。我将训练数据打包成了 `/workspace/data.zip`，你需要先解压它，然后使用解压后的数据进行训练。"
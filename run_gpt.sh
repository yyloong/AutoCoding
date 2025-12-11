rm -rf memory
rm -rf output
mkdir output
cp -r ./examples/gpt_data/data ./output/data
export OPENAI_API_KEY="Your_OpenAI_API_Key_Here"
export SERPER_KEY_ID="Your_Serper_API_Key_Here"
export JINA_API_KEYS="Your_Jina_API_Keys_Here"
python -m ms_agent.cli.cli run --config projects/gpt --trust_remote_code true --openai_api_key ${OPENAI_API_KEY} --query "你需要基于 PyTorch 从零实现一个参数量约为 30M 的 Causal Transformer 模型（即 Mini-GPT），并使用我提供的文本进行训练。最终能使得你训练的模型能输出相对合理的文本。训练数据为金庸的多部小说，路径为 /workspace/data/, 格式为多个 .txt 文件。当前设备下有 GPU 可用，请使用 cuda 加速训练过程, 先运行一个快速训练，然后尽量利用多 gpu，调到合适的 batch size，加速训练。最后你需要能训练出一个输出正常文本的模型权重，并说明如何生成文本。需要保证已经有训练好的模型权重，用户可以直接使用该模型权重进行文本生成。"
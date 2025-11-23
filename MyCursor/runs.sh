export OPENAI_API_KEY="sk-d67a35829268468a8e864369c7540fe7"
export SERPER_KEY_ID="41527f920b7e54b0e8b830ba4983866f33bdfe30"
export JINA_API_KEYS="jina_891b2958b3f04e45a1c12f057c5944f2sUlNYsZ_hfDi-Ov47kJNzfz_I42a"
python -m ms_agent.cli.cli run --config MyCursor --trust_remote_code true --openai_api_key ${OPENAI_API_KEY} --query "用 python 写一个经典的数字华容道游戏"
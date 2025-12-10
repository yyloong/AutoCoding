rm -rf memory
rm -rf output
export OPENAI_API_KEY="Your_OpenAI_API_Key_Here"
export SERPER_KEY_ID="Your_Serper_API_Key_Here"
export JINA_API_KEYS="Your_Jina_API_Keys_Here"
export KAGGLE_API_TOKEN="Your_Kaggle_API_Token_Here"
python -m ms_agent.cli.cli run --config projects/kaggle --trust_remote_code true --openai_api_key ${OPENAI_API_KEY} --query "请你参加 kaggle 竞赛 Spaceship Titanic: https://www.kaggle.com/competitions/spaceship-titanic/leaderboard。尽可能获得更高的排名（尽量将预测准确率提升到 0.81 以上）。最终你需要生成一个用于提交的 csv 文件，包含你的预测结果。"
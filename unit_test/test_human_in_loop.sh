rm -rf memory
rm -rf output
mkdir output
python -m ms_agent.cli.cli run --config unit_test/test_human_in_loop --trust_remote_code true --query "请你写一个 python 程序，打印计数从 0 开始到 100，一秒钟打印一个数字。并且用工具运行该函数。"
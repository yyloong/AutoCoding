rm -rf memory
rm -rf new_output
mkdir new_output
python -m ms_agent.cli.cli run --config unit_test/test_state_memory --query 'auction start' --trust_remote_code true --load_cache true
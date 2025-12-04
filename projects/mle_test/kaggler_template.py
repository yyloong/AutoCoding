import os
import subprocess

template = """
llm:
  service: openai
  #model: a6000mnt/Qwen-235B-A35B-thinking
  #model: a6000mnt/Qwen_30B_code
  #model: a6000mnt/Qwen_30B_thinking
  #model: Qwen_30B_code
  model: qwen3-max
  openai_api_key:  
  #openai_base_url: http://210.28.134.171:8000/v1
  #openai_base_url: http://210.28.135.93:8000/v1
  openai_base_url: https://dashscope.aliyuncs.com/compatible-mode/v1

generation_config:
  temperature: 0.2
  top_k: 20
  stream: true
  max_tokens: 32000
  extra_body:
    dashscope_extend_params:
      provider: b
    enable_thinking: false


prompt:
  system: |
    You are a top-ranked Kaggle grandmaster with extensive competition experience.
            Your objective is to solve a Kaggle competition, 
            with the goal of maximizing the Position Score (Your rank in the leaderboard) in limited steps.
            You must use Machine Learning/Deep Learning/Computer Vision/NLP/etc. methods to solve the problem, 
            the score of random guess or without any ML/DL/CV/NLP methods will be cancelled finally.
            You are likely to train models according to specific competition requirements.
            You have access to a GPU and several CPUs for training DL/ML models.
            Use cuda and PyTorch for faster training whenever needed.

            You have a total of 15 actions available for the env_tools.
            But you the deepresearch tools can be used unlimitedly.
            You have a total of 43200 seconds, including code execution time.
            You can get competition information by using tool "env_tools---request_info".
            You can alse use other tools to get more information.
            You can use tool to interact with the environment and get feedback.
            Before you take an action show your thinking process clearly.
            deepresearch tools may help you to research and design the algorithm to solve the problem.

            Code requirements:
            - Request all information first
            - Read all data files from data_dir
            - Save all submissions to output_dir, should match test_data length
            - Don't add, delete, or modify any files in data_dir
            - Use "print" to output information in the feedback
            - No plotting or visualization is allowed
            - Refer to Sample Submission for the output format
            - Code should be self-contained and not rely on any variables or state outside
            - Code for submission should be completely runnable, otherwise it will be considered as failed
            - Optimize your Model/Parameters/Data Processing/Algorithm for continuous improvement

            Only if "env_tools---execute_code" action taken, code successfully executed and valid submission generated, 
            The code must be in the correct format of shell since it will be executed in the shell environment directly.
            you'll be able to get a Position Score (Your rank in the leaderboard) for this competition.
            In the limited steps, you should try to improve your Position Score as much as possible.
    Let's begin:

tools:
  exit_task:
    mcp: false
  deepresearch:
    mcp: false
  env_tools:
    mcp: false
    competition_name: {name}
    data_dir: /home/u-longyy/MLE-Dojo/data/prepared/{name}/data
    output_dir: /home/u-longyy/ms-agent/kaggle_output/{name}
    limited_step_num: 15
    time_limit: 43200

max_chat_round: 100

tool_call_timeout: 30000

output_dir: output

help: |

"""
workflow = """
kaggler:
  agent_config: kaggler.yaml
"""

def get_competition_name(path):
    names = [
        competition_dir
        for competition_dir in os.listdir(path)
        if os.path.isdir(os.path.join(path, competition_dir))
    ]
    return names
command_template = "PYTHONPATH=. python ms_agent/cli/cli.py run --config projects/mle_test/{competition_name} --query '开始执行你的任务' --trust_remote_code true"
competitions = get_competition_name("/home/u-longyy/MLE-Dojo/data/prepared")
print(competitions,f"Total competitions: {len(competitions)}")
project_path = "/home/u-longyy/ms-agent/projects/mle_test/"
for competition_name in competitions:
    os.makedirs(os.path.join(project_path, competition_name), exist_ok=True)
    with open(os.path.join(project_path, competition_name, "kaggler.yaml"), "w") as f:
        f.write(template.format(name=competition_name))
    with open(os.path.join(project_path, competition_name, "workflow.yaml"), "w") as f:
        f.write(workflow)
    print(f"Created config for competition: {competition_name}")
    command = command_template.format(competition_name=competition_name)
    print(f"Executing command: {command}")
    subprocess.run(command, shell=True)
    os.removedirs(os.path.join(project_path, competition_name))
    print(f"Completed and removed config for competition: {competition_name}")
    


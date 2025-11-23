#### 基本实现pipeline 的一个workflow
##### 关于一些细节问题的临时补丁
- 工作路径，目前在kaggle,run_code,environment_set_up这些工具调用前会先临时修改环境变量
``` python
async def call_tool(self, server_name, *, tool_name, tool_args):
  os.chdir(self.output_dir)
  logger.info(f"Changed working directory to {self.output_dir} for running code.")
  result = await getattr(self, tool_name)(**tool_args)
  os.chdir(os.path.dirname(os.getcwd()))
  logger.info(f"Changed working directory back to {os.getcwd()} after running code.")
  return result
```

- file_system 添加了limit 逻辑，可以在配置文件中配置，设置某些后缀的文件只能读前n个字符
对list files输出长度也限制在1000个字符以内(理论上够用)

- kaggle似乎有时候访问需要梯子?添加了short_http_proxy,short_https_proxy的配置逻辑，设置代理时设置这两个变量即可

- 为了强迫按照files.json文件的格式来写代码，在architect_callback.py中加了一些临时补丁判断写的文件是否符合要求(当然应该有更优雅的写法)

- 目前测试使用query "在kaggle spaceship-titanic比赛中获得超过0.81的score"可以实现完整的pipeline,能够提交-获取结果-修改-提交，虽然最后没能达到0.81(达到0.805后放弃)

- 目前把coding.yaml的memory部分注释掉了(觉得暂时没啥用)
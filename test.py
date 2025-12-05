import docker
import os
import time

# 配置参数（与 docker_shell.py 保持一致）
IMAGE = "my-dev-env:latest"
OUTPUT_DIR = "/home/u-wuhc/AutoCoding/output"  # 你的 output 目录
WORKDIR = "/workspace"
CMD = "cd /workspace && git diff"

# bug: tty = True 时，exec_run 会阻塞

def test_exec():
    client = docker.from_env()
    
    # 获取当前用户 ID
    uid = os.getuid()
    gid = os.getgid()
    
    print(f"Starting session container with image: {IMAGE}")
    print(f"Mounting {OUTPUT_DIR} -> {WORKDIR}")
    
    # 1. 启动长期容器 (Session Container)
    try:
        container = client.containers.run(
            IMAGE,
            command=["/bin/bash"],
            volumes={OUTPUT_DIR: {'bind': WORKDIR, 'mode': 'rw'}},
            working_dir=WORKDIR,
            user=f"{uid}:{gid}",
            environment={"HOME": WORKDIR},
            detach=True,
            tty=True,  # 保持与 docker_shell 一致 (或者改为 False 测试)
            stdin_open=True,
            stderr=True,
            stdout=True,
            auto_remove=False
        )
        print(f"Container started: {container.id[:12]}")
        
        # 等待容器完全启动
        time.sleep(2)
        
        # 2. 执行命令
        print(f"\nExecuting command: {CMD}")
        start_time = time.time()
        
        # 模拟 _exec_in_session 的调用方式
        exec_result = container.exec_run(
            cmd=["/bin/bash", "-lc", CMD],
            stdout=True,
            stderr=True,
            tty=True  # 尝试改为 False 对比
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        output = exec_result.output.decode("utf-8", errors="replace")
        exit_code = exec_result.exit_code
        
        print(f"\nExecution finished in {duration:.2f} seconds")
        print(f"Exit code: {exit_code}")
        print("-" * 40)
        print("Output:")
        print(output)
        print("-" * 40)
        
    except Exception as e:
        print(f"\nError occurred: {e}")
    finally:
        # 清理容器
        if 'container' in locals():
            print(f"\nStopping and removing container {container.id[:12]}...")
            container.stop()
            container.remove()
            print("Cleanup done.")

if __name__ == "__main__":
    test_exec()
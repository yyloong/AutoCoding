# 将字符串保存为 tmp.patch 文件

content = """--- a/django/db/backends/postgresql/client.py\n+++ b/django/db/backends/postgresql/client.py\n@@ -35,8 +35,11 @@\n         if port:\n             args += [\"-p\", str(port)]\n         if dbname:\n-            args += [dbname]\n-        args.extend(parameters)\n+            args += [dbname]\n+        # Add parameters before dbname to ensure psql gets options correctly\n+        args.extend(parameters)\n"""

with open("tmp.patch", "w", encoding="utf-8") as file:
    file.write(content)
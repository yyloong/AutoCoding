import os
import requests
import json
import warnings
import urllib3
from typing import List, Dict, Optional
from ms_agent.tools.base import ToolBase
from ms_agent.llm.utils import Tool


class GitHubCodeSearch(ToolBase):
    """GitHub代码搜索工具"""
    
    GITHUB_API_URL = "https://api.github.com"
    
    def __init__(self, config, **kwargs):
        super(GitHubCodeSearch, self).__init__(config)
        self.github_token = os.getenv("GITHUB_TOKEN")
        self.session = requests.Session()
        # 禁用SSL警告
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        # 禁用SSL验证（仅限开发环境）
        self.session.verify = False
        if self.github_token:
            self.session.headers.update({
                "Authorization": f"token {self.github_token}",
                "Accept": "application/vnd.github.v3+json"
            })
    
    async def get_tools(self):
        tools = {
            "GitHubCodeSearch": [
                Tool(
                    tool_name="search_github_code",
                    server_name="GitHubCodeSearch",
                    description="在GitHub上搜索开源代码仓库和文件内容，支持按语言、仓库、文件名等条件筛选，专门用于查找和学习开源代码实现",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "搜索关键词，例如 'machine learning language:python'，专门用于在GitHub代码中搜索"
                            },
                            "sort": {
                                "type": "string",
                                "enum": ["stars", "forks", "updated"],
                                "description": "排序方式"
                            },
                            "order": {
                                "type": "string",
                                "enum": ["asc", "desc"],
                                "description": "排序顺序"
                            },
                            "per_page": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 100,
                                "description": "每页返回的结果数量，默认30"
                            }
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    tool_name="get_github_file_content",
                    server_name="GitHubCodeSearch",
                    description="获取GitHub上特定文件的内容，用于学习和分析开源代码实现",
                    parameters={
                        "type": "object",
                        "properties": {
                            "owner": {
                                "type": "string",
                                "description": "仓库所有者"
                            },
                            "repo": {
                                "type": "string",
                                "description": "仓库名称"
                            },
                            "path": {
                                "type": "string",
                                "description": "文件路径"
                            },
                            "ref": {
                                "type": "string",
                                "description": "分支、标签或提交SHA，默认为默认分支"
                            }
                        },
                        "required": ["owner", "repo", "path"]
                    }
                )
            ]
        }
        return {
            "GitHubCodeSearch": [
                t for t in tools["GitHubCodeSearch"]
                if t["tool_name"] not in self.exclude_functions
            ]
        }
    
    async def call_tool(self, server_name: str, *, tool_name: str, tool_args: dict) -> str:
        return await getattr(self, tool_name)(**tool_args)
    
    async def search_github_code(self, query: str, sort: str = "updated", order: str = "desc", per_page: int = 30) -> str:
        """
        在GitHub上搜索代码
        
        Args:
            query: 搜索关键词
            sort: 排序方式 (stars, forks, updated)
            order: 排序顺序 (asc, desc)
            per_page: 每页返回的结果数量
            
        Returns:
            搜索结果的JSON字符串
        """
        url = f"{self.GITHUB_API_URL}/search/code"
        params = {
            "q": query,
            "sort": sort,
            "order": order,
            "per_page": min(per_page, 100)
        }
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # 提取关键信息
            results = []
            for item in data.get("items", [])[:10]:  # 限制返回前10个结果
                results.append({
                    "name": item["name"],
                    "path": item["path"],
                    "repository": item["repository"]["full_name"],
                    "url": item["html_url"],
                    "sha": item["sha"]
                })
            
            return json.dumps({
                "total_count": data.get("total_count", 0),
                "items": results
            }, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": f"搜索失败: {str(e)}"}, ensure_ascii=False)
    
    async def get_github_file_content(self, owner: str, repo: str, path: str, ref: str = None) -> str:
        """
        获取GitHub上特定文件的内容
        
        Args:
            owner: 仓库所有者
            repo: 仓库名称
            path: 文件路径
            ref: 分支、标签或提交SHA
            
        Returns:
            文件内容
        """
        url = f"{self.GITHUB_API_URL}/repos/{owner}/{repo}/contents/{path}"
        params = {}
        if ref:
            params["ref"] = ref
            
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # 如果是文件，返回内容
            if data.get("type") == "file":
                import base64
                content = base64.b64decode(data["content"]).decode("utf-8")
                return json.dumps({
                    "content": content,
                    "encoding": data["encoding"],
                    "size": data["size"],
                    "url": data["html_url"]
                }, ensure_ascii=False)
            else:
                return json.dumps({"error": "指定路径不是文件"}, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": f"获取文件内容失败: {str(e)}"}, ensure_ascii=False)
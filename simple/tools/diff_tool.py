import sys
import io
import os
import subprocess
from ms_agent.llm.utils import Tool
from ms_agent.tools.base import ToolBase
from omegaconf import DictConfig

class DiffTool(ToolBase):
    """
    A tool for automatically generating standardized patch files based on local/remote file paths.

    Input parameters:
      - local_original_path: Local original file path (used as the left side of diff)
      - local_modified_path: Local modified file path (used as the right side of diff)
      - remote_original_path: Original file path in the remote repository (used for patch header a/ path)
      - remote_modified_path: Modified file path in the remote repository (used for patch header b/ path)
      - output_patch_path: Output path for the generated patch file (e.g., /workspace/fix.patch)

    Generation rules:
      1. Use `diff -u local_original_path local_modified_path` to generate a unified diff.
      2. Replace the first two lines of the diff result `--- ...` / `+++ ...` with:
         `--- a/{remote_original_path}`
         `+++ b/{remote_modified_path}`
      3. Write the final result to output_patch_path.
    """

    def __init__(self, config: DictConfig):
        super(DiffTool, self).__init__(config)
        self.exclude_func(getattr(config.tools, 'diff_tool', None))

    async def get_tools(self):
        tools = {
            'diff_tool': [
                Tool(
                    tool_name='generate_patch',
                    server_name='diff_tool',
                    description=(
                        'Generate a unified diff patch file from local original/modified '
                        'files, and rewrite headers to remote repo paths.'
                    ),
                    parameters={
                        'type': 'object',
                        'properties': {
                            'local_original_path': {
                                'type': 'string',
                                'description': 'Local original file path (used as diff left side).',
                            },
                            'local_modified_path': {
                                'type': 'string',
                                'description': 'Local modified file path (used as diff right side).',
                            },
                            'remote_original_path': {
                                'type': 'string',
                                'description': 'Original file path in remote repository (used for patch header a/ path).',
                            },
                            'remote_modified_path': {
                                'type': 'string',
                                'description': 'Modified file path in remote repository (used for patch header b/ path).',
                            },
                            'output_patch_path': {
                                'type': 'string',
                                'description': 'Output path for the generated patch file (e.g., /workspace/fix.patch).',
                            },
                        },
                        'required': [
                            'local_original_path',
                            'local_modified_path',
                            'remote_original_path',
                            'remote_modified_path',
                            'output_patch_path',
                        ],
                        'additionalProperties': False,
                    },
                ),
            ],
        }

        return {
            'diff_tool': [
                t for t in tools['diff_tool']
                if t['tool_name'] not in self.exclude_functions
            ]
        }

    async def generate_patch(
        self,
        local_original_path: str,
        local_modified_path: str,
        remote_original_path: str,
        remote_modified_path: str,
        output_patch_path: str,
    ) -> str:
        """
        Use diff -u to generate a unified diff and rewrite header paths to a/ and b/ format.
        """
        # Check if local files exist
        if not os.path.exists(local_original_path):
            return f"Error: local_original_path does not exist: {local_original_path}"
        if not os.path.exists(local_modified_path):
            return f"Error: local_modified_path does not exist: {local_modified_path}"

        try:
            # Call system diff -u
            proc = subprocess.run(
                ['diff', '-u', local_original_path, local_modified_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except FileNotFoundError:
            return "Error: `diff` command not found in environment."
        except Exception as e:
            return f"Error: failed to run diff: {e}"

        # diff return code: 0=no difference, 1=difference found, >1=error
        if proc.returncode not in (0, 1):
            return f"Error: diff failed with code {proc.returncode}: {proc.stderr.strip()}"

        diff_text = proc.stdout
        if not diff_text.strip():
            return "No differences found; patch not generated."

        # Process lines and rewrite the first two header lines
        lines = diff_text.splitlines(keepends=False)
        if len(lines) < 2 or not lines[0].startswith('--- ') or not lines[1].startswith('+++ '):
            # If diff output is not standard unified diff header, return raw diff for debugging
            return (
                "Error: diff output does not start with unified headers.\n"
                f"stdout:\n{diff_text}\n\nstderr:\n{proc.stderr}"
            )

        # Rewrite header lines to required repository paths
        lines[0] = f"--- a/{remote_original_path}"
        lines[1] = f"+++ b/{remote_modified_path}"

        # Ensure each line ends with a newline, including the last line
        patched_text = "".join(line + "\n" for line in lines)

        # Write patch file
        output_dir = os.path.dirname(output_patch_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        with open(output_patch_path, "w", encoding="utf-8") as f:
            f.write(patched_text)

        return (
            f"Patch generated successfully at: {output_patch_path}\n"
            f"From local: {local_original_path} -> {local_modified_path}\n"
            f"To remote: a/{remote_original_path} -> b/{remote_modified_path}"
        )

    async def call_tool(self, server_name: str, tool_name: str, tool_args: dict) -> str:
        if tool_name == 'generate_patch':
            return await self.generate_patch(
                local_original_path=tool_args.get('local_original_path', ''),
                local_modified_path=tool_args.get('local_modified_path', ''),
                remote_original_path=tool_args.get('remote_original_path', ''),
                remote_modified_path=tool_args.get('remote_modified_path', ''),
                output_patch_path=tool_args.get('output_patch_path', ''),
            )
        else:
            return f'Unknown tool: {tool_name}'
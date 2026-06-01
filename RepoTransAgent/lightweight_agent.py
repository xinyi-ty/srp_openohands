import json
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Tuple

import requests
import yaml

logger = logging.getLogger(__name__)


# ---------- 配置加载 ----------
def load_llm_config(config_path: str = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).parent / "llm_config.yaml"
    with open(config_path, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)
    llm_config = config.get("llm", {})
    return {
        "model": llm_config.get("model"),
        "api_key": llm_config.get("api_key"),
        "base_url": llm_config.get("base_url"),
    }


# ---------- LLM 调用 ----------
def call_llm(messages: List[dict], config: dict, temperature: float = 0.2) -> str:
    """调用 DeepSeek API（OpenAI 兼容格式）"""
    url = f"{config['base_url']}/chat/completions"
    headers = {
        "Authorization": f"Bearer {config['api_key']}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": config["model"],
        "messages": messages,
        "temperature": temperature,
        "extra_body": {"thinking": {"type": "disabled"}}
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"LLM 调用失败: {e}")
        raise


# ---------- 工具函数（动作执行）----------
def execute_action(action_name: str, **kwargs) -> str:
    """执行单个动作，返回观察结果（字符串）"""
    try:
        if action_name == "ReadFile":
            path = kwargs["path"]
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        elif action_name == "CreateFile":
            path = kwargs["path"]
            content = kwargs.get("content", "")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"File created: {path}"
        elif action_name == "DeleteFile":
            path = kwargs["path"]
            if os.path.exists(path):
                os.remove(path)
                return f"File deleted: {path}"
            else:
                return f"File not found: {path}"
        elif action_name == "ExecuteCommand":
            cmd = kwargs["command"]
            cwd = kwargs.get("cwd", ".")
            result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=60)
            output = result.stdout + result.stderr
            if result.returncode != 0:
                return f"Command failed (exit {result.returncode}):\n{output[:1000]}"
            return output[:2000]  # 限制长度
        elif action_name == "Finished":
            return "TASK_FINISHED"
        else:
            return f"Unknown action: {action_name}"
    except Exception as e:
        return f"Error executing {action_name}: {str(e)}"


def parse_action(response: str) -> Tuple[str, Dict[str, Any]]:
    """
    从 LLM 响应中解析 Action: 行。
    格式示例：
        Action: ReadFile(path="src/main.py")
    返回 (action_name, kwargs_dict)
    """
    pattern = r'Action:\s*(\w+)\((.*)\)'
    match = re.search(pattern, response, re.DOTALL)
    if not match:
        return None, None
    action_name = match.group(1)
    params_str = match.group(2).strip()
    # 简单解析 kwargs，支持字符串、数字等（eval 安全风险，但仅用于可信环境）
    # 注意：生产环境建议用 ast.literal_eval 或正则解析
    kwargs = {}
    if params_str:
        # 简单处理 key=value 对，value 可能带引号
        # 这里用正则匹配 key="value" 或 key=数字
        for kv in re.findall(r'(\w+)=(".*?"|\d+)', params_str):
            key, val = kv
            if val.startswith('"') and val.endswith('"'):
                val = val[1:-1]
            elif val.isdigit():
                val = int(val)
            kwargs[key] = val
    return action_name, kwargs


# ---------- ReAct 循环 ----------
def run_react_loop(instruction: str, workspace_dir: str, max_steps: int = 50) -> None:
    """执行 ReAct 循环，直到 Finished 或达到最大步数"""
    llm_config = load_llm_config()
    messages = [
        {"role": "system",
         "content": "You are an AI assistant that helps translate code repositories. You can use the following actions: ReadFile, CreateFile, DeleteFile, ExecuteCommand, Finished. Respond with a Thought: and then Action: line. Example:\nThought: I need to list files.\nAction: ExecuteCommand(command=\"ls -la\")"},
        {"role": "user", "content": instruction}
    ]
    step = 0
    while step < max_steps:
        step += 1
        logger.info(f"Step {step}/{max_steps}")
        response = call_llm(messages, llm_config)
        logger.debug(f"LLM response: {response[:500]}")
        messages.append({"role": "assistant", "content": response})

        action_name, kwargs = parse_action(response)
        if action_name is None:
            logger.warning("No valid action found, assuming finished.")
            break
        if action_name == "Finished":
            logger.info("Agent finished task.")
            break

        observation = execute_action(action_name, **kwargs)
        logger.info(f"Action: {action_name} -> {observation[:200]}")
        messages.append({"role": "user", "content": f"Observation: {observation}"})

    if step >= max_steps:
        logger.warning(f"Reached max steps ({max_steps}) without finishing.")


# ---------- 后处理函数（从原 openhands_agent 复用）----------
def cleanup_source_files(target_dir: Path, source_extensions: list, target_extensions: list) -> list:
    """删除源文件并返回映射"""
    deleted_count = 0
    mappings = []
    for ext in source_extensions:
        for file_path in target_dir.rglob(f"*{ext}"):
            base_name = file_path.stem
            parent = file_path.parent
            found = False
            target_file = None
            for t_ext in target_extensions:
                candidate = parent / f"{base_name}{t_ext}"
                if candidate.exists():
                    found = True
                    target_file = candidate
                    break
            if found:
                src_rel = file_path.relative_to(target_dir)
                tgt_rel = target_file.relative_to(target_dir)
                mappings.append((str(src_rel), str(tgt_rel)))
                file_path.unlink()
                deleted_count += 1
                logger.info(f"已删除源文件: {file_path}")
            else:
                logger.warning(f"未找到对应的目标文件，保留源文件: {file_path}")
    logger.info(f"共删除 {deleted_count} 个源文件")
    return mappings


def check_missing_files(source_dir: Path, target_dir: Path, source_extensions: list, target_extensions: list) -> list:
    missing = []
    for ext in source_extensions:
        for src_file in source_dir.rglob(f"*{ext}"):
            rel_path = src_file.relative_to(source_dir)
            found = False
            for t_ext in target_extensions:
                tgt_file = target_dir / rel_path.with_suffix(t_ext)
                if tgt_file.exists():
                    found = True
                    break
            if not found:
                missing.append(rel_path)
    return missing


# ---------- 主翻译函数（替代原 run_translation）----------
def run_translation(
        project_name: str,
        source_language: str,
        target_language: str,
        model_name: str,  # 保留参数，但实际从 config 读取，可忽略
        max_iterations: int,
        source_path: str,
        target_path: str,
) -> None:
    """使用轻量级 ReAct Agent 执行翻译"""
    # 1. 准备目标目录
    target_dir = Path(target_path)
    if target_dir.exists():
        logger.info(f"清空已存在的目标目录: {target_dir}")
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # 2. 复制源项目
    source_dir = Path(source_path)
    if not source_dir.exists():
        raise FileNotFoundError(f"源项目目录不存在: {source_dir}")
    logger.info(f"复制源项目 {source_dir} -> {target_dir}")
    shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)

    # 3. 构建指令（复用原 openhands_agent 中的 instruction）
    instruction = f"""You are an expert in code migration and translation.
Your task is to translate the project '{project_name}' from {source_language} to {target_language}.

The source code has been copied to the current working directory: {target_dir}
You have full access to read, edit, create, and delete files inside this directory.

**Critical rules for file management:**
1. For every source code file (extension specific to {source_language}), you MUST create a corresponding translated file with the **same base name** but the appropriate extension for {target_language}.
   Example: `src/main.cpp` (C++) → `src/main.py` (Python)
2. **After successfully creating the translated file, you MUST delete the original source file.** Do not leave any {source_language} files in the directory.
3. Do not mix {source_language} and {target_language} code in the same file. The translated file must contain only valid {target_language} code.
4. Preserve the directory structure exactly as in the source project.

**Workflow (follow strictly):**
1. Analyze the project structure using `ExecuteCommand(command="find . -type f")` or `ls -la`.
2. **FIRST, translate ALL source files** (one by one) without stopping to run any compile or test commands.
   - For each source file: use ReadFile to read content → CreateFile to write target file → DeleteFile to remove source file.
   - Do NOT run any compile/test commands during this phase.
3. After ALL files have been translated, **then** run the test suite once using `ExecuteCommand` (e.g., `pytest`, `npm test`, `mvn test`). If tests are missing, you may skip.
4. If tests fail, analyze the errors and fix the relevant translated files (ReadFile, CreateFile again).
5. Re-run the tests after fixing. Repeat until all tests pass or you run out of steps.

**Error handling during testing:**
- If a test command fails, carefully read the error output.
- Identify which file(s) caused the failure, fix them, and re-run the test command.
- Do not go back to translating files that are already done unless fixing errors.

You have a maximum of {max_iterations} steps.
When you are finished, output `Action: Finished()`.
"""

    # 4. 运行 ReAct 循环
    run_react_loop(instruction, str(target_dir), max_iterations)

    # 5. 后处理：根据语言类型清理残留源文件并生成映射
    language_extensions = {
        "c++": (['.cpp', '.cxx', '.cc', '.c', '.h', '.hpp', '.hxx'], ['.py']),
        "python": (['.py'], ['.java', '.cpp']),
        "java": (['.java'], ['.py', '.cpp']),
        "javascript": (['.js', '.jsx'], ['.py']),
    }
    src_key = source_language.lower()
    lang_entry = language_extensions.get(src_key)
    if lang_entry:
        source_extensions, target_extensions = lang_entry
    else:
        source_extensions = ['.c', '.cpp', '.h', '.py', '.java', '.js']
        target_extensions = ['.py'] if target_language.lower() == 'python' else ['.java']

    # 清理源文件（兜底）
    mappings = cleanup_source_files(target_dir, source_extensions, target_extensions)

    # 生成 MAPPING.txt
    mapping_file = target_dir / "MAPPING.txt"
    with open(mapping_file, 'w', encoding='utf-8') as f:
        f.write(f"# Translation mapping for project: {project_name}\n")
        f.write(f"# Source language: {source_language} → Target language: {target_language}\n")
        f.write("# Format: source_file -> target_file\n\n")
        if mappings:
            for src, tgt in mappings:
                f.write(f"{src} -> {tgt}\n")
        else:
            f.write("# No source files were deleted (possible incomplete translation).\n")
            for t_ext in target_extensions:
                for target_file in target_dir.rglob(f"*{t_ext}"):
                    f.write(f"# (inferred) {target_file.stem}.* -> {target_file.relative_to(target_dir)}\n")
        f.write("\nNote: Original source files have been deleted where corresponding target files exist.\n")
    logger.info(f"已生成映射文件: {mapping_file}")

    # 检查缺失文件
    missing = check_missing_files(source_dir, target_dir, source_extensions, target_extensions)
    if missing:
        logger.warning(f"发现 {len(missing)} 个未翻译的源文件:")
        for m in missing[:10]:
            logger.warning(f"  - {m}")
        missing_file = target_dir / "MISSING_FILES.txt"
        with open(missing_file, 'w', encoding='utf-8') as f:
            f.write(f"# 以下 {len(missing)} 个源文件未翻译\n")
            for m in missing:
                f.write(f"{m}\n")
        logger.info(f"未翻译文件列表已保存到 {missing_file}")
    else:
        logger.info("所有源文件均已翻译。")

    logger.info(f"翻译任务完成: {project_name} -> {target_dir}")
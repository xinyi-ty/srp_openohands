"""
RepoTransBench 轻量级 ReAct 翻译 Agent
基于原始 RepoTransBench 的简化实现，使用直接 LLM API 调用替代 OpenHands。
"""
import ast
import logging
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# ── 自动加载 .env ──────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, verbose=False)
        logger.info(f"已加载环境变量文件: {env_path}")
except ImportError:
    pass


# ── LLM 配置 ────────────────────────────────────────────────────
def load_llm_config() -> dict:
    return {
        "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
        "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        "model": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
    }


# ── LLM 调用（含自动重试） ─────────────────────────────────────
def call_llm(messages: List[dict], config: dict, temperature: float = 0.2) -> str:
    url = f"{config['base_url']}/chat/completions"
    headers = {
        "Authorization": f"Bearer {config['api_key']}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config["model"],
        "messages": messages,
        "temperature": temperature,
        "extra_body": {"thinking": {"type": "disabled"}},
    }
    for attempt in range(3):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"] or ""
        except requests.exceptions.Timeout:
            logger.warning(f"LLM 超时 (attempt {attempt+1}/3)")
        except requests.exceptions.HTTPError as e:
            s = e.response.status_code
            if s in (429, 502, 503, 504) and attempt < 2:
                logger.warning(f"LLM 返回 {s}，重试 ({attempt+1}/3)")
            else:
                raise
        except Exception as e:
            logger.error(f"LLM 调用失败 (attempt {attempt+1}/3): {e}")
            if attempt >= 2:
                raise
        time.sleep(2 ** attempt)
    raise RuntimeError("LLM 调用在 3 次重试后仍然失败")


# ── 动作执行 ────────────────────────────────────────────────────
def execute_action(action_name: str, **kwargs) -> str:
    try:
        if action_name == "ReadFile":
            path = kwargs.get("path", "")
            if not path or not os.path.exists(path):
                return f"File not found: {path}"
            max_size = 200 * 1024
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read(max_size + 1)
                if len(content) > max_size:
                    return f"Warning: file >200KB, showing first {max_size} bytes:\n{content[:max_size]}\n... (truncated)"
                return content

        elif action_name == "CreateFile":
            path = kwargs.get("path", "")
            if not path:
                return "Error: CreateFile called without a valid 'path' argument"
            content = kwargs.get("content", "")
            dirpath = os.path.dirname(path)
            if dirpath:
                os.makedirs(dirpath, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"File created: {path}"

        elif action_name == "DeleteFile":
            path = kwargs.get("path", "")
            if not path:
                return "Error: DeleteFile called without a valid 'path' argument"
            if os.path.exists(path):
                os.remove(path)
                return f"File deleted: {path}"
            return f"File not found: {path}"

        elif action_name == "ExecuteCommand":
            cmd = kwargs.get("command", "")
            cwd = kwargs.get("cwd", ".")
            result = subprocess.run(
                cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=120
            )
            output = (result.stdout + result.stderr)[:5000]
            if result.returncode != 0:
                if len(output) > 3000:
                    output = output[-3000:]
                return f"Command failed (exit {result.returncode}):\n{output}"
            if len(output) > 4000:
                output = output[:4000] + "\n... (truncated)"
            return output

        elif action_name == "SearchContent":
            keyword = kwargs.get("keyword", "")
            matches = []
            for root, _dirs, files in os.walk("."):
                for fname in files:
                    if not fname.endswith((".py", ".java", ".txt", ".json", ".xml", ".yml", ".yaml", ".sh", ".md")):
                        continue
                    fpath = os.path.join(root, fname)
                    try:
                        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                            if keyword.lower() in f.read().lower():
                                matches.append(os.path.relpath(fpath))
                                if len(matches) >= 10:
                                    break
                    except Exception:
                        pass
                if len(matches) >= 10:
                    break
            return (
                f"Found '{keyword}' in: {', '.join(matches)}" if matches
                else f"No matches found for '{keyword}'"
            )

        elif action_name == "Finished":
            return "TASK_FINISHED"

        return f"Unknown action: {action_name}"

    except Exception as e:
        return f"Error executing {action_name}: {e}"


# ── Action 解析 ────────────────────────────────────────────────
def _strip_quotes(text: str) -> str:
    text = text.strip()
    for q in ('"', "'", "`"):
        if text.startswith(q) and text.endswith(q):
            text = text[1:-1]
            break
    return text.strip()


def parse_action(text: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    解析 LLM 输出中的 Action。返回 (action_name, kwargs) 或 (None, None)。
    支持的格式（按优先级）:
      1. CreateFile(path="x"):  ```lang\ncontent\n```       ← 代码块格式（推荐）
      2. Action: Xxx(key="value", ...)                     ← 行内格式（兼容）
    """
    if not text or not text.strip():
        return None, None

    # ── 1. CreateFile 代码块格式 ──
    m = re.search(
        r'CreateFile\(path=(.*?)\):\s*```[^\n]*\n(.*?)\n```',
        text, re.DOTALL
    )
    if m:
        return "CreateFile", {
            "path": _strip_quotes(m.group(1)),
            "content": m.group(2),
        }

    # ── 2. 行内 Action: Xxx(...) ──
    action_match = re.search(r'(?:^|\n)Action:\s*(\w+)\(', text)
    if not action_match:
        return None, None

    action_name = action_match.group(1)
    err_kwargs: Dict[str, Any] = {}

    if action_name == "Finished":
        return "Finished", {}

    if action_name in ("ReadFile", "DeleteFile"):
        m = re.search(rf"{action_name}\(path=(.*?)\)", text)
        if m:
            return action_name, {"path": _strip_quotes(m.group(1))}
        return action_name, err_kwargs

    if action_name == "ExecuteCommand":
        m = re.search(r"ExecuteCommand\(command=(.*?)\)", text)
        if m:
            return action_name, {"command": _strip_quotes(m.group(1))}
        return action_name, err_kwargs

    if action_name == "SearchContent":
        m = re.search(r"SearchContent\(keyword=(.*?)\)", text)
        if m:
            return action_name, {"keyword": _strip_quotes(m.group(1))}
        return action_name, err_kwargs

    # ── 3. CreateFile 行内兼容 ──
    if action_name == "CreateFile":
        # 先尝试完整 code-block
        m = re.search(r'CreateFile\(path=(.*?)\):\s*```[^\n]*\n(.*?)\n```', text, re.DOTALL)
        if m:
            return "CreateFile", {"path": _strip_quotes(m.group(1)), "content": m.group(2)}

        # 再尝试行内格式
        m = re.search(r'CreateFile\(path=(.*?),\s*content=(.*)', text, re.DOTALL)
        if m:
            raw_path = _strip_quotes(m.group(1))
            raw_content = m.group(2).strip()
            if raw_content.endswith(")"):
                raw_content = raw_content[:-1]
            try:
                content_val = ast.literal_eval(raw_content)
            except Exception:
                content_val = _strip_quotes(raw_content)
            if raw_path:
                return "CreateFile", {"path": raw_path, "content": content_val}

        # 降级：只提取 path（LLM 有时只写 header 不写内容）
        m = re.search(r'CreateFile\(path=(.*?)\)', text)
        if m:
            raw_path = _strip_quotes(m.group(1))
            if raw_path:
                return "CreateFile", {"path": raw_path, "content": ""}

    return None, None


# ── ReAct 循环 ─────────────────────────────────────────────────
def run_react_loop(instruction: str, workspace_dir: str, max_steps: int = 50) -> None:
    llm_config = load_llm_config()
    if not llm_config.get("api_key"):
        raise RuntimeError("请在 .env 中设置 DEEPSEEK_API_KEY")

    original_cwd = os.getcwd()
    os.chdir(workspace_dir)
    logger.info(f"工作目录: {workspace_dir}")

    system_prompt = (
        "You translate code between programming languages. Available actions:\n\n"
        "1. CreateFile(path=\"file.java\"):\n"
        "   ```java\n"
        "   public class File {\n"
        "       // translated code\n"
        "   }\n"
        "   ```\n"
        "2. ReadFile(path=\"file.py\")\n"
        "3. DeleteFile(path=\"file.py\")\n"
        "4. ExecuteCommand(command=\"shell command\")\n"
        "5. SearchContent(keyword=\"search term\")\n"
        "6. Finished()\n\n"
        "Rules:\n"
        "- Always start with 'Thought:' then 'Action:'.\n"
        "- For CreateFile, put content in a code block AFTER '):', NOT in the parentheses.\n"
        "- Read a file first, then CreateFile with translated content, then DeleteFile.\n"
        "- Do NOT run test commands. Only translate.\n"
    )

    messages: List[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": instruction},
    ]

    try:
        for step in range(1, max_steps + 1):
            logger.info(f"Step {step}/{max_steps}")

            response = call_llm(messages, llm_config)
            if not response or not response.strip():
                logger.warning("LLM returned empty, retrying once...")
                response = call_llm(messages, llm_config)
            if not response or not response.strip():
                logger.warning("LLM returned empty twice, aborting loop.")
                break

            messages.append({"role": "assistant", "content": response})

            action_name, kwargs = parse_action(response)
            if action_name is None:
                logger.info(f"Action: none (response: {response[:200]})")
                messages.append({
                    "role": "user",
                    "content": 'Use format: Thought: ...  Action: CreateFile(path="x.java"):  ```java  ...  ```',
                })
                continue

            if action_name == "Finished":
                logger.info("Agent finished task.")
                break

            observation = execute_action(action_name, **kwargs)
            short_obs = observation[:200].replace("\n", " ")
            logger.info(f"Action: {action_name} -> {short_obs}")
            messages.append({"role": "user", "content": f"Observation: {observation}"})

            # 窗口滑动：保留 system + instruction + 最近 2 轮
            while len(messages) > 6:
                messages[2:4] = []

        else:
            logger.warning(f"Reached max steps ({max_steps}) without finishing.")

    finally:
        os.chdir(original_cwd)


# ── 后处理 ─────────────────────────────────────────────────────
def cleanup_source_files(
    target_dir: Path, source_extensions: list, target_extensions: list
) -> list:
    """删除已翻译的源文件，返回 (源文件, 目标文件) 映射列表"""
    deleted, mappings = 0, []
    for ext in source_extensions:
        for file_path in target_dir.rglob(f"*{ext}"):
            base, parent = file_path.stem, file_path.parent
            for t_ext in target_extensions:
                candidate = parent / f"{base}{t_ext}"
                if candidate.exists():
                    mappings.append((
                        str(file_path.relative_to(target_dir)),
                        str(candidate.relative_to(target_dir)),
                    ))
                    file_path.unlink()
                    deleted += 1
                    logger.info(f"已删除源文件: {file_path}")
                    break
    logger.info(f"共删除 {deleted} 个源文件")
    return mappings


def check_missing_files(
    source_dir: Path, target_dir: Path, source_extensions: list, target_extensions: list
) -> list:
    """返回尚未翻译的源文件列表"""
    missing = []
    for ext in source_extensions:
        for src_file in source_dir.rglob(f"*{ext}"):
            rel = src_file.relative_to(source_dir)
            found = any(
                (target_dir / rel.with_suffix(t_ext)).exists()
                for t_ext in target_extensions
            )
            if not found:
                missing.append(rel)
    return missing


# ── 主翻译入口 ────────────────────────────────────────────────
def run_translation(
    project_name: str,
    source_language: str,
    target_language: str,
    model_name: str = "",
    max_iterations: int = 50,
    source_path: str = "",
    target_path: str = "",
) -> None:
    logger.info(f"开始翻译: {source_language} → {target_language} | 项目: {project_name}")
    logger.info(f"源路径: {source_path} | 目标路径: {target_path}")

    # ── 扩展名映射 ──
    ext_map = {
        "c": [".c", ".h"],
        "c++": [".cpp", ".cxx", ".cc", ".c", ".h", ".hpp", ".hxx"],
        "c#": [".cs"],
        "java": [".java"],
        "javascript": [".js", ".jsx", ".mjs"],
        "matlab": [".m"],
        "python": [".py"],
        "rust": [".rs"],
        "go": [".go"],
    }
    tgt_map = {
        "c": [".c", ".h"],
        "c++": [".cpp", ".hpp"],
        "c#": [".cs"],
        "java": [".java"],
        "javascript": [".js"],
        "python": [".py"],
        "rust": [".rs"],
        "go": [".go"],
    }
    src_key = source_language.lower()
    tgt_key = target_language.lower()
    source_exts = ext_map.get(src_key, [".py"])
    target_exts = tgt_map.get(tgt_key, [".java"])

    # ── 准备目标目录 ──
    target_dir = Path(target_path)
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True)

    # ── 复制源文件 ──
    source_dir = Path(source_path)
    if not source_dir.exists():
        raise FileNotFoundError(f"源目录不存在: {source_dir}")
    copied = 0
    for f in source_dir.rglob("*"):
        if f.is_dir():
            continue
        if any(str(f).lower().endswith(e) for e in source_exts):
            tgt = target_dir / f.relative_to(source_dir)
            tgt.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, tgt)
            copied += 1
        elif f.name.lower() in (
            "requirements.txt", "setup.py", "setup.cfg", "pyproject.toml",
            "makefile", "cmakelists.txt", "pom.xml", "build.gradle",
            "cargo.toml", "package.json",
        ):
            tgt = target_dir / f.relative_to(source_dir)
            tgt.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, tgt)
    logger.info(f"复制了 {copied} 个源文件到 {target_dir}")

    # ── 构建指令 ──
    import platform
    ls_cmd = "dir /b" if platform.system() == "Windows" else "ls"
    max_files = max(1, max_iterations // 3)

    instruction = (
        f"Translate project '{project_name}' from {source_language} to {target_language}.\n\n"
        f"Target directory: {target_dir} (you are in this directory)\n"
        f"{copied} {source_language} files were copied here.\n\n"
        f"WORKFLOW:\n"
        f"1. List files: ExecuteCommand(command=\"{ls_cmd}\")\n"
        "2. For each .py file, do: ReadFile → CreateFile → DeleteFile\n"
        "3. Use relative paths like 'chat_router.py' or 'subdir/module.py'\n\n"
        "CreateFile FORMAT:\n"
        'CreateFile(path="ChatRouter.java"):\n'
        "```java\n"
        "public class ChatRouter {\n"
        '    public static void main(String[] args) {}\n'
        "}\n"
        "```\n\n"
        f"You have {max_iterations} steps. You can translate ~{max_files} files.\n"
        "When done, output: Action: Finished()"
    )

    # ── 运行 ReAct 循环 ──
    run_react_loop(instruction, str(target_dir), max_iterations)

    # ── 后处理 ──
    mappings = cleanup_source_files(target_dir, source_exts, target_exts)
    missing = check_missing_files(target_dir, target_dir, source_exts, target_exts)

    # ── 生成报告 ──
    mapping_file = target_dir / "MAPPING.txt"
    with open(mapping_file, "w", encoding="utf-8") as f:
        f.write(f"# Translation: {project_name}  {source_language} → {target_language}\n")
        f.write("# Format: source_file -> target_file\n\n")
        if mappings:
            for src, tgt_m in mappings:
                f.write(f"{src} -> {tgt_m}\n")
        else:
            f.write("# No files were translated.\n")
    logger.info(f"MAPPING.txt -> {mapping_file}")

    if missing:
        logger.warning(f"未翻译: {len(missing)} 个文件 (显示前10个):")
        for m in missing[:10]:
            logger.warning(f"  - {m}")
        missing_file = target_dir / "MISSING_FILES.txt"
        with open(missing_file, "w", encoding="utf-8") as f:
            for m in missing:
                f.write(f"{m}\n")
        logger.info(f"MISSING_FILES.txt -> {missing_file}")
    else:
        logger.info("所有源文件均已翻译！")

    logger.info(f"翻译完成: {project_name} -> {target_dir}")

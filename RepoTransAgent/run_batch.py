#!/usr/bin/env python3
"""批量翻译脚本 —— 调用 run.py，源/目标目录分离，支持并行执行"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed


def read_projects_summary(file_path: str) -> list[dict]:
    """读取 JSONL 格式的项目列表，每条记录需包含：
       project_name, source_language, target_language, source_path, target_path
    """
    projects = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                projects.append(json.loads(line))
    print(f"已从 {file_path} 加载 {len(projects)} 个项目")
    return projects


def group_by_translation_pairs(projects: list[dict]) -> dict[str, list[dict]]:
    pairs = defaultdict(list)
    for p in projects:
        src = p.get('source_language', '')
        tgt = p.get('target_language', '')
        if src and tgt:
            pairs[f"{src}→{tgt}"].append(p)
    return pairs


def select_projects_to_run(translation_pairs: dict, max_per_pair: int = 5) -> list[dict]:
    selected = []
    for pair, projects in translation_pairs.items():
        sorted_projects = sorted(projects, key=lambda x: x.get('project_name', ''))
        chosen = sorted_projects[:max_per_pair]
        selected.extend(chosen)
        print(f"{pair}: 从 {len(projects)} 个项目中选择了 {len(chosen)} 个项目")
    return selected


def run_single_translation(args: tuple) -> dict:
    project, base_dir, model_name, max_iterations, process_id = args
    project_name = project.get('project_name', '')
    source_language = project.get('source_language', '')
    target_language = project.get('target_language', '')
    source_path = project.get('source_path', '')
    target_path = project.get('target_path', '')

    # 如果 JSONL 中未提供 target_path，则根据 base_dir 和项目名自动生成
    if not target_path:
        target_path = os.path.join(base_dir, "target_projects", project_name)
    if not source_path:
        source_path = os.path.join(base_dir, "source_projects", project_name)

    print(f"[P{process_id:02d}] 开始: {project_name} ({source_language} → {target_language})")
    print(f"        源目录: {source_path}")
    print(f"        目标目录: {target_path}")

    cmd = [
        sys.executable, '-m', 'RepoTransAgent.run',
        '--project_name', project_name,
        '--source_language', source_language,
        '--target_language', target_language,
        '--model_name', model_name,
        '--max_iterations', str(max_iterations),
        '--source_path', source_path,
        '--target_path', target_path,
    ]

    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=base_dir,
            capture_output=True,
            text=True,
            timeout=3600,  # 单个项目最长 1 小时
        )
        execution_time = time.time() - start_time

        if result.returncode == 0:
            status = "✅ 成功"
        else:
            status = f"❌ 失败 (code: {result.returncode})"

        print(f"[P{process_id:02d}] {status} - {project_name} - 耗时 {execution_time:.1f}秒")
        return {
            'project_name': project_name,
            'source_language': source_language,
            'target_language': target_language,
            'status': status,
            'return_code': result.returncode,
            'execution_time': execution_time,
            'process_id': process_id,
            'stdout': result.stdout[-2000:],
            'stderr': result.stderr[-2000:],
        }
    except subprocess.TimeoutExpired:
        print(f"[P{process_id:02d}] ⏱️ 超时 - {project_name}")
        return {
            'project_name': project_name,
            'source_language': source_language,
            'target_language': target_language,
            'status': '⏱️ 超时',
            'return_code': 2,
            'execution_time': 3600,
            'process_id': process_id,
        }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="批量运行基于 OpenHands 的代码翻译（源/目标分离）")
    parser.add_argument(
        "--projects_file",
        default="projects_summary.jsonl",
        help="项目列表 JSONL 文件路径",
    )
    parser.add_argument(
        "--model_name",
        default="deepseek-chat",
        help="模型名称（DeepSeek）",
    )
    parser.add_argument(
        "--max_per_pair",
        type=int,
        default=5,
        help="每对翻译组合的最大处理项目数",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=4,   # 根据机器性能调整，避免 API 限流
        help="并行进程数",
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=20,
        help="单个项目 Agent 的最大步数",
    )

    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    projects = read_projects_summary(args.projects_file)
    if not projects:
        print("未找到项目，程序退出。")
        return

    pairs = group_by_translation_pairs(projects)
    selected = select_projects_to_run(pairs, max_per_pair=args.max_per_pair)

    print(f"正在运行 {len(selected)} 个项目，使用 {args.num_processes} 个并行进程...")
    start = time.time()

    tasks = [
        (p, str(base_dir), args.model_name, args.max_iterations, i)
        for i, p in enumerate(selected)
    ]

    with ProcessPoolExecutor(max_workers=args.num_processes) as executor:
        futures = [executor.submit(run_single_translation, t) for t in tasks]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"任务执行失败: {e}")

    elapsed = time.time() - start
    print(f"批量翻译完成，总耗时 {elapsed:.1f} 秒。")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Math Evolve Agent Runner - Refactored to use BasePESRunner.
"""

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type

from agents.math_agent.executor.execute_chat.execute_agent_chat import (
    EvolveExecuteAgentChat,
)
from agents.math_agent.executor.execute_fuse.execute_agent_fuse import (
    EvolveExecuteAgentFuse,
)
from agents.math_agent.executor.execute_react.execute_agent_react import (
    EvolveExecuteAgentReact,
)
from agents.math_agent.planner.plan_agent import EvolvePlanAgent
from agents.math_agent.summary.summary_agent import EvolveSummaryAgent
from loongflow.framework.pes import Worker
from loongflow.framework.pes.base_runner import BasePESRunner


class MathPESAgent(BasePESRunner):
    """
    Math Evolve Agent runner for open-ended math and algorithm optimization tasks.

    Extends BasePESRunner with:
    - Support for initial code file (--initial-file)
    - Multiple executor variants (chat, react, fuse)
    """

    def _add_custom_args(self, parser: argparse.ArgumentParser) -> None:
        """Add math-agent specific CLI arguments."""
        parser.add_argument(
            "--initial-file",
            type=str,
            default=None,
            help="Override the 'initial_code' by reading from a file.",
        )

    def _merge_custom_configs(
        self, args: argparse.Namespace, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle math-agent specific config merging."""
        if args.initial_file is not None:
            try:
                config["evolve"]["initial_code"] = Path(args.initial_file).read_text(
                    encoding="utf-8"
                )
            except FileNotFoundError:
                print(
                    f"Error: Initial code file not found at '{args.initial_file}'",
                    file=sys.stderr,
                )
                sys.exit(1)

        # Optional per-island initial programs. Paths are relative to the task
        # configuration file, so a task bundle remains relocatable.
        evolve_config = config.setdefault("evolve", {})
        metadata = evolve_config.setdefault("metadata", {})
        seed_files = metadata.get("island_initial_files", [])
        if seed_files:
            task_directory = Path(args.config).resolve().parent
            seed_codes = []
            for relative_name in seed_files:
                seed_path = (task_directory / relative_name).resolve()
                if not seed_path.is_relative_to(task_directory):
                    raise ValueError(
                        f"Island seed escapes the task directory: {relative_name}"
                    )
                seed_codes.append(seed_path.read_text(encoding="utf-8"))
            num_islands = evolve_config.get("database", {}).get("num_islands", 1)
            if len(seed_codes) != num_islands:
                raise ValueError(
                    "evolve.metadata.island_initial_files must contain exactly "
                    f"{num_islands} entries; received {len(seed_codes)}."
                )
            metadata["island_initial_codes"] = seed_codes
        # Tasks can opt into dataset-dependent initialization supplied by their
        # trusted evaluator. Candidate programs are never imported here.
        if metadata.get("initialize_from_dataset"):
            evaluator_path = Path(args.eval_file) if args.eval_file else (
                Path(args.config).resolve().parent / "eval_program.py"
            )
            spec = importlib.util.spec_from_file_location("task_initializer", evaluator_path)
            if spec is None or spec.loader is None:
                raise ValueError(f"Cannot load task initializer: {evaluator_path}")
            evaluator = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(evaluator)
            evolve_config["initial_code"], metadata["island_initial_codes"] = (
                evaluator.initialize_programs(
                    evolve_config["initial_code"], metadata.get("island_initial_codes", [])
                )
            )
        return config

    def _get_process_name(self) -> str:
        return "Evolve"

    def _get_worker_registrations(
        self,
    ) -> Tuple[
        List[Tuple[str, Type[Worker]]],
        List[Tuple[str, Type[Worker]]],
        List[Tuple[str, Type[Worker]]],
    ]:
        """Register Math agent workers."""
        planners = [("evolve_planner", EvolvePlanAgent)]
        executors = [
            ("evolve_executor_chat", EvolveExecuteAgentChat),
            ("evolve_executor_react", EvolveExecuteAgentReact),
            ("evolve_executor_fuse", EvolveExecuteAgentFuse),
        ]
        summarizers = [("evolve_summary", EvolveSummaryAgent)]
        return planners, executors, summarizers


if __name__ == "__main__":
    runner = MathPESAgent()
    runner.start()

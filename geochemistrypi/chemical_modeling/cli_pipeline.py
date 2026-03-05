# -*- coding: utf-8 -*-
"""
CLI pipeline for chemical modeling. Supports interactive and non-interactive modes.
Programmatic entrypoint: cli_pipeline(input_path: str, config: dict)
"""
import os
from typing import Dict, Optional

from rich import print

from geochemistrypi.data_mining.utils.base import show_warning

from .dispatcher import discover_tasks, list_method_elements, list_task_methods, run_task_method


def _prompt_choice(prompt: str, options: list) -> Optional[int]:
    """Prompt user to choose from a list of options."""
    print(prompt)
    for i, opt in enumerate(options, start=1):
        print(f"{i}. {opt}")
    s = input("Enter number (or blank to cancel): ").strip()
    if not s:
        return None
    try:
        idx = int(s) - 1
        if 0 <= idx < len(options):
            return idx
    except ValueError:
        pass
    print("Invalid choice.")
    return None


def cli_pipeline(input_path: str, config: Optional[Dict] = None) -> None:
    """
    Run chemical modeling pipeline.
    config keys (optional): task, method, element, out_dir, non_interactive (bool)
    """
    show_warning(False)
    config = config or {}
    out_dir = config.get("out_dir") or os.path.join(os.getcwd(), "results")
    os.makedirs(out_dir, exist_ok=True)

    # In interactive mode, prompt for input file if not provided
    non_interactive = config.get("non_interactive", False)
    if not non_interactive and (not input_path or not input_path.strip() or not os.path.isfile(input_path)):
        # Check if user wants to run auto export tool first
        print("\n[bold]Automation Option[/bold]")
        print("Would you like to run the automated data export tool first?")
        print("This tool helps export data from geochemical instrument software.")
        run_auto_export = input("Run auto export tool? (y/N): ").strip().lower()

        if run_auto_export == "y":
            print("\n[bold]Auto Export Tool[/bold]")
            print("Launching automated data export tool...")
            try:
                # Import and run the auto export tool
                from .model.auto_export_tool.auto_export import main as run_auto_export_main

                run_auto_export_main()
                print("\n[green]Auto export tool completed.[/green]")
                print("You can now proceed with chemical modeling using the exported data.")
            except Exception as e:
                print(f"[red]Error running auto export tool: {e}[/red]")
                print("Continuing with normal chemical modeling workflow...")

        print("\n[bold]Input Data File[/bold]")
        print("Please provide the path to your input data file (Excel format).")
        print("You can:")
        print("1. Enter a file path")
        print("2. Press Enter to use sample data (if available)")
        print("3. Type 'exit' to cancel")

        while True:
            user_input = input("\nFile path (or press Enter for sample data): ").strip()

            if user_input.lower() == "exit":
                print("Cancelled.")
                return

            if not user_input:
                # Use sample data
                sample_path = os.path.join(
                    os.path.dirname(__file__), "..", "data", "sample_data", "hg_fractionation_sample.xlsx"  # Move up from chemical_modeling/cli_pipeline.py to chemical_modeling/
                )
                if os.path.exists(sample_path):
                    input_path = sample_path
                    print(f"[green]Using sample data: {os.path.basename(sample_path)}[/green]")
                    break
                else:
                    print("[yellow]Sample data not found. Please provide a file path.[/yellow]")
                    continue

            # Validate user-provided path
            if not os.path.exists(user_input):
                print(f"[red]File not found: {user_input}[/red]")
                print("Please check the path and try again.")
                continue

            if not user_input.lower().endswith((".xlsx", ".xls")):
                print("[yellow]Warning: File extension suggests this may not be an Excel file.[/yellow]")
                proceed = input("Continue anyway? (y/N): ").strip().lower()
                if proceed != "y":
                    continue

            input_path = user_input
            break

    # Discover tasks
    tasks = discover_tasks()
    if not tasks:
        print("[red]No chemical modeling tasks found.[/red]")
        return

    # Non-interactive: if task/method/element provided in config, run directly
    if config.get("non_interactive") or (config.get("task") and config.get("method") and config.get("element")):
        task = config["task"]
        method = config["method"]
        element = config["element"]
        print(f"Running (non-interactive) -> task={task}, method={method}, element={element}")
        res = run_task_method(task, method, element, input_path, out_dir, **config.get("kwargs", {}))
        print("[green]Done.[/green]", res)
        return

    # Interactive flow
    task_idx = _prompt_choice("\nSelect task:", tasks)
    if task_idx is None:
        print("Cancelled.")
        return
    task = tasks[task_idx]

    methods = list_task_methods(task)
    method_names = [f"{k}: {v}" for k, v in methods.items()]
    m_idx = _prompt_choice("\nSelect method:", method_names)
    if m_idx is None:
        print("Cancelled.")
        return
    method = list(methods.keys())[m_idx]

    elements = list_method_elements(task, method)
    e_idx = _prompt_choice("\nSelect element:", elements)
    if e_idx is None:
        print("Cancelled.")
        return
    element = elements[e_idx]

    print(f"\nRunning -> task={task}, method={method}, element={element}")
    res = run_task_method(task, method, element, input_path, out_dir)
    print("[green]Finished.[/green]", res)

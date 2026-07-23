import json
import os
import io
import sys
import pandas as pd
import numpy as np

def run_notebook(nb_path):
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Change directory to notebook dir to match relative paths behavior
    orig_cwd = os.getcwd()
    nb_dir = os.path.dirname(os.path.abspath(nb_path))
    os.chdir(nb_dir)

    global_env = {}

    for idx, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") == "code":
            source_code = "".join(cell.get("source", []))
            if not source_code.strip():
                cell["outputs"] = []
                cell["execution_count"] = idx
                continue
            
            # Capture stdout and display calls
            buffer = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = buffer

            display_outputs = []
            def custom_display(obj):
                if isinstance(obj, pd.DataFrame):
                    display_outputs.append({
                        "output_type": "execute_result",
                        "execution_count": idx,
                        "data": {
                            "text/plain": str(obj),
                            "text/html": obj._repr_html_()
                        },
                        "metadata": {}
                    })
                else:
                    display_outputs.append({
                        "output_type": "execute_result",
                        "execution_count": idx,
                        "data": {
                            "text/plain": repr(obj)
                        },
                        "metadata": {}
                    })

            global_env["display"] = custom_display

            try:
                # Compile and execute
                compiled = compile(source_code, f"<cell_{idx}>", "exec")
                exec(compiled, global_env)
                
                # Check if the last expression returns a value (like df in jupyter)
                lines = [line for line in source_code.strip().split("\n") if line.strip() and not line.strip().startswith("#")]
                if lines:
                    last_line = lines[-1]
                    if not last_line.startswith(("print", "import", "def", "class", "for", "while", "if", "try", "with", "raise")) and "=" not in last_line:
                        try:
                            last_val = eval(last_line, global_env)
                            if last_val is not None:
                                custom_display(last_val)
                        except Exception:
                            pass
            finally:
                sys.stdout = old_stdout

            stdout_str = buffer.getvalue()
            outputs = []
            if stdout_str:
                outputs.append({
                    "name": "stdout",
                    "output_type": "stream",
                    "text": stdout_str.splitlines(keepends=True)
                })
            outputs.extend(display_outputs)
            
            cell["outputs"] = outputs
            cell["execution_count"] = idx
            cell["id"] = f"cell_{idx}"

    os.chdir(orig_cwd)
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    print(f"Successfully executed and saved outputs to {nb_path}")

if __name__ == "__main__":
    run_notebook("work/notebooks/w02_ml_task_framing.ipynb")

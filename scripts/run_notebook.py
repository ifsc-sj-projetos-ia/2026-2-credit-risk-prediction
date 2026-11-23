"""Executa o notebook inteiro em kernel novo, sem alterar o arquivo-fonte."""

from pathlib import Path
import os
import sys

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT / "notebooks" / "01_credit_risk_prediction.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    environment = dict(os.environ)
    environment["PATH"] = str(Path(sys.executable).parent) + os.pathsep + environment.get("PATH", "")
    client = NotebookClient(
        notebook, timeout=600, kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    print("Executando notebook completo em um kernel novo...", flush=True)
    client.execute(env=environment)
    for cell in notebook.cells:
        for output in cell.get("outputs", []):
            if output.output_type == "stream":
                print(output.text, end="")
    executed = ROOT / "results" / "executed.ipynb"
    nbformat.write(notebook, executed)
    print(f"Notebook executado: {executed}")


if __name__ == "__main__":
    main()

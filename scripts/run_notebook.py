"""Execute and save a notebook in-process when a separate Jupyter kernel is unavailable.

Usage: python scripts/run_notebook.py notebooks/02_image.ipynb
Executes trusted project code; stops on the first error and saves partial outputs.
This utility is not the assignment's integrated CLI app.
"""
import os
from pathlib import Path
import sys
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
import matplotlib


def main():
    path = Path(sys.argv[1]).resolve()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    notebook = nbformat.read(path, 4)
    shell = InteractiveShell.instance()
    matplotlib.use('module://matplotlib_inline.backend_inline')
    from matplotlib_inline.backend_inline import configure_inline_support, flush_figures
    configure_inline_support(shell, 'module://matplotlib_inline.backend_inline')
    count = 0
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
    try:
        for i, cell in enumerate(notebook.cells):
            if cell.cell_type != 'code':
                continue
            count += 1
            print(f'Executing cell {i}', flush=True)
            with capture_output() as captured:
                result = shell.run_cell(cell.source, store_history=True)
                flush_figures()
            cell.execution_count = count
            if captured.stdout:
                cell.outputs.append(nbformat.v4.new_output('stream', name='stdout', text=captured.stdout))
                print(captured.stdout, flush=True)
            if captured.stderr:
                cell.outputs.append(nbformat.v4.new_output('stream', name='stderr', text=captured.stderr))
                print(captured.stderr, file=sys.stderr, flush=True)
            for out in captured.outputs:
                cell.outputs.append(nbformat.v4.new_output('display_data', data=out.data, metadata=out.metadata))
            error = result.error_before_exec or result.error_in_exec
            if error:
                cell.outputs.append(nbformat.v4.new_output('error', ename=type(error).__name__, evalue=str(error), traceback=[]))
                raise error
    finally:
        nbformat.validate(notebook)
        nbformat.write(notebook, path)
    print('Saved executed notebook:', path)


if __name__ == '__main__':
    main()

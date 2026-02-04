import numpy as np
from pyscript import Element

def solve_system(event):
    try:
        rows = int(Element("rows").element.value)
        cols = int(Element("cols").element.value)
        method = Element("method").element.value
        
        A_list = []
        b_list = []
        
        # Collect data from the grid
        for i in range(rows):
            row_data = [float(Element(f"a-{i}-{j}").element.value or 0) for j in range(cols)]
            A_list.append(row_data)
            b_list.append(float(Element(f"b-{i}").element.value or 0))
        
        A = np.array(A_list)
        b = np.array(b_list)
        
        # Solving Logic
        if rows == cols:
            # Square Matrix: Use standard solver
            x = np.linalg.solve(A, b)
            info = "Solved: Square Matrix (Standard Solver)"
        else:
            # Non-Square Matrix: Use Pseudoinverse (Least Squares)
            x = np.linalg.pinv(A) @ b
            info = f"Solved: Non-Square ({rows}x{cols}) via Pseudoinverse"

        display_results(x, info)
    except Exception as e:
        Element("extraInfo").element.innerText = f"Error: {str(e)}"

def display_results(x, info):
    grid = Element("solutionGrid")
    grid.element.innerHTML = ""
    for i, val in enumerate(x):
        grid.element.innerHTML += f"""
            <div class="bg-blue-50 p-4 rounded-xl border border-blue-100 text-center">
                <div class="text-[10px] text-blue-500 font-bold uppercase">X{i+1}</div>
                <div class="text-xl font-mono">{float(val):.4f}</div>
            </div>"""
    Element("resultArea").element.classList.remove("hidden")
    Element("extraInfo").element.innerText = info
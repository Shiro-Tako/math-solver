import numpy as np
from pyscript import display, document

def solve_system(event):
    try:
        # Use document.querySelector to get values from the UI
        rows = int(document.querySelector("#rows").value)
        cols = int(document.querySelector("#cols").value)
        
        A_list = []
        b_list = []
        
        # Collect data from the grid
        for i in range(rows):
            row_data = [float(document.querySelector(f"#a-{i}-{j}").value or 0) for j in range(cols)]
            A_list.append(row_data)
            b_list.append(float(document.querySelector(f"#b-{i}").value or 0))
        
        A = np.array(A_list)
        b = np.array(b_list)
        
        # Solving Logic
        if rows == cols:
            x = np.linalg.solve(A, b)
            info = "Solved: Square Matrix (Standard Solver)"
        else:
            x = np.linalg.pinv(A) @ b
            info = f"Solved: Non-Square ({rows}x{cols}) via Pseudoinverse"

        display_results(x, info)
    except Exception as e:
        document.querySelector("#extraInfo").innerText = f"Error: {str(e)}"

def display_results(x, info):
    grid = document.querySelector("#solutionGrid")
    grid.innerHTML = ""
    for i, val in enumerate(x):
        grid.innerHTML += f"""
            <div class="bg-blue-50 p-4 rounded-xl border border-blue-100 text-center">
                <div class="text-[10px] text-blue-500 font-bold uppercase">X{i+1}</div>
                <div class="text-xl font-mono">{float(val):.4f}</div>
            </div>"""
    document.querySelector("#resultArea").classList.remove("hidden")
    document.querySelector("#extraInfo").innerText = info
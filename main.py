import numpy as np
from pyscript import document

def solve_system(event):
    try:
        # Read inputs using the document module to avoid ImportError
        rows = int(document.querySelector("#rows").value)
        cols = int(document.querySelector("#cols").value)
        method = document.querySelector("#method").value
        
        A_list = []
        b_list = []
        
        # Build A and b from the UI grid
        for i in range(rows):
            row_data = [float(document.querySelector(f"#a-{i}-{j}").value or 0) for j in range(cols)]
            A_list.append(row_data)
            b_list.append(float(document.querySelector(f"#b-{i}").value or 0))
        
        A = np.array(A_list)
        b = np.array(b_list)
        
        # Solving Logic
        if rows == cols:
            if method == "gauss":
                # Standard solver represents Gauss with partial pivoting
                x = np.linalg.solve(A, b)
                info = "Calculation: Gauss Elimination with Partial Pivoting."
            else:
                # Jordan method via matrix inversion (A⁻¹b)
                x = np.linalg.inv(A) @ b
                info = "Calculation: Gauss-Jordan Elimination (via Matrix Inversion)."
        else:
            # Non-square support via Pseudoinverse
            x = np.linalg.pinv(A) @ b
            info = f"Non-Square Matrix detected ({rows}x{cols}). Applied Pseudoinverse."

        # Display results in the solution grid
        grid = document.querySelector("#solutionGrid")
        grid.innerHTML = ""
        for i, val in enumerate(x):
            grid.innerHTML += f"""
                <div class="bg-white p-5 rounded-2xl border border-gray-100 shadow-sm text-center">
                    <div class="text-blue-500 font-bold text-[10px] uppercase mb-1">Variable X{i+1}</div>
                    <div class="text-2xl font-mono font-bold">{float(val):.4f}</div>
                </div>"""
        
        document.querySelector("#resultArea").classList.remove("hidden")
        document.querySelector("#extraInfo").innerText = info
        
    except Exception as e:
        document.querySelector("#extraInfo").innerText = f"Error: {str(e)}"
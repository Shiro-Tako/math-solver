from fractions import Fraction

import numpy as np
from pyscript import document


def parse_fraction(value):
    text = (value or "").strip()
    if not text:
        return Fraction(0)
    return Fraction(text)


def is_integer_fraction(value):
    return isinstance(value, Fraction) and value.denominator == 1


def gauss_elimination(A, b):
    n = len(A)
    A = [row[:] for row in A]
    b = b[:]

    for i in range(n):
        pivot = max(range(i, n), key=lambda r: abs(A[r][i]))
        if A[pivot][i] == 0:
            raise ValueError("Matrix is singular.")
        if pivot != i:
            A[i], A[pivot] = A[pivot], A[i]
            b[i], b[pivot] = b[pivot], b[i]

        for r in range(i + 1, n):
            factor = A[r][i] / A[i][i]
            A[r] = [a - factor * p for a, p in zip(A[r], A[i])]
            b[r] = b[r] - factor * b[i]

    x = [0 for _ in range(n)]
    for i in range(n - 1, -1, -1):
        total = sum(A[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (b[i] - total) / A[i][i]
    return x


def gauss_jordan(A, b):
    n = len(A)
    A = [row[:] for row in A]
    b = b[:]

    for i in range(n):
        pivot = max(range(i, n), key=lambda r: abs(A[r][i]))
        if A[pivot][i] == 0:
            raise ValueError("Matrix is singular.")
        if pivot != i:
            A[i], A[pivot] = A[pivot], A[i]
            b[i], b[pivot] = b[pivot], b[i]

        pivot_value = A[i][i]
        A[i] = [value / pivot_value for value in A[i]]
        b[i] = b[i] / pivot_value

        for r in range(n):
            if r == i:
                continue
            factor = A[r][i]
            A[r] = [a - factor * p for a, p in zip(A[r], A[i])]
            b[r] = b[r] - factor * b[i]

    return b

def solve_system(event):
    try:
        # Read inputs using the document module to avoid ImportError
        rows = int(document.querySelector("#rows").value)
        cols = int(document.querySelector("#cols").value)
        method = document.querySelector("#method").value
        
        A_list = []
        b_list = []
        A_exact = []
        b_exact = []
        
        # Build A and b from the UI grid
        for i in range(rows):
            row_data = []
            row_exact = []
            for j in range(cols):
                value = document.querySelector(f"#a-{i}-{j}").value
                row_exact.append(parse_fraction(value))
                row_data.append(float(value or 0))
            A_list.append(row_data)
            A_exact.append(row_exact)
            b_value = document.querySelector(f"#b-{i}").value
            b_exact.append(parse_fraction(b_value))
            b_list.append(float(b_value or 0))
        
        A = np.array(A_list)
        b = np.array(b_list)
        
        # Solving Logic
        if rows == cols:
            if method == "gauss":
                x_exact = gauss_elimination(A_exact, b_exact)
                info = "Calculation: Gauss Elimination with Partial Pivoting."
            else:
                x_exact = gauss_jordan(A_exact, b_exact)
                info = "Calculation: Gauss-Jordan Elimination."

            x = [float(value) for value in x_exact]
            if all(is_integer_fraction(value) for value in x_exact):
                exact_text = ", ".join(
                    f"x{i+1} = {value.numerator}"
                    for i, value in enumerate(x_exact)
                )
            else:
                exact_text = ", ".join(
                    f"x{i+1} = {value.numerator}/{value.denominator}"
                    for i, value in enumerate(x_exact)
                )
            info = f"{info} Exact solution: {exact_text}."
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

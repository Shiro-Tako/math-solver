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


def format_fraction(value):
    if is_integer_fraction(value):
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def format_system_state(A, b):
    lines = []
    for row, constant in zip(A, b):
        coeffs = ", ".join(format_fraction(value) for value in row)
        lines.append(f"[{coeffs}] | {format_fraction(constant)}")
    return "\n".join(lines)


def format_matrix(matrix):
    return "\n".join(
        "[" + ", ".join(format_fraction(value) for value in row) + "]"
        for row in matrix
    )


def format_permutation(P):
    return "[" + ", ".join(str(value) for value in P) + "]"


def gauss_elimination(A, b):
    n = len(A)
    A = [row[:] for row in A]
    b = b[:]
    steps = ["Initial augmented matrix:\n" + format_system_state(A, b)]

    for i in range(n):
        pivot = max(range(i, n), key=lambda r: abs(A[r][i]))
        if A[pivot][i] == 0:
            raise ValueError("Matrix is singular.")
        if pivot != i:
            A[i], A[pivot] = A[pivot], A[i]
            b[i], b[pivot] = b[pivot], b[i]
            steps.append(
                f"Step {len(steps)}: Swap row {i + 1} with row {pivot + 1}.\n"
                + format_system_state(A, b)
            )

        for r in range(i + 1, n):
            factor = A[r][i] / A[i][i]
            A[r] = [a - factor * p for a, p in zip(A[r], A[i])]
            b[r] = b[r] - factor * b[i]
            steps.append(
                f"Step {len(steps)}: R{r + 1} = R{r + 1} - ({format_fraction(factor)})·R{i + 1}.\n"
                + format_system_state(A, b)
            )

    x = [Fraction(0) for _ in range(n)]
    for i in range(n - 1, -1, -1):
        total = sum(A[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (b[i] - total) / A[i][i]
        steps.append(
            f"Step {len(steps)}: Back substitution for x{i + 1} = {format_fraction(x[i])}."
        )
    return x, steps


def gauss_jordan(A, b):
    n = len(A)
    A = [row[:] for row in A]
    b = b[:]
    steps = ["Initial augmented matrix:\n" + format_system_state(A, b)]

    for i in range(n):
        pivot = max(range(i, n), key=lambda r: abs(A[r][i]))
        if A[pivot][i] == 0:
            raise ValueError("Matrix is singular.")
        if pivot != i:
            A[i], A[pivot] = A[pivot], A[i]
            b[i], b[pivot] = b[pivot], b[i]
            steps.append(
                f"Step {len(steps)}: Swap row {i + 1} with row {pivot + 1}.\n"
                + format_system_state(A, b)
            )

        pivot_value = A[i][i]
        A[i] = [value / pivot_value for value in A[i]]
        b[i] = b[i] / pivot_value
        steps.append(
            f"Step {len(steps)}: Normalize row {i + 1} by dividing by {format_fraction(pivot_value)}.\n"
            + format_system_state(A, b)
        )

        for r in range(n):
            if r == i:
                continue
            factor = A[r][i]
            A[r] = [a - factor * p for a, p in zip(A[r], A[i])]
            b[r] = b[r] - factor * b[i]
            steps.append(
                f"Step {len(steps)}: R{r + 1} = R{r + 1} - ({format_fraction(factor)})·R{i + 1}.\n"
                + format_system_state(A, b)
            )

    return b, steps


def lu_decomposition_with_steps(A):
    n = len(A)
    U = [row[:] for row in A]
    L = [[Fraction(0) for _ in range(n)] for _ in range(n)]
    P = list(range(n))
    steps = [
        "Initial matrices:\n"
        + "A:\n"
        + format_matrix(A)
        + "\n\nL:\n"
        + format_matrix(L)
        + "\n\nU:\n"
        + format_matrix(U)
        + "\n\nPermutation order: "
        + format_permutation(P)
    ]

    for k in range(n):
        pivot_row = max(range(k, n), key=lambda r: abs(U[r][k]))
        if U[pivot_row][k] == 0:
            raise ValueError("Matrix is singular.")

        if pivot_row != k:
            U[k], U[pivot_row] = U[pivot_row], U[k]
            P[k], P[pivot_row] = P[pivot_row], P[k]
            for j in range(k):
                L[k][j], L[pivot_row][j] = L[pivot_row][j], L[k][j]
            steps.append(
                f"Step {len(steps)}: Pivot swap row {k + 1} with row {pivot_row + 1}.\n"
                + "L:\n"
                + format_matrix(L)
                + "\n\nU:\n"
                + format_matrix(U)
                + "\n\nPermutation order: "
                + format_permutation(P)
            )

        L[k][k] = Fraction(1)
        steps.append(
            f"Step {len(steps)}: Set L{k + 1}{k + 1} = 1.\n"
            + "L:\n"
            + format_matrix(L)
        )

        for i in range(k + 1, n):
            factor = U[i][k] / U[k][k]
            L[i][k] = factor
            for j in range(k, n):
                U[i][j] -= factor * U[k][j]
            steps.append(
                f"Step {len(steps)}: Eliminate U{i + 1}{k + 1} using factor {format_fraction(factor)}.\n"
                + "L:\n"
                + format_matrix(L)
                + "\n\nU:\n"
                + format_matrix(U)
            )

    return P, L, U, steps


def apply_permutation(P, b):
    return [b[index] for index in P]


def forward_substitution_with_steps(L, b):
    n = len(L)
    y = [Fraction(0) for _ in range(n)]
    steps = ["Solve Ly = Pb using forward substitution."]
    for i in range(n):
        total = sum(L[i][j] * y[j] for j in range(i))
        y[i] = (b[i] - total) / L[i][i]
        steps.append(f"Step {len(steps)}: y{i + 1} = {format_fraction(y[i])}.")
    return y, steps


def backward_substitution_with_steps(U, y):
    n = len(U)
    x = [Fraction(0) for _ in range(n)]
    steps = ["Solve Ux = y using backward substitution."]
    for i in range(n - 1, -1, -1):
        total = sum(U[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (y[i] - total) / U[i][i]
        steps.append(f"Step {len(steps)}: x{i + 1} = {format_fraction(x[i])}.")
    return x, steps


def lu_solve(A, b):
    P, L, U, decomposition_steps = lu_decomposition_with_steps(A)
    b_permuted = apply_permutation(P, b)
    permutation_step = (
        f"Step {len(decomposition_steps)}: Apply permutation to b => Pb = "
        f"[{', '.join(format_fraction(value) for value in b_permuted)}]."
    )
    y, forward_steps = forward_substitution_with_steps(L, b_permuted)
    x, backward_steps = backward_substitution_with_steps(U, y)

    steps = decomposition_steps + [permutation_step] + forward_steps + backward_steps
    return x, steps


def lu_inverse_with_steps(A):
    n = len(A)
    P, L, U, decomposition_steps = lu_decomposition_with_steps(A)
    inverse_columns = []
    steps = decomposition_steps + [
        f"Step {len(decomposition_steps)}: Solve A·X = I using the LU factors for each identity column."
    ]

    for col in range(n):
        e_col = [Fraction(1) if row == col else Fraction(0) for row in range(n)]
        b_permuted = apply_permutation(P, e_col)

        steps.append(
            f"Step {len(steps)}: Column {col + 1} -> apply permutation to e{col + 1}, "
            + f"Pb = [{', '.join(format_fraction(value) for value in b_permuted)}]."
        )

        y, forward_steps = forward_substitution_with_steps(L, b_permuted)
        x_col, backward_steps = backward_substitution_with_steps(U, y)
        inverse_columns.append(x_col)

        steps.extend(
            f"Step {len(steps)}: Column {col + 1} forward solve - {text}"
            for text in forward_steps
        )
        steps.extend(
            f"Step {len(steps)}: Column {col + 1} backward solve - {text}"
            for text in backward_steps
        )

    inverse_matrix = [
        [inverse_columns[col][row] for col in range(n)]
        for row in range(n)
    ]
    steps.append(
        f"Step {len(steps)}: Assemble inverse matrix from solved columns.\n"
        + format_matrix(inverse_matrix)
    )
    return inverse_matrix, steps


def solve_system(event):
    try:
        rows = int(document.querySelector("#rows").value)
        cols = int(document.querySelector("#cols").value)
        method = document.querySelector("#method").value

        A_list = []
        b_list = []
        A_exact = []
        b_exact = []

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

        if rows == cols:
            if method == "gauss":
                x_exact, steps = gauss_elimination(A_exact, b_exact)
                info = "Calculation: Gauss Elimination with Partial Pivoting."
            elif method == "jordan":
                x_exact, steps = gauss_jordan(A_exact, b_exact)
                info = "Calculation: Gauss-Jordan Elimination."
            else:
                x_exact, steps = lu_solve(A_exact, b_exact)
                inverse_matrix, inverse_steps = lu_inverse_with_steps(A_exact)
                info = "Calculation: LU Decomposition with Partial Pivoting."

            x = [float(value) for value in x_exact]
            if all(is_integer_fraction(value) for value in x_exact):
                exact_text = ", ".join(
                    f"x{i + 1} = {value.numerator}" for i, value in enumerate(x_exact)
                )
            else:
                exact_text = ", ".join(
                    f"x{i + 1} = {value.numerator}/{value.denominator}"
                    for i, value in enumerate(x_exact)
                )
            info = f"{info} Exact solution: {exact_text}."
            if method == "lu":
                inverse_text = format_matrix(inverse_matrix).replace("\n", " ")
                info = f"{info} Inverse matrix A⁻¹: {inverse_text}."
                steps.extend(["", "Inverse matrix calculation:"] + inverse_steps)
            process_text = "\n\n".join(steps)
        else:
            x = np.linalg.pinv(A) @ b
            info = f"Non-Square Matrix detected ({rows}x{cols}). Applied Pseudoinverse."
            process_text = (
                "Initial matrix is non-square, so row-reduction steps are not used.\n"
                "The solver applied the pseudoinverse method to estimate a least-squares solution."
            )

        grid = document.querySelector("#solutionGrid")
        grid.innerHTML = ""
        for i, val in enumerate(x):
            grid.innerHTML += f"""
                <div class=\"bg-white p-5 rounded-2xl border border-gray-100 shadow-sm text-center\">
                    <div class=\"text-blue-500 font-bold text-[10px] uppercase mb-1\">Variable X{i+1}</div>
                    <div class=\"text-2xl font-mono font-bold\">{float(val):.4f}</div>
                </div>"""

        document.querySelector("#resultArea").classList.remove("hidden")
        document.querySelector("#extraInfo").innerText = info
        document.querySelector("#processSteps").innerText = process_text

    except Exception as e:
        document.querySelector("#extraInfo").innerText = f"Error: {str(e)}"
        document.querySelector("#processSteps").innerText = ""

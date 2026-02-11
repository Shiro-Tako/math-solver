from fractions import Fraction

import numpy as np
from pyscript import document


# ฟังก์ชันนี้เอาค่าจากช่องกรอกมาแปลงเป็นเศษส่วน
# ถ้าผู้ใช้เว้นว่างไว้ ให้ถือว่าเป็น 0 ไปเลย

def parse_fraction(value):
    text = (value or "").strip()
    if not text:
        return Fraction(0)
    return Fraction(text)


# เช็กว่าเศษส่วนนี้จริง ๆ แล้วเป็นจำนวนเต็มไหม (ส่วนต้องเป็น 1)
def is_integer_fraction(value):
    return isinstance(value, Fraction) and value.denominator == 1


# จัดรูปเศษส่วนให้อ่านง่ายตอนแสดงผล
# ถ้าเป็นจำนวนเต็มก็โชว์แค่เลขเดียว เช่น 3
# ถ้ายังเป็นเศษส่วนก็โชว์แบบ a/b เช่น 7/5

def format_fraction(value):
    if is_integer_fraction(value):
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


# เอาสถานะสมการ [A|b] มาเรียงเป็นหลายบรรทัดไว้ดูขั้นตอน

def format_system_state(A, b):
    lines = []
    for row, constant in zip(A, b):
        coeffs = ", ".join(format_fraction(value) for value in row)
        lines.append(f"[{coeffs}] | {format_fraction(constant)}")
    return "\n".join(lines)


# แปลงเมทริกซ์ให้กลายเป็นข้อความหลายบรรทัด

def format_matrix(matrix):
    return "\n".join(
        "[" + ", ".join(format_fraction(value) for value in row) + "]"
        for row in matrix
    )


# --- วิธีที่ 1: Gauss Elimination + Partial Pivoting ---
# คืนคำตอบ x (แบบ Fraction) และบันทึกทุกสเต็ป

def gauss_elimination(A, b):
    n = len(A)
    A = [row[:] for row in A]  # ก๊อปปี้ไว้ก่อน จะได้ไม่ไปทับข้อมูลเดิม
    b = b[:]
    steps = ["Initial augmented matrix:\n" + format_system_state(A, b)]

    # วนทีละคอลัมน์ แล้วทำให้ค่าด้านล่าง pivot เป็น 0
    for i in range(n):
        # หาแถวที่เหมาะเป็น pivot ที่สุด (ค่าสัมบูรณ์มากสุดในคอลัมน์นี้)
        pivot = max(range(i, n), key=lambda r: abs(A[r][i]))
        if A[pivot][i] == 0:
            raise ValueError("Matrix is singular.")

        # ถ้า pivot ยังไม่อยู่แถวบนสุดของรอบนี้ ก็สลับแถวก่อน
        if pivot != i:
            A[i], A[pivot] = A[pivot], A[i]
            b[i], b[pivot] = b[pivot], b[i]
            steps.append(
                f"Step {len(steps)}: Swap row {i + 1} with row {pivot + 1}.\n"
                + format_system_state(A, b)
            )

        # จัดการค่าที่อยู่ใต้ pivot ให้กลายเป็น 0
        for r in range(i + 1, n):
            factor = A[r][i] / A[i][i]
            A[r] = [a - factor * p for a, p in zip(A[r], A[i])]
            b[r] = b[r] - factor * b[i]
            steps.append(
                f"Step {len(steps)}: R{r + 1} = R{r + 1} - ({format_fraction(factor)})·R{i + 1}.\n"
                + format_system_state(A, b)
            )

    # หาค่า x ย้อนจากแถวล่างขึ้นบน (back substitution)
    x = [Fraction(0) for _ in range(n)]
    for i in range(n - 1, -1, -1):
        total = sum(A[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (b[i] - total) / A[i][i]
        steps.append(
            f"Step {len(steps)}: Back substitution for x{i + 1} = {format_fraction(x[i])}."
        )
    return x, steps


# --- วิธีที่ 2: Gauss-Jordan Elimination + Partial Pivoting ---
# คืนคำตอบ x (แบบ Fraction) และบันทึกทุกสเต็ป

def gauss_jordan(A, b):
    n = len(A)
    A = [row[:] for row in A]  # ก๊อปปี้ไว้ก่อน จะได้ไม่ไปแก้ของเดิม
    b = b[:]
    steps = ["Initial augmented matrix:\n" + format_system_state(A, b)]

    for i in range(n):
        # เลือก pivot ที่ดีที่สุดในคอลัมน์นี้
        pivot = max(range(i, n), key=lambda r: abs(A[r][i]))
        if A[pivot][i] == 0:
            raise ValueError("Matrix is singular.")

        # ถ้าจำเป็นก็สลับแถว
        if pivot != i:
            A[i], A[pivot] = A[pivot], A[i]
            b[i], b[pivot] = b[pivot], b[i]
            steps.append(
                f"Step {len(steps)}: Swap row {i + 1} with row {pivot + 1}.\n"
                + format_system_state(A, b)
            )

        # ทำให้ pivot ตำแหน่งนี้กลายเป็น 1
        pivot_value = A[i][i]
        A[i] = [value / pivot_value for value in A[i]]
        b[i] = b[i] / pivot_value
        steps.append(
            f"Step {len(steps)}: Normalize row {i + 1} by dividing by {format_fraction(pivot_value)}.\n"
            + format_system_state(A, b)
        )

        # ทำให้ค่าในคอลัมน์ pivot ของแถวอื่นเป็น 0
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

    # พอได้ reduced row-echelon แล้ว คำตอบอยู่ใน b เลย
    return b, steps


# --- [ส่วนที่ 4: ตัวควบคุมหลัก (Main Controller)] ---
# อ่านค่าจากหน้าเว็บ เลือกวิธีคำนวณ แล้วโชว์ผลลัพธ์พร้อมขั้นตอน

def solve_system(event):
    try:
        # ดึงค่าที่ผู้ใช้กรอกมาจากหน้าเว็บ
        rows = int(document.querySelector("#rows").value)
        cols = int(document.querySelector("#cols").value)
        method = document.querySelector("#method").value

        # เตรียมข้อมูล 2 แบบไว้ใช้
        # - แบบ Fraction สำหรับคำนวณให้ตรงเป๊ะ
        # - แบบ float สำหรับใช้กับ pseudoinverse ตอนเมทริกซ์ไม่จัตุรัส
        A_list = []
        b_list = []
        A_exact = []
        b_exact = []

        # วนอ่านค่า A และ b จากช่องกรอกทั้งหมด
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

        # แปลงเป็น NumPy array เผื่อใช้วิธี pseudoinverse
        A = np.array(A_list)
        b = np.array(b_list)

        # ถ้าเป็นเมทริกซ์จัตุรัส ใช้วิธีที่ผู้ใช้เลือกได้เลย
        if rows == cols:
            if method == "gauss":
                x_exact, steps = gauss_elimination(A_exact, b_exact)
                info = "Calculation: Gauss Elimination with Partial Pivoting."
            elif method == "jordan":
                x_exact, steps = gauss_jordan(A_exact, b_exact)
                info = "Calculation: Gauss-Jordan Elimination."
            else:
                raise ValueError(
                    "Method not supported in main.py. Use gauss or jordan "
                    "(LU/Inverse is handled in LU.py)."
                )

            # แปลงคำตอบเป็นทศนิยม เพื่อโชว์ในการ์ดผลลัพธ์
            x = [float(value) for value in x_exact]

            # สร้างข้อความคำตอบแบบ exact (จำนวนเต็ม/เศษส่วน)
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
            process_text = "\n\n".join(steps)

        # ถ้าไม่ใช่เมทริกซ์จัตุรัส ให้ใช้ pseudoinverse หา least-squares
        else:
            x = np.linalg.pinv(A) @ b
            info = f"Non-Square Matrix detected ({rows}x{cols}). Applied Pseudoinverse."
            process_text = (
                "Initial matrix is non-square, so row-reduction steps are not used.\n"
                "The solver applied the pseudoinverse method to estimate a least-squares solution."
            )

        # เอาคำตอบไปแสดงในกริดบนหน้าเว็บ
        grid = document.querySelector("#solutionGrid")
        grid.innerHTML = ""
        for i, val in enumerate(x):
            grid.innerHTML += f"""
                <div class="bg-white p-5 rounded-2xl border border-gray-100 shadow-sm text-center">
                    <div class="text-blue-500 font-bold text-[10px] uppercase mb-1">Variable X{i+1}</div>
                    <div class="text-2xl font-mono font-bold">{float(val):.4f}</div>
                </div>"""

        # แสดงข้อความสรุปกับขั้นตอนการคำนวณ
        document.querySelector("#resultArea").classList.remove("hidden")
        document.querySelector("#extraInfo").innerText = info
        document.querySelector("#processSteps").innerText = process_text

    except Exception as e:
        # ถ้าเจอ error ก็แจ้งบนหน้าเว็บเลย
        document.querySelector("#extraInfo").innerText = f"Error: {str(e)}"
        document.querySelector("#processSteps").innerText = ""

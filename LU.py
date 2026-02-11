import copy

def print_matrix(matrix, name="Matrix"):
    """ฟังก์ชันสำหรับแสดงผลเมทริกซ์ให้สวยงาม"""
    print(f"{name}:")
    for row in matrix:
        print("  " + "  ".join(f"{val:8.3f}" for val in row))
    print()

def identity_matrix(n):
    """สร้างเมทริกซ์เอกลักษณ์ขนาด n x n"""
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

def multiply_matrix_vector(matrix, vector):
    """คูณเมทริกซ์กับเวกเตอร์"""
    result = []
    for i in range(len(matrix)):
        total = 0
        for j in range(len(vector)):
            total += matrix[i][j] * vector[j]
        result.append(total)
    return result

# ---------------------------------------------------------
# ส่วนที่ 1: LU Factorization (สำหรับแก้ระบบสมการ)
# ---------------------------------------------------------

def lu_decomposition(A):
    """
    แยกตัวประกอบ PA = LU โดยใช้ Partial Pivoting
    คืนค่า: P (Permutation), L (Lower), U (Upper)
    """
    n = len(A)
    U = copy.deepcopy(A)       # copy A มาใส่ U (ใช้ทำ U ให้เป็นสามเหลี่ยมบน)
    L = identity_matrix(n)     
    P = identity_matrix(n)     

    for k in range(n):
        # --- Partial Pivoting ---
        # หาแถวที่มีค่าสัมบูรณ์สูงสุดในคอลัมน์ k เพื่อลด error 
        pivot_row = k
        max_val = abs(U[k][k])
        for i in range(k + 1, n):
            if abs(U[i][k]) > max_val:
                max_val = abs(U[i][k])
                pivot_row = i
        
        # สลับแถวใน U, P และ L (เฉพาะส่วนที่คำนวณแล้ว)
        U[k], U[pivot_row] = U[pivot_row], U[k]
        P[k], P[pivot_row] = P[pivot_row], P[k]
        if k > 0:
            for col in range(k):
                L[k][col], L[pivot_row][col] = L[pivot_row][col], L[k][col]

        # ตรวจสอบว่าเป็น Singular Matrix หรือไม่
        if abs(U[k][k]) < 1e-10:
            raise ValueError("Matrix is Singular (cannot solve).")

        # --- Elimination Process ---
        # ทำการกำจัดตัวแปรเพื่อสร้าง Upper Triangular Matrix (U)
        for i in range(k + 1, n):
            factor = U[i][k] / U[k][k]
            L[i][k] = factor # เก็บตัวคูณไว้ใน L
            for j in range(k, n):
                U[i][j] -= factor * U[k][j]
                
    return P, L, U

def forward_substitution(L, b):
    """แก้สมการ Ly = b (หาค่า y)"""
    n = len(L)
    y = [0 for _ in range(n)]
    for i in range(n):
        sum_val = sum(L[i][j] * y[j] for j in range(i))
        y[i] = (b[i] - sum_val) / L[i][i]
    return y

def backward_substitution(U, y):
    """แก้สมการ Ux = y (หาค่า x)"""
    n = len(U)
    x = [0 for _ in range(n)]
    for i in range(n - 1, -1, -1):
        sum_val = sum(U[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (y[i] - sum_val) / U[i][i]
    return x

def solve_linear_system_lu(A, b):
    """ฟังก์ชันหลักสำหรับแก้ระบบสมการด้วย LU"""
    try:
        # 1. แยกตัวประกอบ A เป็น P, L, U
        P, L, U = lu_decomposition(A)
        
        # 2. จัดเรียง b ใหม่ตาม P (Pb)
        b_new = multiply_matrix_vector(P, b)
        
        # 3. หา y จาก Ly = Pb
        y = forward_substitution(L, b_new)
        
        # 4. หา x จาก Ux = y
        x = backward_substitution(U, y)
        return x
    except ValueError as e:
        return str(e)

# ---------------------------------------------------------
# ส่วนที่ 2: Inverse Matrix (โดยใช้ Gauss-Jordan)
# ---------------------------------------------------------

def inverse_matrix(A):
    """
    หา Inverse ของเมทริกซ์โดยใช้วิธี Augmented Matrix [A|I] -> [I|A^-1]
    """
    n = len(A)
    # สร้าง Augmented Matrix [A | Identity]
    M = [A[i] + [1 if i == j else 0 for j in range(n)] for i in range(n)]

    # Forward Elimination (ทำให้เป็น Upper Triangle)
    for k in range(n):
        # Pivoting
        pivot_row = k
        for i in range(k + 1, n):
            if abs(M[i][k]) > abs(M[pivot_row][k]):
                pivot_row = i
        M[k], M[pivot_row] = M[pivot_row], M[k]

        pivot = M[k][k]
        if abs(pivot) < 1e-10:
            return None # หา Inverse ไม่ได้

        # หารแถว k ด้วย pivot เพื่อให้สมาชิกนำเป็น 1
        for j in range(k, 2 * n):
            M[k][j] /= pivot

        # กำจัดแถวอื่นๆ
        for i in range(n):
            if i != k:
                factor = M[i][k]
                for j in range(k, 2 * n):
                    M[i][j] -= factor * M[k][j]

    # แยกส่วน Inverse ออกมา (ครึ่งขวาของ Matrix)
    inv = [row[n:] for row in M]
    return inv

# ---------------------------------------------------------
# ส่วน Main Program: Test Cases
# ---------------------------------------------------------
if __name__ == "__main__":
    print("=== Demo Program Project CSS114 ===")

    # --- ตัวอย่างที่ 1: ระบบสมการ 3 ตัวแปร ---
    print("\n[Test Case 1] Solving 3x3 System using LU Factorization")
    A1 = [
        [2, 1, 3],
        [4, 3, 5],
        [6, 5, 5]
    ]
    b1 = [1, 1, -3]
    result1 = solve_linear_system_lu(A1, b1)
    print(f"Solution x: {result1}")

    # --- ตัวอย่างที่ 2: ระบบสมการ 4 ตัวแปร ---
    print("\n[Test Case 2] Solving 4x4 System using LU Factorization")
    A2 = [
        [2, -1, -3, 1],
        [1, 1, 1, -2],
        [3, 2, -3, -4],
        [-1, -4, 1, 1]
    ]
    b2 = [9, 10, 6, 6]
    result2 = solve_linear_system_lu(A2, b2)
    print(f"Solution x: {result2}")

    # --- ตัวอย่างที่ 3: หา Inverse Matrix ---
    print("\n[Test Case 3] Finding Inverse Matrix")
    A3 = [
        [1, 2, -3],
        [-1, 1, -1],
        [0, -2, 3]
    ]
    inv3 = inverse_matrix(A3)
    if inv3:
        print_matrix(inv3, "Inverse of A3")
    else:
        print("Matrix A3 is singular, no inverse.")

    # --- ตัวอย่างที่ 4: หา Inverse และแก้สมการ Ax=b ---
    print("\n[Test Case 4] Finding Inverse and Solving Ax=b")
    # หมายเหตุ: Matrix นี้ในโจทย์ แถว 1 และ แถว 3 เหมือนกัน (1, 2, 3) 
    # ทำให้ Determinant = 0 (Singular Matrix)
    A4 = [
        [1, 2, 3],
        [-1, -1, -1],
        [1, 2, 3]
    ]
    b4 = [5, 3, -1]
    
    # พยายามหา Inverse
    inv4 = inverse_matrix(A4)
    if inv4:
        print_matrix(inv4, "Inverse of A4")
        # ถ้าหา Inverse ได้ ก็จะคูณเพื่อหา x
        x4 = multiply_matrix_vector(inv4, b4)
        print(f"Solution x (from A^-1 * b): {x4}")
    else:
        print("Result: Matrix A4 is singular (Determinant is 0). Cannot find inverse.")
        
    # ลองแก้ด้วย LU
    result4_lu = solve_linear_system_lu(A4, b4)
    print(f"LU Method Result: {result4_lu}")
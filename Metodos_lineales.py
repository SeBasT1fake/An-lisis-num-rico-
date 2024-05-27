def forward_substitution(L, b):
    n = L.shape[0]
    z = np.zeros(n)

    for i in range(n):
        z[i] = (b[i] - np.dot(L[i, :i], z[:i])) / L[i, i]

    return z

def backward_substitution(U, z):
    n = U.shape[0]
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        x[i] = (z[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]

    return x
    
def gauss_pivoteo(matrix, vector):
    n = matrix.shape[0]
    M = np.hstack([matrix, vector.reshape(-1, 1)])
    for i in range(n):
        max_row_index = np.argmax(np.abs(M[i:, i])) + i
        M[[i, max_row_index]] = M[[max_row_index, i]]
        for j in range(i + 1, n):
            factor = M[j, i] / M[i, i]
            M[j, i:] -= factor * M[i, i:]
    x = np.zeros(n)
    for i in reversed(range(n)):
        x[i] = (M[i, -1] - np.dot(M[i, i+1:n], x[i+1:n])) / M[i, i]
    return {"method": "Gauss Pivoteo", "result": x.tolist()}

def lu_factorization(matrix, vector):
    if len(matrix) != len(matrix[0]) or len(matrix) != len(vector):
        return {"error": "La matriz y el vector deben tener las mismas dimensiones."}
    n = len(matrix)
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    for i in range(n):
        L[i][i] = 1
        for k in range(i, n):
            suma = sum(L[i][j] * U[j][k] for j in range(i))
            U[i][k] = matrix[i][k] - suma
        for k in range(i + 1, n):
            suma = sum(L[k][j] * U[j][i] for j in range(i))
            L[k][i] = (matrix[k][i] - suma) / U[i][i]

    print("Matrices factorizadas")
    print("L= \n" + str(L) + "\nU= \n"+ str(U))
    print("\n\n")


    y = np.linalg.solve(L, vector)
    x = np.linalg.solve(U, y)
    return {"method": "LU Factorization", "result": x.tolist()}

def eliminacion_gaussiana(matrix, vector):
    if len(matrix) != len(matrix[0]) or len(matrix) != len(vector):
        return {"error": "La matriz y el vector deben tener las mismas dimensiones."}
    n = len(matrix)
    M = np.hstack([matrix, np.array(vector).reshape(-1, 1)])
    for i in range(n):
        max_row_index = np.argmax(np.abs(M[i:, i])) + i
        M[[i, max_row_index]] = M[[max_row_index, i]]
        for j in range(i + 1, n):
            factor = M[j, i] / M[i, i]
            M[j, i:] -= factor * M[i, i:]
    print("Matriz final")
    print("L= \n" + str(M))
    print("\n\n")
    x = np.zeros(n)
    for i in reversed(range(n)):
        x[i] = (M[i, -1] - np.dot(M[i, i+1:n], x[i+1:n])) / M[i, i]
    return {"method": "Eliminación Gaussiana", "result": x.tolist()}

def cholesky(matrix, vector):

    if not np.allclose(matrix, matrix.T):
        return {"error": "La matriz no es simétrica."}
    if not np.all(np.linalg.eigvals(matrix) > 0):
        return {"error": "La matriz no es definida positiva."}
    n = matrix.shape[0]
    L = np.eye(n)
    U = np.eye(n)

    for k in range(n):
        addition1 = 0
        for p in range(k):
            addition1 += L[k,p] * U[p,k]
        L[k,k] = (matrix[k,k] - addition1)**0.5   #es asi por cholesky
        U[k,k] = L[k,k]

        for i in range (k,n):
            addition2 = 0
            for p in range(k):
                addition2 += L[i,p]*U[p,k]
            L[i,k] = (matrix[i,k] - addition2)/U[k,k]

        for j in range(k+1,n):
            addition3 = 0
            for p in range(k):
                addition3 += L[k,p]*U[p,j]
            U[k,j] = (matrix[k,j]-addition3)/L[k,k]

    print("Matrices factorizadas")
    print("L= \n" + str(L) + "\nU= \n"+ str(U))
    print("\n\n")

    z = forward_substitution(L,vector)
    x = backward_substitution(U,z)
    return {"method": "Cholesky", "result": x.tolist()}

def crout(matrix, vector):
    n = matrix.shape[0]
    L = np.eye(n)
    U = np.eye(n)

    for k in range(n):
        addition1 = 0
        for p in range(k):
            addition1 += L[k,p] * U[p,k]
        L[k,k] = matrix[k,k] - addition1    #es asi por crout

        for i in range (k,n):
            addition2 = 0
            for p in range(k):
                addition2 += L[i,p]*U[p,k]
            L[i,k] = matrix[i,k] - addition2

        for j in range(k+1,n):
            addition3 = 0
            for p in range(k):
                addition3 += L[k,p]*U[p,j]
            U[k,j] = (matrix[k,j]-addition3)/L[k,k]

    print("Matrices factorizadas")
    print("L= \n" + str(L) + "\nU= \n"+ str(U))
    print("\n\n")

    z = forward_substitution(L,vector)
    x = backward_substitution(U,z)

    return {"method": "Crout", "result": x.tolist()}

def doolittle(matrix, vector):
    n = matrix.shape[0]
    L = np.eye(n)
    U = np.eye(n)

    for k in range(n):
        addition1 = 0
        for p in range(k):
            addition1 += L[k,p] * U[p,k]
        U[k,k] = matrix[k,k] - addition1    #es asi por doolittle

        for i in range (k,n):
            addition2 = 0
            for p in range(k):
                addition2 += L[i,p]*U[p,k]
            L[i,k] =(matrix[i,k] - addition2)/U[k,k]
            

        for j in range(k+1,n):
            addition3 = 0
            for p in range(k):
                addition3 += L[k,p]*U[p,j]
            U[k,j] = (matrix[k,j]-addition3)

    z = forward_substitution(L,vector)
    x = backward_substitution(U,z)
    print("Matrices factorizadas")
    print("L= \n" + str(L) + "\nU= \n"+ str(U))
    print("\n\n")

    return {"method": "Doolittle", "result": x.tolist()}

def gaussiana_sencilla(matrix, vector):
    n = len(matrix)
    for i in range(n):
        for j in range(i + 1, n):
            factor = matrix[j, i] / matrix[i, i]
            vector[j] -= factor * vector[i]
            matrix[j, i:] -= factor * matrix[i, i:]
    x = np.zeros(n)
    for i in reversed(range(n)):
        x[i] = (vector[i] - np.dot(matrix[i, i + 1:], x[i + 1:])) / matrix[i, i]
    return {"method": "Gaussiana Sencilla", "result": x.tolist()}

def gauss_seidel(matrix, vector, tol, max_iter, x0):
    D = np.diag(np.diag(matrix))
    L = -np.tril(matrix) + D
    U = -np.triu(matrix) + D
    T = np.linalg.inv(D - L).dot(U)
    C = np.linalg.inv(D - L).dot(vector)

    table = []
    errors = []

    xant = x0
    e = float('inf')
    ite = 1

    row = f"{0}    |   {x0}    |   {e}"
    table.append(row)

    while e > tol and ite < max_iter:
        xact = T.dot(xant) + C
        e = np.linalg.norm(xant - xact)
        row = f"{ite}    |   {xact}    |   {e}"
        errors.append(e)
        table.append(row)
        print(f"Iteration {ite}: xact = {xact}, e = {e}")  # Debug
        xant = xact
        ite = ite + 1

    if e < tol:
        return {"method": "Gauss-Seidel", "result": xact.tolist()}
    else:
        return {"method": "Gauss-Seidel", "result": "se excedió el máximo de iteraciones"}

def jacobi(matrix, vector, tol, max_iter, x0):
    D = np.diag(np.diag(matrix))
    L = -np.tril(matrix) + D
    U = -np.triu(matrix) + D
    T = np.linalg.inv(D).dot(L + U)
    C = np.linalg.inv(D).dot(vector)

    table = []
    errors = []

    xant = x0
    e = float('inf')
    ite = 1

    row = f"{0}    |   {x0}    |   {e}"
    table.append(row)

    while e > tol and ite < max_iter:
        xact = T.dot(xant) + C
        e = np.linalg.norm(xant - xact)
        row = f"{ite}    |   {xact}    |   {e}"
        errors.append(e)
        table.append(row)
        xant = xact
        ite = ite + 1

    if e < tol:
        return {"method": "Jacobi", "result": xact.tolist()}
    else:
        return {"method": "Jacobi", "result": "se excedió el máximo de iteraciones"}
    

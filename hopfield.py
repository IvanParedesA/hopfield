# Iván Ricardo Paredes Avilez A01705083
# 06/10/2025
# --------------------------------------------------------------
# Red Neuronal de Hopfield para reconocer letras en 5x8
# Lee los archivos directamente desde la carpeta dataset/
# --------------------------------------------------------------

FILAS = 8
COLS = 5

N = FILAS * COLS

# Capacidad teórica: aproximación sin librerías
def capacidad_teorica(N):
    # Aproximación de log2(N) sin librerías: contar divisiones por 2
    x = N
    pasos = 0
    while x > 1:
        x //= 2
        pasos += 1
    p1 = int(0.15 * N)
    p2 = int(N / (2 * pasos)) if pasos > 0 else 0
    return p1, p2

# ---------------- Lectura manual de archivos ----------------
def leer_figura(ruta):
    M = []
    with open(ruta, "r") as f:
        for linea in f:
            numeros = linea.strip().split()
            if numeros:
                fila = [int(x) for x in numeros]
                M.append(fila)
    return M

# ---------------- Funciones básicas ----------------
def imprimir_matriz(M):
    for fila in M:
        linea = ""
        for v in fila:
            if v == 1:
                linea += "#"
            else:
                linea += "."
        print(linea)

def matriz_a_vector(matriz):
    v = []
    for i in range(len(matriz)):
        for j in range(len(matriz[0])):
            if matriz[i][j] == 1:
                v.append(1)
            else:
                v.append(-1)
    return v

def vector_a_matriz(vector):
    M = []
    k = 0
    for i in range(FILAS):
        fila = []
        for j in range(COLS):
            if vector[k] == 1:
                fila.append(1)
            else:
                fila.append(0)
            k += 1
        M.append(fila)
    return M

def hamming(u, v):
    d = 0
    for i in range(len(u)):
        if u[i] != v[i]:
            d += 1
    return d

# ---------------- Clase Hopfield ----------------
class Hopfield:
    def __init__(self, n):
        self.n = n
        self.W = [[0 for _ in range(n)] for _ in range(n)]

    def entrenar(self, patrones):
        for x in patrones:
            for i in range(self.n):
                for j in range(self.n):
                    if i != j:
                        self.W[i][j] += x[i] * x[j]

    def activar(self, u):
        if u >= 0:
            return 1
        else:
            return -1

    def recall_asincrono(self, estado, max_iter=100):
        s = estado[:]
        for _ in range(max_iter):
            cambios = 0
            for i in range(self.n):
                suma = 0
                for j in range(self.n):
                    suma += self.W[i][j] * s[j]
                nuevo = self.activar(suma)
                if nuevo != s[i]:
                    s[i] = nuevo
                    cambios += 1
            if cambios == 0:
                return s
        return s

def mas_parecido(salida, patrones, nombres):
    mejor = nombres[0]
    mejor_d = hamming(salida, patrones[0])
    for i in range(1, len(patrones)):
        d = hamming(salida, patrones[i])
        if d < mejor_d:
            mejor_d = d
            mejor = nombres[i]
    return mejor, mejor_d

# ---------------- Programa principal ----------------
if __name__ == "__main__":
    # Leer las figuras directamente (sin librerías)
    A = matriz_a_vector(leer_figura("dataset/A.txt"))
    B = matriz_a_vector(leer_figura("dataset/B.txt"))
    C = matriz_a_vector(leer_figura("dataset/C.txt"))
    X = matriz_a_vector(leer_figura("dataset/x.txt"))

    patrones = [A, B, C]
    nombres = ["A", "B", "C"]

    # Mostrar capacidad teórica (según la presentación)
    p1, p2 = capacidad_teorica(N)
    print("Capacidad teórica para N =", N)
    print("  p ≈ 0.15N =", p1)
    print("  p ≈ N/(2 log2 N) =", p2)

    red = Hopfield(N)
    red.entrenar(patrones)


    print("\nProbando con figura: x.txt")
    Mx = leer_figura("dataset/x.txt")
    print("Figura de entrada:")
    imprimir_matriz(Mx)

    rec = red.recall_asincrono(X)
    print("\nFigura recuperada:")
    imprimir_matriz(vector_a_matriz(rec))

    mejor, dist = mas_parecido(rec, patrones, nombres)
    print("\nEl patrón más parecido es:", mejor, "con distancia", dist)
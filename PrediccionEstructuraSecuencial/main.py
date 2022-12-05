from utils import *
from Nussinov import Matrix

# ?Link del simulador del algoritmo de Nussinov
# https://rna.informatik.uni-freiburg.de/Teaching/index.jsp?toolName=Nussinov
# Ejemplo de clase:
# GGAACUAUC
# GGAACUAUC
# Ejemplo de diapositivas
# GGGAAAUCC
# GGGAAAUCC
if __name__ == '__main__':
    s1, s2 = readInput()
    print("S1:", s1)
    print("S2:", s2)
    print(len(s1))
    MatrixNussinov = Matrix(s1, s2, debug=False, backtracking=False)
    MatrixNussinov.fun(s1, s2)
    print(MatrixNussinov.values_matrix)

    # print(MatrixNussinov.matrix_coordinates[0][19])
    # print(MatrixNussinov.matrix_coordinates[0, 19])

    MatrixNussinov.alignments(s1, s2)
    secuence, code = MatrixNussinov.getSecuencePatron()
    print(secuence)
    print(code)
    # MatrixNussinov.get
    # print(MatrixNussinov.values_matrix)
    # !Hay error mira con bacteria de 30 la matrix esta bien pero obtener el unico camino esta  mal
    # MatrixNussinov.alignments(s1, s2)
    # MatrixNussinov.saveTXT()
    # print("Termino de escribir")

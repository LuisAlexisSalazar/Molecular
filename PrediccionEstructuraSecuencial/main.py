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
    MatrixNussinov = Matrix(s1, s2, debug=False, backtracking=True)
    MatrixNussinov.fun(s1, s2)
    print(MatrixNussinov.values_matrix)
    MatrixNussinov.alignments(s1, s2)
    MatrixNussinov.saveTXT()

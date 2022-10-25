from utils import *
from Meddleman import Matrix

# ?Link del simulador del algoritmo de Meddleman
# https://bioboot.github.io/bimm143_W20/class-material/nw/
if __name__ == '__main__':
    s1, s2 = readInput()
    # s1,s2 =  "ATTGCCATT", "ATCTTCTT"
    MatrixMeddleman = Matrix(s1, s2, debug=False, backtracking=True)
    MatrixMeddleman.fun(s1, s2)
    MatrixMeddleman.alignments(s1, s2)
    list_per_alignments = MatrixMeddleman.get_aligments()
    # MatrixMeddleman.saveTXT()

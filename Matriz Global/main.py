from utils import *
from Meddleman import Matrix

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    s1, s2 = readInput()
    MatrixMeddleman = Matrix(s1, s2, debug=False)
    MatrixMeddleman.fun(s1, s2)
    MatrixMeddleman.alignments(s1, s2)
    MatrixMeddleman.saveTXT()

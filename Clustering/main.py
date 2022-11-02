from Clustering import Cluster
from Meddleman import MatrixScoreAllString, listNone
from utils import readInputs, print_Winner, readMatrix
import numpy as np

list_cccs = []


# *cophenetic correlation coefficients
def getCopheCorreCoe(matrix, matrixCofenetica, n):
    r, c = np.triu_indices(n, 1)
    list_indexes = list(zip(r, c))
    triangle_matrix = [matrix[i, j] for [i, j] in list_indexes]
    triangle_matrixCofenetica = [matrixCofenetica[i, j] for [i, j] in list_indexes]

    # *Calcular la correlación con numpy
    ccc = np.corrcoef(triangle_matrix, triangle_matrixCofenetica)[0, 1]
    list_cccs.append(ccc)
    # https://people.revoledu.com/kardi/tutorial/Clustering/Cophenetic.htm
    from scipy.stats import pearsonr
    # ?Calcular la correlación con scipy mayor precisión
    # ccc = pearsonr(triangle_matrix, triangle_matrixCofenetica).statistic


DEBUG = True
cluster = [Cluster(debug=DEBUG, criterion="Min"),
           Cluster(debug=DEBUG, criterion="Max"),
           Cluster(debug=DEBUG, criterion="Avg")]

# ?Link del simulador del algoritmo de Meddleman
# https://bioboot.github.io/bimm143_W20/class-material/nw/
if __name__ == '__main__':
    matrix, n = readMatrix(True)

    for c in cluster:
        indexes_per_iterations, minorValue_per_iterations = c.execute(matrix)
        matrixCofenetica = c.getMatrixCofenetica(matrix, indexes_per_iterations, minorValue_per_iterations)
        getCopheCorreCoe(matrix, matrixCofenetica, n)

    print_Winner(list_cccs)

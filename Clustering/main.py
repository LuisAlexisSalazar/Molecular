from copy import copy, deepcopy

from Clustering import Cluster
from utils import *
from Meddleman import Matrix
import numpy as np
import pandas as pd
from ProgresiveAlignment import neighbotJoining

colums_header = []
listNone = []
paresCaminos = []
index_center_star = None


# https://stackoverflow.com/questions/31247198/python-pandas-write-content-of-dataframe-into-text-file
def saveTxt(df):
    with open('output.txt', mode='w') as file_object:
        print("Matrix de Score:\n", file=file_object)
        print(df, file=file_object)


def saveStartTXT(center_string, index_center_star, pares_alignment, multiple_alignment):
    f = open("output.txt", "a")
    f.write("Cadena central: " + center_string + '\n')
    f.write("Indice de la cadena central: " + str(index_center_star) + '\n')
    f.write("\nAlineaciones con la cadena central:\n")

    for i in pares_alignment:
        f.write(i + "\n")

    f.write("\nAlinieación Multiple\n")
    for i in multiple_alignment:
        f.write(i + "\n")


def generateDistance(string1, string2):
    different_positions = 0
    alignments_positions = 0
    for c_1, c_2 in zip(string1, string2):
        if c_1 != "-" and c_2 != "-":
            alignments_positions += 1
        elif (c_1 == "-" and c_2 != "-") or (c_1 != "-" and c_2 == "-"):
            different_positions += 1
    return different_positions / alignments_positions


def MatrixScoreAllString(list_inputs):
    with open('output.txt', 'w') as f:
        f.write("Cadenas Ingresadas:\n")
        for line in list_inputs:
            f.write("%s\n" % line)
    n_string = len(list_inputs)
    # *dtype: https://numpy.org/doc/stable/reference/arrays.dtypes.html
    matrix_distance = np.full(shape=(n_string, n_string), fill_value=0.0, dtype=np.float_)
    matrix_MatrixGlobal = []
    matrix_alignments = np.full(shape=(n_string, n_string), fill_value="").tolist()
    for n in range(len(list_inputs)):
        matrix_MatrixGlobal.append(listNone)

    for i in range(n_string):
        for j in range(n_string):
            if i != j:
                s1, s2 = list_inputs[i], list_inputs[j]
                MatrixGlobal = Matrix(s1, s2)
                MatrixGlobal.fun(s1, s2)
                matrix_MatrixGlobal[i][j] = MatrixGlobal
                matrix_alignments[i][j] = MatrixGlobal.getOneAligment()
                distance = generateDistance(matrix_alignments[i][j][0], matrix_alignments[i][j][1])
                matrix_distance[i][j] = distance

    neighbotJoining(matrix_distance, n_string, colums_header)
    # with open('alignments.txt', 'w') as f:
    #     for line in matrix_alignments:
    #         f.write("%s\n" % line)
    # with open('matrizDistancia.txt', 'w') as f:
    #     for line in matrix_distance:
    #         f.write("%s\n" % line)


# ?Link del simulador del algoritmo de Meddleman
# https://bioboot.github.io/bimm143_W20/class-material/nw/
if __name__ == '__main__':
    # list_inputs, colums_header = readInputs()
    # for i in range(len(list_inputs)):
    #     listNone.append(None)
    # MatrixScoreAllString(list_inputs)
    cluster = Cluster()

    temp_list = [[0, 3, 4, 7, 2], [3, 0, 1, 2, 5], [4, 1, 0, 6, 9], [7, 2, 6, 0, 8], [2, 5, 9, 8, 0]]
    old_distance_matrix = np.array(temp_list)
    old_distance_matrix = old_distance_matrix.reshape(5, 5)
    old_distance_matrix = np.round(old_distance_matrix, decimals=1)
    old_distance_matrix = old_distance_matrix.astype('float64')

    cluster.aglomerativa(old_distance_matrix)

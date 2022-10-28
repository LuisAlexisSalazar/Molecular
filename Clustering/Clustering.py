import numpy as np
from Meddleman import Matrix

"""
Hacer las 3 estrategias maximo,minimo, promedio
Input sera un matrix de distancias
Tambien debemos hacer matrix cogenitica para saber cual es mejor 

Quines se unen y en que valor se unen en cada iteracón
porque esta matrix de congenetica se usan estos valores

Tambien debemos generar la matrix cogenetica
y el ccc de cada matrix cogenitica de cada estrategica
24 x 24
"""

VALUE_PROBLEM = 1


def generateNewDistanceMatrix(oldDMatrix, index_minor_value, n):
    new_matrix_D = np.copy(oldDMatrix)
    # ?mv = Minor value
    i_mv = index_minor_value[0]
    j_mv = index_minor_value[1]
    new_matrix_D = np.delete(new_matrix_D, obj=j_mv, axis=0)
    new_matrix_D = np.delete(new_matrix_D, obj=j_mv, axis=1)
    for i in range(n - 1):
        if i != i_mv:

            if i < j_mv:
                new_matrix_D[i_mv][i] = (1 / 2) * (oldDMatrix[i_mv, i] + oldDMatrix[j_mv, i] - oldDMatrix[i_mv, j_mv])
                new_matrix_D[i, i_mv] = new_matrix_D[i_mv, i]
            else:
                new_matrix_D[i_mv, i] = (1 / 2) * (
                        oldDMatrix[i_mv, i + 1] + oldDMatrix[j_mv, i + 1] - oldDMatrix[i_mv, j_mv])
                new_matrix_D[i, i_mv] = new_matrix_D[i_mv, i]
            # !Problema cuando la distancia es negativa
            if new_matrix_D[i, i_mv] < 0:
                new_matrix_D[i_mv, i] = VALUE_PROBLEM
                new_matrix_D[i, i_mv] = VALUE_PROBLEM

    new_matrix_D = np.round(new_matrix_D, decimals=2)
    return new_matrix_D


def generateDistance(string1, string2):
    different_positions = 0
    alignments_positions = 0
    for c_1, c_2 in zip(string1, string2):
        if c_1 != "-" and c_2 != "-":
            alignments_positions += 1
        elif (c_1 == "-" and c_2 != "-") or (c_1 != "-" and c_2 == "-"):
            different_positions += 1
    return different_positions / alignments_positions


class Cluster:

    def __init__(self, tactic="maximos", list_inputs=[], matrix_distance=[]):
        # def __int__(self, tactic, list_inputs, matrix_distance):
        self.list_inputs = list_inputs
        self.tactic = tactic
        if len(list_inputs) == 0:
            self.matrix_distance = np.zeros((2, 2))
        else:
            print("Debe haber ingresado una matrix de distancias")

    def generate_matrix_distance(self):
        listNone = []

        n_string = len(self.list_inputs)
        matrix_distance = np.full(shape=(n_string, n_string), fill_value=0.0, dtype=np.float_)
        matrix_MatrixGlobal = []
        matrix_alignments = np.full(shape=(n_string, n_string), fill_value="").tolist()
        n_string_inputs = len(self.list_inputs)
        for n in range(n_string_inputs):
            matrix_MatrixGlobal.append(listNone)

        for i in range(n_string):
            for j in range(n_string):
                if i != j:
                    s1, s2 = self.list_inputs[i], self.list_inputs[j]
                    MatrixGlobal = Matrix(s1, s2)
                    MatrixGlobal.fun(s1, s2)
                    matrix_MatrixGlobal[i][j] = MatrixGlobal
                    matrix_alignments[i][j] = MatrixGlobal.getOneAligment()
                    distance = generateDistance(matrix_alignments[i][j][0], matrix_alignments[i][j][1])
                    matrix_distance[i][j] = distance
        self.matrix_distance = matrix_distance

    # def aglomerativa(self, old_distance_matrix, n, list_inputs):
    def aglomerativa(self, old_distance_matrix):
        # neighbotJoining(self.matrix_distance, n_string, colums_header)
        # if n == 2:
        if False:
            print("La matrix ya esta reducido")
        else:
            # ? ---------------Ejemplo del PDF---------------
            # temp_list = [[0, 3, 4, 7, 2], [3, 0, 1, 2, 5], [4, 1, 0, 6, 9], [7, 2, 6, 0, 8], [2, 5, 9, 8, 0]]
            # old_distance_matrix = np.array(temp_list)
            # old_distance_matrix = old_distance_matrix.reshape(5, 5)
            # old_distance_matrix = np.round(old_distance_matrix, decimals=1)
            # old_distance_matrix = old_distance_matrix.astype('float64')
            # ? ----------------------------------------------------------
            # print("D del inicio =  \n", old_distance_matrix)
            # new_matrix_D = np.full(shape=(n - 1, n - 1), fill_value=0.0, dtype=np.float_)
            new_matrix_D = old_distance_matrix
            n_veces_reducidas = 0
            while new_matrix_D.shape[0] != 2:
                # matrix_with_sum = getMatrixWithSum(old_distance_matrix)
                print(old_distance_matrix, end="\n\n")
                matrix_tria_Upper = np.triu(old_distance_matrix, 1)
                print(matrix_tria_Upper)
                # result = np.where(old_distance_matrix == np.amin(old_distance_matrix))
                # listOfCordinates = list(zip(result[0], result[1]))
                # f.write("\nMatrix D con la columna de sumatoria\n")
                # for line in matrix_with_sum:
                #     f.write("%s \n" % line)

                # print("Resultado:", listOfCordinates)
                break
                # * Q =
            #     distance_promedia = generateQ(matrix_with_sum, n)
            #     f.write("Matrix Q\n")
            #     for line in distance_promedia:
            #         f.write("%s \n" % line)
            #     # print("Q = \n", distance_promedia)
            #     index_minor_value = getIndexMinorValue(distance_promedia, n)
            #     print("Unión de los string:", list_inputs[index_minor_value[0]], " ",
            #           list_inputs[index_minor_value[1]])
            #     f.write("Menor valor:" + str(distance_promedia[index_minor_value]) + "\n")
            #     f.write("Indice del menor valor:" + str(index_minor_value) + "\n")
            #     f.write(
            #         "String a unirse:" + list_inputs[index_minor_value[0]] + "\t" + list_inputs[
            #             index_minor_value[1]] + "\n")
            #
            #     NewElement = list_inputs[index_minor_value[0]] + list_inputs[index_minor_value[1]]
            #     elements_to_remove = [list_inputs[index_minor_value[0]], list_inputs[index_minor_value[1]]]
            #     list_inputs = [e for e in list_inputs if e not in elements_to_remove]
            #     list_inputs.insert(index_minor_value[0], NewElement)
            #     print("Nuevos strings:", list_inputs)
            #     f.write("Nuevos strings:\n")
            #     for line in list_inputs:
            #         f.write("%s \t" % line)
            #
            #     # *New matrix D =
            #     new_matrix_D = generateNewDistanceMatrix(old_distance_matrix, index_minor_value, n)
            #     old_distance_matrix = new_matrix_D
            #     # print("Nueva D =  \n", new_matrix_D)
            #     n = n - 1
            #     n_veces_reducidas += 1
            #     f.write("\n\n")
            #     f.write("\nMatrix D\n")
            #
            # print("Ultima Matrix de Distancia\n", old_distance_matrix)
            # for line in old_distance_matrix:
            #     f.write("%s \n" % line)

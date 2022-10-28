import numpy as np
import pandas as pd

VALUE_PROBLEM = 1


def getMatrixWithSum(distance_matrix):
    df = pd.DataFrame(data=distance_matrix)
    df['Sum'] = df.sum(axis=1)

    matrix_with_sum = df.values
    matrix_with_sum = np.round(matrix_with_sum, decimals=1)
    return matrix_with_sum


def generateQ(matrix_distance, n):
    # print("N:", n)
    new_matrix_Q = np.full(shape=(n, n), fill_value=0.0, dtype=np.float_)
    # print(new_matrix_Q)
    index_column_sum = n
    for i in range(n):
        for j in range(i + 1, n):
            if i != j:
                new_matrix_Q[i, j] = matrix_distance[i, j] - (1 / (n - 2)) * (
                        matrix_distance[i, index_column_sum] + matrix_distance[j, index_column_sum])
                new_matrix_Q[j, i] = new_matrix_Q[i, j]
    new_matrix_Q = np.round(new_matrix_Q, decimals=1)
    return new_matrix_Q


def getIndexMinorValue(matrix_Q, n):
    menor = 100000000
    index_i = 0
    index_j = 0
    for i in range(n):
        for j in range(i + 1, n):
            if menor > matrix_Q[i, j] and i != j:
                menor = matrix_Q[i, j]
                index_i = i
                index_j = j
    indexsMinorValues = (index_i, index_j)
    return indexsMinorValues


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


def neighbotJoining(old_distance_matrix, n, list_inputs):
    with open('output.txt', 'a') as f:
        f.write("Cadenas Reducidas:\n")
        for line in list_inputs:
            f.write("%s \t" % line)

        if n == 2:
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
                matrix_with_sum = getMatrixWithSum(old_distance_matrix)
                f.write("\nMatrix D con la columna de sumatoria\n")
                for line in matrix_with_sum:
                    f.write("%s \n" % line)
                # * Q =
                distance_promedia = generateQ(matrix_with_sum, n)
                f.write("Matrix Q\n")
                for line in distance_promedia:
                    f.write("%s \n" % line)
                # print("Q = \n", distance_promedia)
                index_minor_value = getIndexMinorValue(distance_promedia, n)
                print("Unión de los string:", list_inputs[index_minor_value[0]], " ", list_inputs[index_minor_value[1]])
                f.write("Menor valor:" + str(distance_promedia[index_minor_value]) + "\n")
                f.write("Indice del menor valor:" + str(index_minor_value) + "\n")
                f.write(
                    "String a unirse:" + list_inputs[index_minor_value[0]] + "\t" + list_inputs[
                        index_minor_value[1]] + "\n")

                NewElement = list_inputs[index_minor_value[0]] + list_inputs[index_minor_value[1]]
                elements_to_remove = [list_inputs[index_minor_value[0]], list_inputs[index_minor_value[1]]]
                list_inputs = [e for e in list_inputs if e not in elements_to_remove]
                list_inputs.insert(index_minor_value[0], NewElement)
                print("Nuevos strings:", list_inputs)
                f.write("Nuevos strings:\n")
                for line in list_inputs:
                    f.write("%s \t" % line)

                # *New matrix D =
                new_matrix_D = generateNewDistanceMatrix(old_distance_matrix, index_minor_value, n)
                old_distance_matrix = new_matrix_D
                # print("Nueva D =  \n", new_matrix_D)
                n = n - 1
                n_veces_reducidas += 1
                f.write("\n\n")
                f.write("\nMatrix D\n")

            print("Ultima Matrix de Distancia\n", old_distance_matrix)
            for line in old_distance_matrix:
                f.write("%s \n" % line)

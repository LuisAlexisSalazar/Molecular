from itertools import groupby
import numpy as np


# https://bio.libretexts.org/Bookshelves/Computational_Biology/Book%3A_Computational_Biology_-_Genomes_Networks_and_Evolution_(Kellis_et_al.)/02%3A_Sequence_Alignment_and_Dynamic_Programming/2.05%3A_The_Needleman-Wunsch_Algorithm
# Ejemplo del Link: AAGC AGT
# Ejemplo Clase: AAAC AGC
# https://stackoverflow.com/questions/7618858/how-to-to-read-a-matrix-from-a-given-file
def readMatrix(n, debug=False):
    # with open('data/matrix.txt', 'r') as f:
    #     matrix = [[float(num) for num in line.split(',')] for line in f]
    # n = len(matrix[0])
    # matrix = np.array(matrix)
    # matrix = matrix.reshape(n, n)
    # matrix = matrix.astype('float64')
    #
    # if debug:
    #     print(matrix, end="\n\n")
    #
    import pandas as pd

    df = pd.read_csv('data/matrix.csv', sep=";", header=None)
    # print(df)
    matrix = df.to_numpy()
    # print(matrix)
    # print(matrix.shape)

    matrix = np.delete(matrix, range(n, 24), 0)
    matrix = np.delete(matrix, range(n, 24), 1)
    if debug:
        print(matrix, end="\n\n")

    # print(matrix.shape)

    # print(df)

    return matrix


def readInputs():
    with open('data/input.txt') as f:
        lines = f.readlines()
    n_string = len(lines)
    # print(lines)
    for i in range(n_string):
        lines[i] = lines[i].replace(" ", "")
        if i != n_string - 1:
            # eliminar \n en input.txt
            lines[i] = lines[i][:-1]

    print("String ingresados:")
    colums_headers = []
    for i, s in enumerate(lines):
        header = "S" + str(i)
        colums_headers.append(header)
        print(header + " = ", s)
    print("Strings en Lista:", lines)
    print("Strings Reducidas en Lista:", colums_headers)
    return lines, colums_headers


def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


def KeepWay(values_conditions):
    list_values = [values_conditions[0].value, values_conditions[1].value, values_conditions[2].value]
    if all_equal(list_values):
        return values_conditions
    elif values_conditions[0].value == values_conditions[1].value:
        del values_conditions[2]
        return values_conditions
    elif values_conditions[0].value == values_conditions[2].value:
        del values_conditions[1]
        return values_conditions
    return [values_conditions[0]]


# ----------------------Generar Matrix de Distancias a partir de strings--------------------


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

    f.write("\nAlinieaci??n Multiple\n")
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


# https://www.ugr.es/~gallardo/pdf/cluster-3.pdf
# https://slideplayer.com/slide/14121882/
# *Estrategias
def min_distance(distance_1, distance_2):
    if distance_1 < distance_2:
        return distance_1
    return distance_2


def max_distance(distance_1, distance_2):
    if distance_1 > distance_2:
        return distance_1
    return distance_2


def average_distance(distance_1, distance_2):
    return (distance_2 + distance_1) / 2


dict_strategics = {0: "Distancia Minima", 1: "Distancia Maxima", 2: "Distancia Promedio"}


def print_Winner(list_cccs):
    file = "output/result.txt"
    with open(file, "w") as f:
        f.write("Min distance:\n")
        np.savetxt(f, [list_cccs[0]], fmt='%1.6f')
        f.write("Max distance:\n")
        np.savetxt(f, [list_cccs[1]], fmt='%1.6f')
        f.write("Promedio ponderado:\n")
        np.savetxt(f, [list_cccs[2]], fmt='%1.6f')

        max_ccc = max(list_cccs)
        indexes = [i for i, x in enumerate(list_cccs) if x == max_ccc]
        print("\n\n")
        if len(indexes) == 1:
            # print("Estrategia Ganador es ", dict_strategics[indexes[0]], " con ", max_ccc)
            f.write("Estrategia Ganador es " + dict_strategics[indexes[0]] + "\n")
            np.savetxt(f, [max_ccc], fmt='%1.6f')
        else:
            # print("Empate... Las estrategias")
            f.write("\nEmpate... Las estrategias ")
            for index in indexes:
                # print(dict_strategics[index])
                f.write(dict_strategics[index] + " ")
            # print("Con el valor de ", max_ccc)
            f.write("\nCon el valor de:\n")
            np.savetxt(f, [max_ccc], fmt='%1.6f')

from copy import copy
from itertools import groupby
# https://bio.libretexts.org/Bookshelves/Computational_Biology/Book%3A_Computational_Biology_-_Genomes_Networks_and_Evolution_(Kellis_et_al.)/02%3A_Sequence_Alignment_and_Dynamic_Programming/2.05%3A_The_Needleman-Wunsch_Algorithm
# Ejemplo del Link: AAGC AGT
# Ejemplo Clase: AAAC AGC
from multiprocessing import cpu_count


def get_amount_cores():
    n_cores = cpu_count()
    # report the number of logical cpu cores
    print(f'Número de nucleos disponibles en de CPU Logicos: {n_cores}')
    return n_cores


def readInput():
    with open('data/input.txt') as f:
        lines = f.readlines()

    string_to_axis_y = lines[0]
    string_to_axis_x = lines[1]
    string_to_axis_y = string_to_axis_y.replace(" ", "")
    string_to_axis_x = string_to_axis_x.replace(" ", "")
    # string_to_axis_y = "-" + (string_to_axis_y[:-1])
    # string_to_axis_x = "-" + string_to_axis_x

    string_to_axis_y = (string_to_axis_y[:-1])
    string_to_axis_x = string_to_axis_x
    # if len(string_to_axis_x) > len(string_to_axis_y):
    #     string_to_axis_y, string_to_axis_x = string_to_axis_x, string_to_axis_y
    print("Cadena 1:", string_to_axis_y)
    print("Cadena 2:", string_to_axis_x)
    print("Len Cadena 1:", len(string_to_axis_y))
    print("Len Cadena 2:", len(string_to_axis_x))
    return string_to_axis_y, string_to_axis_x


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


# !Solo recorrido solo fase "Former"
def travel_antidiagonal(value_matrix):
    len_s1, len_s2 = value_matrix.shape

    anti_diagonals = []
    anti_diagonal = []

    for i in range(0, len_s1):
        anti_diagonal = []
        j = 0
        i_initial = copy(i)
        if i_initial == 0:
            anti_diagonal.append(value_matrix[i][j])
        else:
            while i + 1 != 0 and j - 1 != i_initial:
                print("i= ", i)
                print("j=", j)
                anti_diagonal.append(value_matrix[i][j])
                i += -1
                j += 1

        anti_diagonals.append(anti_diagonal)

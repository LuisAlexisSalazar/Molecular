import numpy as np
# from treelib import Node, Tree
# import graphviz
import networkx as nx
# import matplotlib.pyplot as plt
from utils import KeepWay

G = nx.DiGraph()
path_simple = []


class ValueCondition:
    def __init__(self, value, index):
        self.value = value
        self.index = index

    def __str__(self):
        return self.value


# *Recursivo para total de alineamientos
def travel_matrix(matrix, x, y):
    if x == 0 and y == 0:
        return 0

    amount_tuples = len(matrix[x][y])
    for i in range(amount_tuples):
        travel_matrix(matrix, matrix[x][y][i][0], matrix[x][y][i][1])
        G.add_edge((x, y), matrix[x][y][i])


# *Recursivo a bucle
def travel_matrix_to_one_simple_path(matrix, x_index, y_index):
    while not (x_index == 0 and y_index == 0):
        path_simple.append(matrix[x_index][y_index][0])
        x_index = path_simple[-1][0]
        y_index = path_simple[-1][1]


def create_graph(matrix):
    # *Iniciamos en las esquina
    x_start = len(matrix) - 1
    y_start = len(matrix[0]) - 1
    travel_matrix(matrix, x_start, y_start)


def get_one_path(matrix):
    x_start = len(matrix) - 1
    y_start = len(matrix[0]) - 1
    path_simple.append((len(matrix) - 1, len(matrix[0]) - 1))
    travel_matrix_to_one_simple_path(matrix, x_start, y_start)


# *way: Lista de tuplas de indices
def bool_list(way):
    index = way[0]
    list_bool = []
    way = way[1::]

    for i, next_node in enumerate(way):
        index_diagonal = (index[0] - 1, index[1] - 1)
        if index_diagonal == next_node:
            list_bool.append(1)
        else:
            list_bool.append(0)
        index = next_node
    return list_bool


def fix_bool_list(way):
    index = way[0]
    list_bool = []
    way = way[1::]

    # list_bool.append(index)
    for i, next_node in enumerate(way):
        # if i == len(way) - 1:
        #     break
        index_diagonal = (index[0] - 1, index[1] - 1)
        index_izquierda = (index[0], index[1] - 1)
        # index_arriba = (index[0], index[1] - 1)
        # if index_diagonal[0] == next_node[0] and index_diagonal[1] == next_node[1]:
        if index_diagonal == next_node:
            list_bool.append(1)
        elif index_izquierda == next_node:
            list_bool.append(2)
        else:  # Index Arriba
            list_bool.append(3)
        index = next_node
    # print(len(list_bool))
    return list_bool


class Matrix:
    value_interval = -2  # gap
    values_matrix = None  # scores
    # matrix_node = None
    matrix_coordinates = None
    debug = False
    ways = []
    string1 = None
    string2 = None
    backtracking = False

    def __init__(self, string1, string2, debug=False, backtracking=False):
        self.string1 = string1
        self.string2 = string2
        self.backtracking = backtracking
        # ? el +1  es por simular el añadido de al inicio "-"
        n, m = len(string1) + 1, len(string2) + 1
        self.debug = debug
        # --------------Matrix de Valores--------------
        self.values_matrix = np.zeros((n, m), int)
        # ? Rellenar las primera fila y columna con la serie
        # * 0 -2 -4 -6 ...
        self.values_matrix[0] = np.arange(m) * self.value_interval
        self.values_matrix[:, 0] = np.arange(n) * self.value_interval

        # --------------Matrix de Coordenadas--------------
        # ? Matrix donde se guarda lista de tuplas (indices)
        self.matrix_coordinates = []
        for i in range(n):
            self.matrix_coordinates.append([])

        # *Rellenar la primera fila
        for i in range(m):
            tuple_index = (0, i - 1)
            self.matrix_coordinates[0].append([tuple_index])
        # *Rellenar la primera columna sin el 0,0
        for i in range(1, n):
            tuple_index = (i - 1, 0)
            self.matrix_coordinates[i].append([tuple_index])
        self.matrix_coordinates[0][0] = [()]
        # -------------------------------------------------

        # -------Debug--------
        if self.debug:
            print("Cadena 1:", string1)
            print("Cadena 2:", string2)
            print("Matrix de Valores Inicial:", self.values_matrix, end="\n\n")
            print("Matrix de Coordenadas Inicial:", self.matrix_coordinates, end="\n\n")
            print("N = ", n, "M = ", m)

    # ?Link del simuladore del algoritmo de Meddleman
    # https://bioboot.github.io/bimm143_W20/class-material/nw/
    def fun(self, string1, string2):
        # +1  es por simular el añadido de al inicio "-"
        n, m = len(string1) + 1, len(string2) + 1

        for i in range(1, n):
            for j in range(1, m):
                # ---------Obtener valores de condiciones---------
                value_first_condition = 1
                if string1[i - 1] != string2[j - 1]:
                    value_first_condition = -1

                # *Valores en orden de Esquina izquierda, solo Derecha y solo izquierda
                index_1, index_2, index_3 = (i - 1, j - 1), (i - 1, j), (i, j - 1)
                value_1 = self.values_matrix[i - 1][j - 1]
                value_2 = self.values_matrix[i - 1][j]
                value_3 = self.values_matrix[i][j - 1]

                # *Guardar el valor junto al indice de donde proviene
                values_matrix = [ValueCondition(value_1 + value_first_condition, index_1),
                                 ValueCondition(value_2 - 2, index_2),
                                 ValueCondition(value_3 - 2, index_3)]
                # ------Mantener solo el mayor valor-----
                # *Ordenar
                sorted_values_conditions = sorted(values_matrix, key=lambda x: x.value)
                # ?Reverse no retorna nada solo actualiza los indices
                sorted_values_conditions.reverse()
                # *Filtrar
                sorted_values_conditions = KeepWay(sorted_values_conditions)
                list_value_indexs = [classValue.index for classValue in sorted_values_conditions]

                # ------Agregar a la matrix de valores y coordenadas-----
                self.matrix_coordinates[i].append(list_value_indexs)
                self.values_matrix[i][j] = sorted_values_conditions[0].value

                # -------Debug--------
                if self.debug:
                    print("-" * 7, i, "-", j, "-" * 7)
                    print("Valores ordenados de mayor a menor:")
                    [print(classValue.__str__(), end=" ") for classValue in sorted_values_conditions]
                    print("Mayores valor con sus indices:", list_value_indexs)
        # *Generar grafo a travez de la matrix de coordenadas
        if self.backtracking:
            create_graph(self.matrix_coordinates)

        if self.debug:
            print("Matrix de Valores:", self.values_matrix)
            print("Matrix de Coordenadas:", self.matrix_coordinates)

    def alignments(self, string1, string2):
        if self.backtracking:
            n, m = len(string1) + 1, len(string2) + 1

            for path in nx.all_simple_paths(G, source=(n - 1, m - 1), target=(0, 0)):
                self.ways.append(path)

            if self.debug:
                print("Caminos para las alineaciones:", self.ways)
        else:
            get_one_path(matrix=self.matrix_coordinates)
            if self.debug:
                print("Camino simple:", path_simple)

    def getOneAligment(self):

        list_bool_to_alignment = fix_bool_list(path_simple)
        alignment = self.getAlignmentFix(list_bool_to_alignment)
        return alignment

    def getAlignmentFix(self, list_bool):
        stringAlignment1 = ""
        stringAlignment2 = ""

        string1_inverse = self.string1[::-1]
        string2_inverse = self.string2[::-1]
        # if len(min_string) > len(max_string):
        #     max_string, min_string = min_string, max_string
        j = 0
        k = 0
        for i, bool_way in enumerate(list_bool):
            if bool_way == 1:
                stringAlignment1 += string1_inverse[j]
                stringAlignment2 += string2_inverse[k]
                j += 1
                k += 1

            elif bool_way == 2:
                stringAlignment1 += "-"
                stringAlignment2 += string2_inverse[k]
                k += 1

            elif bool_way == 3:
                stringAlignment1 += string1_inverse[j]
                stringAlignment2 += "-"
                j += 1

        stringAlignment1, stringAlignment2 = stringAlignment1[::-1], stringAlignment2[::-1]
        return [stringAlignment1, stringAlignment2]

    def get_aligments(self):
        alignments = []
        if self.backtracking:
            for i in range(len(self.ways)):
                list_bool_to_alignment = fix_bool_list(self.ways[i])
                alignment = self.getAlignmentFix(list_bool_to_alignment)
                alignments.append(alignment)
        else:
            alignment = self.getOneAligment()
            alignments.append(alignment)
        return alignments

    def saveTXT(self):
        # np.savetxt('Arreglo de valores.txt', self.values_matrix, fmt='%.0f')
        np.savetxt('output.txt', self.values_matrix, fmt='%.0f', header="Matrix de Valores:")
        f = open("output.txt", "a")
        n, m = len(self.string1), len(self.string2)
        f.write("Score: " + str(self.values_matrix[n, m]) + '\n')
        f.write("Cantidad de alineamientos: " + str(len(self.ways)) + "\n")
        f.write("Alineamientos: " + "\n")

        alignments = self.get_aligments()
        for par_alig in alignments:
            [f.write(alig + "\n") for alig in par_alig]
            f.write("-" * 10)
            f.write("\n")

        f.close()

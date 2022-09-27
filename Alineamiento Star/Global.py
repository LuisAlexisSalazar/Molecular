import numpy as np
from treelib import Node, Tree
import graphviz
import networkx as nx
import matplotlib.pyplot as plt
from utils import KeepWay


# h = graphviz.Digraph('H', filename='hello.gv')


class ValueCondition:
    def __init__(self, value, index):
        self.value = value
        self.index = index

    def __str__(self):
        # return f'{self.value}'
        return self.value


# *way: Lista de tuplas de indices
def bool_list(way):
    index = way[0]
    list_bool = []
    way = way[1::]

    # list_bool.append(index)
    for i, next_node in enumerate(way):
        # if i == len(way) - 1:
        #     break
        index_diagonal = (index[0] - 1, index[1] - 1)
        # if index_diagonal[0] == next_node[0] and index_diagonal[1] == next_node[1]:
        if index_diagonal == next_node:
            list_bool.append(1)
        else:
            list_bool.append(0)
        index = next_node
    # print(len(list_bool))
    return list_bool


class Matrix:
    value_interval = -2  # gap
    values_matrix = None  # scores
    matrix_coordinates = None
    debug = False
    ways = None
    string1 = None
    string2 = None
    score = None

    def __init__(self, string1, string2, debug=False):
        self.G = nx.DiGraph()
        self.ways = []
        # print(self.matrix_coordinates)
        # print(self.ways)

        self.string1 = string1
        self.string2 = string2
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

    def travel_matrix(self, matrix, x, y):
        if x == 0 and y == 0:
            return 0

        amount_tuples = len(matrix[x][y])
        for i in range(amount_tuples):
            self.travel_matrix(matrix, matrix[x][y][i][0], matrix[x][y][i][1])
            self.G.add_edge((x, y), matrix[x][y][i])

    def create_graph(self, matrix):
        # *Iniciamos en las esquina
        x_start = len(matrix) - 1
        y_start = len(matrix[0]) - 1
        # print("Score = ", self.values_matrix[x_start][y_start])
        self.score = self.values_matrix[x_start][y_start]
        self.travel_matrix(matrix, x_start, y_start)
        # nx.draw(G, with_labels=True)
        # plt.savefig("generador.png")
        # plt.show()

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
        self.create_graph(self.matrix_coordinates)
        if self.debug:
            print("Matrix de Valores:", self.values_matrix)
            print("Matrix de Coordenadas:", self.matrix_coordinates)

    def alignments(self, string1, string2):
        n, m = len(string1) + 1, len(string2) + 1

        for path in nx.all_simple_paths(self.G, source=(n - 1, m - 1), target=(0, 0)):
            self.ways.append(path)

        if self.debug:
            print("Caminos para las alineaciones:", self.ways)

    def getAlignment(self, list_bool):
        list_bool.reverse()
        stringAlignment = ""
        i = 0
        i_s2 = 0
        # print("Lista boleana:", list_bool)
        # print("Lista len:", len(list_bool))
        # print("S1:", self.string1)
        # print("S2:", self.string2)
        max_string = self.string1
        min_string = self.string2
        if len(self.string2) > len(self.string1):
            max_string, min_string = self.string2, self.string1

        # while i < len(max_string):
        while i < len(max_string):
            # if list_bool[i] == 1 or i >= len(min_string):
            # if i_s2 == len(self.string2):
            if i_s2 == len(self.string1) - 1 and i_s2 != len(self.string2) - 1:
                # print("Entro condicional nuevo")
                for j in range(i_s2, len(self.string2)):
                    stringAlignment += self.string2[i_s2]

            elif i_s2 == len(self.string2) - 1:
                for j in range(i_s2, len(self.string2)):
                    stringAlignment += self.string2[i_s2]
                # *Completación de alienación
                len_stringAlig = len(stringAlignment)
                len_string1 = len(self.string1)
                # print("Entro a la nueva alineación")
                # print(len_stringAlig)
                # print(len_string1)
                if len_stringAlig != len_string1 and len_string1 > len_stringAlig:
                    fill_gap = len_string1 - len_stringAlig
                    for j in range(0, fill_gap):
                        stringAlignment += "-"
                break
            #
            # elif list_bool[i] == 1 or i >= len(min_string):
            elif list_bool[i] == 1:
                stringAlignment += self.string2[i_s2]
                i_s2 += 1
            else:
                stringAlignment += "-"
            i += 1
        # print("SA:", stringAlignment)

        return stringAlignment

    def saveTXT(self):
        # np.savetxt('Arreglo de valores.txt', self.values_matrix, fmt='%.0f')
        np.savetxt('output.txt', self.values_matrix, fmt='%.0f', header="Matrix de Valores:")
        f = open("output.txt", "a")
        n, m = len(self.string1), len(self.string2)
        f.write("Score: " + str(self.values_matrix[n, m]) + '\n')
        f.write("Cantidad de alineamientos: " + str(len(self.ways)) + "\n")
        f.write("Alineamientos: " + "\n")
        for i in range(len(self.ways)):
            list_bool_to_alignment = bool_list(self.ways[i])
            # print("Tamaño de la Lista booleana:", len(list_bool_to_alignment))
            # print(self.string1)
            # print(self.string2)
            alignment = self.getAlignment(list_bool_to_alignment)
            f.write(self.string1)
            f.write("\n")
            f.write(alignment)
            f.write("\n")
            f.write("-" * 10)
            f.write("\n")
        f.close()

    def getFistAlignment(self):

        self.fun(self.string1, self.string2)
        self.alignments(self.string1, self.string2)
        list_bool_to_alignment = bool_list(self.ways[0])
        print("Cantidad de Caminos:", len(self.ways))
        print("list_bool_to_alignment antes del get:", list_bool_to_alignment)
        alignment = self.getAlignment(list_bool_to_alignment)
        print(self.string1)
        print(alignment)
        print("*" * 10)
        # print("Alineamientos:")
        # for i in range(len(self.ways)):
        #     list_bool_to_alignment = bool_list(self.ways[i])
        #     alignment = self.getAlignment(list_bool_to_alignment)
        #     print(self.string1)
        #     print(alignment)
        #     print("*" * 10)

        return alignment

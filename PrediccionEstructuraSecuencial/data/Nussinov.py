import numpy as np
# from treelib import Node, Tree
# import graphviz
import networkx as nx
# import matplotlib.pyplot as plt
from PrediccionEstructuraSecuencial.utils import KeepWay, isMinValueUniqueInLastIndex
import matplotlib.pyplot as plt

PLOT = False


# PLOT = True


# import sys
# sys.setrecursionlimit(1500)
# from utils import KeepWay


# G = nx.DiGraph()
# path_simple = []


# def clear_Global():
#     G.clear()
#     path_simple.clear()


class ValueCondition:
    def __init__(self, value, index):
        self.value = value
        self.index = index

    def __str__(self):
        return str(self.value)


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
    string_result = []

    for i, next_node in enumerate(way):
        # if i == len(way) - 1:
        #     break
        index_diagonal = (index[0] + 1, index[1] - 1)
        index_izquierda = (index[0], index[1] - 1)
        # index_arriba = (index[0], index[1] - 1)
        # if index_diagonal[0] == next_node[0] and index_diagonal[1] == next_node[1]:
        if index_diagonal == next_node:  # <- /
            list_bool.append(1)
        elif index_izquierda == next_node:  # <-
            # list_bool.append(2)
            list_bool.append(2)
        else:  # ↓
            # list_bool.append(3)
            list_bool.append(3)
        index = next_node
    # print(len(list_bool))
    return list_bool


def index_long_diagonal(n):
    diagonal = []
    row = 0
    col = 0

    while col < n and row < n:
        diagonal.append((row, col))
        row += 1
        col += 1
    return diagonal


def get_all_index_diagonals(n):
    all_indexs = []
    for i in range(1, n):
        row = 0
        col = i
        diag = []
        while col < n and row < n:
            diag.append((row, col))
            row += 1
            col += 1
        all_indexs.append(diag)
    # for index in all_indexs:
    #     for i, j in index:
    #         print(self.values_matrix[i, j], end=" ")
    #     print("")
    return all_indexs


class Matrix:
    dict_complement = {"G": 'C', 'C': 'G', 'A': 'U', 'U': 'A'}
    values_matrix = None  # scores
    # matrix_node = None
    matrix_coordinates = None
    debug = False
    ways = []
    string1 = None
    string2 = None
    backtracking = False
    G = nx.DiGraph()
    path_simple = []
    indexes_end = set()

    def __init__(self, string1, string2, debug=False, backtracking=False):
        self.debug = debug
        self.string1 = string1
        self.string2 = string2
        self.backtracking = backtracking
        # n, m = len(string1) + 1, len(string2) + 1
        n, m = len(string1), len(string2)
        self.n = n
        # --------------Matrix de Valores--------------
        self.values_matrix = np.zeros((n, m), int)
        # --------------Matrix de Coordenadas--------------
        # ? Matrix donde se guarda lista de tuplas (indices)
        self.matrix_coordinates = []
        for i in range(n):
            self.matrix_coordinates.append([])
        for i in range(n):
            for j in range(m):
                self.matrix_coordinates[i].append([])

        self.G.clear()
        self.path_simple.clear()
        self.ways.clear()
        # -------Debug--------
        if self.debug:
            print("Cadena 1:", string1)
            print("Cadena 2:", string2)
            print("Matrix de Valores Inicial:\n", self.values_matrix, end="\n\n")
            print("Matrix de Coordenadas Inicial:", self.matrix_coordinates, end="\n\n")
            print("N = ", n, "M = ", m)

    # *Recursivo para total de alineamientos
    def travel_matrix(self, matrix, x, y, indexes_diagonal):
        # if x == 0 and y == 0:
        if (x, y) in indexes_diagonal:
            self.indexes_end.add((x, y))
            # print(x, y)
            return 0

        amount_tuples = len(matrix[x][y])
        for i in range(amount_tuples):
            next_index = matrix[x][y][i]
            if self.is_over_diag2(next_index):
                # Ingresa elementos que no estan pasando la diagonal pero tiene un camino a la diagonal
                # print("Continue: ", (x, y))
                self.indexes_end.add((x, y))
                continue
            else:
                self.travel_matrix(matrix, next_index[0], next_index[1], indexes_diagonal)
                self.G.add_edge((x, y), matrix[x][y][i])

    def is_over_diag2(self, index):
        indexes_triangle_lower = np.tril_indices(self.n)
        indexes_triangle_lower = list(zip(indexes_triangle_lower[0], indexes_triangle_lower[1]))
        indexes_diagonal = index_long_diagonal(self.n)
        indexes_triangle_lower_withouth_diagn = [x for x in indexes_triangle_lower if x not in indexes_diagonal]

        if index in indexes_triangle_lower_withouth_diagn:
            return True
        return False

    # !Eror porque esta tomando en cuenta la digonal misma cuando deberia ser permitido un indice en la misma diagonal
    def is_over_diag(self, index):
        indexes_triangle_lower = np.tril_indices(self.n)
        indexes_triangle_lower = list(zip(indexes_triangle_lower[0], indexes_triangle_lower[1]))
        # print("Diagonal:", indexes_triangle_lower)
        if index in indexes_triangle_lower:
            return True
        return False

    # *Recursivo a bucle
    def travel_matrix_to_one_simple_path(self, matrix, x_index, y_index, indexes_diagonal):
        print("Triangle inferior:\n")
        # for index in indexes_triangle_lower_withouth_diagn:
        #     print(indexes_triangle_lower_withouth_diagn)
        #
        while not self.is_over_diag2((x_index, y_index)):
            print("Index Actual: (", x_index, " , ", y_index, ")")
            next_directions = matrix[x_index][y_index]
            if (5, 6) == (x_index, y_index):
                print("No funca")
            if len(next_directions) == 0:
                break
            else:
                next_direction = next_directions[0]
                if self.is_over_diag2(next_direction):
                    break
                # if next_direction in indexes_diagonal or next_direction

                self.path_simple.append(next_direction)
                x_index = next_direction[0]
                y_index = next_direction[1]
            # print("Paths a escoger:", matrix[x_index][y_index][0])

    def create_graph(self, matrix, indexes_start):
        # *Iniciamos en las esquina
        x_start = indexes_start[0].item()
        y_start = indexes_start[1].item()
        # print(x_start, y_start)
        indexes_diagonal = index_long_diagonal(self.n)
        # print("Indices Diagonal:", indexes_diagonal)
        # self.is_over_diag2((x_start, y_start))
        self.travel_matrix(matrix, x_start, y_start, indexes_diagonal)
        # print("Nodes:", self.G.nodes)
        # print("Finales:", self.indexes_end)

        # for n in self.G.nodes:
        #     if self.is_over_diag(n):
        #         print("Index:", n)

        if PLOT:
            nx.draw(self.G, with_labels=True)
            plt.show()

    def get_one_path(self, matrix, indexes_star):
        x_start = indexes_star[0]
        y_start = indexes_star[1]
        self.path_simple.append((x_start, y_start))
        indexes_diagonal = index_long_diagonal(self.n)
        print("Diagonal Grande:", indexes_diagonal)
        self.travel_matrix_to_one_simple_path(matrix, x_start, y_start, indexes_diagonal)
        print("Path Simlpe:", self.path_simple)
        # return self.path_simple

    def getValueComplement(self, letter1, letter2):
        complement_to_letter1 = self.dict_complement.get(letter1)
        if complement_to_letter1 == letter2:
            return -1
        return 0

    def getValues_to_4_condition(self, i, j):
        values = []
        # print("Indices:", i, j)
        for k in range(i + 1, j):
            if (i, j) == (0, 19):
                # print("k=",k)
                print(self.values_matrix[i, k], " - ", self.values_matrix[k + 1, j])
                # print(i, k)
                # print(self.values_matrix[k + 1, j])
            values.append(self.values_matrix[i, k] + self.values_matrix[k + 1, j])
        # print("Values:", values)
        minValue = (min(values))
        # print("Min Values:", minValue)
        return minValue

    def fun(self, string1, string2):
        n, m = len(string1), len(string2)

        all_indexs = get_all_index_diagonals(n)
        for diag_index in all_indexs:
            for i, j in diag_index:
                # ---------Obtener valores de condiciones---------
                # *Valores en orden de Esquina izquierda, solo Derecha y solo izquierda
                # if (i,j) == (0,19):
                #     print("No funca ;V")
                index_1, index_2, index_3, index_4 = (i + 1, j), (i, j - 1), (i + 1, j - 1), (i, j)

                value_1 = self.values_matrix[i + 1, j]
                value_2 = self.values_matrix[i, j - 1]
                value_3 = self.values_matrix[i + 1, j - 1] + self.getValueComplement(string1[i], string2[j])

                # *Guardar el valor junto al indice de donde proviene
                values_matrix = [ValueCondition(value_1, index_1),
                                 ValueCondition(value_2, index_2),
                                 ValueCondition(value_3, index_3)]
                # print("Len Values:", len(values_matrix))
                if i + 1 != j:
                    value_4 = self.getValues_to_4_condition(i, j)
                    values_matrix.append(ValueCondition(value_4, index_4))
                # if len(values_matrix) == 4:
                #     print(i, j)
                # for s in values_matrix:
                #     print(s.value, "\t", s.index)
                # print("Len Values Final:", len(values_matrix))
                # ------Mantener solo el minimo valor-----
                # *Ordenar
                sorted_values_conditions = sorted(values_matrix, key=lambda x: x.value)

                if self.debug:
                    print("-------------------")
                    print("Valores Iniciales:", (i, j))
                    for s in sorted_values_conditions:
                        print(s.value, "\t", s.index)

                sorted_values_conditions, isLastIndex = isMinValueUniqueInLastIndex(sorted_values_conditions, (i, j))
                list_value_indexes = []
                if not isLastIndex:
                    sorted_values_conditions_tmp = []
                    for v in sorted_values_conditions:
                        if v.index != (i, j):
                            sorted_values_conditions_tmp.append(v)
                    list_value_indexes = [classValue.index for classValue in sorted_values_conditions_tmp]
                if self.debug:
                    print("Valores Finales:", (i, j))
                    for s in sorted_values_conditions:
                        print(s.value, "\t", s.index)

                # ------Agregar a la matrix de valores y coordenadas-----
                self.matrix_coordinates[i][j] = list_value_indexes
                self.values_matrix[i][j] = sorted_values_conditions[0].value

                # -------Debug--------
                # if self.debug:
                #     print("Valores ordenados de mayor a menor:")
                #     [print(classValue.__str__(), end=" ") for classValue in sorted_values_conditions]
                #     print("\nMayores valor con sus indices:", list_value_indexes)
        # *Generar grafo a travez de la matrix de coordenadas
        if self.backtracking:  # Backtracking desde 1 elemento menor
            minValue = np.amin(self.values_matrix)
            result = np.where(self.values_matrix == minValue)
            # ?Puede ver varios menores valores, solo usamos 1
            listOfCordinates = list(zip(result[0], result[1]))
            self.create_graph(self.matrix_coordinates, listOfCordinates[0])

        if self.debug:
            print("Matrix de Valores:", self.values_matrix)
            print("Matrix de Coordenadas:", self.matrix_coordinates)

    def alignments(self, string1, string2):
        minValue = np.amin(self.values_matrix)
        result = np.where(self.values_matrix == minValue)
        listOfCordinates = list(zip(result[0], result[1]))
        indexes_star = (listOfCordinates[0][0].item(), listOfCordinates[0][1].item())

        # *No funciona el backtracking supera la recrusión
        if self.backtracking:
            targets = index_long_diagonal(self.n)

            for path in nx.all_simple_paths(self.G, source=indexes_star, target=targets):
                self.ways.append(path)
            # for p in self.ways:
            #     print("Caminos para las alineaciones:", p)
            if self.debug:
                print("Caminos para las alineaciones:", self.ways)
        else:
            self.get_one_path(self.matrix_coordinates, indexes_star)
            if self.debug:
                print("Camino simple:", self.path_simple)

    def get_score(self, string1, string2):
        n = len(string1)
        m = len(string2)
        return self.values_matrix[n, m]

    def getOneAligment(self):
        list_bool_to_alignment = fix_bool_list(self.path_simple)
        # print("Codificación Path Simple:", list_bool_to_alignment)
        alignment = self.getAlignmentFix(list_bool_to_alignment)
        return alignment

    def getSecuencePatron(self):
        list_bool_to_alignment = fix_bool_list(self.path_simple)
        secuence, code = self.getAlignmentFix(list_bool_to_alignment)
        return secuence, code

    def getOneCode(self):
        code = fix_bool_list(self.path_simple)
        return code

    def getAlignmentFix(self, list_bool):
        stringAlignment1 = ""
        stringAlignment2 = ""
        code_visual_1 = ""
        code_visual_2 = ""
        # string1_inverse = self.string1[::-1]
        string1_inverse = self.string1
        string2_inverse = self.string2[::-1]
        j = 0
        k = 0
        for i, bool_way in enumerate(list_bool):
            if bool_way == 1:
                stringAlignment1 += string1_inverse[j]
                stringAlignment2 += string2_inverse[k]
                code_visual_2 += "("
                code_visual_1 += ")"
                j += 1
                k += 1

            elif bool_way == 2:
                # stringAlignment1 += "-"
                stringAlignment2 += string2_inverse[k]
                code_visual_1 += "."
                k += 1

            elif bool_way == 3:
                stringAlignment1 += string1_inverse[j]
                code_visual_2 += "."
                # stringAlignment2 += "-"
                j += 1
        code = code_visual_2 + code_visual_1
        secuence = stringAlignment1 + stringAlignment2
        # stringAlignment1, stringAlignment2 = stringAlignment1[::-1], stringAlignment2[::-1]
        return secuence, code

    def getCodeficate(self):
        codes = []
        if self.backtracking:
            self.ways = self.ways[:5]
            for i in range(len(self.ways)):
                list_bool_to_alignment = fix_bool_list(self.ways[i])
                codes.append(list_bool_to_alignment)
        else:
            code = self.getOneCode()
            codes.append(code)
        return codes

    def get_aligments(self):
        alignments = []
        if self.backtracking:
            # *La cantidad de alineaciones son 5
            self.ways = self.ways[:5]
            for i in range(len(self.ways)):
                list_bool_to_alignment = fix_bool_list(self.ways[i])
                alignment = self.getAlignmentFix(list_bool_to_alignment)
                alignments.append(alignment)
        else:
            alignment = self.getOneAligment()
            # alignments.append(alignment)
        return alignments

    def saveTXT(self):
        # np.savetxt('Arreglo de valores.txt', self.values_matrix, fmt='%.0f')
        # print(type(self.values_matrix))
        np.savetxt('output.txt', self.values_matrix, fmt='%d', header="Matrix de Valores:", delimiter='\t')
        f = open("output.txt", "a")
        print("Escribiendo")
        n, m = len(self.string1), len(self.string2)
        f.write("Len del Strin1: " + str(n))
        minValue = np.amin(self.values_matrix)
        result = np.where(self.values_matrix == minValue)
        listOfCordinates = list(zip(result[0], result[1]))
        listOfCordinates = listOfCordinates[0]
        f.write("\nMinimo Valor: " + str(minValue) + '\n')
        f.close()
        return 0
        f.write(
            "Una unica alineación desde indice (" + str(listOfCordinates[0]) + "," + str(listOfCordinates[1]) + ')\n')
        f.write("Significado de la codificacion: " + "\n")
        f.write("1:Diagonal \n")
        f.write("2:Izquierda \n")
        f.write("3:Abajo \n\n")
        f.write("Codificacion del camino: " + "\n")
        codes = self.getCodeficate()
        for code in codes:
            [f.write(str(c) + ",") for c in code]
            f.write("\n")

        f.close()

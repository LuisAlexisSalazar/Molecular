import numpy as np
import graphviz
import networkx as nx
import matplotlib.pyplot as plt
from utils import KeepWay

h = graphviz.Digraph('H', filename='hello.gv')
G = nx.DiGraph()
list_G = []

saveAligments = False


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

    for next_node in way:
        index_diagonal = (index[0] - 1, index[1] - 1)
        if index_diagonal[0] == next_node[0] and index_diagonal[1] == next_node[1]:
            list_bool.append(1)
        else:
            list_bool.append(0)
        index = next_node
    return list_bool


class Matrix:
    value_interval = -2  # gap
    values_matrix = None  # scores
    matrix_coordinates = None
    debug = False
    ways = []
    string1 = None
    string2 = None
    # Local
    max_value = None
    x_start = None
    y_start = None
    end_x = None
    end_y = None
    list_index_start = []
    list_index_end = []

    def __init__(self, string1, string2, debug=False, plot=False):
        self.string1 = string1
        self.string2 = string2
        # ? el +1  es por simular el añadido de al inicio "-"
        n, m = len(string1) + 1, len(string2) + 1
        self.debug = debug
        self.plot = plot
        # --------------Matrix de Valores--------------
        self.values_matrix = np.zeros((n, m), int)
        # ? Rellenar las primera fila y columna con la serie
        # * 0 -2 -4 -6 ...
        # self.values_matrix[0] = np.arange(m) * self.value_interval
        # self.values_matrix[:, 0] = np.arange(n) * self.value_interval

        # --------------Matrix de Coordenadas--------------
        # ? Matrix donde se guarda lista de tuplas (indices)
        self.matrix_coordinates = []
        for i in range(n):
            self.matrix_coordinates.append([])

        # *Rellenar la primera fila
        for i in range(m):
            # tuple_index = (0, i - 1)
            tuple_index = {}
            self.matrix_coordinates[0].append([tuple_index])
        # *Rellenar la primera columna sin el 0,0
        for i in range(1, n):
            # tuple_index = (i - 1, 0)
            tuple_index = {}
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

    def travel_matrix2(self, matrix, Graph, x, y, matrix_to_mark=None):
        # if self.values_matrix[x, y] == 0:
        if self.values_matrix[x, y] == 0:
            self.list_index_end.append((x, y))
            return 0

        amount_tuples = len(matrix[x][y])
        for i in range(amount_tuples):

            if matrix_to_mark[x][y] == 1:
                continue
            matrix_to_mark[x][y] = 1

            self.travel_matrix2(matrix, Graph, matrix[x][y][i][0], matrix[x][y][i][1], matrix_to_mark)
            # except:
            #     print("Error:")
            #     print("x =", x)
            #     print("y =", y)
            #     print(self.matrix_coordinates[x][y])
            #     print(self.matrix_coordinates[50][49])
            #     print(self.matrix_coordinates[50][48])
            #     self.travel_matrix2(matrix, Graph, matrix[x][y][i][0], matrix[x][y][i][1])
            Graph.add_edge((x, y), matrix[x][y][i])

    def findMaxValue(self):
        self.max_value = np.amax(a=self.values_matrix)

    def create_graph(self, matrix):
        # indices = np.where(self.values_matrix == self.max_value)
        # indices = np.argmax(self.values_matrix)
        # indices = np.unravel_index(np.argmax(self.values_matrix, axis=None), self.values_matrix.shape)
        indices = np.argwhere(self.values_matrix == self.max_value)
        # print(self.values_matrix)
        # print(indices)

        # for i in indices:
        #     print(i)
        # print(self.values_matrix.shape)
        # print(indices[0])
        # print(indices[1])
        for i in range(len(indices)):
            list_G.append(nx.DiGraph())

        index_graph = 1
        for G_item, index in zip(list_G, indices):
            x_start = index[0]
            y_start = index[1]
            self.list_index_start.append((x_start, y_start))
            matrix_to_mark = np.zeros(shape=self.values_matrix.shape)
            # self.travel_matrix2(matrix, G_item, x_start, y_start)
            self.travel_matrix2(matrix, G_item, x_start, y_start, matrix_to_mark)
            if self.plot:
                nx.draw(G_item, with_labels=True)
                plt.savefig("generador" + str(index_graph) + ".png")
                plt.show()
            index_graph += 1

        global saveAligments
        saveAligments = True

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
                index_1, index_2, index_3, index_4 = (i - 1, j - 1), (i - 1, j), (i, j - 1), ()
                value_1 = self.values_matrix[i - 1][j - 1]
                value_2 = self.values_matrix[i - 1][j]
                value_3 = self.values_matrix[i][j - 1]
                value_4 = 0

                # *Guardar el valor junto al indice de donde proviene
                values_matrix = [ValueCondition(value_1 + value_first_condition, index_1),
                                 ValueCondition(value_2 - 2, index_2),
                                 ValueCondition(value_3 - 2, index_3),
                                 ValueCondition(value_4, index_4),
                                 ]
                # ------Mantener solo el mayor valor-----
                # *Ordenar
                sorted_values_conditions = sorted(values_matrix, key=lambda x: x.value)
                # ?Reverse no retorna nada solo actualiza los indices
                sorted_values_conditions.reverse()
                list_value_indexes = None
                if sorted_values_conditions[0].value == 0:
                    list_value_indexes = [()]
                else:  # Diferente de 0
                    sorted_values_conditions = KeepWay(sorted_values_conditions)
                    list_value_indexes = [classValue.index for classValue in sorted_values_conditions]

                # ------Agregar a la matrix de valores y coordenadas-----
                self.matrix_coordinates[i].append(list_value_indexes)
                self.values_matrix[i][j] = sorted_values_conditions[0].value

                # -------Debug--------
                if self.debug:
                    print("-" * 7, i, "-", j, "-" * 7)
                    print("Valores ordenados de mayor a menor:")
                    [print(classValue.__str__(), end=" ") for classValue in sorted_values_conditions]
                    print("Mayores valor con sus indices:", list_value_indexes)
        self.findMaxValue()
        # *Generar grafo a travez de la matrix de coordenadas
        # print(self.matrix_coordinates)
        self.create_graph(self.matrix_coordinates)
        if self.debug:
            print("Matrix de Valores:", self.values_matrix)
            print("Matrix de Coordenadas:", self.matrix_coordinates)

    def alignments(self, string1, string2):
        n, m = len(string1) + 1, len(string2) + 1
        # print("Indices:")
        # print(self.list_index_start)
        # print(self.list_index_end)

        for index, G_item in enumerate(list_G):
            for path in nx.all_simple_paths(G_item,
                                            source=(self.list_index_start[index][0], self.list_index_start[index][1]),
                                            target=(self.list_index_end[index][0], self.list_index_end[index][1])):
                self.ways.append(path)
        if self.debug:
            print("Caminos para las alineaciones:", self.ways)

    def getAlignment(self, list_bool):
        print("List Booleanos: ", list_bool)
        list_bool.reverse()
        stringAlignment = ""
        i = 0
        n = 0
        while i < len(self.string1):
            if list_bool[i] == 1:
                stringAlignment += self.string2[n]
                n += 1
            else:
                stringAlignment += "-"
            i += 1
        return stringAlignment

    def getAlignmentLocal(self, way):
        alignment = ""
        # ?Eliminar la columna donde el valor es 0 significa el fin del subcadena
        del way[-1]

        # ? -1 is simular el "-" añadido
        for index_i, index_j in way:
            if self.string1[index_i - 1] != self.string2[index_j - 1]:
                alignment = "-" + alignment
            else:
                alignment = self.string1[index_i - 1] + alignment

        return alignment

    def get_bool_list(self, way):
        index = way[0]
        list_bool = []

        for next_node in way:
            index_diagonal = (index[0] - 1, index[1] - 1)
            if index_diagonal[0] == next_node[0] and index_diagonal[1] == next_node[1]:
                list_bool.append(1)
            else:
                list_bool.append(0)
            index = next_node
        return list_bool

    def saveTXT(self, substring_same=False):
        # np.savetxt('Arreglo de valores.txt', self.values_matrix, fmt='%.0f')
        np.savetxt('output.txt', self.values_matrix, fmt='%.0f', header="Matrix de Valores:")
        f = open("output.txt", "a")

        f.write("Maximo Valor (Local) =" + str(self.max_value) + "\n")
        f.write("Cadena 1:" + self.string1 + "\n")
        f.write("Cadena 2:" + self.string2 + "\n")

        if saveAligments:
            if substring_same:
                # f.write("Cantidad de alineamientos: " + str(len(self.ways)) + "\n")
                # f.write("Alineamientos Local: " + "\n")
                list_substring_same = []
                print("Ways:", self.ways)
                for way in self.ways:

                    alignment = ""
                    nex_path = False
                    len_way = len(way)
                    for i in range(len_way):
                        if i == len_way - 1:
                            break
                        if self.values_matrix[way[i][0], way[i][1]] < self.values_matrix[way[i + 1][0], way[i + 1][1]]:
                            nex_path = True
                            break

                    if nex_path:
                        continue

                    del way[-1]

                    for x, y in way:
                        if self.string1[x - 1] == self.string2[y - 1]:
                            alignment = self.string1[x - 1] + alignment

                    list_substring_same.append(alignment)

                f.write("Cantidad de alineamientos: " + str(len(list_substring_same)) + "\n")
                f.write("Alineamientos Local: " + "\n")
                for alig in list_substring_same:
                    f.write(alig + "\n")
                    f.write("-" * 10 + "\n")
                f.close()
            else:
                f.write("Cantidad de alineamientos: " + str(len(self.ways)) + "\n")
                f.write("Alineamientos Local: " + "\n")

                for way in self.ways:
                    print(way)
                    # list_bool = self.get_bool_list(way)
                    # print("Lista diagonal:", list_bool)
                    alignment = self.getAlignmentLocal(way)
                    f.write(alignment + "\n")
                    f.write("-" * 10 + "\n")
                f.close()

from copy import copy
import time
from multiprocessing import Pool, cpu_count, Manager
from Meddleman import ValueCondition
from utils import KeepWay
import multiprocessing as mp
import numpy as np
from threading import Thread, active_count
import threading

# !Hacer la fase de inicialización aqui sin usar otro archivo de python
# https://stackoverflow.com/questions/64791736/how-to-modify-global-numpy-array-safely-with-multithreading-in-python
values_matrix = np.zeros((1, 1))
matrix_coordinates = []
s1 = ""
s2 = ""
n_cores_run = 1
len_s1 = 0
len_s2 = 0
n_cores_available = 1
flag_to_sleep = False
chunk_lists = lambda it, n: (l for l in ([],) for i, g in enumerate((it, ((),))) for e in g for l in
                             (l[:len(l) % n] + [e][:1 - i],) if (len(l) % n == 0) != i)

main_thread = threading.current_thread()


# Si el flag esta en Falso obtiene los valores
def get_anti_diagonal(index_i, flag_indexs=True):
    # global values_matrix
    j = 0
    i_initial = copy(index_i)
    anti_diagonal_index = []
    anti_diagonal_values = []
    while index_i + 1 != 0 and j - 1 != i_initial:
        if flag_indexs == True:
            anti_diagonal_index.append((index_i, j))
        else:
            anti_diagonal_values.append(values_matrix[index_i][j])
        index_i += -1
        j += 1

    return anti_diagonal_index if flag_indexs else anti_diagonal_values


# 10 12, len = 12 -> 10,3 <-> 0, 12
# def new_get_anti_diagonal(index_i, index_j, lens_s2):
def new_get_anti_diagonal(index_i, index_j, step):
    anti_diagonal_index = []
    if step == -1:
        anti_diagonal_index.append((index_i, index_j))
        return anti_diagonal_index
    else:
        # global values_matrix
        # j = 0
        i_initial = copy(index_i)
        j_initial = copy(index_j)

        # print("I y J inicial:", i_initial, j_initial)
        # print("Len S2: ", lens_s2)
        # 5 y 1   s1 = 7 y s2 = 7
        # while index_i + 1 != 0 and index_j - 1 != i_initial:
        # while index_i + 1 != j_initial - 1 and index_j - 1 != lens_s2:

        while index_i + 1 != i_initial - (step + 1) and index_j - 1 != index_j + (step):
            anti_diagonal_index.append((index_i, index_j))
            index_i -= 1
            index_j += 1
        # print("\n\n")
        return anti_diagonal_index


def calculate_score(list_indexs):
    for i, j in list_indexs:
        value_first_condition = 1
        if s1[i - 1] != s2[j - 1]:
            value_first_condition = -1

        # *Valores en orden de Esquina izquierda, solo Derecha y solo izquierda
        index_1, index_2, index_3 = (i - 1, j - 1), (i - 1, j), (i, j - 1)
        value_1 = values_matrix[i - 1][j - 1]
        value_2 = values_matrix[i - 1][j]
        value_3 = values_matrix[i][j - 1]

        # *Guardar el valor junto al indice de donde proviene
        values_local_matrix = [ValueCondition(value_1 + value_first_condition, index_1),
                               ValueCondition(value_2 - 2, index_2),
                               ValueCondition(value_3 - 2, index_3)]
        # ------Mantener solo el mayor valor-----
        # *Ordenar
        sorted_values_conditions = sorted(values_local_matrix, key=lambda x: x.value)
        # ?Reverse no retorna nada solo actualiza los indices
        sorted_values_conditions.reverse()
        # *Filtrar
        sorted_values_conditions = KeepWay(sorted_values_conditions)
        list_value_indexs = [classValue.index for classValue in sorted_values_conditions]

        # ------Agregar a la matrix de valores y coordenadas-----
        # matrix_coordinates[i].append(list_value_indexs)
        values_matrix[i, j] = sorted_values_conditions[0].value
        # print("Fin del calculo de score local")


def launch_threads(CD):
    len_CD = len(CD)
    q = len_CD // n_cores_run
    r = len_CD % n_cores_run
    if r == 0:
        chunk = [e for e in chunk_lists(CD, q)]
    else:
        chunk = [e for e in chunk_lists(CD, q + r)]

    len_chunk = len(chunk)
    threads = []
    for i in range(len_chunk):
        t = Thread(target=calculate_score, args=(chunk[i],))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # amount_threads_run = active_count()
    # print("Iterable: ", threading.enumerate())
    # if amount_threads_run != 1:
    #     print("Cantidad de Threads Corriendo:", active_count())
    #     print("\n\n")


def phase_former():
    # set_len_chunks = set()
    global flag_to_sleep
    for i_CD in range(2, len_s1 + 1):
        CD = get_anti_diagonal(i_CD, flag_indexs=True)
        del CD[0]
        del CD[-1]
        launch_threads(CD)


# Bien 10 11 -> index final 10 11
# Mla 10 12 ->  index final 10 12
def phase_mid_latter():
    index_i = len_s1
    copy_len_s1 = copy(len_s1)

    # for f in range(1, len(matrix[0])):

    for f in range(1, len(values_matrix[0])):
        CD = []
        # for k in range(matrix.shape[1] - f):
        # for k in range(values_matrix.shape[1] - f):
        for k in range(len_s2 + 1 - f):
            # s.append(matrix[len(matrix) - k - 1, f + k])
            # s.append(matrix[len(matrix) - k - 1, f + k])
            # CD.append((len(values_matrix) - k - 1, f + k))
            CD.append((len_s1 + 1 - k - 1, f + k))
            if len(CD) == len_s2 + 1 - ((len_s2 + 1) - (len_s1 + 1)):
                del CD[-1]
                break
        # print(CD)
        # print(values_matrix)
        launch_threads(CD)
        # for i, j in CD:
        #     print(values_matrix[i, j], end=" ")
        # print("\n")
        # my_list.append(s)
    # print("Debajo de la diagonal")

    # print("Debajo de la diagonal")
    # for i in my_list:
    #     print(i)

    # !Error con la antidiagonal
    # for j in range(1, len_s2 + 1):
    #     # CD = new_get_anti_diagonal(index_i, j, len_s2)
    #     print("step:", copy_len_s1)
    #     CD = new_get_anti_diagonal(index_i, j, copy_len_s1)
    #     copy_len_s1 -= 1
    #     for v, u in CD:
    #         print(values_matrix[v, u], end=" ")
    #     print("\n")
    #     try:
    #         launch_threads(CD)
    #     except:
    #         print("Excepción encontrada", CD)
    #         print(index_i, j)
    #         print("CD de Exp:", CD)
    #         print("Fin de la excepción")


def call_Cores_calculate_score(n_cores):
    # print("Valor Inicial de Matrx:\n", values_matrix)
    # print("Shape:", values_matrix.shape)
    global n_cores_run, len_s1, len_s2
    if n_cores <= n_cores_available:
        # n_cores_run = n_cores - 1
        n_cores_run = n_cores
    else:
        print("Exedio el número de nucleos porfavor reduzca")
        return
    len_s1 = len(s1)
    len_s2 = len(s2)

    marker = time.time()
    phase_former()
    phase_mid_latter()
    print("Multithreading", n_cores, "spent", time.time() - marker)

    print("\n\n")
    print("Matrix de Valores Resultante con Threading:")
    print(values_matrix)

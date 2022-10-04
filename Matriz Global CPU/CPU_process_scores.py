from copy import copy
from multiprocessing import Pool, cpu_count, Manager
from Meddleman import ValueCondition
from utils import KeepWay
import multiprocessing as mp
import numpy as np

# !Hacer la fase de inicialización aqui sin usar otro archivo de python
# https://stackoverflow.com/questions/64791736/how-to-modify-global-numpy-array-safely-with-multithreading-in-python
values_matrix = np.zeros((2, 2))
matrix_coordinates = []
s1 = ""
s2 = ""


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


def new_get_anti_diagonal(index_i, index_j, lens_s2):
    # global values_matrix
    # j = 0
    i_initial = copy(index_i)
    j_initial = copy(index_j)
    anti_diagonal_index = []
    # print("I y J inicial:", i_initial, j_initial)
    # print("Len S2: ", lens_s2)
    # 5 y 1   s1 = 7 y s2 = 7
    # while index_i + 1 != 0 and index_j - 1 != i_initial:
    while index_i + 1 != j_initial - 1 and index_j - 1 != lens_s2:
        # print("Index :", index_i, index_j)
        anti_diagonal_index.append((index_i, index_j))
        index_i -= 1
        index_j += 1
    # print("\n\n")
    return anti_diagonal_index


def calculate_single_score(list_indexs):
    # global values_matrix, matrix_coordinates, s1, s2
    # print("Core T M:", type(values_matrix))
    # print("Core Lista Indices:", list_indexs)
    # print("Core Shape:", values_matrix.shape)
    # print("Core S1:", s1)
    # print("Core S2:", s2)

    if type(list_indexs) == list:
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
            matrix_coordinates[i].append(list_value_indexs)
            values_matrix[i, j] = sorted_values_conditions[0].value

    else:  # Tuple
        i, j = list_indexs[0], list_indexs[1]
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
        matrix_coordinates[i].append(list_value_indexs)
        values_matrix[i, j] = sorted_values_conditions[0].value
        # print("Fin del calculo de score local")


# !Globals and Arrays with numpy
# https://stackoverflow.com/questions/64222805/how-to-pass-2d-array-as-multiprocessing-array-to-multiprocessing-pool
# !Simple Array and Value
# https://stackoverflow.com/questions/39322677/python-how-to-use-value-and-array-in-multiprocessing-pool
# def init_worker(share_values_matrix, share_matrix_coordinates, share_s1, share_s2):
def init_worker(share_values_matrix, share_s1, share_s2):
    # def init_worker(arr):
    # global values_matrix, matrix_coordinates, s1, s2
    # values_matrix = share_values_matrix
    # matrix_coordinates = share_matrix_coordinates
    # s1 = share_s1
    # s2 = share_s2
    # print("Core Valor de la matrix:", share_values_matrix)
    globals()['values_matrix'] = np.frombuffer(share_values_matrix, dtype='int32').reshape(len(share_s1) + 1,
                                                                                           len(share_s2) + 1)
    global values_matrix
    # 110 por yipo? int32
    # globals()['values_matrix'] = np.frombuffer(share_values_matrix, dtype='int32').reshape(5, 5)
    # globals()['values_matrix'] = np.frombuffer(share_values_matrix, dtype='float32').reshape(5, 5)
    # globals()['matrix_coordinates'] = np.frombuffer(share_values_matrix, dtype='float64').reshape(len(share_s1),len(share_s2))
    globals()['s1'] = share_s1
    globals()['s2'] = share_s2


def calculate_score(list_indexs):
    # global values_matrix, matrix_coordinates, s1, s2
    global values_matrix, s1, s2

    # print("Core T M:", type(values_matrix))
    # print("Core Lista Indices:", list_indexs)
    # print("Core Shape:", values_matrix.shape)

    # print("Core S1:", s1)
    # print("Core S2:", s2)

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


# else:  # Tuple
#     i, j = list_indexs[0], list_indexs[1]
#     value_first_condition = 1
#     if s1[i - 1] != s2[j - 1]:
#         value_first_condition = -1
#
#     # *Valores en orden de Esquina izquierda, solo Derecha y solo izquierda
#     index_1, index_2, index_3 = (i - 1, j - 1), (i - 1, j), (i, j - 1)
#     value_1 = values_matrix[i - 1][j - 1]
#     value_2 = values_matrix[i - 1][j]
#     value_3 = values_matrix[i][j - 1]
#
#     # *Guardar el valor junto al indice de donde proviene
#     values_local_matrix = [ValueCondition(value_1 + value_first_condition, index_1),
#                            ValueCondition(value_2 - 2, index_2),
#                            ValueCondition(value_3 - 2, index_3)]
#     # ------Mantener solo el mayor valor-----
#     # *Ordenar
#     sorted_values_conditions = sorted(values_local_matrix, key=lambda x: x.value)
#     # ?Reverse no retorna nada solo actualiza los indices
#     sorted_values_conditions.reverse()
#     # *Filtrar
#     sorted_values_conditions = KeepWay(sorted_values_conditions)
#     list_value_indexs = [classValue.index for classValue in sorted_values_conditions]
#
#     # ------Agregar a la matrix de valores y coordenadas-----
#     # matrix_coordinates[i].append(list_value_indexs)
#     values_matrix[i, j] = sorted_values_conditions[0].value
#     # print("Fin del calculo de score local")


def phase_former(arr, n_cores, s1, s2):
    len_s1 = len(s1)
    # Llegue el indice al mismo tamaño del Strinng 1
    for i_CD in range(2, len_s1 + 1):
        # print("Index para antidiagonal:", i_CD)
        j = 0
        CD = get_anti_diagonal(i_CD, flag_indexs=True)
        del CD[0]
        del CD[-1]
        # print("List Indices:", CD)
        len_CD = len(CD)
        q = len_CD // n_cores
        step_one = 1
        chunks = [CD[x:x + q] for x in range(0, len_CD, q)] if q != 0 else [CD[x:x + step_one] for x in
                                                                            range(0, len_CD, step_one)]
        # print("Chunks:", chunks)
        # *Se puede dividr entre los nucleos
        if q != 0:
            # print("IF")
            pool = Pool(processes=n_cores, initializer=init_worker,
                        initargs=(arr, s1, s2))
            pool.map(calculate_score, chunks, chunksize=q)
        else:
            # print("ELSE")
            pool = Pool(processes=n_cores, initializer=init_worker,
                        initargs=(arr, s1, s2))
            # pool.map(calculate_score, CD, chunksize=q)
            pool.map(calculate_score, chunks, chunksize=1)
        pool.close()
        pool.join()


def phase_mid_latter(arr, n_cores, s1, s2):
    len_s2 = len(s2)
    len_s1 = len(s1)
    # Llegue el indice al mismo tamaño del Strinng 1

    # index_i = len_s1 - 1
    index_i = len_s1
    for j in range(1, len_s2 + 1):
        CD = new_get_anti_diagonal(index_i, j, len_s2)
        len_CD = len(CD)
        # print("Lista Indices mid latter:", CD)
        q = len_CD // n_cores
        step_one = 1
        chunks = [CD[x:x + q] for x in range(0, len_CD, q)] if q != 0 else [CD[x:x + step_one] for x in
                                                                            range(0, len_CD, step_one)]
        # print("Chunks:", chunks)
        # *Se puede dividr entre los nucleos
        if q != 0:
            # print("IF")
            pool = Pool(processes=n_cores, initializer=init_worker,
                        initargs=(arr, s1, s2))
            pool.map(calculate_score, chunks, chunksize=q)
        else:
            # print("ELSE")
            pool = Pool(processes=n_cores, initializer=init_worker,
                        initargs=(arr, s1, s2))
            pool.map(calculate_score, chunks, chunksize=1)
        pool.close()
        pool.join()


def call_Cores_calculate_score():
    # https://stackoverflow.com/questions/68373535/global-variable-access-during-python-multiprocessing
    print("S1:", s1)
    print("S2:", s2)
    print("Valor de Matrx:\n", values_matrix)
    print("Shape:", values_matrix.shape)
    # return
    # d = mp.managers.dict()
    # d["s1"] = s1
    # d["s2"] = s2
    n_cores = cpu_count()
    # n_cores_run = 2
    len_s1, len_s2 = values_matrix.shape

    # CD = get_anti_diagonal(2, flag_indexs=True)
    # del CD[0]
    # del CD[-1]
    # CD = CD
    # calculate_single_score(CD)
    temp_np_array = values_matrix.flatten()
    arr = mp.Array('i', temp_np_array, lock=False)

    phase_former(arr, n_cores, s1, s2)
    phase_mid_latter(arr, n_cores, s1, s2)
    # print("Antidionales:")
    # index_i = len_s1 - 1
    # for j in range(1, len_s2):
    #     print(new_get_anti_diagonal(index_i, j, len_s2))

    REF_D = []
    REF_HV = []
    CD = []

    # print("\n\n")

    arr = np.frombuffer(arr, dtype='int32').reshape(values_matrix.shape)
    print(f"{mp.current_process().name}\n{arr}")


# *Copia de Seguridad con error en los valores pero si hace simulateneo
def call_Cores_calculate_score2():
    # https://stackoverflow.com/questions/68373535/global-variable-access-during-python-multiprocessing
    #
    # print("Type Value Matrix:", type(values_matrix))
    #
    print("S1:", s1)
    print("S2:", s2)
    print("Valor de Matrx:\n", values_matrix)
    print("Shape:", values_matrix.shape)
    # return
    # d = mp.managers.dict()
    # d["s1"] = s1
    # d["s2"] = s2
    # n_cores_run = cpu_count()
    n_cores = 2
    len_s1, len_s2 = values_matrix.shape

    temp_np_array = values_matrix.flatten()
    arr = mp.Array('i', temp_np_array, lock=False)
    # v1 = mp.Value('i', 3)
    # v2 = mp.Value('i', 3)
    # arr2 = mp.Array('i', np.zeros(len_s1 * len_s2, dtype='float64'), lock=False)
    # return
    REF_D = []
    REF_HV = []
    CD = []
    # old_CD = []

    # anti_diagonals = []
    # anti_diagonal = []

    for i_CD in range(2, len_s1):
        REF_D = REF_HV
        REF_HV = copy(CD)
        # CD = []
        j = 0
        # i_initial = copy(i)
        # *Primera vez
        if i_CD == 2:
            REF_D = values_matrix[0][0]
            # REF_HV = get_anti_diagonal(i_CD - 1, values_matrix, flag_indexs=True)
            REF_HV = get_anti_diagonal(i_CD - 1, flag_indexs=True)
            # CD = get_anti_diagonal(i_CD, values_matrix, flag_indexs=True)
            CD = get_anti_diagonal(i_CD, flag_indexs=True)
            del CD[0]
            del CD[-1]
            # print("CD:", CD)
            calculate_single_score(CD)
            # print("REF_D:", REF_D)
            # print("REF_HV", REF_HV)
            # print("CD", CD)
        else:
            # print("Conjunto de Pool")
            # CD = get_anti_diagonal(i_CD, values_matrix, flag_indexs=True)
            CD = get_anti_diagonal(i_CD, flag_indexs=True)
            del CD[0]
            del CD[-1]
            # print("CD:", CD)
            len_CD = len(CD)
            if len_CD < n_cores:
                print("IF")
                pool = Pool(processes=len_CD, initializer=init_worker,
                            initargs=(arr, s1, s2))
                pool = Pool(processes=len_CD)
                pool.map(calculate_score, CD, chunksize=1)
                pool.close()
                pool.join()
            else:

                print("ELSE")
                q = len_CD // n_cores
                chunks = [CD[x:x + q] for x in range(0, len_CD, q)]
                # print("Chunks:", chunks)
                # print("Chunksize:", q)
                # print("Residup:", len_CD % n_cores_run)
                pool = Pool(processes=n_cores, initializer=init_worker,
                            initargs=(arr, s1, s2))
                # pool.map(calculate_score, CD, chunksize=q)
                pool.map(calculate_score, chunks, chunksize=q)
                pool.close()
                pool.join()
            # i_begin_slice += q
            # i_start_slice += q

            # calculate_score(CD)
            # print("REF_D:", REF_D)
            # print("REF_HV", REF_HV)
            # print("CD", CD)
        print("\n\n")

    arr = np.frombuffer(arr, dtype='int32').reshape(values_matrix.shape)
    print(f"{mp.current_process().name}\n{arr}")

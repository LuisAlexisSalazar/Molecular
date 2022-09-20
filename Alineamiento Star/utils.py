from itertools import groupby


# https://bio.libretexts.org/Bookshelves/Computational_Biology/Book%3A_Computational_Biology_-_Genomes_Networks_and_Evolution_(Kellis_et_al.)/02%3A_Sequence_Alignment_and_Dynamic_Programming/2.05%3A_The_Needleman-Wunsch_Algorithm
# Ejemplo del Link: AAGC AGT
# Ejemplo Clase: AAAC AGC

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
    print(lines)
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

from itertools import groupby


# https://bio.libretexts.org/Bookshelves/Computational_Biology/Book%3A_Computational_Biology_-_Genomes_Networks_and_Evolution_(Kellis_et_al.)/02%3A_Sequence_Alignment_and_Dynamic_Programming/2.05%3A_The_Needleman-Wunsch_Algorithm
# Ejemplo del Link: AAGC AGT
# Ejemplo Clase: AAAC AGC

def readInput():
    with open('data/input.txt') as f:
        lines = f.readlines()

    string_to_axis_x = lines[0]
    string_to_axis_x = string_to_axis_x.strip()
    # string_to_axis_x = lines[1]
    # string_to_axis_y = string_to_axis_y.replace(" ", "")
    string_to_axis_x = string_to_axis_x.replace(" ", "")
    # string_to_axis_y = string_to_axis_y.upper()
    string_to_axis_x = string_to_axis_x.upper()
    # string_to_axis_y = "-" + (string_to_axis_y[:-1])
    # string_to_axis_x = "-" + string_to_axis_x
    # print("S P1:",len(string_to_axis_x))
    # for i,s in enumerate(string_to_axis_x):
    #     print(i, "c:", s)
    # string_to_axis_y = (string_to_axis_y[:-1])
    # string_to_axis_x = string_to_axis_x
    # if len(string_to_axis_x) > len(string_to_axis_y):
    #     string_to_axis_y, string_to_axis_x = string_to_axis_x, string_to_axis_y
    # print("Cadena 1:", string_to_axis_y)
    # print("Cadena 2:", string_to_axis_x)
    return string_to_axis_x, string_to_axis_x


def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


def isMinValueUniqueInLastIndex(values_conditions, index):
    minValue = values_conditions[0].value
    values_conditions_filter = []

    for v in values_conditions:
        if v.value == minValue:
            values_conditions_filter.append(v)

    if len(values_conditions_filter) == 1 and values_conditions_filter[0].index == index:
        return values_conditions_filter, True
    else:
        return values_conditions_filter, False


def KeepWay(values_conditions):
    list_values = [values_conditions[0].value, values_conditions[1].value, values_conditions[2].value]
    if all_equal(list_values):
        return [values_conditions[3]]
    elif values_conditions[0].value == values_conditions[1].value:
        del values_conditions[2]
        return values_conditions
    elif values_conditions[0].value == values_conditions[2].value:
        del values_conditions[1]
        return values_conditions
    return [values_conditions[0]]

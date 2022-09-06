from itertools import groupby


def readInput():
    with open('data/input.txt') as f:
        lines = f.readlines()

    string_to_axis_y = lines[0]
    string_to_axis_x = lines[1]

    # string_to_axis_y = "-" + (string_to_axis_y[:-1])
    # string_to_axis_x = "-" + string_to_axis_x

    string_to_axis_y = (string_to_axis_y[:-1])
    string_to_axis_x = string_to_axis_x
    print("Cadena 1:", string_to_axis_y)
    print("Cadena 2:", string_to_axis_x)
    if len(string_to_axis_x) > len(string_to_axis_y):
        string_to_axis_y, string_to_axis_x = string_to_axis_x, string_to_axis_y
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
"""Basic sort algorithms"""

def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in  range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]

    return arr


def selection_sort(arr):
    for  i in range(len(arr) -1, 0, -1):
        max_value_index = 0
        for k in range(1, i + 1):
            if arr[k] > arr[max_value_index]:
                 max_value_index = k

        temp = arr[i]
        arr[i] = arr[max_value_index]
        arr[max_value_index] = temp
    return arr


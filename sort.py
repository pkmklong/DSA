"""Basic sort algorithms"""

def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in  range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]

    return arr


def selection_sort(arr):
    
    for start_index in range(len(arr) -1, 0, -1):
        
        max_index = 0
        
        for scan_index in range(1, start_index + 1):
            if arr[scan_index] > arr[max_index]:
                 max_index = scan_index
                    
        arr[start_index], arr[max_index] = \
        arr[max_index], arr[start_index]
        
    return arr


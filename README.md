## About 
This package impliments classic data structures and algorithms for review and exploration.

## Table of Contents
- [DSA from Scratch](#dsa-from-scratch)
  * [About](#About)
  * [Data Structures & Abstract Data Types](#Data-Structures-&-Abstract-Data-Types)
    + [Stack](#stack)
    + [Queue](#queue)
    + [Deque](#deque)
    + [Linked Lists](#linked-lists)
    + [Doubly Linked Lists](#doubly-linked-lists)
    + [Heap](#Heap)
  * [Sort Algorithms](#Sort-Algorithms)
    + [Bubble Sort](#Bubble-Sort)
    + [Selection Sort](#Selection-Sort)
    + [Insertion Sort](#Insertion-Sort)
    + [Quick Sort](#Quick-Sort)
    + [Merge Sort](#Merge-Sort)
  * [Search Algorithms](#Search-Algorithms)
    + [Linear Search](#Linear-Search)
    + [Binary Search](#Binary-Search)
  * [Recursion](#Recursion)
    + [Fibonacci](#Fibonacci)

## Data Structures & Abstract Data Types
### Stack
<details>
 <summary>Code</summary>
 
    class Stack:
 
       def __init__(self):
            self.items = []

       def checkEmpty(self):
           return self.items == []

       def push(self, item):
           self.items.append(item)

       def pop(self):
           return self.items.pop()

       def peek(self):
           return self.items[self.size -1]

       def size(self):
           return  len(self.items)
 
</details>

### Queue
<details>
 <summary>Code</summary>
  
    class Queue:

       def __init__(self):
           self.items = []

       def checkEmpty(self):
           return self.items == []

       def front(self):
           return self.items[-1]

       def back(self):
           return self.items[0]

       def enqueue(self, x: int):
           self.x = x
           self.items.insert(0, x)       

       def dequeue(self):
           self.items.pop()
</details> 

### Deque
<details>
 <summary>Code</summary>
 
    class Deque:
 
       def __init__(self):
           self.items = []

       def checkEmpty(self):
           return self.items == []

       def addFront(self, item):
           self.items.append(item)

       def addRear(self, item):
           self.items.insert(0,item)

       def popFront(self):
           return self.items.pop()

       def popRear(self):
           return self.items.pop(0)

       def size(self):
           return len(self.items)

</details>

### Linked Lists
<details>
 <summary>Code</summary>
 
     class Node:
        def __init__(self, val=None):
            self.val = val
            self.next = None

    class LinkedList:
        def __init__(self, val):
            nn = Node(val)
            self.head = nn
            self.tail = nn
            self.length = 1

        def print_list(self):
            temp = self.head
            while temp:
                print(temp.val)
                temp = temp.next

        def append(self, val):
            nn = Node(val)
            if self.length == 0:
                self.head, self.tail = nn, nn
            else:
                self.tail.next = nn
                self.tail = nn
            self.length += 1

        def prepend(self, val):
            nn = Node(val)
            if self.length == 0:
                self.head, self.tail = nn, nn
            else:
                nn.next = self.head
                self.head = nn
            self.length += 1

        def pop_first(self):
            if self.length == 0:
                return None
            temp = self.head
            self.head = temp.next
            temp.next = None
            self.length -= 1

        def pop(self):
            temp, prev = self.head, self.head
            while temp.next:
                prev = temp
                temp = temp.next
            self.tail = prev
            prev.next = None
            self.length -= 1
            if self.length == 0:
                self.head, self.tail = None, None
            return temp

        def get(self, ind):
            if ind < 0 or ind >= self.length:
                return None
            else:
                temp = self.head
                for _ in range(ind):
                    temp = temp.next
                return temp

        def set_val(self, ind, val):
            temp = self.get(ind)
            if temp:
                temp.val = val
                return True
            return False

        def insert(self, ind, val):
            if ind < 0 or ind > self.length:
                return None
            elif ind == 0:
                return self.prepend(val)
            elif ind == self.length:
                return self.append(val)
            else:
                nn = Node(val)
                temp = self.get(ind - 1)
                nn.next = temp.next
                temp.next = nn
            self.length += 1
            return True

        def remove(self, ind):
            if ind < 0 or ind >= self.length:
                return None
            elif ind == 0:
                return self.pop_first()
            elif ind == self.length - 1:
                return self.pop()
            else:
                prev = self.get(ind - 1)
                temp = prev.next
                prev.next = prev.next.next
                temp.next = None
            self.length -= 1
            return temp

        def reverse(self):
            prev = None
            current = self.head
            self.tail = current
            while current:
                temp = current.next
                current.next = prev
                prev = current
                current = temp
            self.head = prev

</details>

### Doubly Linked Lists
<details>
 <summary>Code</summary>
</details>

### Heap
<details>
 <summary>Code</summary>

    class Heap:
        """Heap data structure with list implementation"""

        def __init__(self, data: List):
            self.data = data

        def first_node(self):
            return self.data[0]

        def last_node(self):
            return self.data[-1]

        def left_child_index(self, index):
            return (index * 2) + 1

        def right_child_index(self, index):
            return (index * 2) + 2

        def parent_index(self, index):
            return (index - 1) / 2
            
            
            
    def heapify(arr:list, n:int, i:int) -> int:
        # Find largest among root and children
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2
        if left < n and arr[i] < arr[left]:
            largest = left
        if right < n and arr[largest] < arr[right]:
            largest = right
        # If root is not largest, swap with largest and continue heapifying
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            heapify(arr, n, largest)

    def heap_sort(arr:list) -> list:
        n = len(arr)
        # Build max heap
        for i in range(n // 2, -1, -1):
            heapify(arr, n, i)
        for i in range(n - 1, 0, -1):
            # Swap
            arr[i], arr[0] = arr[0], arr[i]
            # Heapify root element
            heapify(arr, i, 0)
        return arr
 </details>

## Sort Algorithms
### Bubble Sort
<details>
 <summary>Code</summary>
 
    from typing import List
 
    def bubble_sort(arr:List):
        n = len(arr)
        swapped = True
        for i in range(0, n-1):
            for j in range(0, n-1-i):
                if arr[j] > arr[j+1]:
                    arr[j], arr[j+1], = arr[j+1], arr[j]
                    swapped = False
                if swapped == False:
                    break
        return arr
</details>

### Selection Sort
<details>
 <summary>Code</summary>


    def selection_sort(l):

        length = len(l)

        for i in range(length):
            min_ind = i

            for j in range(i+1, length):
                if l[min_ind] > l[j]:
                    min_ind = j
            if min_ind > i:
                l[min_ind], l[i] = l[i], l[min_ind]
                
        return l

        
    def selection_sort(arr: List):

       for start_index in range(len(arr) -1, 0, -1):

           max_index = 0

           for scan_index in range(1, start_index + 1):
               if arr[scan_index] > arr[max_index]:
                    max_index = scan_index

           arr[start_index], arr[max_index] = \
           arr[max_index], arr[start_index]

       return arr
</details>     
       
### Insertion Sort
<details>
 <summary>Code</summary>
 
    from typing import List

    def insertion_sort(arr: List[int]):

        for i in range(0,len(arr)):
            ref = arr[i]
            j = i - 1
            while j >= 0 and arr[j] > ref:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j+1] = ref
        return arr
</details>   

### Quick Sort

### Merge Sort
<details>
<summary>Code</summary>

    def merge(l1, l2):
    
        # vars
        combined = [] 
        i, j = 0, 0 

        # merge lists
        while i < len(l1) and j < len(l2):
            if l1[i] <= l2[j]:
                combined.append(l1[i])
                i+=1
            else:
                combined.append(l2[j])
                j+=1

        # append remainer to sorted list
        while i < len(l1):
            combined.append(l1[i])
            i+=1
        while j < len(l2):
            combined.append(l2[j])
            j+=1

        return combined

    def merge_sort(l):
        
        # base case
        if len(l) <= 1:
            return l

        # recursion
        mid_ind = len(l) // 2
        left = merge_sort(l[:mid_ind])
        right = merge_sort(l[mid_ind:])

        return merge(left, right)

</details> 

## Search Algorithms
### Linear Search
<details>
 <summary>Code</summary>
 
    from typing import List

    def linear_search(nums: List[int], target = int)  -> bool:

        for i in range(len(nums)):

            if nums[i] == target:
                return i

        return False
</details>     

### Binary Search
<details>
 <summary>Code</summary>
 
    from typing import List

    def binary_search_recursive(nums: List[int], target = int) -> bool:
        if len(nums) == 0:
            return False
        mid  = len(nums) // 2
        if target == nums[mid]:
            return True
        elif target < nums[mid]:
            return binary_rsearch(nums[:mid], target)
        elif target > nums[mid]:
            return binary_rsearch(nums[mid+1:], target)
 
 
    def binary_search(nums: List[int], target = int) -> int:
       first = 0
       last = len(nums) - 1

       while first <= last:
           mid = (first + last) // 2
           if target == nums[mid]:
               return mid
           elif target < nums[mid]:
               last = mid -1
           elif target > nums[mid]:
               first = mid + 1
       return -1
</details>
 
## Recursion
### Fibonacci
<details>
 <summary>Code</summary>
 
    def fib(n, memo = {}):
     # Returns nth Fibonacci value using recursion and memoization
       if n == 0: 
           return 0
       if n == 1: 
           return 1
       if not memo.get(n):
                memo[n] = fib(n-1, memo) + fib(n-2, memo) 
       return memo[n]
</details>

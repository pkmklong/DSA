## About 
This package impliments classic data structures and algorithms for review and exploration.

## Table of Contents
- [DSA from Scratch](#dsa-from-scratch)
  * [About](#About)
  * [Data Structures & Abstract Data Types](#Data-Structures-&-Abstract-Data-Types)
    + [Stack](#stack)
    + [Queue](#Queue)
    + [Deque](#Deque)
    + [Linked Lists](#Linked-Lists)
    + [Heap](#Heap)
  * [Sort Algorithms](#Sorting-Algorithms)
    + [Bubble Sort](#Bubble-Sort)
    + [Selection Sort](#Selection-Sort)
    + [Insertion Sort](#Insertion-Sort)
    + [Quick Sort](#Quick-Sort)
    + [Merge Sort](#Merge-Sort)
  * [Search Algorithms](#Searching-Algorithms)
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
 
        def __init__(self, value = None):
            self.next_node = None
            self.value = value


    class LinkedList:
 
        def __init__(self):
            self.head = None

        def print_list(self):
            print_value = self.head
            while print_value != None:
                print(print_value.value)
                print_value = print_value.next_node

        def insert_end(self, value):
            new_node = Node(value)
            if not self.head:
                self.head = new_node
                return
            current_node = self.head
            while current_node.next_node:
                current_node = current_node.next_node
            current_node.next_node = new_node

        def insert_start(self, value):
            new_node = Node(value)
            new_node.next_node = self.head
            self.head = new_node

        def delete_start(self):
            if self.head.next_node:
                self.head = self.head.next_node
            else:
                self.head = None

        def delete_end(self):
            current_node = self.head
            while current_node.next_node.next_node:
                current_node = current_node.next_node
            current_node.next_node = None

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
 </details>

## Sort Algorithms
### Bubble Sort
<details>
 <summary>Code</summary>
 
    from typing import List

    def bubble_sort(arr: List):
        n = len(arr)
        for i in range(n):
            for j in  range(0, n-i-1):
                if arr[j] > arr[j+1]:
                    arr[j], arr[j+1] = arr[j+1], arr[j]
        return arr 
</details>

### Selection Sort
<details>
 <summary>Code</summary>

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
### Quick Sort
### Merge Sort

## Search Algorithms
### Linear Search

### Binary Search
<details>
 <summary>Code</summary>
 
    from typing import List

    def binary_rsearch(nums: List[int], target = int) -> bool:
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

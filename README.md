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

## Sort Algorithms
### Bubble Sort
### Selection Sort
### Insertion Sort
### Quick Sort
### Merge Sort

## Search Algorithms
### Linear Search
### Binary Search

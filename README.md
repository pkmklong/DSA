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
### Deque
### Linked Lists
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

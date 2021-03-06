"""Basic data structures"""

from typing import List

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

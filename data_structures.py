"""Basic data structures"""

class Stack:
    def __init__(self):
         self.items = []

    def  checkEmpty(self):
        return self.items == []

    def  push(self, item):
        self.items.append(item)

    def  pop (self):
        return self.items.pop()

    def peek(self):
        return self.items[self.size -1]

    def size(self):
        return  len(self.items)
       


class Node:
    def  __init__(self, value = None): 
        self.next_node = None
        self.value = value

class LinkedList():
     def __init__(self):
         self.head = None

     def print_list(self):
         print_value = self.head
         while print_value != None:
             print(print_value.value)
             print_value = print_value.next_node

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
        return self.items(self.size -1]

    def size(self):
        return  len(self.items)

## About 
This package impliments classic data structures and algorithms for review and exploration.

## Table of Contents
- [DSA from Scratch](#dsa-from-scratch)
  * [About](#about)
  * [Data Structures and Abstract Data Types](#data-structures-and-abstract-data-types)
    + [Stack](#stack)
    + [Queue](#queue)
    + [Deque](#deque)
    + [Linked Lists](#linked-lists)
    + [Doubly Linked Lists](#doubly-linked-lists)
    + [Graph](#graph)
    + [Heap](#heap)
    + [BST](#bst)
  * [Sort Algorithms](#sort-algorithms)
    + [Bubble Sort](#bubble-sort)
    + [Selection Sort](#selection-sort)
    + [Insertion Sort](#insertion-sort)
    + [Quick Sort](#quick-sort)
    + [Merge Sort](#merge-sort)
  * [Search Algorithms](#search-algorithms)
    + [Linear Search](#linear-search)
    + [Binary Search](#binary-search)
  * [Recursion](#recursion)
    + [Fibonacci](#fibonacci)
  * [Hash Tables](#hash-tables)
  * [Patterns for Linear Data Structures](#patterns-for-linear-data-structures)
    + [Two Pointers](#two-pointers)
      + [Palidrome](#palidrome)
      + [Sum Three](#sum-three)
      + [Max Area](#max-area)
      + [Product Except Self](#product-except-self)
      + [Remove nth Node from Tail](#remove-nth-node-from-tail)
    + [Two Pointers Fast Slow](#two-pointers-fast-slow)
      + [Detect Cycle](#detect-cycle)
    + [Modified Binary Search](#modified-binary-search)
      + [Binary Search Rotated](#binary-search-rotated)
      + [Recursive Binary Search Rotated](#recursive-binary-search-rotated)
      + [Find Min Val Rotated](#find-min-val-rotated)
    + [In Place Reversals of Linked Lists](#in-place-reversals-of-linked-lists)
      + [Fold Linked List](#fold-linked-list)
      + [Reverse  Linked List](#reverse-linked-list)
    + [Stacks Valid Parentheses](#stacks-valid-parentheses)
    + [Matrices](#matrices)
      + [Set to Zero](#set-to-zero)
      + [Rotate 90 Degrees](#rotate-90-degrees)
      + [Spiral Matrix to Array](#spiral-matrix-to-array)
    + [BFS](#bfs)
    + [DFS](#dfs)
      + [Pre Order](#pre-order)
      + [In Order](#in-order)
      + [Post Order](#post-order)
      + [Serialize BST](#serialize-bst)
      + [Deserialize BST](#deserialize-bst)
      + [Serialize Deserialize BST Exact](#serialize-deserialize-bst-exact)
      + [Max Sum Path](#max-sum-path)
      + [Build BST Pre-Order and In-Order Lists](#build-bst-pre-order-in-order-lists)
      + [Invert Binary Tree Depth-First](#invert-binary-tree-depth-first)
      + [Invert Binary Tree Breadth-First](#invert-binary-tree-breadth-first)
      + [Find kth Smallest](#find-kth-smallest)
      + [Find Lowest Common Ancestor LCA](#find-lowest-common-ancestor-lca)
      + [Max Depth of Binary Tree](#max-depth-of-binary-tree)
      + [Same Tree](#same-tree)
      + [Is Subtree](#is-subtree)
      + [Validate BST](#validate-bst)
    + [Word Search Using Backtracking](#word-search-using-backtracking)
    + [Heaps, Hashing, Tracking](#heaps-hashing-tracking)
    + [Contains Duplicates](#contains-duplicates)
- [ML](#ml)
    + [Linear Regression](#linear-regression)
    + [Closed Form](#closed-form)
    + [Logistic Regression](#logistic-regression)

## Data Structures and Abstract Data Types
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


    def find_middle_of_linkedlist(head):
        slow, fast = head, head
        # even fast reaches the last node so fast.next is None
        # odd fast skips end so itself becomes none
        while fast and fast.next: 
            slow = slow.next
            fast = fast.next.next
        return slow
        
</details>

### Doubly Linked Lists
<details>
 <summary>Code</summary>

    class Node:
        def __init__(self, val):
            self.val = val
            self.next = None
            self.prev = None

    class DoublyLinkedList:
        def __init__(self, val):
            nn = Node(val)
            self.head = nn
            self.tail = nn
            self.length = 1

        def append(self, val):
            nn = Node(val)
            if self.head is None:
                self.head, self.tail = nn, nn
            else:
                self.tail.next = nn
                nn.prev = self.tail
                self.tail = nn
            self.length += 1
            return True

        def pop(self):
            if self.length == 0:
                return None
            temp = self.tail
            self.tail = self.tail.prev
            self.tail.next = None
            temp.prev = None
            self.length -= 1
            if self.length == 0:
                self.head, self.tail = None, None
            return temp

        def prepend(self, val):
            nn = Node(val)
            if self.length == 0:
                self.head, self.tail = nn, nn
            else:
                nn.next = self.head
                self.head.prev = nn
                self.head = nn
            self.length += 1
            return True

        def pop_first(self):
            if self.length == 0:
                return None
            temp = self.head
            if self.length == 1:
                self.head, self.tail = None, None
            else:
                self.head = self.head.next
                self.head.prev = None
                temp.next = None
            self.length -= 1
            return temp

        def get(self, ind):
            if ind < 0 or ind >= self.length:
                return None
            temp = self.head
            if ind < self.length / 2:
                for _ in range(ind):
                    temp = temp.next
            else:
                temp = self.tail
                for _ in range(self.length - 1, ind, -1):
                    temp = temp.prev
            return temp

        def set_val(self, ind, val):
            temp = self.get(ind)
            if temp:
                temp.val = val
                return True
            else:
                return False

        def insert(self, ind, val):
            if ind < 0 or ind > self.length:
                return False
            if ind == 0:
                return self.prepend(val)
            if ind == self.length:
                return self.append(val)
            nn = Node(val)
            before = self.get(ind - 1)
            after = before.next
            nn.prev = before
            nn.next = after
            before.next = nn
            after.prev = nn
            self.length += 1
            return True

        def remove(self, ind):
            if ind < 0 or ind >= self.length:
                return None
            if ind == 0:
                return self.pop_first()
            if ind == self.length - 1:
                return self.pop()
            temp = self.get(ind)
            temp.next.prev = temp.prev
            temp.prev.next = temp.next
            temp.next, temp.prev = None, None
            self.length -= 1
            return True

 
</details>

### Graph
<details>
 <summary>Code</summary>
    
    class Graph:

        def __init__(self):
            self.adj_list = {}


        def add_vertex(self, v):
            if v not in self.adj_list.keys():
                self.adj_list[v] = []
                return True
            return False

        def print_graph(self):
            for v in self.adj_list:
                print(f"{v}:{self.adj_list[v]}")

        def add_edge(self, v1, v2):
            if v1 not in self.adj_list.keys() and v2 not in self.adj_list.keys():
                return False
            else:
                self.adj_list[v1].append(v2)
                self.adj_list[v2].append(v1)
                return True

        def remove_edge(self, v1, v2):
            if v1 in self.adj_list.keys() and v2 in slef.adj_list.keys():
                self.adj_list[v1].remove(v2)
                self.adj_list[v2].remove(v1)
                return True
            return False

        def remove_vertex(self,  v):
            if v in self.adj_list.keys():
                for other_v in self.adj_list[v]:
                    self.adj_list[other_v].remove(v)
                del self.adj_list[v]
                return True
            return False
 
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

### BST
<details>
 <summary>Code</summary>

    class Node:
        def __init__(self, val):
            self.val = val
            self.left = None
            self.right = None


    class BST:

        def __init__(self):
            self.root = None

        def insert(self, val):
            nn = Node(val)
            if self.root == None:
                self.root = nn
                return True

            temp = self.root

            while True:
                if nn.val == temp.val:
                    return False

                if nn.val < temp.val:
                    if temp.left == None:
                        temp.left = nn
                        return True
                    else:
                        temp == temp.left
                if nn.val > temp.val:
                    if temp.right == None:
                        temp.right = nn
                        return True
                    else:
                        temp = temp.right
                else:
                    return False


        def contains(self, val):
            temp = self.root
            while temp is not None:
                if val < temp.val:
                    temp = temp.left
                if val > temp.val:
                    temp = temp.right
                else:
                    return True
            return False
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


     def insertion_sort(l):
        for i in range(1, len(l)):
            temp = l[i]
            j = i-1
            while (temp < l[j]) and (j > -1):
                l[j+1] = l[j]
                l[j] = temp
                j -= 1
        return l

        
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

## Hash-Tables

<details>
 <summary>Code</summary>

    class HashTable:

        def __init__(self, size = 7):
            self.data_map = [None] * size

        def _hash(self, key):
            my_hash = 0
            for i, letter in enumerate(key):
                my_hash = (my_hash + ord(letter) * (i+1)) % len(self.data_map)
            return my_hash

</details>

### Patterns for Linear Data Structures
### Two Pointers
### Palidrome
 <details>
 <summary>Code</summary>  
  
    def check_palidrome(s):
        assert isinstance(s, str)
        if len(s) == 1:
            return True
        l = 0
        r = len(s) - 1
        while s[l] == s[r]:
            print(s[l])
            print(s[r])
            l+=1
            r-=1
            if l > r:
                return True
        return False

    def is_palindrome(s):
        left = 0
        right = len(s) - 1
        while left < right:
            if s[left] != s[right]:
                return False
            left = left + 1 
            right = right - 1
        return True
  
  </details>

### Sum Three
 <details>
 <summary>Code</summary>  
      
    def check_sum_of_three(a, target):
        a.sort()

        for i in range(len(a) - 2):
            low = i + 1
            high = len(a) - 1

            while low < high:
                current_sum = a[i] + a[low] + a[high]

                if current_sum == target:
                    return True
                elif current_sum > target:
                    high -= 1
                else:
                    low += 1

        return False
        
  </details>

### Max Areea
 <details>
 <summary>Code</summary>  
  
      def calc_max_area(s):
        lower = 0
        upper = len(s) - 1
        max_vol = 0
        while lower < upper:
            length = upper - lower
            height = min(s[lower], s[upper])
            vol = length * height
            if vol > max_vol:
                max_vol = vol
            if s[lower] >= s[upper]:
                upper -= 1
            else:
                lower += 1
        return max_vol

  </details>

### Product Except Self
 <details>
 <summary>Code</summary> 

    def product_except_self(s):

        n = len(s)
        out = [1] * n

        left_prod = 1
        for i in range(n):
            out[i] *= left_prod
            left_prod *= s[i]

        right_prod = 1
        for i in range(n-1, -1, -1):
            out[i] *= right_prod
            right_prod *= s[i]

        return out


    def product_except_self(nums):
        n = len(nums)
        res = [1] * n
        left_product, right_product = 1, 1
        l = 0
        r = n - 1

        while l < n and r > -1:
            res[l] *= left_product
            res[r] *= right_product

            left_product *= nums[l]
            right_product *= nums[r]

            l += 1
            r -= 1

        return res

  </details>

### Remove  nth Node From Tail
 <details>
 <summary>Code</summary> 
  
      def remove_nth_last_node(head, n):
        l = head
        r = head

        for i in range(n):
            r = r.next

        if not r:
            return head.next

        while r.next:
            r = r.next
            l = l.next

        l.next = l.next.next

        return head
        
  </details>


### Two Pointers Fast Slow
### Detect Cycle
 <details>
 <summary>Code</summary> 
  
    def detect_cycle(head):

       if head is None:
          return False

       fast,  slow = head, head
       while fast.next:
          fast = fast.next.next
          slow = slow.next
          if fast == slow:
             return True
       return False

  </details>

### Modified Binary Search
### Binary Search Rotated
 <details>
 <summary>Code</summary> 
  
      def binary_search_rotated(nums, target):
        low = 0
        high = len(nums) - 1

        while low <= high:
            mid = low + (high - low) // 2

            if nums[mid] == target:
                return mid

            elif nums[low] <= nums[mid]:
                if nums[low] <= target and target < nums[mid]:
                    high = mid - 1
                else:
                    low = mid + 1
            elif nums[mid] <= nums[high]:
                if nums[mid] < target and target <= nums[high]:
                    low = mid + 1
                else:
                    high = mid - 1

        return False
  </details>

### Recursive Binary Search Rotated
 <details>
 <summary>Code</summary> 

      def binary_search(nums, low, high, target):

        if low > high:
            return False

        mid = low + (high - low) // 2

        if nums[mid] == target:
            return mid

        if nums[low] <= nums[mid]:
            if nums[low] <= target and target < nums[mid]:
                return binary_search(nums, low, mid-1, target)
            return binary_search(nums, mid+1, high, target)
        elif nums[mid] <= nums[high]:
            if nums[mid] < target and target <= nums[high]:
                return binary_search(nums, mid+1, high, target)
            binary_search(nums, low, mid-1, target)
        return False


    def binary_search_rotated(nums, target):
        return binary_search(nums, 0, len(nums) -1, target)

  </details>

### Find Min Val Rotated
 <details>
 <summary>Code</summary> 

      def find_min_val_rotated(nums):
        if len(nums) == 1:
            return nums[0]
        left = 0
        right = len(nums) - 1
        while right >= left:
            mid = left + (right - left) // 2
            if nums[mid] > nums[mid + 1]:
                return nums[mid + 1]
            if nums[mid] < nums[mid-1]:
                return nums[mid]
            if nums[left] < nums[mid]:
                left = mid + 1
            else:
                right = mid - 1

  </details>

### In Place Reversals of Linked Lists
### Fold Linked List
 <details>
 <summary>Code</summary> 

      def fold_linked_list(head):
        if not head:
            return head
        slow = fast = head

        # find middle node
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next 
        prev, curr = None, slow

        # reverse second half of linked list
        while curr:
            curr.next, prev, curr = prev, curr, curr.next     
        first, second = head, prev

        # merge lists by alternating links
        while second.next:
            first.next, first = second, first.next
            second.next, second = first, second.next

        return head

  </details>

### Reverse Linked List
 <details>
 <summary>Code</summary> 

      def reverse(head):
        prev = None
        cur = head
        while cur:
            temp = cur.next # store next
            cur.next = prev # reverse link
            prev = cur # incriment prev
            cur = temp
        return prev   
  </details>

### Stacks Valid Parentheses
 <details>
 <summary>Code</summary>

      def is_valid(string):
      stack = []
      hashmap = {")": "(", "}": "{", "]": "["}

      for char in string:
          if char not in hashmap:
              stack.append(char)
          else:
              if stack:
                  popped_element = stack.pop()
              else:
                  popped_element = "*"

              if hashmap[char] != popped_element:
                  return False

      return not stack

  </details>

### Matrices
### Set to Zero
 <details>
 <summary>Code</summary>

      def set_matrix_zeros(mat):
        rows = len(mat)
        cols = len(mat[0])
        fcol = False
        frow = False

        # Check if there is a zero in first column, set fcol to True.
        for i in range(rows):
            if mat[i][0] == 0:
                fcol = True
        # Check if there is a zero in first row, set frow to True.
        for i in range(cols):
            if mat[0][i] == 0:
                frow = True

        # Check row elements (by ignoring first row and first column). If zero is found,
        # set corresponding row's and column's first element to zero.
        for i in range(1, rows):
            for j in range(1, cols):
                if mat[i][j] == 0:
                    mat[0][j] = mat[i][0] = 0

        # Check every row's first element starting from second row.
        # Set complete row to zero if zero is found.
        for i in range(1, rows):
            if mat[i][0] == 0:
                for j in range(1, cols):
                    mat[i][j] = 0

        # Check every column's first element starting from second column.
        # Set complete column to zero if zero is found.
        for j in range(1, cols):
            if mat[0][j] == 0:
                for i in range(1, rows):
                    mat[i][j] = 0

        # If fcol is true, set first column to zero.
        if fcol:
            for i in range(rows):
                mat[i][0] = 0

        # If frow is true, set first row to zero.
        if frow:
            for j in range(cols):
                mat[0][j] = 0
        return mat


    def set_to_zero(m):
        rows = len(m)
        cols = len(m[0])
        f_row = False
        f_col = False

        # Check if zero in first row
        if 0 in m[0]:
            f_row = True
        # Check if zero in first col
        for i in range(rows):
            if m[i][0] == 0:
                f_col = True

        # Iterate through matrix and if zero found at index i,j then set i,0 to zero and 0,j to zero
        for i in range(1, rows):
            for j in range(1, cols):
                if m[i][j] == 0:
                    m[i][0] = m[0][j] = 0

        # Iterate through the matrix to set entire row i to zero if m[i][0] is zero
        # i remains fixed
        for i in range(1, rows): 
            if m[i][0] == 0:
                for j in range(cols):
                    m[i][j] = 0

        # Iterate through the matrix to set entire column j to zero if m[0][j] is zero
        # j remains fixed
        for j in range(1, cols):
            if m[0][j] == 0:
                for i in range(rows):
                    m[i][j] = 0

        # If f_row is True then set first row to zero
        if f_row:
            for j in range(cols):
                m[0][j] = 0

        # If f_col is True then set first col to zero
        if f_col:
            for i in range(rows):
                m[i][0] = 0

        return m
        
  </details>

### Rotate 90 Degrees
 <details>
 <summary>Code</summary>

      def rotate_90_degrees(matrix):

        n = len(matrix)

        # Traverse the matrix
        for row in range(n // 2):
            for col in range(row, n - row - 1):
                # Swap the top-left and top-right cells in the current group
                matrix[row][col], matrix[col][n - 1 - row] = matrix[col][n - 1 - row], matrix[row][col]

                # Swap the top-left and bottom-right cells in the current group 
                matrix[row][col], matrix[n - 1 - row][n - 1 - col] = matrix[n - 1 - row][n - 1 - col], matrix[row][col]

                # Swap the top-left and bottom-left cells in the current group  
                matrix[row][col], matrix[n - 1 - col][row] = matrix[n - 1 - col][row], matrix[row][col] 

        return matrix

  </details>

### Spiral Matrix to Array
 <details>
 <summary>Code</summary>

      def spiral_order(m):
        direction = "right"
        rows = len(m)
        cols = len(m[0])
        left_boundary, right_boundary = 0, cols - 1
        top_boundary, bottom_boundary = 0, rows - 1
        output_array = []

        while left_boundary <= right_boundary and top_boundary <= bottom_boundary:
            if direction == "right":
                for c in range(left_boundary, right_boundary + 1):
                    output_array.append(m[top_boundary][c])
                top_boundary += 1  # Adjust top_boundary after processing the row.
                direction = "down"  # Change direction.

            elif direction == "down":
                for r in range(top_boundary, bottom_boundary + 1):
                    output_array.append(m[r][right_boundary])
                right_boundary -= 1  # Adjust right_boundary after processing the column.
                direction = "left"  # Change direction.

            elif direction == "left":
                for c in range(right_boundary, left_boundary - 1, -1):
                    output_array.append(m[bottom_boundary][c])
                bottom_boundary -= 1  # Adjust bottom_boundary after processing the row.
                direction = "up"  # Change direction.

            elif direction == "up":
                for r in range(bottom_boundary, top_boundary - 1, -1):
                    output_array.append(m[r][left_boundary])
                left_boundary += 1  # Adjust left_boundary after processing the column.
                direction = "right"  # Change direction.

        return output_array

     def spiral_order(matrix):
        rows, cols = len(matrix), len(matrix[0])
        row, col = 0, -1
        direction = 1 
        result = []

        while rows > 0 and cols > 0:
            for _ in range(cols):
                col += direction
                result.append(matrix[row][col])
            rows -= 1

            for _ in range(rows):
                row += direction
                result.append(matrix[row][col])
            cols -= 1

            direction *= -1 

        return result     

  </details>

### BFS
 <details>
 <summary>Code</summary>

    from collections import deque  # Import the deque data structure for the queue.

    def bfs(root):
        if not root:
            return []

        result = []  # Initialize a list to store the BFS traversal result.
        queue = deque()  # Create a queue to keep track of nodes to be visited.

        queue.append(root)  # Enqueue the root node to start the traversal.

        while queue:
            current_node = queue.popleft()  # Dequeue the first node in the queue.
            result.append(current_node.value)  # Append the current node's value to the result.

            # Enqueue the child nodes (if they exist) for future exploration.
            if current_node.left:
                queue.append(current_node.left)
            if current_node.right:
                queue.append(current_node.right)

        return result  # Return the BFS traversal result


            from collections import deque

    class TreeNode:
        def __init__(self, data):
            self.data = data
            self.left = None
            self.right = None

    def level_order_traversal(root):
        if not root:
            return []

        result = []  # Initialize a list to store the BFS traversal result.
        queue = deque()  # Create a queue to keep track of nodes to be visited.

        queue.append(root)  # Enqueue the root node to start the traversal.
        queue.append(None)  # Use None as a level delimiter.

        while queue:
            current_node = queue.popleft()

            if current_node is None:
                # When encountering None, it means we've finished one level. Append ":" as a delimiter.
                result.append(":")
                if queue:  # Check if there are more nodes in the queue.
                    queue.append(None)  # Add a new level delimiter for the next level.
            else:
                result.append(current_node.data)  # Append the current node's data to the result.

                # Enqueue the child nodes (if they exist) for future exploration.
                if current_node.left:
                    queue.append(current_node.left)
                if current_node.right:
                    queue.append(current_node.right)

        return result  # Return the BFS traversal result


    def level_order_traversal(root):
        result = ""
        if not root:
            result = "None"
            return result
        else:
            queues = [deque(), deque()]
            current_queue = queues[0]
            next_queue = queues[1]

            current_queue.append(root)
            level_number = 0

            while current_queue:
                temp = current_queue.popleft()
                result += str(temp.data)

                if temp.left:
                    next_queue.append(temp.left)

                if temp.right:
                    next_queue.append(temp.right)

                if not current_queue:
                    level_number += 1

                    if next_queue:
                        result += " : "
                    current_queue = queues[level_number % 2]
                    next_queue = queues[(level_number + 1) % 2]

                else:
                    result += ", "

            return result


    from collections import deque

    def level_order_traversal(root):
        result = ""
        # Print 'None' if the root is empty
        if not root:
            result = "None"
            return result
        else:
            # Initializing the current queue
            current_queue = deque()

            # Initializing the dummy node
            dummy_node = TreeNode(0)

            current_queue.append(root)
            current_queue.append(dummy_node)

            # Printing nodes in level-order until the current queue remains
            # empty
            while current_queue:
                # Dequeuing and printing the first element of queue
                temp = current_queue.popleft()
                result += str(temp.data)

                # Adding dequeued node's children to the next queue
                if temp.left:
                    current_queue.append(temp.left)

                if temp.right:
                    current_queue.append(temp.right)

                # When the dummyNode comes next in line, we print a new line and dequeue
                # it from the queue
                if current_queue[0] == dummy_node:
                    current_queue.popleft()

                    # If the queue is still not empty we add back the dummy node
                    if current_queue:
                        result += " : "
                        current_queue.append(dummy_node)
                else:
                    result += ", "
            return result

  </details>

## DFS
### Pre-Order
 <details>
 <summary>Code</summary>
  
    def preorder_dfs(root):
        # base case
        if root is None:
            return

        # visit node
        print(root.val)
        # recursive call left
        preorder_dfs(root.left)
        # recursive call right
        preorder_dfs(root.right)
        
   </details>

### In-Order
 <details>
 <summary>Code</summary>

    def inorder_dfs(root):
        # base case
        if root is None:
            return

        # recursive call left
        inorder_dfs(root.left)
        # visit node
        print(root.val)
        # recursive call right
        inorder_dfs(root.right)
        
   </details>

### Post-Order
 <details>
 <summary>Code</summary>

     def postorder_dfs(root):
        # base case
        if root is None:
            return

        # recursive call left
        postorder_dfs(root.left)
        # recursive call right
        postorder_dfs(root.right)
        # visit node
        print(root.val)

   </details>

### Serialize BST
 <details>
 <summary>Code</summary>

    def preorder_dfs(node, serialized_tree):
        if node is None:
            serialized_tree.append(None)
            return
        serialized_tree.append(node.val)
        preorder_dfs(node.left, serialized_tree)
        preorder_dfs(node.right, serialized_tree)

    def serialize_with_preorder_dfs(root):
        serialized_tree = []
        preorder_dfs(root, serialized_tree)
        return serialized_tree

   </details>


### Deserialize BST
 <details>
 <summary>Code</summary>

    def deserialize_bst(values):
        if not values:
            return None

        # Create the root of the BST.
        root = TreeNode(values[0])

        # Iterate through the remaining values to insert them into the BST.
        for val in values[1:]:
            insert_into_bst(root, val)

        return root

    def insert_into_bst(node, val):
        if node is None:
            return TreeNode(val)

        # Insert value into the appropriate subtree based on BST property.
        if val < node.val:
            node.left = insert_into_bst(node.left, val)
        else:
            node.right = insert_into_bst(node.right, val)

        return node
  
   </details>

### Serialize Deserialize BST Exact
 <details>
 <summary>Code</summary>

     from BinaryTree import *
    from TreeNode import *

    # Initializing our marker
    MARKER = "M"
    m = 1

    def serialize_rec(node, stream):
        global m

        if node is None:
            stream.append(MARKER + str(m))
            m += 1
            return

        stream.append(node.data)

        serialize_rec(node.left, stream)
        serialize_rec(node.right, stream)

    # Function to serialize tree into list of integers.
    def serialize(root):
        stream = []
        serialize_rec(root, stream)
        return stream

    def deserialize_helper(stream):
        val = stream.pop()

        if type(val) is str and val[0] == MARKER:
            return None

        node = TreeNode(val)

        node.left = deserialize_helper(stream)
        node.right = deserialize_helper(stream)

        return node

    # Function to deserialize integer list into a binary tree.
    def deserialize(stream):
        stream.reverse()
        node = deserialize_helper(stream)
        return node

   </details>

### Max Sum Path
 <details>
 <summary>Code</summary>

       def maxPathSum(root):
        # Initialize a variable to keep track of the global maximum sum.
        max_sum = float('-inf')

        # Define a recursive function to compute the maximum path sum for a node.
        def max_path_sum(node):
            nonlocal max_sum  # Use the nonlocal keyword to modify the global max_sum.

            # Base case: If the node is None, return 0 (no contribution to the path).
            if not node:
                return 0

            # Recursively compute the maximum path sums for the left and right subtrees.
            left_sum = max(0, max_path_sum(node.left))  # Ensure negative values are not included.
            right_sum = max(0, max_path_sum(node.right))

            # Calculate the local maximum including the current node.
            local_max = node.val + left_sum + right_sum

            # Update the global maximum if the local maximum is greater.
            max_sum = max(max_sum, local_max)

            # Return the maximum path sum starting from the current node upwards.
            return node.val + max(left_sum, right_sum)

        # Start the recursive traversal from the root node.
        max_path_sum(root)

        # The maximum path sum is stored in max_sum after the traversal.
        return max_sum

   </details>

### Build BST Pre-Order In-Order Lists
 <details>
 <summary>Code</summary>

       def build_tree_helper(p_order, i_order, left, right, mapping, p_index):
        # Base case: If the left index exceeds the right index, there are no nodes to create.
        if left > right:
            return None

        # Get the current root value from p_order using the p_index pointer.
        curr = p_order[p_index[0]]
        p_index[0] += 1

        # Create a TreeNode with the current root value.
        root = TreeNode(curr)

        # If left and right are equal, it's a leaf node, so return the root.
        if left == right:
            return root

        # Find the index of the current root value in i_order (in_index).
        in_index = mapping[curr]

        # Recursively build the left and right subtrees.
        root.left = build_tree_helper(p_order, i_order, left, in_index - 1, mapping, p_index)
        root.right = build_tree_helper(p_order, i_order, in_index + 1, right, mapping, p_index)

        return root

    def build_tree(p_order, i_order):
        # Initialize a list containing a single element as a pointer to the next value in p_order.
        p_index = [0]

        # Create a mapping dictionary to efficiently find the index of values in i_order.
        mapping = {}

        # Populate the mapping dictionary by iterating through p_order and i_order.
        for i in range(len(p_order)):
            mapping[i_order[i]] = i

        # Call the build_tree_helper to construct the binary tree.
        return build_tree_helper(p_order, i_order, 0, len(p_order) - 1, mapping, p_index)

   </details>
   
### Invert Binary Tree Depth-First 
 <details>
 <summary>Code</summary>

    # pre-order traversal
    def mirror_binary_tree_df(root):
      # base case
      if root is None:
        return None

      # perform swap
      root.left, root.right = root.right, root.left

      # recursive call
      mirror_binary_tree_df(root.left)
      mirror_binary_tree_df(root.right)

      return root

    # post-order traversal
    def mirror_binary_tree(root):

        if not root:
            return None

        mirror_binary_tree(root.left)
        mirror_binary_tree(root.right)

        root.left, root.right = root.right, root.left

        return root

   </details>


### Invert Binary Tree Breadth-First 
 <details>
 <summary>Code</summary>
  
       def invert_tree_breadth_first(root):
        if not root:
            return None

        queue = deque([root])

        while queue:
            current_node = queue.popleft()

            # Swap the children of the current node
            current_node.left, current_node.right = current_node.right, current_node.left

            # Add the children to the queue for subsequent processing
            if current_node.left:
                queue.append(current_node.left)
            if current_node.right:
                queue.append(current_node.right)

        return root

   </details>

### Find kth Smallest 
 <details>
 <summary>Code</summary>
  
    def kth_smallest_element(root, k):
        # Helper function for in-order traversal of the tree
        def inorder(node):
            # Base case: return if node is None or if we've found k elements already
            if node is None or len(traversal) >= k:
                return

            # Recursive call on the left subtree
            inorder(node.left)

            # Process the current node
            # Only add to traversal list if fewer than k elements have been found
            if len(traversal) < k:
                traversal.append(node.data)

            # Recursive call on the right subtree
            inorder(node.right)

        # Initialize an empty list to store the traversed elements
        traversal = []

        # Start in-order traversal from the root
        inorder(root)

        # Check if we found k elements and return the kth smallest
        # If the list is not empty, return the last element (kth smallest)
        if traversal:
            return traversal[-1]
        else: 
            # If the list is empty (e.g., if k is larger than the number of nodes),
            # return None indicating the kth smallest element doesn't exist
            return None



    def kth_smallest_element(root, k):
        # Call the recursive helper function with the root and k wrapped in a list
        # The list is used to maintain the state of k across recursive calls
        return kth_smallest_rec(root, [k]).data

    # Recursive helper function for finding the kth smallest element
    def kth_smallest_rec(node, k):
        # Base case: if the node is None, return None
        if not node:
            return None

        # Recurse on the left subtree
        left = kth_smallest_rec(node.left, k)
        # If a node was returned from the left subtree, it is the kth smallest
        # Hence, return it up the call stack
        if left:
            return left

        # Process the current node
        # Decrement the counter (k[0]) since we've visited one more node
        k[0] -= 1
        # If the counter reaches 0, we've found the kth smallest element
        # Return the current node
        if k[0] == 0:
            return node

        # Recurse on the right subtree if the kth smallest hasn't been found yet
        return kth_smallest_rec(node.right, k)

   </details>

### Find Lowest Common Ancestor LCA
 <details>
 <summary>Code</summary>

    def lowest_common_ancestor(root, p, q):
        # Base case: If we reach the end of a path (root is None), 
        # or find either p or q, return root (which could be None, p, or q)
        if not root or root == p or root == q:
            return root

        # Recursively search for p and q in the left subtree
        left = lowest_common_ancestor(root.left, p, q)

        # Recursively search for p and q in the right subtree
        right = lowest_common_ancestor(root.right, p, q)

        # If both left and right are not None, it means we found p and q in 
        # different subtrees, so the current node is the LCA
        if left and right:
            return root

        # If only one of left or right is not None, return the one that is not None.
        # This could be a situation where:
        #   1. One of p or q is in the subtree, and the other is the current node.
        #   2. Both p and q are in one subtree, and we're returning the LCA found in that subtree.
        return left if left else right



    def lowest_common_ancestor(root, p, q):
        # If the current node is None, or the current node matches either p or q, 
        # return the current node. This acts as a base case for recursion and also 
        # checks if we have found one of the nodes we're looking for.
        if not root or root == p or root == q:
            return root

        # Recursively search for p and q in the left subtree of the current node.
        # If either p or q is found in the left subtree, 'left' will hold that node; 
        # otherwise, it will be None.
        left = lowest_common_ancestor(root.left, p, q)

        # Similarly, recursively search for p and q in the right subtree.
        # If either p or q is found in the right subtree, 'right' will hold that node; 
        # otherwise, it will be None.
        right = lowest_common_ancestor(root.right, p, q)

        # If both 'left' and 'right' are non-None, it means that we have found both p and q in 
        # different subtrees of the current node. Therefore, the current node is the LCA.
        if left and right:
            return root

        # If only one of 'left' or 'right' is non-None, it means either one of the nodes was found
        # and the other was not, or one node is an ancestor of the other. In both cases, 
        # return the non-None node.
        return left if left else right



    def lowest_common_ancestor(root, p, q):
        # Mutable container to hold the LCA
        lca = [None]
        # Start the recursive function
        lowest_common_ancestor_rec(root, p, q, lca)
        # Return the found LCA
        return lca[0]
        

    def lowest_common_ancestor_rec(current_node, p, q, lca):
        # Base case: if current node is None, return False
        if not current_node:
            return False

        # Check recursively if the left subtree contains either p or q
        left = lowest_common_ancestor_rec(current_node.left, p, q, lca)
        # Check recursively if the right subtree contains either p or q
        right = lowest_common_ancestor_rec(current_node.right, p, q, lca)

        # Check if the current node itself is either p or q
        mid = current_node == p or current_node == q

        # If any two of the three checks (left, right, mid) are True,
        # it means this is the common ancestor of p and q
        if mid + left + right >= 2:
            lca[0] = current_node

        # Return True if the current node or any node in its subtrees is p or q
        return mid or left or right

   </details>

### Max Depth of Binary Tree
 <details>
 <summary>Code</summary>

    def max_depth(root):
        # Base case: If the current node is None, it means we have reached 
        # beyond a leaf node, or the tree is empty. In either case, return 0.
        if not root:
            return 0

        # Recursively find the depth of the left subtree. This call will 
        # go down to the leftmost leaf, calculating the depth along the way.
        left_depth = max_depth(root.left)

        # Similarly, recursively find the depth of the right subtree.
        right_depth = max_depth(root.right)

        # The depth of the current tree is the maximum of the depths of the left and 
        # right subtrees, plus 1 for the current node.
        # This "+1" accounts for the edge between the current node and its parent.
        return max(left_depth, right_depth) + 1


    from collections import deque

    def find_max_depth(root):
        if not root:
            return 0

        nodes_stack = deque([(root, 1)])

        max_depth = 0

        while nodes_stack:
            node, depth = nodes_stack.pop()

            if node.left:
                nodes_stack.append((node.left, depth + 1))

            if node.right:
                nodes_stack.append((node.right, depth + 1))

            if not node.left and not node.right:
                max_depth = max(max_depth, depth)

        return max_depth
        
   </details>


### Same Tree
 <details>
 <summary>Code</summary>

    def is_same_tree(p, q):
        # Base case: If both nodes are None, they are the same.
        if not p and not q:
            return True

        # If one node is None and the other isn't, or if the values differ, the trees aren't the same.
        if not p or not q or p.value != q.value:
            return False

        # Recursively compare the left and right subtrees.
        return is_same_tree(p.left, q.left) and is_same_tree(p.right, q.right)


    def same_tree(p, q):
        if (not p) and (not q):
            return True
        elif (not p) or (not q):
            return False
        elif p.data != q.data:
            return False

        return same_tree(p.left, q.left) and same_tree(p.right, q.right)
        
   </details>

### Is Subtree
 <details>
 <summary>Code</summary>

    def isSubtree(root, sub_root):
        # If sub_root is None, it's universally considered a subtree of any tree, including an empty tree.
        if not sub_root:
            return True
        # If root is None but sub_root is not, then sub_root can't be a subtree of root.
        if not root:
            return False

        # Use a helper function to check if the tree rooted at 'root' is the same as sub_root.
        if isSameTree(root, sub_root):
            return True

        # Recursively check if sub_root is a subtree of either the left or right subtree of root.
        # The subtree is found if it matches either left or right subtree of the current node.
        return isSubtree(root.left, sub_root) or isSubtree(root.right, sub_root)

    def isSameTree(p, q):
        # Base case: If both nodes are None, they are the same (end of branches).
        if not p and not q:
            return True
        # If one node is None and the other isn't, or if their values differ,
        # the trees rooted at these nodes are not the same.
        if not p or not q or p.value != q.value:
            return False
        # Recursively check the left and right subtrees of p and q.
        return isSameTree(p.left, q.left) and isSameTree(p.right, q.right)



    def is_subtree(root, sub_root):
        if not root:
            return False

        if is_identical(root, sub_root):
            return True

        return is_subtree(root.left, sub_root) or is_subtree(root.right, sub_root)

    def is_identical(node1, node2):
        if not node1 and not node2:
            return True
        if not node1 or not node2:
            return False

        return (node1.data == node2.data and
                is_identical(node1.left, node2.left) and
                is_identical(node1.right, node2.right))

   </details>

### Validate BST
 <details>
 <summary>Code</summary>
  
    def isValidBST(root):
        # Start the validation process with the full range of valid values.
        # The entire valid range is considered initially (from negative to positive infinity).
        return validate(root, float('-inf'), float('inf'))

    def validate(node, low, high):
        # Base case: An empty node is considered valid.
        if not node:
            return True

        # Check if the current node's value is within the valid range.
        # It must be greater than the low limit and less than the high limit.
        if node.value <= low or node.value >= high:
            return False

        # Recursively check the left subtree.
        # For the left subtree, the current node's value becomes the new high limit.
        # The left child must have a value less than the current node's value.
        return validate(node.left, low, node.value) and \
               # Recursively check the right subtree.
               # For the right subtree, the current node's value becomes the new low limit.
               # The right child must have a value greater than the current node's value.
               validate(node.right, node.value, high)



    ### with Inoder traversal - is always increasing (left, root, right)
    import math

    def validate_bst(root):
        # Initialize a list with one element, -infinity, to keep track of the last visited node's value.
        # Using a list allows the value to be mutable and updated across recursive calls.
        prev = [-math.inf]
        # Call the helper function to validate the BST.
        return validate_bst_helper(root, prev)

    # Helper function to validate if a binary tree is a BST.
    def validate_bst_helper(root, prev):
        # Base case: If the current node is None, return True, as an empty tree is a valid BST.
        if not root:
             return True

        # Recursively validate the left subtree.
        # If the left subtree is not a valid BST, return False immediately.
        if not validate_bst_helper(root.left, prev):
            return False

        # Check the current node's value against the last visited node's value (stored in prev).
        # If the current node's value is not greater than the last visited node's value,
        # the tree is not a valid BST.
        if root.data <= prev[0]:
            return False

        # Update prev with the current node's value before moving to the right subtree.
        prev[0] = root.data

        # Recursively validate the right subtree.
        # The result of this call determines the validity of the entire 
        # subtree rooted at the current node.
        return validate_bst_helper(root.right, prev)

   </details>


### Word Search Using Backtracking
 <details>
 <summary>Code</summary>
  
    def exist(board, word):
        # Define the backtracking function that will be used to check
        # if the word exists starting from a specific cell
        def backtrack(row, col, index):
            # If the entire word is matched, return True
            if index == len(word):
                return True
            # Check if the current cell is out of bounds or the character doesn't match
            # the current character in the word
            if (row < 0 or row >= len(board) or col < 0 or col >= len(board[0]) or
                    word[index] != board[row][col]):
                return False

            # Temporarily mark the current cell as visited by replacing its value
            temp, board[row][col] = board[row][col], '#'

            # Recursively explore all four adjacent directions (up, down, left, right)
            # and check if the word can be formed from this cell
            found = (backtrack(row + 1, col, index + 1) or  # Down
                     backtrack(row - 1, col, index + 1) or  # Up
                     backtrack(row, col + 1, index + 1) or  # Right
                     backtrack(row, col - 1, index + 1))    # Left

            # Restore the original value of the cell (unmark it as visited)
            board[row][col] = temp

            # Return True if the word is found in any direction
            return found

        # Iterate through each cell in the grid as a potential starting point
        for i in range(len(board)):
            for j in range(len(board[0])):
                # Start the backtracking process from the current cell
                if backtrack(i, j, 0):
                    # If the word is found, return True
                    return True

        # If the word is not found in any path, return False
        return False



    def word_search(grid, word):
        # Get the dimensions of the grid
        n = len(grid)
        m = len(grid[0])

        # Iterate over each cell in the grid as potential starting points
        for row in range(n):
            for col in range(m):
                # If the word is found starting from this cell, return True
                if depth_first_search(row, col, word, grid):
                    return True

        # If the word is not found in any path, return False
        return False

    # Apply backtracking on every element to search the required word
    def depth_first_search(row, col, word, grid):
        # Base case: if the entire word is matched, return True
        if len(word) == 0:
            return True

        # Check if the current cell is out of bounds, or the character in the grid cell
        # does not match the first character of the word
        if row < 0 or row >= len(grid) or col < 0 or col >= len(grid[0]) \
                or grid[row][col].lower() != word[0].lower():
            return False

        # Temporarily mark the current cell as visited by changing its value
        grid[row][col] = '*'

        # Explore all four adjacent directions (up, down, left, right)
        for rowOffset, colOffset in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            # Recursively check the next character of the word in the adjacent cells
            if depth_first_search(row + rowOffset, col + colOffset, word[1:], grid):
                # If the word is found in any direction, return True
                return True

        # Restore the original value of the cell (backtracking)
        grid[row][col] = word[0]

        # Return False if the word is not found in any direction from this cell
        return False
        
   </details>
   
### Heaps, Hashing, Tracking
 <details>
 <summary>Code</summary>

    def is_anagram(str1, str2):
        """
        Check if two strings are anagrams of each other.

        An anagram is a word or phrase formed by rearranging the letters of a different word or phrase,
        typically using all the original letters exactly once. This function checks if the two provided
        strings are anagrams of each other.

        Parameters:
        str1 (str): The first string to be compared.
        str2 (str): The second string to be compared.

        Returns:
        bool: True if str1 and str2 are anagrams, False otherwise.

        Example:
        >>> is_anagram("listen", "silent")
        True
        >>> is_anagram("hello", "world")
        False
        """

        if len(str1) == 0 or len(str1) != len(str2):
            return False

        char_dict = {}

        for char in str1:
            if char not in char_dict:
                char_dict[char] = 1
            else:
                char_dict[char] += 1

        for char in str2:
            if char not in char_dict:
                return False  # This character was not in str1
            else:
                char_dict[char] -= 1

        return all(value == 0 for value in char_dict.values())


    def is_anagram(str1, str2):
        if len(str1) != len(str2):
            return False

        table = {}

        for i in str1:
          if i in table:
            table[i] += 1

          else:
            table[i] = 1

        for i in str2:
          if i in table:
            table[i] -= 1

          else:
            return False

        for key in table:
            if table[key] != 0:
                return False

        return True
        
   </details>

### Contains Duplicates
 <details>
 <summary>Code</summary>

    def contains_duplicate(nums):
        """
        Check if the list contains any duplicates.

        Parameters:
        nums (list): A list of integers to be checked for duplicates.

        Returns:
        bool: True if any value appears at least twice in the list, False if every element is distinct.

        Example:
        >>> contains_duplicate([1, 2, 3, 4])
        False
        >>> contains_duplicate([1, 2, 3, 3])
        True
        """

        seen = set()

        for n in nums:
            if n in seen:
                return True
            seen.add(n)

        return False


    def contains_duplicate(nums):
        """
        Check if the list contains any duplicates.

        Parameters:
        nums (list): A list of integers to be checked for duplicates.

        Returns:
        bool: True if any value appears at least twice in the list, False if every element is distinct.

        Example:
        >>> contains_duplicate([1, 2, 3, 4])
        False
        >>> contains_duplicate([1, 2, 3, 3])
        True
        """

        records = {}
        for i in nums:
            if i in records:
                return True

            records[i] = i
        return False

   </details>

## ML

### Linear Regression
 <details>
 <summary>Code</summary>
    import numpy as np

    # Linear Regression Class from scratch
    class LinearRegression:
        def __init__(self, learning_rate=0.01, epochs=1000):
            self.learning_rate = learning_rate  # Learning rate for gradient descent
            self.epochs = epochs  # Number of iterations for training
            self.W = None  # Weights
            self.b = None  # Bias

        def fit(self, X, y):
            """
            Train the linear regression model using gradient descent.

            Args:
            X (numpy.ndarray): Input features (NxD matrix, where N is the number of samples and D is the number of features).
            y (numpy.ndarray): True labels (Nx1 vector).
            """
            # Initialize weights and bias
            num_samples, num_features = X.shape
            self.W = np.zeros((num_features, 1))  # Initialize weights to zeros
            self.b = 0  # Initialize bias to zero

            # Reshape y to make sure it's a column vector
            y = y.reshape(-1, 1)

            # Gradient Descent
            for epoch in range(self.epochs):
                # Step 1: Make predictions (forward pass)
                yhat = X.dot(self.W) + self.b  # Linear hypothesis

                # Step 2: Compute the Mean Squared Error (MSE) loss
                error = yhat - y
                mse = np.mean(error ** 2)

                # Step 3: Compute the gradients
                grad_W = (1 / num_samples) * X.T.dot(error)  # Gradient with respect to weights
                grad_b = (1 / num_samples) * np.sum(error)   # Gradient with respect to bias

                # Step 4: Update the weights and bias using the gradients
                self.W -= self.learning_rate * grad_W
                self.b -= self.learning_rate * grad_b

                # Optionally, print the loss to see progress
                if epoch % 100 == 0:
                    print(f"Epoch {epoch}, MSE: {mse}")

        def predict(self, X):
            """
            Make predictions using the trained linear regression model.

            Args:
            X (numpy.ndarray): Input features (NxD matrix).

            Returns:
            numpy.ndarray: Predicted values (Nx1 vector).
            """
            return X.dot(self.W) + self.b

    # Example Usage
    if __name__ == "__main__":
        # Create some example data (NxD matrix)
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])  # Features
        y = np.array([3, 5, 7, 9])  # True labels

        # Initialize and train the linear regression model
        model = LinearRegression(learning_rate=0.01, epochs=1000)
        model.fit(X, y)

        # Predict values for the input data
        predictions = model.predict(X)
        print("Predictions:", predictions.flatten())
 </details>

### Closed Form
 <details>
 <summary>Code</summary>

import numpy as np

    def normal_equation(X, y):
     """
     Solve for weights W using the normal equation.
 
     Args:
     X (numpy.ndarray): Input features, augmented with a column of ones (NxD+1 matrix).
     y (numpy.ndarray): Output/target values (Nx1 vector).
 
     Returns:
     numpy.ndarray: Weights vector (Dx1 vector).
     """
     # Compute the matrix inverse of X'X
     X_transpose = X.T  # Transpose of X
     X_transpose_X = np.dot(X_transpose, X)  # X'X
     X_transpose_X_inv = np.linalg.inv(X_transpose_X)  # (X'X)^-1
 
     # Compute the final weights
     X_transpose_y = np.dot(X_transpose, y)  # X'y
     W = np.dot(X_transpose_X_inv, X_transpose_y)  # (X'X)^-1 X'y
 
     return W
 </details>


### Logistic Regression
 <details>
 <summary>Code</summary>
    import numpy as np

    # Logistic Regression Class from scratch
    class LogisticRegression:
        def __init__(self, learning_rate=0.01, epochs=1000):
            self.learning_rate = learning_rate  # Learning rate for gradient descent
            self.epochs = epochs  # Number of iterations for training
            self.W = None  # Weights
            self.b = None  # Bias

        def sigmoid(self, z):
            """
            Compute the sigmoid function for the input z.
            """
            return 1 / (1 + np.exp(-z))

        def fit(self, X, y):
            """
            Train the logistic regression model using gradient descent.

            Args:
            X (numpy.ndarray): Input features (NxD matrix, where N is the number of samples and D is the number of features).
            y (numpy.ndarray): True labels (Nx1 vector).
            """
            # Initialize weights and bias
            num_samples, num_features = X.shape
            self.W = np.zeros((num_features, 1))  # Initialize weights to zeros
            self.b = 0  # Initialize bias to zero

            # Reshape y to make sure it's a column vector
            y = y.reshape(-1, 1)

            # Gradient Descent
            for epoch in range(self.epochs):
                # Step 1: Make predictions (forward pass)
                z = X.dot(self.W) + self.b  # Linear combination
                yhat = self.sigmoid(z)  # Sigmoid to get probabilities

                # Step 2: Compute the binary cross-entropy (BCE) loss
                error = yhat - y
                bce = -np.mean(y * np.log(yhat + 1e-9) + (1 - y) * np.log(1 - yhat + 1e-9))

                # Step 3: Compute the gradients
                grad_W = (1 / num_samples) * X.T.dot(error)  # Gradient with respect to weights
                grad_b = (1 / num_samples) * np.sum(error)   # Gradient with respect to bias

                # Step 4: Update the weights and bias using the gradients
                self.W -= self.learning_rate * grad_W
                self.b -= self.learning_rate * grad_b

                # Optionally, print the loss to see progress
                if epoch % 100 == 0:
                    print(f"Epoch {epoch}, BCE Loss: {bce}")

        def predict_prob(self, X):
            """
            Predict the probabilities for input data X using the trained logistic regression model.

            Args:
            X (numpy.ndarray): Input features (NxD matrix).

            Returns:
            numpy.ndarray: Predicted probabilities (Nx1 vector).
            """
            z = X.dot(self.W) + self.b
            return self.sigmoid(z)

        def predict(self, X, threshold=0.5):
            """
            Predict binary class labels (0 or 1) based on a probability threshold.

            Args:
            X (numpy.ndarray): Input features (NxD matrix).
            threshold (float): Threshold to convert probabilities to class labels (default is 0.5).

            Returns:
            numpy.ndarray: Predicted class labels (0 or 1).
            """
            probabilities = self.predict_prob(X)
            return (probabilities >= threshold).astype(int)

 </details>

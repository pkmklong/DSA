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
    + [Graph](#graph)
    + [Heap](#heap)
    + [BST](#bst)
  * [Sort Algorithms](#sort-algorithms)
    + [Bubble Sort](#bubble-sort)
    + [Selection Sort](#Selection-sort)
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
      + [Product Except Self](#prod-except-self)
      + [Remove nth Node from Tail](#remove-nth-node-from-tail)
    + [Two Pointers Fast Slow](#two-pointers-fast-slow)
      + [Detect Cycle](#detect-cycle)
    + [Modified Binary Search](#modified-binary-search)
      + [Binary Search Rotated](#binary-search-rotated)
      + [Recursive Binary Search Rotated](#recursive-binary-search-rotated)
      + [Find Min Val Rotated](#find-min-val-rotated)

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

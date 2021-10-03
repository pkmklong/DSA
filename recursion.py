
def fib(n, memo = {}):
  # Returns nth Fibonacci value using recursion and memoization
    if n == 0: 
        return 0
    if n == 1: 
        return 1
    if not memo.get(n):
             memo[n] = fib(n-1, memo) + fib(n-2, memo) 
    return memo[n]

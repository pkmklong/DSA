"""Search algorithms"""
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

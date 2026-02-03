# LeetCode Notes

- [LeetCode Notes](#leetcode-notes)
  - [General](#general)
    - [In place sort](#in-place-sort)
    - [Use hash map](#use-hash-map)
    - [Count same element](#count-same-element)
    - [copy entire list](#copy-entire-list)
    - [create an list with length n](#create-an-list-with-length-n)
    - [initial value for loop](#initial-value-for-loop)
    - [Hash map in python](#hash-map-in-python)
    - [Divide and Mod](#divide-and-mod)
    - [functions](#functions)
    - [zip and get in dictionary](#zip-and-get-in-dictionary)
  - [Questions](#questions)
    - [169. Majority Element](#169-majority-element)
    - [189. Rotate Array](#189-rotate-array)

## General

### In place sort

```python
nums.sort()
```

### Use hash map

```python
counts = collections.counter(nums)
return max(counts.keys(), key=counts.get)
```

>**Note**
For in place changes, start from the end

### Count same element

```ruby
count = sum(1 for elem in nums if elem == num)
```

### copy entire list

```python
nums[:] = a
```

### create an list with length n

```python
a = [0] * n
```

### initial value for loop

```python
min_price = float("inf")
```

### Hash map in python

```python
hashmap = {}
### check if a key in it
if "val" in hashmap
### get the value for a key
value = hashmap.get("key", 0) # return 0 if not found
### loop
for key, value in hashmap.items()
### search by value
key = next((k for k, v in hashmap.items() if v == 42), None)
### delete by key
del hashmap["key"]
hashmap.pop("key", None)
### delete all
hashmap.clear()
```

### Divide and Mod

```python
count, num = divmod(num, value)
```

```python
        result = []
        for dig, sym in digits:
            if num == 0:
                break
            count, num = divmod(num, dig)
            result.append(sym * count)
        return "".join(result)
```

### functions

```python
reversed()
floor()
ceil()
isalnum()
isalpha()
find()
filtered_chars = filter(lambda ch: ch.isalnum(), s) # string
lowercase_filtered_chars = map(lambda ch: ch.lower(), filtered_chars) # list
filtered_chars_list = list(lowercase_filtered_chars) # no need

```

### zip and get in dictionary

get return None if not in dictionary, or you can specify default value e.g. get(key, 0)

```python
for p, s in zip(pattern, sentence):
    if p not in pp and s not in ss:
        pp[p] = s
        ss[s] = p
    else:
        if pp.get(p) != s or ss.get(s) != p:
            return False

```

### Array sort and equal

```python
    char[] str1 = s.toCharArray();
    char[] str2 = t.toCharArray();
    Arrays.sort(str1);
    Arrays.sort(str2);
    return Arrays.equals(str1, str2);
```

### hashmaps

```python
l = collections.default(list)
i = collections.default(int) # default is 0
s = collections.default(set)
list(l.values())
```

### get each digit of a number

```python
digits = [ int(d) for d in str(n)]
```

### remove element from hashmap

```python
my_dict.pop(key)
# pop is safe and optional default
```

## Questions

### [169. Majority Element](https://leetcode.com/problems/majority-element)

- method 1: Bit Manipulation

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        n = len(nums)
        majority_element = 0

        bit = 1
        for i in range(31):
            # Count how many numbers have the current bit set.
            bit_count = sum(bool(num & bit) for num in nums)

            # If this bit is present in more than n / 2 elements
            # then it must be set in the majority element.
            if bit_count > n // 2:
                majority_element += bit

            # Shift bit to the left one space. i.e. '00100' << 1 = '01000'
            bit = bit << 1

        # In python 1 << 31 will automatically be considered as positive value
        # so we will count how many numbers are negative to determine if
        # the majority element is negative.
        is_negative = sum(num < 0 for num in nums) > (n // 2)

        # When evaluating a 32-bit signed integer, the values of the 1st through
        # 31st bits are added to the total while the value of the 32nd bit is
        # subtracted from the total. This is because the 32nd bit is responsible
        # for signifying if the number is positive or negative.
        if is_negative:
            majority_element -= bit

        return majority_element
```

- Boyer-Moore Voting Algorithm
  
```python
class Solution:
    def majorityElement(self, nums):
        count = 0
        candidate = None

        for num in nums:
            if count == 0:
                candidate = num
            count += 1 if num == candidate else -1

        return candidate
```

### [189. Rotate Array](https://leetcode.com/problems/rotate-array/description/)

- Approach 4: Using Reverse

```python
class Solution:
    def reverse(self, nums: list, start: int, end: int) -> None:
        while start < end:
            nums[start], nums[end] = nums[end], nums[start]
            start, end = start + 1, end - 1

    def rotate(self, nums: List[int], k: int) -> None:
        n = len(nums)
        k %= n

        self.reverse(nums, 0, n - 1)
        self.reverse(nums, 0, k - 1)
        self.reverse(nums, k, n - 1)
```

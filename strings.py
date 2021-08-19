"""Basic  string manipulation algorithms"""

def reverse_integer(n):
    s = str(n)
    if s[0] == "-":
        return int("-" + s[:0:-1])
    else:
        return int(s[::-1])


def ave_word_length(s):
    for i in "!?',;:.":
        s = s.replace(i, "")
    words = s.split(" ")
    return round(sum(len(w) for w in words), 2)

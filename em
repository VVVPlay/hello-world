a = input()
def is_em(a):
    if a[0] != "." and a[0] != "@":
        if a[-4] == ".":
            b = a[:-4]
            if b[-1] != "@":

                return True
            else:
                return False
    else:
        return False

res = is_em(a)
print(res)

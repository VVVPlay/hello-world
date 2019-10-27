# hello-world
just calculator
class Calculator:
    def pl(self, a, b):
        print(a + b)
    def mn(self, a, b):
        print(a - b)
    def umn(self, a, b):
            print(a * b)
    def dl(self, a, b):
        if  b != 0:
            print(a / b)
        else:
            print("Error!")
    def dln(self, a, b):
        if  b != 0:
            print(a // b)
        else:
            print("Error!")
    def dlst(self, a, b):
        if  b != 0:
            print(a % b)
        else:
            print("Error!")
    def st(self, a, b):
        print(a ** b)
    def sq(self, a):
        print(a ** .5)

Calc = Calculator()
a = int(input("a: "))
b = int(input("b: "))
Calc.pl(a, b)
Calc.mn(a, b)
Calc.umn(a, b)
Calc.dl(a, b)
Calc.dln(a, b)
Calc.dlst(a, b)
Calc.st(a, b)
Calc.sq(a)

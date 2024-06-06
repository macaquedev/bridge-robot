#names = {
#    0: 4, 1: 3, 2: 3, 3: 5, 4: 4, 5: 4, 6: 3, 7: 5, 8: 5, 9: 4,
#    10: 3, 11: 6, 12: 6, 13: 8, 14: 8, 15: 7, 16: 7, 17: 9, 18: 8, 19: 8,
#    20: 6, 30: 6, 40: 5, 50: 5, 60: 5, 70: 7, 80: 6, 90: 6,
#    100: 7, 1000: 8, 1000000: 7
#}
#
#def calc(n):
#    if n < 21:
#        return names[n]
#    elif n < 100:
#        return names[10*(n//10)] + names[n%10]
#    elif n < 1000:
#        return names[100] + names[n//100] + calc(n % 100)
#    elif n < 1000000:
#        return names[1000] + calc(n//1000) + calc(n%1000)
#    else:
#        return names[1000000] + calc(n//1000000) + calc(n%1000000)
#
#def solve(n):
#    for i in range(0, 1000000):
#        if n == 4:
#            return i
#
#        n = calc(n)
#    
#n, k = [int(i) for i in input().strip().split()]
#for i in range(n):
#    if solve(i) == k:
#        print(i)
#        break
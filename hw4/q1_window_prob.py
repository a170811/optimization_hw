from golden_search import golden_search

pi = 3.1415926

my_func = lambda r, k: ( -(0.5*pi + 2)* r**2 + 10*r )*k
min_func = lambda r: my_func(r, -1)
opt_r = golden_search(min_func, [0, 10])

print('The optimal solution of radius(r) is {:.3f}'.format(opt_r))


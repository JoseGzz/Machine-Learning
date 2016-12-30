import matplotlib.pyplot as plt

population_ages = [22,55,62,45,21,22,34,42,42,4,99,102,110,120,121,122,120,111,115,112,80,75,65,54,44,43,42,48]
#ids = [x for x in range(len(population_ages))]
bins = [0,10,20,30,40,50,60,70,80,90,100,110,120,130]

plt.hist(population_ages, bins, histtype='bar', rwidth=0.8)

x = [2,4,6,8,10]
y = [6,7,8,2,4]

x2 = [1,3,5,7,9]
y2 = [7,8,2,4,2]

plt.bar(x,y, label='bars1', color='r')
plt.bar(x2,y2, label='bars2', color='c')

plt.xlabel('x')
plt.ylabel('y')
plt.title('TITLE')
plt.legend()
plt.show()

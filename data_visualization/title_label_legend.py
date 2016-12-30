import matplotlib.pyplot as plt

x = [1,2,3]
y = [5,7,4]

plt.plot(x, y, label='first line')
plt.xlabel('x vals')
plt.ylabel('y vals')
plt.legend()
plt.title('the title')
plt.show()

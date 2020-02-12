import numpy as np
spacing = np.linspace(0,1,num=101)
n = 10
x = [1,1,1,1,1,1,0,0,0,0]
x_10 = [1,1,1,1,1,0,0,0,0,0]
x_100 = [0] * 100
for i in range(60):
    x_100[i] = 1

x_5 = [1,1,1,0,0]
likelihoods = []
lh_100 = []
lh_10 = []
lh_5 = []
for theta in spacing:
    likelihood = 1
    for x_i in x_100:
        likelihood *= ((theta ** x_i) * ((1 - theta) ** (1 - x_i)))
    lh_100.append(likelihood)

    likelihood = 1
    for x_i in x_10:
        likelihood *= ((theta ** x_i) * ((1 - theta) ** (1 - x_i)))
    lh_10.append(likelihood)

    likelihood = 1
    for x_i in x_5:
        likelihood *= ((theta ** x_i) * ((1 - theta) ** (1 - x_i)))
    lh_5.append(likelihood)
    likelihood = 1

    for x_i in x:
        likelihood *= ((theta ** x_i) * ((1-theta) ** (1-x_i)))
    likelihoods.append(likelihood)



import matplotlib.pyplot as plt
plt.subplot(3,1,1)
plt.plot(spacing, lh_100)
plt.xlabel(r'$\theta$')
plt.ylabel(r'$L(\theta)$ for n=100')
plt.subplot(3,1,2)
plt.plot(spacing, lh_10)
plt.xlabel(r'$\theta$')
plt.ylabel(r'$L(\theta)$ for n=10')
plt.subplot(3,1,3)
plt.plot(spacing, lh_5)
plt.xlabel(r'$\theta$')
plt.ylabel(r'$L(\theta)$ for n=5')
#plt.annotate(r'$\hat{\theta}_{MLE}$',
#            xy=(0.6, 0),
#            xytext=(0.6, 0.0002),
#            arrowprops = dict(facecolor='black', shrink=0.05))
plt.show()
# clustering-with-PSO-and-neural-nets
# clustering using particle swarm optimization and feed forward neural networks for unsupervised learning
from tkinter import *
import time
from itertools import permutations
import pandas as pd
import random
import numpy as np

print("creating the probability converter")


def prob_convert(x, logic=False):
    if logic is True:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


print("creating radial bias")


def radial_bias(centroid_number, particle_position, data):
    fit_s = 0
    fits = 0
    for m in range(centroid_number):
        for k in range(len(data)):
            fit = abs(particle_position[m] - data[k])
            fits = fits + fit
        fit_s = fit_s + fits
    return fit_s


name = list(permutations("abc"))
names = dict()

for i in range(len(name)):
    names[i] = name[i]

returns = dict()
purchase = dict()
invoice = dict()
print(len(names))
lena = len(names)

print("initializing data")

for i in range(len(names)):
    purchase[i] = random.randint(0, 1000)
    returns[i] = random.randint(0, 1000)
    invoice[i] = ["".join(names[i]), purchase[i], returns[i]]

df = pd.DataFrame([invoice[k] for k in range(len(names))], columns=["names", "purchases", "returns"])
print(df)

print("converting the inputs to matrix forms")

v_pur = dict()
for i in range(lena):
    v_pur[i] = str("{0:09b}".format(purchase[i]))

purchase_matrix = np.array(np.zeros((lena, 9)))

for j in range(lena):
    for k in range(9):
        purchase_matrix[j][k] = v_pur[j][k]

v_ret = dict()
for i in range(len(names)):
    v_ret[i] = str("{0:09b}".format(returns[i]))

returns_matrix = np.array(np.zeros((lena, 9)))

for i in range(len(names)):
    for k in range(9):
        returns_matrix[i][k] = v_ret[i][k]

print(purchase_matrix, returns_matrix)

print("establishing constants")

w = 0.9
c1 = 0.3
c2 = 0.5
r = random.uniform(0, 1)
fitness_p = dict()
fitness_r = dict()
particle_position_r = dict()
particle_position_p = dict()
velocity_p = dict()
velocity_r = dict()
pp_best = dict()
pr_best = dict()

print(w, c1, c2)
print("initializing the particles")

for i in range(100):
    particle_position_p[i] = dict()
    particle_position_r[i] = dict()
    velocity_p[i] = dict()
    velocity_r[i] = dict()
    for j in range(7):
        particle_position_p[i][j] = random.randint(0, 1000)
        particle_position_r[i][j] = random.randint(0, 1000)
        velocity_p[i][j] = 0
        velocity_r[i][j] = 0

    fitness_p[i] = radial_bias(7, particle_position_p[i], purchase)
    fitness_r[i] = radial_bias(7, particle_position_r[i], returns)

print("starting the swarm")
new_particle_position_p = dict()
new_particle_position_r = dict()

new_velocity_p = dict()
new_velocity_r = dict()

for k in range(500):

    new_velocity_p[k] = dict()
    new_velocity_r[k] = dict()

    for i in range(100):

        new_velocity_p[k][i] = dict()
        new_velocity_r[k][i] = dict()

        for j in range(7):
            new_particle_position_p[j] = random.uniform(0, 1000)

        new_fitness_p = radial_bias(7, new_particle_position_p, purchase)

        if new_fitness_p < fitness_p[i]:
            pp_best[i] = new_particle_position_p
        else:
            pp_best[i] = particle_position_p[i]

        for l in range(7):
            new_particle_position_r[l] = random.uniform(0, 1000)

        new_fitness_r = radial_bias(7, new_particle_position_r, returns)

        if new_fitness_r < fitness_r[i]:
            pr_best[i] = new_particle_position_r
        else:
            pr_best[i] = particle_position_r[i]

    vp = list(fitness_p.values())
    kp = list(fitness_p.keys())
    vr = list(fitness_r.values())
    kr = list(fitness_r.keys())
    gp_best_key = kp[vp.index(min(vp))]
    gr_best_key = kr[vr.index(min(vr))]
    gp_best = particle_position_p[gp_best_key]
    gr_best = particle_position_r[gr_best_key]

    for i in range(100):
        fitness_p[i] = radial_bias(7, particle_position_p[i], purchase)
        fitness_r[i] = radial_bias(7, particle_position_r[i], returns)
        for j in range(7):
            new_velocity_p[k][i][j] = w * velocity_p[i][j] * r + c1 * (pp_best[i][j] - particle_position_p[i][j]) * r + c2 * (
                gp_best[j] - particle_position_p[i][j]) * r
            new_velocity_r[k][i][j] = w * velocity_r[i][j] * r + c1 * (pr_best[i][j] - particle_position_r[i][j]) * r + c2 * (
                gr_best[j] - particle_position_r[i][j]) * r
            particle_position_p[i][j] = particle_position_p[i][j] + new_velocity_p[k][i][j]
            particle_position_r[i][j] = particle_position_r[i][j] + new_velocity_r[k][i][j]

print("initializing the display screen")

tk = Tk()
canvas = Canvas(tk, width=1000, height=1000)
tk.title("particle swarm optimization using radial bias function for clustering of purchase and returns")
canvas.pack()

print("creating the display particles")

particles = dict()

for j in range(100):
    particles[j] = dict()
    for k in range(7):
        particles[j][k] = canvas.create_oval(particle_position_p[j][k], particle_position_r[j][k],
                                             particle_position_p[j][k] + 10, particle_position_r[j][k] + 10, fill="green")
print("starting the display")

for i in range(500):
    for j in range(100):
        for k in range(7):
            canvas.move(particles[j][k], new_velocity_p[i][j][k], new_velocity_r[i][j][k])
            tk.update()
            time.sleep(0.01)

tk.mainloop()

print("the job is done")
print("the centroids")
print(particle_position_p)
print(particle_position_r)

print("the clusters")
cluster_p = dict()
cluster_r = dict()
for i in range(7):
    cluster_p[i] = dict()
    cluster_r[i] = dict()

counts_p = dict()
counts_r = dict()
for i in range(7):
    counts_p[i] = 0
    counts_r[i] = 0

cl_p = dict()

for p in range(len(purchase)):
    cl_p[p] = dict()
    for i in range(7):
        cl_p[p][i] = abs(particle_position_p[1][i] - purchase[p])

for p in range(len(purchase)):
    cl_vp = list(cl_p[p].values())
    cl_kp = list(cl_p[p].keys())
    cl_min_kp = cl_kp[cl_vp.index(min(cl_vp))]
    cluster_p[cl_min_kp][counts_p[cl_min_kp]] = purchase[p]
    counts_p[cl_min_kp] += 1

cl_r = dict()

for r in range(len(returns)):
    cl_r[r] = dict()
    for i in range(7):
        cl_r[r][i] = abs(particle_position_r[1][i] - returns[r])

for r in range(len(returns)):
    cl_vr = list(cl_r[r].values())
    cl_kr = list(cl_r[r].keys())
    cl_min_kr = cl_kr[cl_vr.index(min(cl_vr))]
    cluster_r[cl_min_kr][counts_r[cl_min_kr]] = returns[r]
    counts_r[cl_min_kr] += 1

print(cluster_p)
print(cluster_r)
mat_p = dict()
mat_r = dict()
lp = dict()
lr = dict()

print("making all the clusters matrices")

vp = dict()
vr = dict()

for i in range(7):
    vp[i] = dict()
    vr[i] = dict()
    for j in range(len(cluster_p[i])):
        vp[i][j] = str("{0:09b}".format(cluster_p[i][j]))
    for j in range(len(cluster_r[i])):
        vr[i][j] = str("{0:09b}".format(cluster_r[i][j]))

for i in range(7):
    lp[i] = len(cluster_p[i])
    mat_p[i] = np.array(np.zeros((lp[i], 9)))
    for l in range(lp[i]):
        for k in range(9):
            mat_p[i][l][k] = vp[i][l][k]
    lr[i] = len(cluster_r[i])
    mat_r[i] = np.array(np.zeros((lr[i], 9)))
    for j in range(lr[i]):
        for k in range(9):
            mat_r[i][j][k] = vr[i][j][k]

print(mat_p, mat_r)

print("creating neural networks for the clusters")

La0_p = dict()
La1_p = dict()
La2_p = dict()

sy0_p = dict()
sy1_p = dict()

La1Dp = dict()
La2Dp = dict()

La1Dr = dict()
La2Dr = dict()

La0_r = dict()
La1_r = dict()
La2_r = dict()

sy0_r = dict()
sy1_r = dict()

yp = dict()
yr = dict()

print("training the neural networks")

for i in range(7):

    xp = purchase_matrix
    yp[i] = mat_p[i]

    sy0_p[i] = 1.5 * np.random.random((len(purchase), len(purchase))) - 1.17
    sy1_p[i] = 1.5 * np.random.random((len(cluster_p[i]), len(purchase))) - 1.17

    for j in range(1000000):
        La0_p[i] = xp
        La1_p[i] = prob_convert(np.dot(sy0_p[i], La0_p[i]))
        La2_p[i] = prob_convert(np.dot(sy1_p[i], La1_p[i]))

        La2E = yp[i] - La2_p[i]
        if j % 200000 == 0:
            print("error: ", np.mean(np.abs(La2E)))

        La2Dp[i] = La2E * prob_convert(La2_p[i], logic=True)
        La1E = np.dot(sy1_p[i].T, La2Dp[i])
        La1Dp[i] = La1E * prob_convert(La1_p[i], logic=True)

        sy1_p[i] += np.dot(La2Dp[i], La1_p[i].T)
        sy0_p[i] += np.dot(La1Dp[i], La0_p[i].T)

    print("output after the training")
    print(La2_p[i])

for i in range(7):

    xr = returns_matrix
    yr[i] = mat_r[i]

    sy0_r[i] = 1.5 * np.random.random((len(returns), len(returns))) - 1.17
    sy1_r[i] = 1.5 * np.random.random((len(cluster_r[i]), len(returns))) - 1.17

    for j in range(1000000):
        La0_r[i] = xr
        La1_r[i] = prob_convert(np.dot(sy0_r[i], La0_r[i]))
        La2_r[i] = prob_convert(np.dot(sy1_r[i], La1_r[i]))

        La2E = yr[i] - La2_r[i]
        if j % 200000 == 0:
            print("error: ", np.mean(np.abs(La2E)))

        La2Dr[i] = La2E * prob_convert(La2_r[i], logic=True)
        La1E = np.dot(sy1_r[i].T, La2Dr[i])
        La1Dr[i] = La1E * prob_convert(La1_r[i], logic=True)

        sy1_r[i] += np.dot(La2Dr[i], La1_r[i].T)
        sy0_r[i] += np.dot(La1Dr[i], La0_r[i].T)

    print("output after the training")
    print(La2_r[i])


print("calling the neural networks")

for i in range(7):

    La0_p[i] = purchase_matrix
    La1_p[i] = prob_convert(np.dot(sy0_p[i], La0_p[i]))
    La2_p[i] = prob_convert(np.dot(sy1_p[i], La1_p[i]))

    print(La2_p[i])

    La0_r[i] = returns_matrix
    La1_r[i] = prob_convert(np.dot(sy0_r[i], La0_r[i]))
    La2_r[i] = prob_convert(np.dot(sy1_r[i], La1_r[i]))

    print(La2_r[i])


print("creating the fault tolerance networks")

La0_pf = dict()
La1_pf = dict()
La2_pf = dict()

sy0_pf = dict()
sy1_pf = dict()

La1Dpf = dict()
La2Dpf = dict()

La1Drf = dict()
La2Drf = dict()

La0_rf = dict()
La1_rf = dict()
La2_rf = dict()

sy0_rf = dict()
sy1_rf = dict()

ypf = dict()
yrf = dict()

print("training the fault tolerance neural networks")

for i in range(7):
    for q in range(7):
        La0_pf[i] = dict()
        La1_pf[i] = dict()
        La2_pf[i] = dict()

        sy0_pf[i] = dict()
        sy1_pf[i] = dict()

        La1Dpf[i] = dict()
        La2Dpf[i] = dict()

        ypf[i] = dict()
        xp = purchase_matrix
        ypf[i][q] = mat_p[i]

        sy0_pf[i][q] = 1.9 * np.random.random((len(purchase), len(purchase))) - 1.17
        sy1_pf[i][q] = 1.9 * np.random.random((len(cluster_p[i]), len(purchase))) - 1.17

        for j in range(1000000):
            La0_pf[i][q] = xp
            La1_pf[i][q] = prob_convert(np.dot(sy0_pf[i][q], La0_pf[i][q]))
            La2_pf[i][q] = prob_convert(np.dot(sy1_pf[i][q], La1_pf[i][q]))

            La2E = ypf[i][q] - La2_pf[i][q]
            if j % 200000 == 0:
                print("error: ", np.mean(np.abs(La2E)))

            La2Dpf[i][q] = La2E * prob_convert(La2_pf[i][q], logic=True)
            La1E = np.dot(sy1_pf[i][q].T, La2Dpf[i][q])
            La1Dpf[i][q] = La1E * prob_convert(La1_pf[i][q], logic=True)

            sy1_pf[i][q] += np.dot(La2Dpf[i][q], La1_pf[i][q].T)
            sy0_pf[i][q] += np.dot(La1Dpf[i][q], La0_pf[i][q].T)

        print("output after the training")
        print(La2_pf[i][q])

    for q in range(7):
        La1Drf[i] = dict()
        La2Drf[i] = dict()

        La0_rf[i] = dict()
        La1_rf[i] = dict()
        La2_rf[i] = dict()

        sy0_rf[i] = dict()
        sy1_rf[i] = dict()

        yrf[i] = dict()
        xr = returns_matrix
        yrf[i][q] = mat_r[i]

        sy0_rf[i][q] = 1.5 * np.random.random((len(returns), len(returns))) - 1.17
        sy1_rf[i][q] = 1.5 * np.random.random((len(cluster_r[i]), len(returns))) - 1.17

        for j in range(1000000):
            La0_rf[i][q] = xr
            La1_rf[i][q] = prob_convert(np.dot(sy0_rf[i][q], La0_rf[i][q]))
            La2_rf[i][q] = prob_convert(np.dot(sy1_rf[i][q], La1_rf[i][q]))

            La2E = yrf[i][q] - La2_rf[i][q]
            if j % 200000 == 0:
                print("error: ", np.mean(np.abs(La2E)))

            La2Drf[i][q] = La2E * prob_convert(La2_rf[i][q], logic=True)
            La1E = np.dot(sy1_rf[i][q].T, La2Drf[i][q])
            La1Drf[i][q] = La1E * prob_convert(La1_rf[i][q], logic=True)

            sy1_rf[i][q] += np.dot(La2Drf[i][q], La1_rf[i][q].T)
            sy0_rf[i][q] += np.dot(La1Drf[i][q], La0_rf[i][q].T)

        print("output after the training")
        print(La2_rf[i][q])

print("the fault tolerance is now initialized")

for i in range(7):
    if La2_p[i] != yp[i]:
        for q in range(7):
            La0_pf[i][q] = purchase_matrix
            La1_pf[i][q] = prob_convert(np.dot(sy0_pf[i][q], La0_pf[i][q]))
            La2_pf[i][q] = prob_convert(np.dot(sy1_pf[i][q], La1_pf[i][q]))
        if La2_pf[i] == ypf[i]:
            print(i)
            break

    if La2_r[i] != yr[i]:
        for q in range(7):
            La0_rf[i][q] = returns_matrix
            La1_rf[i][q] = prob_convert(np.dot(sy0_rf[i][q], La0_rf[i][q]))
            La2_rf[i][q] = prob_convert(np.dot(sy1_rf[i][q], La1_rf[i][q]))
        if La2_rf[i] == yrf[i]:
            print(i)
            break

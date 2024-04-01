import sys
import math
import numpy as np
import matplotlib.pyplot as plt


# problem instance
def policeman_and_burglar_matrix(n, th=0.8, seed=0):
    np.random.seed(seed)
    w = np.abs(np.random.randn(n))
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            C[i][j] = 1 - np.exp(-th * abs(i - j))
    return w * C


# projection onto simplex
def projection(y, a=1):
    d = len(y)
    if np.sum(y) == a and np.all(y >= 0):
        tmp = y
    else:
        u = np.sort(y)[::-1]
    cumsum = 0
    for i in range(d):
        if cumsum + u[i] < u[i] * (i + 1) + a:
            cumsum += u[i]
            tau = (cumsum - a) / (i + 1)
        else:
            break
    tmp = np.zeros(d)
    for i in range(d):
        tmp[i] = max(y[i] - tau, 0)


# Extragradient update (full)
def EG_update(A, x, y, eta):
    x_ = projection(x - eta * A.T.dot(y))
    y_ = projection(y + eta * A.dot(x))
    grad_x = A.T.dot(y_)
    grad_y = A.dot(x_)
    x_tmp = x - eta * grad_x
    y_tmp = y + eta * grad_y
    x = projection(x_tmp)
    y = projection(y_tmp)
    res = np.sqrt(np.linalg.norm(x_ - projection(x_ - A.T.dot(y_)))**2 + np.linalg.norm(y_ - projection(y_ + A.dot(x_)))**2)
    return x, y, res

# EG-Halpern update (full)
def ex_Halpern_update(A, x, y, x_0, y_0, lam, eta):
    x_ = projection(lam * x_0 + x - lam * x - eta * A.T.dot(y))
    y_ = projection(lam * y_0 + y - lam * y + eta * A.dot(x))
    x_tmp = lam * x_0 + x - lam * x - eta * A.T.dot(y_)
    y_tmp = lam * y_0 + y - lam * y + eta * A.dot(x_)
    x = projection(x_tmp)
    y = projection(y_tmp)
    return x, y

# Halpern update (with PAGE)
def Halpern_mini_update(A, x, y, x_0, y_0, lam, eta, batch_size, gx, gy, x_prev, y_prev):
    n = A.shape[0]
    if batch_size == n:
        batch = np.arange(n)
        gx, gy = A.T.dot(y), -A.dot(x)
    else:
        batch = np.random.choice(n, batch_size, replace=False)
        gx += A[batch,].T.dot(y[batch] - y_prev[batch]) * n / batch_size
        gy -= A[:, batch].dot(x[batch] - x_prev[batch]) * n / batch_size
    x_tmp = lam * x_0 + x - lam * x - eta * gx
    y_tmp = lam * y_0 + y - lam * y - eta * gy
    return projection(x_tmp), projection(y_tmp), gx, gy, x, y

# FoRB: the subsolver for Algorithm 2
def subsolver_FoRB(A, x, y, M, c, cnt, p, alpha, gamma):
    m, n = A.shape[0], A.shape[1]

    vx, vy = x, y
    wx, wy = x, y
    wx_prev, wy_prev = x, y
    Ax, Ay = -c * A.dot(wx) + wy - y, c * A.T.dot(wy) + wx - x
    toss = np.random.binomial(1, p, M)
    b_list = np.random.choice(n, size=M, replace=True)
    for k in range(M):
        idx = b_list[k]
        vxx = alpha * vx + wx - alpha * wx
        vyy = alpha * vy + wy - alpha * wy
        vx_tmp = vxx - gamma * (Ay + (c * n * A[idx, ].squeeze() * (vy[idx] - wy_prev[idx]) + vx - wx_prev))
        vy_tmp = vyy - gamma * (Ax - (c * n * A[:, idx].squeeze() * (vx[idx] - wx_prev[idx]) + vy - wy_prev))
        vx = projection(vx_tmp)
        vy = projection(vy_tmp)
        cnt += 2

        wx_prev, wy_prev = wx, wy
        if toss[k] == 1:
            wx, wy = x, y
            Ax, Ay = -c * A.dot(wx) + wy - y, c * A.T.dot(wy) + wx - x
            cnt += n

    return vx, vy, cnt




if __name__ == "__main__":
    # Initialize the problem
    n, m = 500, 500
    A = policeman_and_burglar_matrix(n, seed=1)
    op_norm = np.linalg.norm(A, ord=2)
    l2_norm = np.linalg.norm(A, ord='fro')
    print(op_norm, l2_norm)


    # deterministic EG
    x = np.ones(n) / n
    y = np.ones(m) / m

    eta = 5 / op_norm
    N = 2500
    res_EG = list()
    res = np.sqrt(np.linalg.norm(x - projection(x - A.T.dot(y)))**2 + np.linalg.norm(y - projection(y + A.dot(x)))**2)
    res_EG.append(res)
    for i in range(N):
        x, y, res = EG_update(A, x, y, eta)
        res_EG.append(res)
        if i % 100 == 0:
            print(np.log10(res))
    
    res_EG = np.array(res_EG)
    np.save("./EG_res_pb.npy", res_EG)


    # variance-reduced EG
    x = np.ones(n) / n
    y = np.ones(m) / m
    p = 1 / n
    alpha = 1 - p
    gamma = 3 * np.sqrt(p) / l2_norm

    res_vrEG = list()
    cnt_vrEG = list()
    cnt = 0
    wx, wy = x, y
    Ax, Ay = -A.dot(wx), A.T.dot(wy)
    res = np.sqrt(np.linalg.norm(wx - projection(wx - Ay))**2 + np.linalg.norm(wy - projection(wy - Ax))**2)
    res_vrEG.append(res)
    cnt_vrEG.append(cnt)
    print(np.log10(res), cnt / n)
    N = 5000 * n
    cnt += n
    toss = np.random.binomial(1, p, int(N * 2))
    b_list = np.random.choice(n, int(N * 2), replace=True)
    i = 0
    while cnt < N:
        idx = b_list[i]
        xx = alpha * x + wx - alpha * wx
        yy = alpha * y + wy - alpha * wy
        x_ = projection(xx- gamma * Ay)
        y_ = projection(yy - gamma * Ax)
        x_tmp = xx - gamma * (Ay + A[idx, ].squeeze() * (y_[idx] - wy[idx]) * n)
        y_tmp = yy - gamma * (Ax - A[:, idx].squeeze() * (x_[idx] - wx[idx]) * n)
        x = projection(x_tmp)
        y = projection(y_tmp)
        cnt += 2

        if toss[i] == 1:
            wx, wy = x, y
            Ax, Ay = -A.dot(wx), A.T.dot(wy)
            res = np.sqrt(np.linalg.norm(wx - projection(wx - Ay))**2 + np.linalg.norm(wy - projection(wy - Ax))**2)
            res_vrEG.append(res)
            cnt_vrEG.append(cnt)
            cnt += n
        if i % (500 * n) == 0:
            print(np.log10(res), cnt / n)

        i += 1
    
    res_vrEG = np.array(res_vrEG)
    np.save("./vrEG_res_pb.npy", res_vrEG)
    cnt_vrEG = np.array(cnt_vrEG)
    np.save("./vrEG_cnt_pb.npy", cnt_vrEG)


    # EAG
    x_0 = np.ones(n) / n
    y_0 = np.ones(m) / m
    x, y = x_0, y_0

    eta = 5 / (op_norm)
    N = 2500
    res_Halpern = list()
    res = np.sqrt(np.linalg.norm(x - projection(x - A.T.dot(y)))**2 + np.linalg.norm(y - projection(y + A.dot(x)))**2)
    res_Halpern.append(res)
    for i in range(N):
        lam = 1 / (i + 2)
        x, y = ex_Halpern_update(A, x, y, x_0, y_0, lam, eta)
        res = np.sqrt(np.linalg.norm(x - projection(x - A.T.dot(y)))**2 + np.linalg.norm(y - projection(y + A.dot(x)))**2)
        res_Halpern.append(res)
        if i % 1000 == 0:
            print(np.log10(res))
    
    res_Halpern = np.array(res_Halpern)
    np.save("./EAG_res_PB.npy", res_Halpern)


    # Algorithm 1 in our paper
    x_0 = np.ones(n) / n
    y_0 = np.ones(m) / m
    x, y = x_0, y_0
    x_prev, y_prev = x, y

    eta = 5 / (l2_norm)
    batch_size = int(np.sqrt(n))
    N = 5000 * n
    gx = np.zeros(n)
    gy = np.zeros(m)
    i = 0
    cnt = 0
    res_cocoHalpern = list()
    res_cnt = list()
    res = np.sqrt(np.linalg.norm(x - projection(x - A.T.dot(y)))**2 + np.linalg.norm(y - projection(y + A.dot(x)))**2)
    res_cocoHalpern.append(res)
    res_cnt.append(cnt)

    while cnt <= N:
        lam = 2 / (i + 4)
        if i <= 1:
            toss = 1
        elif i - 1 <= np.sqrt(n):
            p = 4 / (i + 4)
            toss = np.random.binomial(1, p)
        else:
            p = 4 / (np.sqrt(n) + 5)
            toss = np.random.binomial(1, p)
        if toss == 1:
            x, y, gx, gy, x_prev, y_prev = Halpern_mini_update(A, x, y, x_0, y_0, lam, eta, n, gx, gy, x_prev, y_prev)
            cnt += n
        else:
            x, y, gx, gy, x_prev, y_prev = Halpern_mini_update(A, x, y, x_0, y_0, lam, eta, batch_size, gx, gy, x_prev, y_prev)
            cnt += batch_size * 2
        if i % 10 == 0:
            res = np.sqrt(np.linalg.norm(x - projection(x - A.T.dot(y)))**2 + np.linalg.norm(y - projection(y + A.dot(x)))**2)
            res_cocoHalpern.append(res)
            res_cnt.append(cnt)
        if i % 500 == 0 or cnt > N:
            print(np.log10(res))

        i += 1
    
    res_cocoHalpern = np.array(res_cocoHalpern)
    np.save("./cocoHalpern_res_pb.npy", res_cocoHalpern)
    res_cnt = np.array(res_cnt)
    np.save("./cnt_res_pb.npy", res_cnt)

    # Algorithm 2 in our paper
    p = 1 / n
    alpha = 1 - p

    x_0 = np.ones(n) / n
    y_0 = np.ones(m) / m
    x, y = x_0, y_0

    N = 5000 * n
    c = np.sqrt(n) / l2_norm
    gamma = np.sqrt(p * (1 - p)) / (np.sqrt(n) + 1) * 5

    i = 0
    cnt = 0
    res_inexactHalpern = list()
    res_cnt_inexact = list()
    res = np.sqrt(np.linalg.norm(x - projection(x - A.T.dot(y)))**2 + np.linalg.norm(y - projection(y + A.dot(x)))**2)
    res_inexactHalpern.append(res)
    res_cnt_inexact.append(cnt)
    while cnt <= N:
        lam = 1 / (i + 2)
        M = int(0.05 * n * np.log(i + 2))
        resol_x, resol_y, cnt = subsolver_FoRB(A, x, y, M, c, cnt, p, alpha, gamma)
        x = lam * x_0 + resol_x - lam * resol_x
        y = lam * y_0 + resol_y - lam * resol_y
        res = np.sqrt(np.linalg.norm(x - projection(x - A.T.dot(y)))**2 + np.linalg.norm(y - projection(y + A.dot(x)))**2)
        res_inexactHalpern.append(res)
        res_cnt_inexact.append(cnt)
        
        if i % 1000 == 0:
            print(np.log10(res))
        
        i += 1
    
    res_inexactHalpern = np.array(res_inexactHalpern)
    np.save("./inexactHalpern_res_pb.npy", res_inexactHalpern)
    res_cnt_inexact = np.array(res_cnt_inexact)
    np.save("./cnt_inexact_res_pb.npy", res_cnt_inexact)


    # Plot
    plt.rc('text', usetex=True)
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    plt.rcParams["figure.figsize"] = (12,8)
    markers = ["v","^","<",">","o","s","p","P","*"]

    plt.plot((np.arange(len(res_EG)) * 2 + 1), np.array(res_EG), "-", color="orangered", linewidth=5.0, label="EG", marker=markers[0], markersize=20, markevery=400, fillstyle="none", markeredgewidth=3)
    plt.plot((np.array(cnt_vrEG) / n), np.array(res_vrEG), "-", color="yellowgreen", linewidth=5.0, label="VR-EG", marker=markers[1], markersize=20, markevery=200, fillstyle="none", markeredgewidth=3)
    plt.plot((np.arange(len(res_Halpern)) * 2 + 1), np.array(res_Halpern), "-", color="violet", linewidth=5.0, label="EAG", marker=markers[5], markersize=20, markevery=400, fillstyle="none", markeredgewidth=3)
    plt.plot((np.array(res_cnt) / (n)), np.array(res_cocoHalpern), "-", color="orange", linewidth=8.0, label="Algorithm 1", marker=markers[2], markersize=20, markevery=800, fillstyle="none", markeredgewidth=3)
    plt.plot((np.array(res_cnt_inexact) / (n)), np.array(res_inexactHalpern), "-", color="skyblue", linestyle="dashed", linewidth=5.0, label="Algorithm 2", marker=markers[3], markersize=20, markevery=800, fillstyle="none", markeredgewidth=3)

    plt.yscale("log")
    plt.xlabel("\# of epochs", fontsize=50)
    plt.ylabel("Operator norm", fontsize=50)
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=45)
    plt.ylim()
    plt.legend(loc='upper right', fontsize=25)
    plt.savefig("finite_halpern_PB.png", bbox_inches='tight')
    plt.show()
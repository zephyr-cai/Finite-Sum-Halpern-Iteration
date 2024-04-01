import sys
import math
import numpy as np
import matplotlib.pyplot as plt


# Compute the grad
def grad(x, y, H, A, b, h):
    gx = H.dot(x) - h - A.T.dot(y)
    gy = A.dot(x) - b
    return gx, gy

def grad_uniform(x, y, x_prev, y_prev, H, A, b, h, bs):
    n = A.shape[0]
    batch = np.random.choice(n, bs, replace=False)
    ggx = (H[:, batch].dot(x[batch] - x_prev[batch]) - A[batch,].T.dot(y[batch] - y_prev[batch])) * n / bs
    ggy = (A[:, batch].dot(x[batch] - x_prev[batch])) * n / bs
    return ggx, ggy

def grad_uniform_single(x, y, x_prev, y_prev, H, A, b, h, idx):
    n = A.shape[0]
    ggx = (H[:, idx].dot(x[idx] - x_prev[idx]) - A[idx,].T.dot(y[idx] - y_prev[idx])) * n
    ggy = (A[:, idx].dot(x[idx] - x_prev[idx])) * n
    return ggx, ggy

# FoRB: subsolver used in our Algorithm 2
def subsolver_FoRB(A, x, y, M, c, cnt, p, alpha, gamma):
    n = A.shape[0]
    vx, vy = x, y
    wx, wy = x, y
    wx_prev, wy_prev = x, y
    grad_wx, grad_wy = grad(x, y, H, A, b, h)
    cnt += n
    grad_wx = c * grad_wx + wx - x
    grad_wy = c * grad_wy + wy - y
    b_list = np.random.choice(n, size=M, replace=True)
    toss = np.random.binomial(1, p, M)
    for k in range(M):
        vxx = alpha * vx + wx - alpha * wx
        vyy = alpha * vy + wy - alpha * wy
        gradx, grady = grad_uniform_single(vx, vy, wx_prev, wy_prev, H, A, b, h, b_list[k])
        vx = vxx - gamma * (grad_wx + c * gradx + (vx - wx_prev))
        vy = vyy - gamma * (grad_wy + c * grady + (vy - wy_prev))
        cnt += 2

        wx_prev, wy_prev = wx, wy
        if toss[k] == 1:
            wx, wy = x, y
            grad_wx, grad_wy = grad(wx, wy, H, A, b, h)
            grad_wx = c * grad_wx + wx - x
            grad_wy = c * grad_wy + wy - y
            cnt += n

    return vx, vy, cnt



if __name__ == "__main__":
    # Initialize the problem
    n = 200
    A = np.zeros((n, n))
    for i in range(n):
        A[i][n - 1 - i] = 1
        if i < n - 1:
            A[i][n - 2 - i] = -1
    A /= 4
    H = 2 * A.T.dot(A)
    b = np.ones(n) / 4
    h = np.zeros(n)
    h[-1] = 1 / 4

    # deterministic EG
    x = np.ones(n) / n
    y = np.ones(n) / n

    eta = 0.5
    N = 500000
    res_EG = list()
    for i in range(N):
        grad_x, grad_y = grad(x, y, H, A, b, h)
        if i == 0:
            res = np.sqrt(np.linalg.norm(grad_x)**2 + np.linalg.norm(grad_y)**2)
            res_EG.append(res)
        x_ = x - eta * grad_x
        y_ = y - eta * grad_y
        gradx, grady = grad(x_, y_, H, A, b, h)
        x -= eta * gradx
        y -= eta * grady
        res = np.sqrt(np.linalg.norm(gradx)**2 + np.linalg.norm(grady)**2)
        res_EG.append(res)
        if i % 100000 == 0:
            print(np.log10(res), i)
    

    # variance-reduced EG
    x = np.ones(n) / n
    y = np.ones(n) / n

    N = n * 1e6
    res_vrEG = list()
    cnt_vrEG = list()

    wx, wy = x, y
    grad_wx, grad_wy = grad(x, y, H, A, b, h)
    cnt = 0
    res = np.sqrt(np.linalg.norm(grad_wx)**2 + np.linalg.norm(grad_wy)**2)
    print(np.log10(res), cnt / n)
    res_vrEG.append(res)
    cnt_vrEG.append(cnt)

    p = 1 / n
    alpha = 1 - p
    gamma = 0.01
    cnt += n
    i = 0
    toss = np.random.binomial(1, p, int(n * 1e6 * 2))
    b_list = np.random.choice(n, int(n * 1e6 * 2), replace=True)
    while cnt < N:
        xx = alpha * x + wx - alpha * wx
        yy = alpha * y + wy - alpha * wy
        x_ = xx - gamma * grad_wx
        y_ = yy - gamma * grad_wy
        gradx, grady = grad_uniform_single(x_, y_, wx, wy, H, A, b, h, b_list[i])
        x = xx - gamma * (grad_wx + gradx)
        y = yy - gamma * (grad_wy + grady)
        cnt += 1

        cnt_tmp = cnt / n
        if toss[i] == 1:
            wx, wy = x, y
            grad_wx, grad_wy = grad(wx, wy, H, A, b, h)
            res = np.sqrt(np.linalg.norm(grad_wx)**2 + np.linalg.norm(grad_wy)**2)
            res_vrEG.append(res)
            cnt_vrEG.append(cnt)
            cnt += n
        if i % (50000 * n) == 0:
            print(np.log10(res), cnt_tmp)
        i += 1
    
    res_vrEG = np.array(res_vrEG)
    np.save("./vrEG_res_qp.npy", res_vrEG)
    cnt_vrEG = np.array(cnt_vrEG)
    np.save("./vrEG_cnt_qp.npy", cnt_vrEG)


    # EAG
    x_0 = np.ones(n) / n
    y_0 = np.ones(n) / n
    x, y = x_0, y_0

    eta = 0.3
    N = 500000
    res_EAG = list()
    grad_x, grad_y = grad(x, y, H, A, b, h)
    res = np.sqrt(np.linalg.norm(grad_x)**2 + np.linalg.norm(grad_y)**2)
    res_EAG.append(res)
    for i in range(N):
        lam = 1 / (i + 2)
        x_ = lam * x_0 + x - lam * x - eta * grad_x
        y_ = lam * y_0 + y - lam * y - eta * grad_y
        gradx, grady = grad(x_, y_, H, A, b, h)
        x = lam * x_0 + x - lam * x - eta * gradx
        y = lam * y_0 + y - lam * y - eta * grady
        grad_x, grad_y = grad(x, y, H, A, b, h)
        res = np.sqrt(np.linalg.norm(grad_x)**2 + np.linalg.norm(grad_y)**2)
        res_EAG.append(res)
        if i % 100000 == 0:
            print(np.log10(res))
    

    # Algorithm 1 in our paper
    x_0 = np.ones(n) / n
    y_0 = np.ones(n) / n
    x, y = x_0, y_0
    x_prev, y_prev = x, y

    eta = 0.3
    batch_size = int(np.sqrt(n))
    N = 1000000 * n
    gx = np.zeros(n)
    gy = np.zeros(n)
    i = 0
    cnt = 0
    res_cocoHalpern = list()
    res_cnt = list()

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
            gx, gy = grad(x, y, H, A, b, h)
            if i == 0:
                res = np.sqrt(np.linalg.norm(gx)**2 + np.linalg.norm(gy)**2)
                res_cocoHalpern.append(res)
                res_cnt.append(cnt)
                x_prev, y_prev = x, y
            x = lam * x_0 + x - lam * x - eta * gx
            y = lam * y_0 + y - lam * y - eta * gy
            cnt += n
        else:
            gx_diff, gy_diff = grad_uniform(x, y, x_prev, y_prev, H, A, b, h, batch_size)
            gx += gx_diff
            gy += gy_diff
            x_prev, y_prev = x, y
            x = lam * x_0 + x - lam * x - eta * gx
            y = lam * y_0 + y - lam * y - eta * gy
            cnt += batch_size * 2
        if i % n == 0 or cnt > N:
            gradx, grady = grad(x, y, H, A, b, h)
            res = np.sqrt(np.linalg.norm(gradx)**2 + np.linalg.norm(grady)**2)
            res_cocoHalpern.append(res)
            res_cnt.append(cnt)
        if i % 100000 == 0 or cnt > N:
            print(np.log10(res))
        
        i += 1
    

    # Algorithm 2 in our paper
    p = 1 / n
    alpha = 1 - p

    x_0 = np.ones(n) / n
    y_0 = np.ones(n) / n
    x, y = x_0, y_0

    N = n * 1e6
    c = np.sqrt(n)
    gamma = 0.001

    i = 0
    cnt = 0
    res_inexactHalpern = list()
    res_cnt_inexact = list()
    grad_x0, grad_y0 = grad(x_0, y_0, H, A, b, h)
    res = np.sqrt(np.linalg.norm(grad_x0)**2 + np.linalg.norm(grad_y0)**2)
    print(np.log10(res), cnt / n)
    res_inexactHalpern.append(res)
    res_cnt_inexact.append(cnt)
    while cnt <= N:
        lam = 1 / (i + 2)
        M = int(0.05 * n * np.log(i + 2))
        resol_x, resol_y, cnt = subsolver_FoRB(A, x, y, M, c, cnt, p, alpha, gamma)
        x = lam * x_0 + resol_x - lam * resol_x
        y = lam * y_0 + resol_y - lam * resol_y
        grad_x, grad_y = grad(x, y, H, A, b, h)
        res = np.sqrt(np.linalg.norm(grad_x)**2 + np.linalg.norm(grad_y)**2)
        res_inexactHalpern.append(res)
        res_cnt_inexact.append(cnt)
        if i % 5000 == 0:
            print(np.log10(res), cnt / n)
        i += 1
    
    res_inexactHalpern = np.array(res_inexactHalpern)
    np.save("./inexactHalpern_res_qp.npy", res_inexactHalpern)
    res_cnt_inexact = np.array(res_cnt_inexact)
    np.save("./inexactHalpern_cnt_qp.npy", res_cnt_inexact)

    
    # plot
    plt.rc('text', usetex=True)
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    plt.rcParams["figure.figsize"] = (12,8)
    markers = ["v","^","<",">","o","s","p","P","*"]

    len_1 = len(res_cocoHalpern)
    len_2 = len(res_inexactHalpern)
    len_3 = len(res_vrEG)
    idx_vrEG = np.arange(len_3, step=100)
    idx_coco = np.arange(len_1, step=10)
    idx_inexact = np.arange(len_2, step=2000)
    idx_EG = np.array([2 * i + 1 for i in range(len(res_EG) - 1)])
    idx_EG = np.insert(idx_EG, 0, 0)
    idx_EAG = np.array([2 * i + 2 for i in range(len(res_EAG) - 1)])
    idx_EAG = np.insert(idx_EAG, 0, 0)
    iddx_EAG = np.arange(len(res_EAG), step=2000)

    plt.plot(idx_EG, np.array(res_EG), "-", color="orangered", linewidth=5.0, label="EG", marker=markers[0], markersize=20, markevery=100000, fillstyle="none", markeredgewidth=3)
    plt.plot((np.array(cnt_vrEG) / n)[idx_vrEG], np.array(res_vrEG)[idx_vrEG], "-", color="yellowgreen", linewidth=5.0, label="VR-EG", marker=markers[1], markersize=20, markevery=1000, fillstyle="none", markeredgewidth=3)
    plt.plot(idx_EAG[iddx_EAG], np.array(res_EAG)[iddx_EAG], "-", color="violet", linewidth=5.0, label="EAG", marker=markers[5], markersize=20, markevery=25, fillstyle="none", markeredgewidth=3)
    plt.plot((np.array(res_cnt_inexact) / (n))[idx_inexact], np.array(res_inexactHalpern)[idx_inexact], "-", color="cadetblue", linewidth=5.0, label="Algorithm 2", marker=markers[3], markersize=20, markevery=10, fillstyle="none", markeredgewidth=3)
    plt.plot((np.array(res_cnt) / (n))[idx_coco], np.array(res_cocoHalpern)[idx_coco], "-", color="orange", linewidth=5.0, label="Algorithm 1", marker=markers[2], markersize=20, markevery=250, fillstyle="none", markeredgewidth=3)


    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    order = [0, 1, 2, 4, 3]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
    plt.yscale("log")
    # plt.xscale("log")
    plt.xlabel("\# of epochs", fontsize=50)
    plt.ylabel("Operator norm", fontsize=50)
    plt.xticks(fontsize=45)
    plt.yticks(fontsize=45)
    plt.ylim()
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='lower right', bbox_to_anchor=(1, 0.2), fontsize=25)
    ax.xaxis.offsetText.set_fontsize(36)
    plt.savefig("./finite_halpern_qp.png", bbox_inches='tight')
    plt.show()

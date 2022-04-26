import numpy as np
import cvxpy as cp
from numba import jit, objmode

NOPYTHON = True
NOGIL = True
BOUNDSCHECK = False
FASTMATH = True
PARALLEL = False

jit_kwargs = {
    "nopython": NOPYTHON,
    "nogil": NOGIL,
    "boundscheck": BOUNDSCHECK,
    "fastmath": FASTMATH,
}


@jit(**jit_kwargs)
def hardthresh(u, k):
    abs_u = np.abs(u)

    thresh = np.partition(abs_u, -k)[-k]
    for i in range(len(abs_u)):
        if abs_u[i] < thresh:
            u[i] = 0.0
    # return u_s


def solve_SDP(Sigma, k):
    H = cp.Variable(Sigma.shape)  # , symmetric=True)
    # The operator >> denotes matrix inequality.
    constraints = [H >> 0]
    constraints += [cp.trace(H) == 1]
    constraints += [cp.pnorm(H, 1) <= k]
    prob = cp.Problem(cp.Maximize(cp.scalar_product(Sigma, H)),
                      constraints)
    prob.solve()
    return prob.value, H.value

@jit(**jit_kwargs)
def RSGE(Xss, k_p, ro_sep, cl, grad_ph, pph, pTau):
    n, d = Xss.shape
    Xs = Xss#.copy()
    id = np.random.randint(10000)
    ph = pph
    Tau = pTau
    G = grad_ph#np.empty(d)
    N_eliminated = 0
    while N_eliminated < n*(cl + 0.05):
        print("loop", id)
        G.fill(0.0)
        for j in range(d):
            for i in range(n - N_eliminated):
                G[j] += Xs[i, j]
            G[j] /= n - N_eliminated
        hardthresh(G, 2 * k_p)

        for i in range(n - N_eliminated):
            ph[i, :] = Xs[i, :] - G
        ph = ph[:n - N_eliminated]
        Sigma = (ph.T @ ph) / n - N_eliminated

        with objmode(lamda='float64', H='float64[:,:]'):  # annotate return type
            # this region is executed by object-mode.
            lamda, H = solve_SDP(Sigma, k_p)
            H = np.array(H)
        #lamda, H = solve_SDP(Sigma, k_p)
        # print("rank is ", np.linalg.matrix_rank(H))
        if not np.isfinite(lamda):
            print("lamda infinite ")
            return
        print(lamda, ro_sep)
        if lamda < ro_sep:
            return #G
        else:
            #Tau = np.abs(np.einsum("ij, jk, ik -> i", ph, H, ph))
            Tau.fill(0.0)
            for i in range(n - N_eliminated):
                for j in range(d):
                    for k in range(d):
                        Tau[i] += ph[i, j] * H[j, k] * ph[i, k]
                Tau[i] = np.abs(Tau[i])
            Tau = Tau[:n - N_eliminated]
            arg = np.argmax(Tau) #np.random.choice(len(Tau), p=Tau / Tau.sum())  #
            N_eliminated += 1
            #return RSGE(np.vstack((Xs[:arg, :], Xs[arg + 1:, :])), k, ro_sep)

            #Xs = np.vstack((Xs[:arg, :], Xs[arg + 1:, :]))
            for j in range(d):
                Xs[arg, j] = Xs[n - N_eliminated, j]
            Xs = Xs[:n - N_eliminated]
    return #G

@jit(**jit_kwargs)
def liuliu18_solver(X, y, step_size, k_prime, n_iter, loss_deriv, theta_star, sigma, C_gamma = 1, corrupt_lvl=0.0):
    n, d = X.shape
    beta = np.zeros(d)
    gradients_ph = np.empty_like(X)
    grad_ph = np.empty(d)
    inner_products = np.empty(n)
    ph = np.empty_like(X)
    Tau = np.empty(n)

    for t in range(n_iter):

        np.dot(X, beta, out=inner_products)

        for i in range(n):
            deriv = loss_deriv(inner_products[i], y[i])
            for j in range(d):
                gradients_ph[i, j] = deriv * X[i, j]

        rho_sep = C_gamma * (np.sum((theta_star - beta)**2) + sigma**2)
        RSGE(gradients_ph, k_prime, rho_sep, corrupt_lvl, grad_ph, ph, Tau)

        beta -= step_size * grad_ph
        hardthresh(beta, k_prime)

    return beta
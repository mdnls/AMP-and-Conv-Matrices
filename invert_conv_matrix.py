from tramp.variables import SISOVariable as V, SILeafVariable as O
from tramp.priors import GaussBernouilliPrior
from tramp.likelihoods import GaussianLikelihood
from tramp.ensembles.base_ensemble import Ensemble
from tramp.ensembles.gaussian_ensemble import GaussianEnsemble
from tramp.channels import LinearChannel, BiasChannel, GaussianChannel
from tramp.algos import ExpectationPropagation, EarlyStopping
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import scipy.linalg
from scipy.fftpack import dct
'''
Use a convex solver and TRAMP priors to:
1. generate a convolutional measurement matrix and a sparse signal
2. solve l1 minimization convex recovery property 
'''

def dctmtx(n, normalize=False):
    d = dct(np.eye(n), axis=0)
    if (normalize):
        d = d / np.sqrt(np.sum(d ** 2, axis=1)).reshape((-1, 1))
    return d

class ConvEnsemble(Ensemble):
    def __init__(self, N, k):
        '''
        Random convolutional matrix ensemble. Samples from this distribution are of
            the form W, an N by N matrix scaled so E[ |Wx|^2 ] = |x|^2

        Parameters
        ----------
        N - Signal dimensionality
        k - Size of random convolutional filters
        '''
        self.N = N
        self.k = k
        self.repr_init()

    def generate(self):
        sigma_x = 1 / np.sqrt(self.k)
        filter = np.random.normal(size=(self.k,), scale=sigma_x)
        padded_filter = np.zeros(shape=(self.N,))
        padded_filter[0:self.k] = filter
        X = scipy.linalg.circulant(np.roll(padded_filter, shift=-(self.k-1)//2)).T
        return X

class ChannelConvEnsemble(Ensemble):
    def __init__(self, width, in_channels, out_channels, k):
        self.W = width
        self.C_in = in_channels
        self.C_out = out_channels
        self.k = k

    def generate(self):
        sigma_x = 1/np.sqrt(self.k)
        filter = sigma_x * np.random.normal(size=(self.C_out, self.C_in, self.k))
        padded_filter = np.zeros(shape=(self.C_out, self.C_in, self.W))
        padded_filter[:, :, 0:self.k] = filter
        padded_filter = np.roll(padded_filter, shift=-(self.k-1)//2, axis=-1)
        transf_tensor = np.apply_along_axis(scipy.linalg.circulant, axis=-1, arr=padded_filter)
        blocks = list([ list(r) for r in transf_tensor])
        return np.block(blocks)

def conv_model(A, C, prior):
    return prior @ V(id="x") @ LinearChannel(W=C) @ V(id="Cx") @ LinearChannel(W=A) @ V(id="z")

def conv_sensing(measurement_ratios, filter_size, solver="AMP", sparse_in_dct=False,
                 width=50, depth=20, delta=1e-2, sparsity=0.5, n_rep=10):

    N = width * depth
    signal = GaussBernouilliPrior(size=(N,), rho=sparsity)

    def sample_trnsf():
        if(sparse_in_dct):
            D = scipy.linalg.block_diag(*[dctmtx(width).T for _ in range(depth)])
        else:
            D = np.eye(N)
        return D
    recovery_per_alpha = []

    for alpha in measurement_ratios:
        recoveries = []
        for rep in range(n_rep):
            out_channels = int(np.rint(alpha * depth))
            ensemble = ChannelConvEnsemble(width=width, in_channels=depth, out_channels=out_channels, k=filter_size)
            A = ensemble.generate()
            C = sample_trnsf()

            teacher = conv_model(A, C, signal) @ GaussianChannel(var=delta) @ O(id="y")
            teacher = teacher.to_model()
            sample = teacher.sample()

            if (solver == "AMP"):

                max_iter = 20
                damping = 0.1

                student = conv_model(A, C, signal) @ GaussianLikelihood(y=sample['y'], var=delta)
                student = student.to_model_dag()
                student = student.to_model()
                ep = ExpectationPropagation(student)
                ep.iterate(max_iter=max_iter, damping=damping, callback=EarlyStopping(tol=1e-8))
                data_ep = ep.get_variables_data((['x']))
                mse = np.mean((data_ep['x']['r'] - sample['x']) ** 2)
                recoveries.append(mse)

            elif (solver == "CVX"):

                reg_param = 0.001

                x = cp.Variable(shape=(N,), name="x")
                lmbda = cp.Parameter(nonneg=True)
                objective = cp.norm2(A @ C @ x - sample['y']) ** 2 + lmbda * cp.norm1(x)
                problem = cp.Problem(cp.Minimize(objective))
                lmbda.value = reg_param
                problem.solve(abstol=1e-6)
                mse = np.mean((x.value - sample['x']) ** 2)
                recoveries.append(mse)
            else:
                raise ValueError("Solver must be 'AMP' or 'CVX'")
        recovery_per_alpha.append(recoveries)
    return recovery_per_alpha


def recovery(measurement_ratios, filter_size, solver="AMP", prior="conv",
             sparse_in_dct=False, N=1000, delta=1e-2, sparsity=0.5, n_rep=10):

    signal = GaussBernouilliPrior(size=(N,), rho=sparsity)
    prior_conv_ens = ConvEnsemble(N, filter_size)

    def sample_trnsf():
        if(sparse_in_dct):
            D = dctmtx(N).T
        else:
            D = np.eye(N)

        if(prior == "conv"):
            C = prior_conv_ens.generate()
        elif(prior == "sparse"):
            C = np.eye(N)
        else:
            raise ValueError("Prior must be 'conv' or 'sparse'")
        return D @ C
    recovery_per_alpha = []

    for alpha in measurement_ratios:
        recoveries = []
        for rep in range(n_rep):
            M = int(alpha * N)
            ensemble = GaussianEnsemble(M, N)
            A = ensemble.generate()
            C = sample_trnsf()

            teacher = conv_model(A, C, signal) @ GaussianChannel(var=delta) @ O(id="y")
            teacher = teacher.to_model()
            sample = teacher.sample()

            if(solver == "AMP"):

                max_iter = 20
                damping = 0.1

                student = conv_model(A, C, signal) @ GaussianLikelihood(y=sample['y'], var=delta)
                student = student.to_model_dag()
                student = student.to_model()
                ep = ExpectationPropagation(student)
                ep.iterate(max_iter=max_iter, damping=damping, callback=EarlyStopping(tol=1e-8))
                data_ep = ep.get_variables_data((['x']))
                mse = np.mean((data_ep['x']['r'] - sample['x'])**2)
                recoveries.append(mse)

            elif(solver == "CVX"):

                reg_param = 0.001

                x = cp.Variable(shape=(N,), name="x")
                lmbda = cp.Parameter(nonneg=True)
                objective = cp.norm2(A @ C @ x - sample['y'])**2 + lmbda * cp.norm1(x)
                problem = cp.Problem(cp.Minimize(objective))
                lmbda.value = reg_param
                problem.solve(abstol=1e-6)
                mse = np.mean((x.value - sample['x'])**2)
                recoveries.append(mse)
            else:
                raise ValueError("Solver must be 'AMP' or 'CVX'")
        recovery_per_alpha.append(recoveries)
    return recovery_per_alpha



def plot_recovery(rhos, alphas, N=1000, k=5, sparse_in_dct=True, delta=1e-4, n_rep=10, prior="conv", omit_cvx=False):
    cm = plt.cm.get_cmap("tab20")
    for i, rho in enumerate(rhos):
        amp_recoveries = recovery(alphas, filter_size=k, delta=delta, solver="AMP", sparse_in_dct=sparse_in_dct, N=N, n_rep=n_rep, prior=prior, sparsity=rho)
        avg_amp_recoveries = [np.mean(r) for r in amp_recoveries]
        plt.plot(alphas, avg_amp_recoveries, "-", label=r"$\rho=" + str(rho) + str("$"), color=cm.colors[2*i])

        if(not omit_cvx):
            cvx_recoveries = recovery(alphas, filter_size=k, delta=delta, solver="CVX", sparse_in_dct=sparse_in_dct, N=N, n_rep=n_rep, prior=prior, sparsity=rho)
            avg_cvx_recoveries = [np.mean(r) for r in cvx_recoveries]
            plt.plot(alphas, avg_cvx_recoveries, "--", color=cm.colors[2*i+1])
    if(prior == "conv" and sparse_in_dct):
        plt.title("Gaussian CS for Sparse-DCT-Conv Prior, $N=" + str(N) + "$")
    elif(prior == "conv"):
        plt.title("Gaussian CS for Sparse-Conv Prior, $N=" + str(N) + "$")
    elif(prior == "sparse"):
        plt.title("Gaussian CS for Gauss-Bernouilli Prior, $N=" + str(N) + "$")
    plt.legend()
    plt.xlabel("Measurement Ratios")
    plt.ylabel("MSE")
    plt.xlim(1, 0)

def plot_conv_sensing(rhos, alphas, in_channels, n=50, k=5, sparse_in_dct=True, delta=1e-4, n_rep=10, omit_cvx=False):
    cm = plt.cm.get_cmap("tab20")
    for i, rho in enumerate(rhos):
        amp_recoveries = conv_sensing(alphas, filter_size=k, solver="AMP", sparse_in_dct=sparse_in_dct, sparsity=rho,
                                      width=n, depth=in_channels, delta=delta, n_rep=n_rep)
        avg_amp_recoveries = [np.mean(r) for r in amp_recoveries]
        plt.plot(alphas, avg_amp_recoveries, "-", label=r"$\rho=" + str(rho) + str("$"), color=cm.colors[2 * i])

        if (not omit_cvx):
            cvx_recoveries = conv_sensing(alphas, filter_size=k, solver="CVX", sparse_in_dct=sparse_in_dct, sparsity=rho,
                                      width=n, depth=in_channels, delta=delta, n_rep=n_rep)
            avg_cvx_recoveries = [np.mean(r) for r in cvx_recoveries]
            plt.plot(alphas, avg_cvx_recoveries, "--", color=cm.colors[2 * i + 1])

    if(sparse_in_dct):
        plt.title("Multichannel Conv Sensing for DCT-Sparse Signal, $N=" + f"{in_channels} \\times {n}" + "$")
    else:
        plt.title("Multichannel Conv Sensing for Sparse Signal, $N=" + f"{in_channels} \\times {n}" + "$")
    plt.legend()
    plt.xlabel("Measurement Ratios")
    plt.ylabel("MSE")
    plt.xlim(1, 0)

if __name__ == "__main__":

    N = 1000
    alpha = 1
    rho = 0.1
    delta = 1e-2
    rhos = [0.25, 0.5, 0.75]
    alphas = 1 - np.arange(0, 1, 0.05)
    plot_conv_sensing(rhos, alphas, in_channels=20, n=50, k=5, sparse_in_dct=False, delta=1e-4, n_rep=1, omit_cvx=True)
    plot_recovery([0.5, 0.25], [1, 0.9, 0.8, 0.7, 0.6, 0.6, 0.4, 0.3, 0.2, 0.1], n_rep=1)
    plt.show()

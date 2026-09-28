"""Source distributions for the interpolant.

Flow matching transports a source p0 to the data p1 along
x_t = (1-t)*x0 + t*x1, and the conditional target u_t = x1 - x0 does not depend
on what p0 is. So the Gaussian is a choice, not a requirement, and a source
closer to the data leaves the velocity field less work to do.

The catch is that it also leaves the field less to DO. Push the source all the
way onto the data and the flow learns the identity, and what you are really
sampling from is the source. That is why the `prior` arm -- source decoded with
no ODE at all -- is the diagnostic that matters here: it measures how much of the
result the source is responsible for, and (fm - prior) is what flow matching
actually contributed. A richer source that raises `prior` and `fm` by the same
amount has bought nothing.

Fitting happens on the TRAIN cloud only, in the Standardizer's frame, exactly
like every other fitted object in this pipeline.
"""
from __future__ import annotations

import torch
from torch import Tensor


class GaussianSource:
    """Standard N(0, I) -- the default, and the baseline everything else is judged against."""

    name = "gauss"

    def fit(self, x: Tensor) -> "GaussianSource":
        self.d = x.shape[1]
        return self

    def sample(self, n: int, device=None) -> Tensor:
        return torch.randn(n, self.d, device=device)

    def state_dict(self):
        return {"kind": "gauss", "d": self.d}

    def load_state_dict(self, sd):
        self.d = sd["d"]; return self


class FullGaussianSource:
    """One Gaussian with full covariance, fitted in closed form.

    In a standardized frame this adds exactly one thing over N(0, I): the linear
    correlation between latent dimensions. §6.7 measured the cloud's effective
    rank at ~8 of 16, so those correlations are real and a diagonal source
    ignores them.
    """

    name = "full"

    def fit(self, x: Tensor) -> "FullGaussianSource":
        x = x.double()
        self.d = x.shape[1]
        self.mean = x.mean(0)
        c = torch.cov(x.T) + 1e-5 * torch.eye(self.d, dtype=torch.float64)
        self.L = torch.linalg.cholesky(c)
        return self

    def sample(self, n: int, device=None) -> Tensor:
        z = torch.randn(n, self.d, dtype=torch.float64)
        return (self.mean + z @ self.L.T).float().to(device or "cpu")

    def state_dict(self):
        return {"kind": "full", "d": self.d, "mean": self.mean, "L": self.L}

    def load_state_dict(self, sd):
        self.d, self.mean, self.L = sd["d"], sd["mean"], sd["L"]; return self


class GaussianMixtureSource:
    """K-component diagonal-covariance mixture, fitted by EM.

    Diagonal rather than full on purpose: with 3,447 points in 16 dimensions, K
    full covariances is enough parameters to start memorizing the cloud, and a
    source that memorizes the training set will show up as high plausibility with
    collapsed novelty -- which is the failure this is meant to detect, not cause.
    """

    name = "gmm"

    def __init__(self, k: int = 8, iters: int = 200, seed: int = 0, var_floor: float = 1e-3):
        self.k, self.iters, self.seed, self.var_floor = k, iters, seed, var_floor

    def fit(self, x: Tensor) -> "GaussianMixtureSource":
        g = torch.Generator().manual_seed(self.seed)
        x = x.double()
        n, d = x.shape
        self.d = d
        # k-means++-ish init: first centre random, rest far from those chosen
        idx = [int(torch.randint(n, (1,), generator=g))]
        for _ in range(self.k - 1):
            dist = torch.cdist(x, x[idx]).min(1).values ** 2
            p = (dist / dist.sum().clamp_min(1e-12)).cpu()
            idx.append(int(torch.multinomial(p, 1, generator=g)))
        mu = x[idx].clone()
        var = x.var(0, keepdim=True).repeat(self.k, 1)
        w = torch.full((self.k,), 1.0 / self.k, dtype=torch.float64)

        prev = -float("inf")
        for it in range(self.iters):
            # E step, in log space
            lp = (-0.5 * (((x[:, None, :] - mu[None]) ** 2) / var[None]).sum(-1)
                  - 0.5 * var.log().sum(-1)[None] - 0.5 * d * torch.log(torch.tensor(2 * torch.pi)))
            lp = lp + w.log()[None]
            ll = torch.logsumexp(lp, dim=1)
            r = (lp - ll[:, None]).exp()
            # M step
            nk = r.sum(0).clamp_min(1e-8)
            w = nk / n
            mu = (r.T @ x) / nk[:, None]
            var = ((r.T @ (x ** 2)) / nk[:, None] - mu ** 2).clamp_min(self.var_floor)
            cur = float(ll.mean())
            if abs(cur - prev) < 1e-7:
                break
            prev = cur
        self.mu, self.var, self.w, self.ll = mu, var, w, prev
        self.n_iter = it + 1
        return self

    def sample(self, n: int, device=None) -> Tensor:
        comp = torch.multinomial(self.w, n, replacement=True)
        z = torch.randn(n, self.d, dtype=torch.float64)
        return (self.mu[comp] + z * self.var[comp].sqrt()).float().to(device or "cpu")

    def state_dict(self):
        return {"kind": "gmm", "k": self.k, "d": self.d,
                "mu": self.mu, "var": self.var, "w": self.w}

    def load_state_dict(self, sd):
        self.k, self.d = sd["k"], sd["d"]
        self.mu, self.var, self.w = sd["mu"], sd["var"], sd["w"]
        return self


class StudentTSource:
    """Multivariate Student-t with the same mean and covariance as FullGaussianSource.

    Motivated by measurement, not taste: per-band diagnostics show high-melting ILs
    sit at increasing distance from the train-cloud centroid (radius 3.66 -> 4.52
    across melting-point bands) and are generated worst (plausibility 0.976 -> 0.908)
    even though the autoencoder reconstructs them BETTER. The weakness is therefore
    the flow's, at the periphery of the latent distribution -- exactly where a
    Gaussian source supplies least mass.

    t = mu + L z / sqrt(g/nu),  z ~ N(0,I),  g ~ chi2(nu). Lower nu = heavier tails;
    nu -> inf recovers the Gaussian. L is scaled so the covariance matches `full`
    rather than being inflated by the t factor, so the only change is tail weight.
    """

    name = "studentt"

    def __init__(self, nu: float = 5.0):
        self.nu = float(nu)

    def fit(self, x: Tensor) -> "StudentTSource":
        x = x.double()
        self.d = x.shape[1]
        self.mean = x.mean(0)
        c = torch.cov(x.T) + 1e-5 * torch.eye(self.d, dtype=torch.float64)
        # a t with nu d.o.f. has covariance Sigma*nu/(nu-2); rescale so we match Sigma
        if self.nu > 2:
            c = c * (self.nu - 2.0) / self.nu
        self.L = torch.linalg.cholesky(c)
        return self

    def sample(self, n: int, device=None) -> Tensor:
        z = torch.randn(n, self.d, dtype=torch.float64)
        g = torch.distributions.Chi2(torch.tensor(self.nu, dtype=torch.float64)).sample((n,))
        t = z / (g / self.nu).sqrt().unsqueeze(1)
        return (self.mean + t @ self.L.T).float().to(device or "cpu")

    def state_dict(self):
        return {"kind": "studentt", "nu": self.nu, "d": self.d, "mean": self.mean, "L": self.L}

    def load_state_dict(self, sd):
        self.nu, self.d, self.mean, self.L = sd["nu"], sd["d"], sd["mean"], sd["L"]
        return self


def make_source(kind: str, k: int = 8, seed: int = 0, nu: float = 5.0):
    if kind == "gauss":
        return GaussianSource()
    if kind == "full":
        return FullGaussianSource()
    if kind == "gmm":
        return GaussianMixtureSource(k=k, seed=seed)
    if kind == "studentt":
        return StudentTSource(nu=nu)
    raise ValueError(f"unknown source {kind!r}")

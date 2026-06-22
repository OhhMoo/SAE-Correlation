"""Reproduce Figs 1-4 of Poole et al. 2016 -- theory lines vs simulation dots."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")                # save-only, no display
import matplotlib.pyplot as plt
from mft import (length_map, q_star, c_map, chi1, curv_evolve, _ortho,
                 sim_length, sim_corr, sim_geometry, propagate)

SW = [1.3, 2.5, 4.0]                 # ordered / critical / chaotic
COL = ['b', 'g', 'r']
SB, N = 0.3, 1000

def fig_length():
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    qg = np.linspace(0, 15, 80)
    for sw, c in zip(SW, COL):
        ax[0].plot(qg, [length_map(q, sw, SB) for q in qg], c, label=f"$\\sigma_w$={sw}")
    ax[0].plot(qg, qg, 'k--', lw=1)
    ax[0].set(xlabel="$q^{l-1}$", ylabel="$q^l$", title="length map $\\mathcal{V}$"); ax[0].legend()

    D = 15
    for sw, c in zip(SW, COL):
        for q1 in [0.3, 8.0]:
            traj = [q1]
            for _ in range(D - 1):
                traj.append(length_map(traj[-1], sw, SB))
            ax[1].plot(range(1, D + 1), traj, c, lw=1)
            ax[1].plot(range(1, D + 1), sim_length(D, N, sw, SB, q1), c + 'o', ms=3)
    ax[1].set(xlabel="layer $l$", ylabel="$q^l$", title="dynamics of $q$ (line=theory, dots=sim)")

    swg, sbg = np.linspace(0.1, 4, 50), np.linspace(0, 4, 50)
    Z = np.array([[q_star(sw, sb) for sw in swg] for sb in sbg])
    fig.colorbar(ax[2].pcolormesh(swg, sbg, Z, shading='auto'), ax=ax[2])
    ax[2].set(xlabel="$\\sigma_w$", ylabel="$\\sigma_b$", title="$q^*$")
    fig.tight_layout(); fig.savefig("out/fig1_length.png", dpi=130); plt.close(fig)

def fig_corr():
    fig, ax = plt.subplots(1, 4, figsize=(17, 4))
    cg = np.linspace(0, 1, 50)
    for sw, c in zip(SW, COL):
        qs = q_star(sw, SB)
        ax[0].plot(cg, [c_map(x, qs, sw, SB) for x in cg], c, label=f"$\\sigma_w$={sw}")
    ax[0].plot(cg, cg, 'k--', lw=1)
    ax[0].set(xlabel="$c^{l-1}$", ylabel="$c^l$", title="correlation map $\\mathcal{C}$"); ax[0].legend()

    D = 30
    for sw, c in zip(SW, COL):
        qs = q_star(sw, SB)
        for c0 in [0.1, 0.6, 0.95]:
            traj = [c0]
            for _ in range(D - 1):
                traj.append(np.clip(c_map(traj[-1], qs, sw, SB), -1, 1))
            ax[1].plot(range(1, D + 1), traj, c, lw=1)
            ax[1].plot(range(1, D + 1), sim_corr(D, N, sw, SB, qs, c0), c + 'o', ms=2)
    ax[1].set(xlabel="layer $l$", ylabel="$c^l$", title="dynamics of $c$")

    swg, sbg = np.linspace(0.1, 4, 50), np.linspace(0, 4, 50)
    Cs, X1 = np.zeros((50, 50)), np.zeros((50, 50))
    for i, sb in enumerate(sbg):
        for j, sw in enumerate(swg):
            qs = q_star(sw, sb)
            X1[i, j] = chi1(qs, sw)
            c = 0.5
            for _ in range(300):
                c = np.clip(c_map(c, qs, sw, sb), -1, 1)
            Cs[i, j] = c
    fig.colorbar(ax[2].pcolormesh(swg, sbg, Cs, shading='auto'), ax=ax[2])
    ax[2].set(xlabel="$\\sigma_w$", ylabel="$\\sigma_b$", title="$c^*$")
    fig.colorbar(ax[3].pcolormesh(swg, sbg, X1, shading='auto'), ax=ax[3])
    ax[3].contour(swg, sbg, X1, [1.0], colors='r')
    ax[3].set(xlabel="$\\sigma_w$", ylabel="$\\sigma_b$", title="$\\chi_1$ (red: $\\chi_1=1$)")
    fig.tight_layout(); fig.savefig("out/fig2_corr.png", dpi=130); plt.close(fig)

def _theory_ac(qs, sw, sb, L, dth):  # autocorrelation: iterate C-map on cos(dtheta)
    c = np.cos(dth)
    for _ in range(L - 1):
        c = np.clip([c_map(x, qs, sw, sb) for x in c], -1, 1)
    return c

def fig_manifold():
    P, L = 512, 10
    th = np.linspace(0, 2 * np.pi, P, endpoint=False)
    half = slice(0, P // 2)
    fig, ax = plt.subplots(3, 4, figsize=(16, 11))
    for r, sw in enumerate(SW):
        qs = q_star(sw, SB)
        U = _ortho(N, 2)
        X = np.sqrt(N * qs) * (U[:, 0:1] * np.cos(th) + U[:, 1:2] * np.sin(th))
        hs = propagate(16, N, sw, SB, X)
        for k, lyr in enumerate([5, 10, 15]):
            h = hs[lyr - 1] - hs[lyr - 1].mean(1, keepdims=True)
            _, S, Vt = np.linalg.svd(h, full_matrices=False)
            p = S[:2, None] * Vt[:2]
            ax[r, k].scatter(p[0], p[1], c=th, cmap='hsv', s=3)
            ax[r, k].set(xticks=[], yticks=[])
            if r == 0: ax[r, k].set_title(f"layer {lyr}")
            if k == 0: ax[r, k].set_ylabel(f"$\\sigma_w$={sw}")
        h = hs[L - 1]
        F = np.fft.rfft(h, axis=1)
        ac = np.fft.irfft((F * F.conj()).real, n=P, axis=1).mean(0); ac /= ac[0]
        ax[r, 3].plot(th[half], _theory_ac(qs, sw, SB, L, th)[half], 'k-')
        ax[r, 3].plot(th[half], ac[half], 'r.', ms=2)
        ax[r, 3].set(xlabel="$\\Delta\\theta$", title=f"autocorr @ L={L}")
    fig.tight_layout(); fig.savefig("out/fig3_manifold.png", dpi=120); plt.close(fig)

def fig_curvature():
    D = 25
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ls = range(1, D + 1)
    for sw, c in zip(SW, COL):
        gE, kap = curv_evolve(sw, SB, D)
        Lg = 2 * np.pi * kap * np.sqrt(gE)
        skap, sgE, sLg = sim_geometry(D, N, sw, SB)
        ax[0].semilogy(ls, kap, c); ax[0].semilogy(ls, skap, c + 'o', ms=3)
        ax[1].semilogy(ls, gE, c);  ax[1].semilogy(ls, sgE, c + 'o', ms=3)
        ax[2].semilogy(ls, Lg, c, label=f"$\\sigma_w$={sw}"); ax[2].semilogy(ls, sLg, c + 'o', ms=3)
    ax[0].set(xlabel="layer", title="curvature $\\bar\\kappa$")
    ax[1].set(xlabel="layer", title="expansion $\\bar g^E$")
    ax[2].set(xlabel="layer", title="Grassmannian length $\\mathcal{L}^G$"); ax[2].legend()
    fig.tight_layout(); fig.savefig("out/fig4_curvature.png", dpi=130); plt.close(fig)

def main():
    np.random.seed(0)
    os.makedirs("out", exist_ok=True)
    fig_length(); fig_corr(); fig_manifold(); fig_curvature()
    print("wrote out/fig1_length.png .. fig4_curvature.png")

if __name__ == "__main__":
    main()

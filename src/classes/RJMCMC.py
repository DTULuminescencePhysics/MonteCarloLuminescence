from __future__ import annotations
import numpy as np
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass, field, fields
from src.classes.physics.crystal import Box 
from src.classes.monte_carlo import MCBase


@dataclass
class ReverseJmpMCMC:
    obs: np.ndarray
    steps: int 
    reps:int = field(default=10)
    MC: MCBase | None = field(default=None)
    crrnt_T: dict  = field(init=False)
    test_T: dict  = field(init=False)
    k: int  = field(init=False)
    test_k: int = field(init=False)
    rng: np.random.Generator = field(init=False)


    @classmethod
    def from_config(cls, obs: np.ndarray, steps:int, cfg: DictConfig, reps: int = 10) -> "ReverseJmpMCMC":
        """Intialise RJMCMC from input dictionary"""
        obj = cls(obs, steps, reps)
        seed = cfg.setup.seed+cfg.mc.reps
        obj.set_random_generator(seed)
        obj.k = obj.intialise_k()
        obj.crrnt_T = obj.initialise_temperature_profile(obj.k, cfg.temp.unit,cfg.temp.celsius,cfg.temp.duration)
        obj.MC = MCBase.RJMCMC_setup(cfg, obj.crrnt_T, reps, seed+1)
        obj.test_T = obj.crrnt_T
      
        return obj 
    
    def intialise_values(self, MC: MCBase, seed):
        """Intialise RJMCMC class from an already exisiting 
        Monte Carlo Class"""
        self.MC = MC
        self.k = self.intialise_k()
        self.crrnt_T = self.initialise_temperature_profile(self.k, self.MC.crystal.unit, self.MC.crystal.celsius, 
                                                           self.MC.crystal.duration)
        self.MC.crystal.set_temperature_profile("linearsteps",self.crrnt_T)
        self.set_random_generator(seed)
        self.MC.seed = seed+1
        self.MC.repetion = self.reps

    def initialise_temperature_profile(self, k: int, unit: str, celsius: bool, duration: float) -> dict:
        """Function that sets intial temperature profile parameters"""
        crrnt_T={'unit': unit, 
              'celsius': celsius, 
              'kind': 'linearsteps', 
              'T0': self.initialise_T0(), 
              'duration': duration, 
              'times': self.initialise_times(k,duration), 
              'dT_step': self.initialise_dT_step(k), 
              'dT': self.initialise_dT(k)}
   
        return crrnt_T

    def set_random_generator(self, seed: int | None = None) -> None:
        """Set the random number genreator using the seed. The seed 
        will be a user set value plus the current simulation number"""
        self.rng = np.random.default_rng(seed=seed)

    def intialise_k(self,p_geom: float = 0.3, K_max: int = 10) -> int:
        """Intialises k which is the number of step points but could also be seen as the point the current 
        temperature behaviour changes. """
        K = 0
        while K < K_max and self.rng.random() > p_geom:
            K += 1
        return K

   
    def initialise_times(self, K: int, t_end: float, min_gap: float = 1.0) -> np.ndarray:
        """
        Sample K ordered internal breakpoints in (0, t_end) with minimal spacing.
        Returns taus with shape (K+2): [0, tau1, ..., tauK, t_end].
        """
        if K == 0:
            return np.array([])
        # sample K points via sorted uniform, then enforce min_gap by jittering/rejecting
        for _ in range(10_000):  # simple rejection w/ retries
            pts = np.sort(self.rng.uniform(0, t_end, size=K))
            taus = np.concatenate([[0.0], pts, [t_end]])
            gaps = np.diff(taus)
            if np.all(gaps > min_gap):
                return taus
        # fallback: spread evenly if rejection struggled
        pts = np.linspace(t_end/(K+1), K*t_end/(K+1), K)
        return pts

    def initialise_dT_step(self, K: int, mean_drop: float = -30.0, sd_drop: float = 20.0) -> np.ndarray:
        """
        Initialize the K step drops at each breakpoint. Negative = step down.
        """
        if K == 0:
            return np.zeros(0)
        steps = self.rng.normal(loc=mean_drop, scale=sd_drop, size=K)
        # enforce “down” steps (≤ 0)
        steps = np.minimum(steps, 0.0)
        return steps

    def initialise_dT(self, K: int, mean_slope: float = 0.1, sd_slope: float = 0.05,
                    non_increasing: bool = True) -> np.ndarray:
        """
        Initialize slopes for the K+1 sections. If non_increasing, clip to ≤ 0.
        """
        g = self.rng.normal(loc=mean_slope, scale=sd_slope, size=K+1)
        # if non_increasing:
        #     g = np.minimum(g, 0.0)
        return g

    def initialise_T0(self, mean_T0: float = 100.0, sd_T0: float = 100.0) -> float:
        """Starting temperature before first interval."""
        return float(self.rng.normal(mean_T0, sd_T0))

 
    def propose_new_time(self, step_tau: float = 0.5, min_gap: float = 1.0) -> None:
        """Randomly jitter one internal breakpoint, keeping order and min gap."""
        self.test_T=self.crrnt_T
        if self.test_k == 0:
            return
        
        idx = self.rng.integers(0,  self.test_k+1) 
        lo = self.test_T["times"][idx-1] + min_gap
        hi = self.test_T["times"][idx+1] - min_gap
        if hi <= lo:
            return 
        prop = np.clip(self.test_T["times"][idx] + self.rng.normal(0, step_tau), lo, hi)
        self.test_T["times"][idx] = prop
 

    def propose_new_dT_size(self, step_sd: float = 1.0) -> None:
        """Perturb one step drop; keep it ≤ 0 for 'down' steps."""
        self.test_T=self.crrnt_T
        if self.test_k == 0:
            return 
       
        j = self.rng.integers(self.test_k)
        self.test_T['dT_step'][j] = min(0.0, self.test_T['dT_step'][j] + self.rng.normal(0, step_sd))
      
    def propose_new_dT(self, grad_sd: float = 0.02,
                            non_increasing: bool = True) -> None:
        """Perturb one interval slope; optionally enforce ≤ 0."""
        self.test_T=self.crrnt_T
        if self.test_k == 0:
            j = self.rng.integers(self.test_k)
        else: 
            j = 0 
        val = self.test_T["dT"][j] + self.rng.normal(0, grad_sd)
        self.test_T["dT"][j] =val
        
    def propose_new_T0(self, sd_T0: float = 1.0) -> None:
        """Random-walk on starting temperature."""
        self.test_T=self.crrnt_T
        self.test_T['T0'] += self.rng.normal(0, sd_T0)


    def propose_birth_step(self, min_gap: float = 1.0,
                        new_step_mean: float = -8.0, new_step_sd: float = 4.0,
                        new_grad_mean: float = -0.1, new_grad_sd: float = 0.05,
                        non_increasing: bool = True) ->  float:
        """
        Insert a new step: choose a segment to split, sample a breakpoint inside it,
        add a new step drop and a new gradient (we create one extra interval).
        Returns (proposal_profile, log_proposal_correction_for_RJ).
        """
        self.test_T=self.crrnt_T
        # choose a segment [tau_s, tau_{s+1}) to place a new breakpoint
        s = self.rng.integers(self.test_k)
        if s == self.test_k:
            hi = self.test_T["duration"]
        else: 
            hi = self.test_T["times"][s + 1]
        lo = self.test_T["times"][s]
        if hi - lo <= 2 * min_gap:
            return -np.inf
        
        new_time = self.rng.uniform(lo + min_gap, hi - min_gap)

        self.test_T["times"] = np.insert(self.test_T["times"], s + 1, new_time)
       
        step_dt_step = min(0.0, self.rng.normal(new_step_mean, new_step_sd))
        self.test_T["dT_step"] = np.insert(self.test_T["dT_step"], s, step_dt_step)
        
        dT_new = self.rng.normal(new_grad_mean, new_grad_sd)
        # if non_increasing:
        #     grad_new = min(grad_new, 0.0)
        self.test_T["dT"] = np.insert(self.test_T["dT"], s + 1, dT_new)
        self.test_k += 1
      
        return np.log(self.test_k)

    def propose_death_step(self) ->  float:
        """
        Remove a randomly chosen step (and its breakpoint and one gradient).
        Returns (proposal_profile, log_proposal_correction_for_RJ).
        """
        self.test_T=self.crrnt_T
        if self.test_k == 0:
            return -np.inf
       
     
        j = self.rng.integers(self.test_k)       # step index
        self.test_T["times"] = np.delete(self.test_T["times"], j)
        self.test_T["dT_step"] = np.delete(self.test_T["dT_step"], j)
        
        # self.test_T["dT"][j] = (self.test_T["dT"][j]+self.test_T["dT"][j+1])/2
        self.test_T["dT"] = np.delete(self.test_T["dT"], j + 1)
        self.test_k -= 1

        return -np.log(self.test_k)
    
    from math import log
    from scipy.stats import truncnorm
    # ---- Your forward model: from T(t) to expected ratio ----
    def forward_ratio(t_obs: np.ndarray, prof: TempProfile) -> np.ndarray:
        Tt = temperature_at(t_obs, prof)  # piecewise linear with steps
        # Replace this with your physics/surrogate:
        return Tt  # placeholder (identity)

    # ---- Likelihood: Gaussian noise ----
    def loglik_gaussian(r_obs: np.ndarray, r_hat: np.ndarray, sigma: float) -> float:
        n = r_obs.size
        res = (r_obs - r_hat) / sigma
        return -0.5*np.sum(res**2) - n*np.log(sigma)  # + const (omitted)

    # ---- Priors (edit to your domain) ----
    def log_prior_profile(prof: TempProfile,
                        non_increasing: bool = True,
                        min_gap: float = 1.0) -> float:
        lp = 0.0
        # prior on K (geometric on K>=0)
        p = 0.3
        lp += prof.K*np.log(1-p) + np.log(p)

        # ordered taus with min gap
        gaps = np.diff(prof.taus)
        if np.any(gaps <= min_gap) or prof.taus[0] != 0 or abs(prof.taus[-1]-prof.t_end) > 1e-12:
            return -np.inf

        # step sizes: encourage downward steps (≤0) with N(mean=-10, sd=5) truncated at 0
        if prof.K > 0:
            if np.any(prof.step_sizes > 0):
                return -np.inf
            lp += np.sum(-0.5*((prof.step_sizes + 10.0)/5.0)**2)

        # gradients: N(mean=-0.1, sd=0.05), trunc at 0 if cooling
        if non_increasing and np.any(prof.grads > 0):
            return -np.inf
        lp += np.sum(-0.5*((prof.grads + 0.1)/0.05)**2)

        # start temp T0 ~ N(150, 20^2)
        lp += -0.5*((prof.T0 - 150.0)/20.0)**2
        return lp

    def log_prior_sigma(sigma: float) -> float:
        if sigma <= 0: return -np.inf
        # Half-Cauchy(10) up to const:
        return -np.log(1.0 + (sigma/10.0)**2)

    def log_truncnorm_pdf(x, mean, sd, upper=0.0):
        # truncated at (-inf, upper]
        a = (-np.inf - mean)/sd
        b = (upper - mean)/sd
        # scipy truncnorm uses [a,b] in standard normal units
        return truncnorm.logpdf(x, a, b, loc=mean, scale=sd)

    def log_q_birth_forward(old: TempProfile,
                            new: TempProfile,
                            s_chosen: int,
                            tau_new: float,
                            min_gap: float,
                            new_step_mean: float, new_step_sd: float,
                            new_grad_mean: float, new_grad_sd: float) -> float:
        # choose segment uniformly among old.K+1 segments
        lq = -log(old.K + 1)

        # continuous tau proposal uniform in (lo+min_gap, hi-min_gap)
        lo, hi = old.taus[s_chosen], old.taus[s_chosen+1]
        L = (hi - min_gap) - (lo + min_gap)
        lq += -log(L)

        # step size from truncated normal (≤0)
        j = s_chosen  # inserted step index
        step_new = new.step_sizes[j]
        lq += log_truncnorm_pdf(step_new, new_step_mean, new_step_sd, upper=0.0)

        # new gradient for the new right interval (inserted at s+1)
        grad_new = new.grads[s_chosen+1]
        # clipped at ≤0 (cooling)
        if grad_new > 0: return -np.inf
        # Treat as truncated normal as well
        lq += log_truncnorm_pdf(grad_new, new_grad_mean, new_grad_sd, upper=0.0)

        return lq

    def log_q_death_forward(old: TempProfile, j_removed: int) -> float:
        # choose which step to remove uniformly among old.K steps
        return -log(old.K)
     
    def rjmcmc_temperature(t_obs: np.ndarray,r_obs: np.ndarray,iters: int = 5000,t_end: float | None = None,
                       min_gap: float = 1.0,
                       non_increasing: bool = True,
                       birth_prob: float = 0.25,
                       death_prob: float = 0.25,
                       seed: int | None = None):
        """
        Returns lists of samples of (TempProfile, sigma).
        """
        global rng
        if seed is not None:
            rng = default_rng(seed)
        if t_end is None:
            t_end = float(np.max(t_obs))

        # --- init state ---
        prof = make_initial_profile(rng, t_end=t_end, p_geom=0.3, K_max=10, min_gap=min_gap)
        sigma = np.std(r_obs) + 1e-6
        rhat  = forward_ratio(t_obs, prof)

        samples = []
        acc_within = acc_birth = acc_death = 0

        for it in range(iters):
            # ---------------- within-model (fixed K) ----------------
            # Randomly choose one of the local perturbations
            move = rng.choice(["break", "step", "grad", "T0"])
            if move == "break":
                prop = propose_perturb_breaks(rng, prof, step_tau=0.5, min_gap=min_gap)
            elif move == "step":
                prop = propose_perturb_steps(rng, prof, step_sd=1.0)
            elif move == "grad":
                prop = propose_perturb_grads(rng, prof, grad_sd=0.02, non_increasing=non_increasing)
            else:
                prop = propose_perturb_T0(rng, prof, sd_T0=1.0)

            if prop is not prof:  # might be identical if invalid
                rhat_p = forward_ratio(t_obs, prop)
                la = (loglik_gaussian(r_obs, rhat_p, sigma) + log_prior_profile(prop, non_increasing, min_gap)
                    - loglik_gaussian(r_obs, rhat,   sigma) - log_prior_profile(prof, non_increasing, min_gap))
                if np.log(rng.random()) < la:
                    prof, rhat = prop, rhat_p
                    acc_within += 1

            # ---------------- trans-dimensional RJ move ----------------
            u = rng.random()
            if u < birth_prob:
                # choose segment and try birth
                s = rng.integers(prof.K + 1)
                lo, hi = prof.taus[s], prof.taus[s+1]
                if hi - lo > 2*min_gap:
                    tau_new = rng.uniform(lo + min_gap, hi - min_gap)

                    # construct birth proposal deterministically from (s, tau_new)
                    # (we’ll reuse your convenience helper that also returns a coarse log_q)
                    prop_b, _ = propose_birth_step(
                        rng, prof, min_gap=min_gap,
                        new_step_mean=-8.0, new_step_sd=4.0,
                        new_grad_mean=-0.1, new_grad_sd=0.05,
                        non_increasing=non_increasing
                    )
                    if prop_b is not prof:
                        rhat_p = forward_ratio(t_obs, prop_b)
                        # Posterior ratio
                        la = (loglik_gaussian(r_obs, rhat_p, sigma) + log_prior_profile(prop_b, non_increasing, min_gap)
                            - loglik_gaussian(r_obs, rhat,   sigma) - log_prior_profile(prof,  non_increasing, min_gap))
                        # Proposal ratio q(reverse)/q(forward)
                        # reverse death: choose one of (K+1) steps uniformly
                        lq_rev = -np.log(prop_b.K)  # death forward on new state
                        # forward birth density (must mirror propose_birth_step)
                        lq_fwd = log_q_birth_forward(
                            old=prof, new=prop_b, s_chosen=s, tau_new=tau_new, min_gap=min_gap,
                            new_step_mean=-8.0, new_step_sd=4.0, new_grad_mean=-0.1, new_grad_sd=0.05
                        )
                        la += (lq_rev - lq_fwd)  # |J| = 1

                        if np.log(rng.random()) < la:
                            prof, rhat = prop_b, rhat_p
                            acc_birth += 1

            elif u < birth_prob + death_prob and prof.K > 0:
                # choose a step to remove
                j = rng.integers(prof.K)
                prop_d, _ = propose_death_step(rng, prof)
                if prop_d is not prof:
                    rhat_p = forward_ratio(t_obs, prop_d)
                    la = (loglik_gaussian(r_obs, rhat_p, sigma) + log_prior_profile(prop_d, non_increasing, min_gap)
                        - loglik_gaussian(r_obs, rhat,   sigma) - log_prior_profile(prof,  non_increasing, min_gap))
                    # Proposal ratio q(reverse)/q(forward)
                    # reverse birth: choose segment uniformly and pick tau uniformly inside admissible interval,
                    # and draw new step & gradient from the same truncated normals.
                    # For a simple, consistent pairing with the birth above, approximate with:
                    lq_rev = 0.0  # (you can compute the exact reverse birth density as in log_q_birth_forward on prop_d→prof)
                    lq_fwd = log_q_death_forward(old=prof, j_removed=j)
                    la += (lq_rev - lq_fwd)  # |J| = 1

                    if np.log(rng.random()) < la:
                        prof, rhat = prop_d, rhat_p
                        acc_death += 1

            # ---------------- noise update (optional) ----------------
            sigma_p = sigma * np.exp(rng.normal(0, 0.05))
            la = (loglik_gaussian(r_obs, rhat, sigma_p) + log_prior_sigma(sigma_p)
                - loglik_gaussian(r_obs, rhat, sigma)   - log_prior_sigma(sigma))
            if np.log(rng.random()) < la:
                sigma = sigma_p

            samples.append((prof.copy(), float(sigma)))

        stats = {
            "acc_within": acc_within / iters,
            "acc_birth": acc_birth / max(1, int(iters*birth_prob)),
            "acc_death": acc_death / max(1, int(iters*death_prob)),
        }
        return samples, stats
#     # ----------------------- Optional: evaluate T(t) -----------------------

#     def temperature_at(t: np.ndarray, prof: TempProfile) -> np.ndarray:
#         """
#         Compute T(t) given piecewise-linear intervals with instantaneous drops at taus[1..K].
#         Convention:
#         - On each interval [taus[s], taus[s+1]) temperature evolves as:
#             T(t) = T_start(s) + grads[s] * (t - taus[s])
#         - At t = taus[s] for s>=1, add the drop 'step_sizes[s-1]' (usually ≤ 0).
#         """
#         t = np.asarray(t, dtype=float)
#         out = np.empty_like(t)
#         # precompute cumulative drops applied up to each interval start
#         cum_drop = np.zeros(prof.K + 1)
#         if prof.K > 0:
#             cum_drop[1:] = np.cumsum(prof.step_sizes)
#         # value at start of first interval:
#         base0 = prof.T0 + cum_drop[0]
#         for s in range(prof.K + 1):
#             mask = (t >= prof.taus[s]) & (t < prof.taus[s + 1] if s + 1 < prof.taus.size else True)
#             base_s = prof.T0 + cum_drop[s]
#             out[mask] = base_s + prof.grads[s] * (t[mask] - prof.taus[s])
#         # Apply instantaneous drops exactly at step times (if sampling includes exact taus)
#         # (Optional: depends on your convention.)
#         return out

# # ----------------------- Example usage -----------------------

# if __name__ == "__main__":
#     rng = default_rng(0)
#     prof = make_initial_profile(rng, t_end=300.0, p_geom=0.4, K_max=5, min_gap=5.0)

#     # Local tweaks
#     prof1 = propose_perturb_breaks(rng, prof, step_tau=2.0, min_gap=5.0)
#     prof2 = propose_perturb_steps(rng, prof1, step_sd=2.0)
#     prof3 = propose_perturb_grads(rng, prof2, grad_sd=0.02)
#     prof4 = propose_perturb_T0(rng, prof3, sd_T0=1.5)

#     # Birth/death (change K and associated params)
#     prof_b, log_q_b = propose_birth_step(rng, prof4, min_gap=5.0)
#     prof_d, log_q_d = propose_death_step(rng, prof_b)
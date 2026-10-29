# Guidance consistency: what to test after the September 15 comparisons

The current priority is agreement between a predicted interaction and the result
of executing its gripper motion. The development runs can reach their requested
positions in the model while failing the independent MuJoCo replay. That replay
uses approximate contact and inertia, so it does not isolate model error from
every simulator/controller mismatch. Recorded controls and two timestep checks
remain necessary. See the [campaign evidence](PUSH_2026-09-15.md).

## What the literature suggests, and what transfers

[Diffusion Noise Optimization (DNO)](https://arxiv.org/html/2312.11994)
optimizes a fully decoded sample by differentiating through its denoising chain.
It also studies normalized gradients, preservation of a reference noise sample,
and temporal noise decorrelation for human-motion refinement. Our existing
initial-noise optimizer follows the first principle, with a distance prior and
per-window RMS cap. Its optional normalized-gradient and learning-rate controls
already exist. A noise-change cap does not preserve all Gaussian statistics;
the [saved-noise audit](../../validation/guidance_push_2026-09-15/LATENT_NOISE_AUDIT.md)
measures this separately. Human-motion improvements do not establish grasp-force
consistency in this model.

[Diffusion Forcing's Monte Carlo Tree Guidance](https://arxiv.org/html/2407.01392v3#S3.SS2)
averages gradients from multiple possible futures when guiding an earlier token.
Its construction uses a causal model and a suitable sampling schedule. Our
bidirectional windows and actuator-inpainting probe do not implement that
construction. The useful transferable question is whether a candidate gripper
motion works across several possible object outcomes, rather than only one
jointly optimized outcome. Inpainting is an approximate way to investigate this
with the present checkpoint, not a qualified conditional dynamics model.

[ACID](https://gawon1224.github.io/ACID/) adds an inverse-dynamics action-consistency
cost to planning with action-conditioned world models. A direct adaptation needs
care here: our state explicitly includes the gripper poses. An inverse model
could recover the action from those poses while ignoring impossible object
motion. This is an architectural inference, not a result reported by ACID.
Removing the actuator from its input would avoid that shortcut, but passive
object motion alone may not identify a unique action, especially before contact.
A small reconstruction error would still need independent validation.

## Concrete next experiments

1. **The fixed-gripper diagnostic is complete.** All four probes reproduce the
   original sample exactly and preserve actuator paths across windows. None of
   eight fresh passive-noise draws per probe reaches the held-position tolerance.
   The original teacup's 1.6 mm error becomes 38.1 mm with inpainting alone and
   97.2–126.7 mm with fresh passive noise. See [all results and
   movies](../../validation/guidance_push_2026-09-15/FIXED_GRIPPER_SURFACES.md).
   This motivates consistency testing while retaining the distinction between
   inpainting effects, sampling variation and physically correct dynamics.
2. **Test candidate consistency before optimizing more aggressively.** Use the
   same generated gripper path across several passive redraws, then rank by both
   target progress and the distribution of physical violations. Compare against
   ordinary single-sample selection at matched model evaluations. Keep fresh
   evaluation draws separate from any draws used during selection.
3. **Qualify a differentiable multi-draw objective only after the probe works.**
   A generated actuator trajectory can be supplied as differentiable known
   channels to several inpainted rollouts. Their passive noise is fixed within
   each optimization, while their actuator trajectory is shared. Check exact
   actuator agreement, finite-difference gradients, single-draw parity and
   independent evaluation seeds before interpreting results. This would cost
   several full denoising rollouts per update and is not implemented by the
   diagnostic script.
4. **Use simulator-assisted selection as an explicitly separate experiment.**
   Independent replay can filter ordinary proposals or saved optimized iterates.
   This tests whether a usable interaction already exists among candidates.
   It gives the planner access to a simulator and must be reported as such;
   it cannot establish success for planning with the learned model alone.

The replay adapter also matters. Its default finger preload uses proximity to
the **predicted** object to modify joint commands. Thus equal predicted gripper
poses do not necessarily imply equal replay commands when object predictions
differ. Any multi-draw physics comparison must either freeze the resulting
commands or explicitly include that adapter difference. The current fixed-gripper
probe compares model predictions only. The optional latched preload was tested
separately and did not repair the development failures.

That interface deserves its own validation: the model generates achieved gripper
poses, while execution needs motor commands. A geometry-triggered preload is an
approximate conversion between those quantities. On the later leftward jar
replica, the hard proximity gate commands about 6 mm of opening at peak lift.
Latched preload prevents the subsequent drop and improves endpoint error from
174.7/178.7 mm to 32.5/35.2 mm, while still failing the full criteria. The
[four-target controller comparison](../../validation/guidance_push_2026-09-15/LATCHED_REPLICATION.md)
keeps the original-policy results separate. A proposed controller improvement
should therefore be checked for command continuity, actual tracking and contact,
and recorded-control agreement across samples before assigning the residual gap
to learned dynamics or changing the default policy. The separate
[timing probe](../../validation/guidance_push_2026-09-15/TIMING_PROBE.md) finds that
simply doubling execution duration does not repair the five tested generated plans.

Items 2–4 remain follow-up proposals. The current campaign's frozen experiment
packets remain the record of what was actually run; those proposals are not
claims of completed implementation or improved task success.

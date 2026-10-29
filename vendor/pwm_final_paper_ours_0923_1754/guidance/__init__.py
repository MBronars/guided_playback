"""Guidance on the unified model: inverse dynamics through the world model.

Layer 0 of docs/guidance/DESIGN.md. The actuator's future is the model's input
(`TrajectoryBatch.pose_actions` for pose-native rigid bodies); planners choose it,
or the sampler's noise, so that the predicted passive outcome meets an objective.
"""

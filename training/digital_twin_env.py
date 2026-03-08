"""Offline digital twin for UAV-assisted WSN routing policy training.

This environment models per-forwarding decisions centrally at the base station.
It is intentionally lightweight so large numbers of episodes can be trained offline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class RewardWeights:
    w1_delivery: float = 5.0
    w2_energy: float = 1.0
    w3_delay: float = 2.0
    w4_drop: float = 5.0
    w5_overflow: float = 4.0
    w6_uav_distance: float = 1.25


@dataclass
class DomainConfig:
    num_nodes: int = 500
    area_side_m: float = 1000.0
    tx_range_min_m: float = 80.0
    tx_range_max_m: float = 100.0
    uav_speed_min: float = 10.0
    uav_speed_max: float = 20.0
    mission_duration_min: float = 180.0
    mission_duration_max: float = 480.0
    max_candidates: int = 15
    buffer_capacity_packets: int = 30


class DigitalTwinRoutingEnv:
    """Synthetic digital twin for centralized offline DQN training.

    State-action feature vector used for Q(s, a) approximation:
    [energy_ratio, queue_ratio, neighbor_density_norm, tau_norm,
     link_quality, neighbor_energy_ratio, neighbor_failure_ratio,
     uav_distance_norm]
    """

    feature_min = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    feature_max = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)

    def __init__(
        self,
        max_steps: int = 240,
        reward: RewardWeights | None = None,
        domain: DomainConfig | None = None,
        seed: int = 13,
    ) -> None:
        self.max_steps = max_steps
        self.reward = reward if reward is not None else RewardWeights()
        self.domain = domain if domain is not None else DomainConfig()
        self.rng = np.random.default_rng(seed)

        self.energy_ratio = 1.0
        self.queue_ratio = 0.0
        self.neighbor_density_norm = 0.5
        self.tau_norm = 0.5

        self.channel_noise = 0.1
        self.traffic_rate = 0.3
        self.node_failure_rate = 0.0
        self.tx_range_m = 90.0
        self.uav_speed = 15.0
        self.mission_duration = 300.0
        self.uav_distance_norm = 0.5

        self.step_idx = 0
        self.current_candidates = np.zeros((0, 8), dtype=np.float32)

    def reset(self) -> np.ndarray:
        self._randomize_domain()

        self.step_idx = 0
        self.energy_ratio = float(self.rng.uniform(0.75, 1.0))
        self.queue_ratio = float(self.rng.uniform(0.05, 0.35))
        self.neighbor_density_norm = float(self.rng.uniform(0.25, 0.8))
        self.tau_norm = float(self.rng.uniform(0.2, 1.0))
        self.uav_distance_norm = float(self.rng.uniform(0.1, 0.9))

        self.current_candidates = self._sample_candidates()
        return self.current_candidates.copy()

    def step(self, action_index: int) -> Tuple[np.ndarray, float, bool, Dict[str, float]]:
        if self.current_candidates.shape[0] == 0:
            return np.zeros((0, 8), dtype=np.float32), 0.0, True, {
                "delivery": 0.0,
                "energy_cost": 0.0,
                "delay_penalty": 0.0,
                "drop": 1.0,
                "overflow": 0.0,
            }

        action_index = int(np.clip(action_index, 0, self.current_candidates.shape[0] - 1))
        action_feat = self.current_candidates[action_index]

        # Candidate-local metrics
        lq = float(action_feat[4])
        neighbor_energy = float(action_feat[5])
        neighbor_fail = float(action_feat[6])
        uav_dist = float(action_feat[7])

        p_success = (
            0.60 * lq
            + 0.15 * neighbor_energy
            + 0.10 * (1.0 - self.queue_ratio)
            + 0.10 * (1.0 - self.channel_noise)
            + 0.05 * (1.0 - self.node_failure_rate)
            + 0.10 * (1.0 - uav_dist)
            - 0.18 * neighbor_fail
        )
        p_success = float(np.clip(p_success, 0.02, 0.98))
        delivery_success = 1.0 if self.rng.random() < p_success else 0.0

        energy_cost = (
            0.025
            + 0.045 * (1.0 - lq)
            + 0.010 * self.traffic_rate
            + 0.005 * self.channel_noise
        )

        delay_penalty = (
            0.45 * self.queue_ratio
            + 0.35 * self.tau_norm
            + 0.25 * neighbor_fail
            + 0.10 * self.channel_noise
        )

        packet_drop = 0.0
        if delivery_success < 0.5 and self.rng.random() < (0.55 + 0.25 * self.queue_ratio):
            packet_drop = 1.0

        buffer_overflow = 1.0 if (self.queue_ratio > 0.85 and self.rng.random() < 0.55) else 0.0

        reward = (
            self.reward.w1_delivery * delivery_success
            - self.reward.w2_energy * energy_cost
            - self.reward.w3_delay * delay_penalty
            - self.reward.w4_drop * packet_drop
            - self.reward.w5_overflow * buffer_overflow
            - self.reward.w6_uav_distance * uav_dist
        )

        self._advance_state(delivery_success, energy_cost, packet_drop, buffer_overflow)

        self.step_idx += 1
        done = bool(self.step_idx >= self.max_steps or self.energy_ratio <= 0.02)

        if done:
            self.current_candidates = np.zeros((0, 8), dtype=np.float32)
        else:
            self.current_candidates = self._sample_candidates()

        info = {
            "delivery": delivery_success,
            "energy_cost": energy_cost,
            "delay_penalty": delay_penalty,
            "drop": packet_drop,
            "overflow": buffer_overflow,
            "uav_distance": uav_dist,
            "p_success": p_success,
        }
        return self.current_candidates.copy(), float(reward), done, info

    def _randomize_domain(self) -> None:
        self.tx_range_m = float(self.rng.uniform(self.domain.tx_range_min_m, self.domain.tx_range_max_m))
        self.uav_speed = float(self.rng.uniform(self.domain.uav_speed_min, self.domain.uav_speed_max))
        self.mission_duration = float(
            self.rng.uniform(self.domain.mission_duration_min, self.domain.mission_duration_max)
        )

        self.channel_noise = float(self.rng.uniform(0.02, 0.35))
        self.traffic_rate = float(self.rng.uniform(0.15, 0.95))
        self.node_failure_rate = float(self.rng.uniform(0.0, 0.12))

    def _sample_candidates(self) -> np.ndarray:
        # Neighbor count scales with local density and transmission range.
        density_scale = 0.5 * self.neighbor_density_norm + 0.5 * (
            (self.tx_range_m - self.domain.tx_range_min_m)
            / max(1e-6, (self.domain.tx_range_max_m - self.domain.tx_range_min_m))
        )
        k = int(np.clip(np.round(2 + density_scale * (self.domain.max_candidates - 2)), 2, self.domain.max_candidates))

        candidates = np.zeros((k, 8), dtype=np.float32)

        for i in range(k):
            lq_mu = 0.45 + 0.40 * (1.0 - self.channel_noise) + 0.10 * self.neighbor_density_norm
            lq = float(np.clip(self.rng.normal(lq_mu, 0.12), 0.02, 0.99))

            neighbor_energy = float(np.clip(self.rng.normal(0.70, 0.20), 0.02, 1.0))

            fail_ratio_mu = 0.12 + 0.45 * (1.0 - lq) + 0.20 * self.channel_noise + 0.10 * self.node_failure_rate
            neighbor_fail = float(np.clip(self.rng.normal(fail_ratio_mu, 0.10), 0.0, 1.0))

            candidates[i, 0] = np.float32(self.energy_ratio)
            candidates[i, 1] = np.float32(self.queue_ratio)
            candidates[i, 2] = np.float32(self.neighbor_density_norm)
            candidates[i, 3] = np.float32(self.tau_norm)
            candidates[i, 4] = np.float32(lq)
            candidates[i, 5] = np.float32(neighbor_energy)
            candidates[i, 6] = np.float32(neighbor_fail)
            # Candidate-level proximity to the current UAV location.
            uav_dist = np.clip(self.rng.normal(self.uav_distance_norm, 0.15), 0.0, 1.0)
            candidates[i, 7] = np.float32(uav_dist)

        return candidates

    def _advance_state(
        self,
        delivery_success: float,
        energy_cost: float,
        packet_drop: float,
        overflow: float,
    ) -> None:
        self.energy_ratio = float(np.clip(self.energy_ratio - 0.22 * energy_cost, 0.0, 1.0))

        arrivals = int(self.rng.poisson(lam=max(0.05, self.traffic_rate * 1.6)))
        departures = 1 if delivery_success > 0.5 else 0
        penalty_packets = int(packet_drop + overflow)

        queue_packets = self.queue_ratio * self.domain.buffer_capacity_packets
        queue_packets = queue_packets + arrivals - departures + penalty_packets
        queue_packets = float(np.clip(queue_packets, 0.0, float(self.domain.buffer_capacity_packets)))
        self.queue_ratio = float(queue_packets / self.domain.buffer_capacity_packets)

        self.neighbor_density_norm = float(
            np.clip(
                self.neighbor_density_norm + self.rng.normal(0.0, 0.05) - 0.10 * self.node_failure_rate,
                0.0,
                1.0,
            )
        )

        # UAV contact time is a countdown with periodic refresh beacons.
        tau_decay = 1.0 / max(8.0, self.mission_duration / max(self.uav_speed, 0.1))
        self.tau_norm = float(np.clip(self.tau_norm - tau_decay, 0.0, 1.0))
        if self.rng.random() < 0.08:
            self.tau_norm = float(self.rng.uniform(0.0, 1.0))

        # Dynamic UAV location proxy (distance from node to UAV) with smooth perturbation.
        drift = 0.10 * (0.5 - self.tau_norm)
        self.uav_distance_norm = float(
            np.clip(self.uav_distance_norm + drift + self.rng.normal(0.0, 0.05), 0.0, 1.0)
        )

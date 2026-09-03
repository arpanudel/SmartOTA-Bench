import gym
from gym import spaces
import numpy as np
import hashlib
import os
import bsdiff4
from dataclasses import dataclass

from deployment.semantics import DeploymentConfig, DeploymentSimulator
from encoding.operations import (
    DeltaCodec,
    make_append_operation,
    make_backup_operation,
    make_checkpoint_operation,
    make_commit_operation,
    make_copy_operation,
    make_delete_operation,
    make_delta_operation,
    make_keep_operation,
    make_raw_insert_operation,
    make_rollback_operation,
    make_truncate_operation,
    make_verify_operation,
)
from encoding.replay import (
    blocks_to_bytes,
    validate_update,
)


@dataclass(frozen=True)
class RewardConstraintConfig:
    package_size_weight: float = 0.0
    network_bytes_weight: float = 0.0
    flash_write_weight: float = 0.0
    peak_ram_weight: float = 0.0
    peak_storage_weight: float = 0.0
    budget_violation_weight: float = 0.0
    rollback_required_penalty: float = 0.0

    def enabled(self):
        return any(
            value > 0.0
            for value in (
                self.package_size_weight,
                self.network_bytes_weight,
                self.flash_write_weight,
                self.peak_ram_weight,
                self.peak_storage_weight,
                self.budget_violation_weight,
                self.rollback_required_penalty,
            )
        )


class OTAEnv(gym.Env):
    ACTION_M = 0
    ACTION_MB = 1
    ACTION_KEEP = 2
    ACTION_COPY = 3
    ACTION_DELTA = 4
    ACTION_RAW_INSERT = 5
    ACTION_BACKUP = 6
    ACTION_APPEND = 7
    ACTION_DELETE = 8
    ACTION_TRUNCATE = 9
    ACTION_CHECKPOINT = 10
    ACTION_VERIFY = 11
    ACTION_COMMIT = 12
    ACTION_ROLLBACK = 13

    ACTION_NAMES = (
        "M",
        "MB",
        "keep",
        "copy",
        "delta",
        "raw_insert",
        "backup",
        "append",
        "delete",
        "truncate",
        "checkpoint",
        "verify",
        "commit",
        "rollback",
    )
    ACTION_NAME_TO_ID = {name: index for index, name in enumerate(ACTION_NAMES)}
    BUDGET_VIOLATION_PENALTY = 5.0
    BUDGET_VIOLATION_STATE_PENALTY = 0.25
    UNSAFE_OVERWRITE_PENALTY = 3.0
    ROLLBACK_UNREADY_STATE_PENALTY = 0.5
    FLASH_WRITE_COST_WEIGHT = 0.05
    EXCESS_FLASH_WRITE_PENALTY = 2.0
    INVALID_REPLAY_PENALTY = 10.0
    REPLAY_ERROR_PENALTY = 1.0
    VALID_REPLAY_BONUS = 1.0

    def __init__(
        self,
        old_file,
        new_file,
        block_size=4096 * 16,
        deployment_config=None,
        reward_constraints=None,
        similarity_top_k=3,
        similarity_sample_bytes=1024,
    ):
        self._identical_block_found = False

        self._old_file = old_file
        self._new_file = new_file
        self._old_size_bytes = os.path.getsize(self._old_file)
        self._new_size_bytes = os.path.getsize(self._new_file)

        self._block_size = block_size
        self._deployment_config = deployment_config or DeploymentConfig()
        self._reward_constraints = self._coerce_reward_constraints(reward_constraints)
        self._similarity_top_k = max(0, int(similarity_top_k))
        self._similarity_sample_bytes = max(1, int(similarity_sample_bytes))

        self._old_blocks = self.read_blocks(self._old_file, self._block_size)
        self._new_blocks = self.read_blocks(self._new_file, self._block_size)

        self._old_block_md5s = self.md5_blocks(self._old_file, self._block_size)
        self._new_block_md5s = self.md5_blocks(self._new_file, self._block_size)

        self._block_num = len(self._new_block_md5s)
        self._configure_observation_normalizers()
        self._similarity_candidates = self._precompute_similarity_candidates()

        self._mem_cost = len(self._old_blocks) * self._block_size
        self._encoding_cost = 0

        self._current_block_pos = 0

        self._current_block = self._old_blocks[0] if self._old_blocks else None
        self._action_to_direction = dict(enumerate(self.ACTION_NAMES))
        self.action_space = spaces.Discrete(len(self.ACTION_NAMES))
        self.legacy_action_space = spaces.Discrete(2)

        self._configure_observation_space()

        self._reward = 0.0
        self._encoding = ''
        self._encoding_ops = []
        self._backup_index_to_id = {}
        self._deployment = DeploymentSimulator(
            self._old_blocks,
            self._block_size,
            config=self._deployment_config,
            expected_blocks=self._new_blocks,
        )
        self._steps = 0
        self._action_counts = self._empty_action_counts()
        self._last_budget_violation_count = 0
        self._last_unsafe_overwrite_count = 0
        self._last_flash_write_bytes = 0
        self._last_package_size_bytes = 0
        self._last_network_bytes = 0
        self._last_peak_ram_bytes = 0
        self._last_peak_storage_bytes = 0

        # Table for storing yet-to-be-processed target/new block indices.
        self._blocks_remaining = [i for i in range(self._block_num)]

        self._blocks_mask = [1 for i in range(self._block_num)]
        '''
        M : modify current block in place, either via copy (if match found) or delta update from closest block
        MB: modify current block in place and backup old block (add to end of old blocks)
        '''
        
        self._current_action = 0

    def _coerce_reward_constraints(self, reward_constraints):
        if reward_constraints is None:
            return RewardConstraintConfig()
        if isinstance(reward_constraints, RewardConstraintConfig):
            return reward_constraints
        return RewardConstraintConfig(**dict(reward_constraints))

    def get_block_num(self):
        return self._block_num

    def get_old_blocks_md5(self):
        return self._old_block_md5s

    def get_new_blocks_md5(self):
        return self._new_block_md5s

    def get_observation_feature_names(self):
        return list(self._observation_feature_names)

    def get_observation_feature_layout(self):
        return dict(self._observation_feature_layout)

    def get_similarity_candidates(self, block_index=None):
        if block_index is None:
            block_index = self._current_block_pos
        if block_index < 0 or block_index >= len(self._similarity_candidates):
            return [self._empty_similarity_candidate() for _ in range(self._similarity_top_k)]
        return [dict(candidate) for candidate in self._similarity_candidates[block_index]]

    def get_action_family_names(self):
        return list(self.ACTION_NAMES)

    def action_id(self, action_name):
        return self.ACTION_NAME_TO_ID[action_name]

    def action_name(self, action_id):
        return self._action_to_direction[int(action_id)]

    def _empty_action_counts(self):
        return {name: 0 for name in self.ACTION_NAMES}

    def get_action_mask(self, target_index=None):
        if target_index is None:
            target_index = self._blocks_remaining[0] if self._blocks_remaining else None

        mask = np.zeros(self.action_space.n, dtype=np.float32)
        has_pending_target = target_index is not None and target_index in self._blocks_remaining

        if has_pending_target:
            mask[self.ACTION_M] = 1.0
            mask[self.ACTION_MB] = 1.0
            if self._can_keep(target_index):
                mask[self.ACTION_KEEP] = 1.0
            if self._default_copy_source(target_index) is not None:
                mask[self.ACTION_COPY] = 1.0
            if self._default_delta_source(target_index) is not None:
                mask[self.ACTION_DELTA] = 1.0
            mask[self.ACTION_RAW_INSERT] = 1.0
            if target_index == len(self._old_blocks):
                mask[self.ACTION_APPEND] = 1.0

        if self._old_blocks:
            mask[self.ACTION_BACKUP] = 1.0
        if self._has_trailing_installed_blocks():
            mask[self.ACTION_DELETE] = 1.0
        if len(blocks_to_bytes(self._old_blocks)) > self._new_size_bytes:
            mask[self.ACTION_TRUNCATE] = 1.0

        mask[self.ACTION_CHECKPOINT] = 1.0
        if self.is_transformed():
            mask[self.ACTION_VERIFY] = 1.0
            mask[self.ACTION_COMMIT] = 1.0
        if self._deployment.state.backup_area or self._deployment.state.installed_blocks:
            mask[self.ACTION_ROLLBACK] = 1.0
        return mask

    def get_encoding_ops(self):
        return [dict(operation) for operation in self._encoding_ops]

    def get_deployment_metrics(self):
        return self._deployment.metrics()

    def validate_encoding(self):
        return validate_update(
            self._old_file,
            self._new_file,
            self._encoding_ops,
            self._block_size,
        )

    def get_metrics(self):
        return {
            "steps": self._steps,
            "blocks_total": self._block_num,
            "blocks_remaining": len(self._blocks_remaining),
            "memory_cost": self._mem_cost,
            "encoding_cost": self._encoding_cost,
            "reward": self._reward,
            "action_counts": dict(self._action_counts),
            "transformation_valid": self.is_transformed(),
            "deployment": self.get_deployment_metrics(),
        }

    def is_transformed(self):
        return blocks_to_bytes(self._deployment.state.installed_blocks) == blocks_to_bytes(self._new_blocks)

    def _write_old_block(self, block_index, block, block_md5):
        if block_index < len(self._old_blocks):
            self._old_blocks[block_index] = block
            self._old_block_md5s[block_index] = block_md5
            return

        empty_md5 = hashlib.md5(b"").hexdigest()
        while len(self._old_blocks) < block_index:
            self._old_blocks.append(b"")
            self._old_block_md5s.append(empty_md5)
            self._mem_cost += self._block_size

        self._old_blocks.append(block)
        self._old_block_md5s.append(block_md5)
        self._mem_cost += self._block_size

    def _sync_installed_from_deployment(self):
        self._old_blocks = list(self._deployment.state.installed_blocks)
        self._old_block_md5s = [
            hashlib.md5(block).hexdigest()
            for block in self._old_blocks
        ]

    def _has_trailing_installed_blocks(self):
        return len(self._deployment.state.installed_blocks) > self._block_num

    def _can_keep(self, target_index):
        return (
            0 <= target_index < len(self._old_blocks)
            and 0 <= target_index < len(self._new_blocks)
            and self._old_blocks[target_index] == self._new_blocks[target_index]
        )

    def _backup_source_refs(self):
        refs = []
        for backup_id, record in sorted(self._deployment.state.backup_area.items()):
            refs.append({
                "source": record.index,
                "source_area": "backup",
                "source_backup_id": backup_id,
                "block": record.block,
            })
        return refs

    def _installed_source_refs(self):
        return [
            {
                "source": source_index,
                "source_area": "installed",
                "block": block,
            }
            for source_index, block in enumerate(self._old_blocks)
        ]

    def _source_refs(self):
        return self._installed_source_refs() + self._backup_source_refs()

    def _operation_source_kwargs(self, source_ref):
        kwargs = {
            "source": source_ref["source"],
            "source_area": source_ref.get("source_area", "installed"),
        }
        if source_ref.get("source_backup_id") is not None:
            kwargs["source_backup_id"] = source_ref["source_backup_id"]
        return kwargs

    def _default_copy_source(self, target_index):
        if target_index < 0 or target_index >= len(self._new_blocks):
            return None
        target_block = self._new_blocks[target_index]
        for source_ref in self._source_refs():
            if source_ref["block"] == target_block:
                return source_ref
        return None

    def _default_delta_source(self, target_index):
        if target_index < 0 or target_index >= len(self._new_blocks):
            return None
        seen = set()
        for candidate in self.get_similarity_candidates(target_index):
            source_index = candidate["source_index"]
            key = ("installed", source_index)
            if key in seen:
                continue
            if 0 <= source_index < len(self._old_blocks):
                seen.add(key)
                return {
                    "source": source_index,
                    "source_area": "installed",
                    "block": self._old_blocks[source_index],
                }
        for source_ref in self._source_refs():
            return source_ref
        return None

    def _resolve_source_ref(self, source=None, source_area="installed", source_backup_id=None):
        if source_area == "installed":
            if source is None:
                raise ValueError("installed source action requires a source index")
            source = int(source)
            if source < 0 or source >= len(self._old_blocks):
                raise ValueError(
                    f"source index {source} is out of range for {len(self._old_blocks)} installed blocks"
                )
            return {
                "source": source,
                "source_area": "installed",
                "block": self._old_blocks[source],
            }

        if source_area == "backup":
            backup_id = source_backup_id if source_backup_id is not None else source
            if backup_id not in self._deployment.state.backup_area:
                raise ValueError(f"backup source '{backup_id}' is not available")
            record = self._deployment.state.backup_area[backup_id]
            return {
                "source": record.index,
                "source_area": "backup",
                "source_backup_id": backup_id,
                "block": record.block,
            }

        raise ValueError(f"unsupported source_area '{source_area}'")

    def _require_pending_target(self, target_index):
        if target_index is None:
            raise ValueError("target action requires a target index")
        if target_index < 0 or target_index >= len(self._new_blocks):
            raise ValueError(
                f"target index {target_index} is out of range for {len(self._new_blocks)} target blocks"
            )
        if target_index not in self._blocks_remaining:
            raise ValueError(f"target index {target_index} has already been processed")

    def _mark_target_processed(self, target_index):
        self._blocks_mask[target_index] = 0
        self._blocks_remaining.remove(target_index)

    def _default_backup_id(self, source):
        return f"step-{self._steps + 1}-source-{source}"

    def _configure_observation_space(self):
        feature_names = []
        feature_layout = {}

        def add_features(prefix, count):
            start = len(feature_names)
            if count == 1:
                feature_names.append(prefix)
            else:
                feature_names.extend(f"{prefix}_{index}" for index in range(count))
            feature_layout[prefix] = (start, len(feature_names))

        add_features("target_position", self._block_num)
        add_features("last_action", self.action_space.n)
        add_features("block_count_fraction", 2)
        add_features("remaining_mask", self._block_num)
        add_features("exact_match_mask", self._block_num)
        add_features("remaining_block_count_fraction", 1)
        add_features("remaining_fraction", 1)
        add_features("block_size_fraction", 1)
        add_features("encoding_cost_scaled", 1)
        add_features("memory_cost_scaled", 1)
        add_features("step_count_fraction", 1)
        add_features("progress_fraction", 1)
        add_features("remaining_storage_budget_fraction", 1)
        add_features("remaining_ram_budget_fraction", 1)
        add_features("bandwidth_bytes_per_s_scaled", 1)
        add_features("rollback_ready", 1)
        add_features("checkpoint_interval_progress", 1)
        add_features("unsafe_overwrite_count_scaled", 1)
        add_features("peak_storage_bytes_scaled", 1)
        add_features("peak_ram_bytes_scaled", 1)
        for rank in range(self._similarity_top_k):
            add_features(f"top_{rank}_source_index_fraction", 1)
            add_features(f"top_{rank}_similarity_score", 1)
            add_features(f"top_{rank}_exact_match", 1)

        self._observation_feature_names = tuple(feature_names)
        self._observation_feature_layout = feature_layout
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(feature_names),),
            dtype=np.float32,
        )

    def _configure_observation_normalizers(self):
        self._observation_block_count_scale = max(1, len(self._old_blocks) + self._block_num)
        self._observation_byte_scale = max(1, self._block_size * self._observation_block_count_scale)
        self._observation_step_scale = max(1, self._block_num)
        self._observation_source_index_scale = max(1, len(self._old_blocks) - 1)

    def _scale_count(self, value):
        return float(value) / self._observation_block_count_scale

    def _scale_bytes(self, value):
        if value <= 0:
            return 0.0
        return float(np.log1p(value) / np.log1p(self._observation_byte_scale))

    def _scale_step(self, value):
        return float(value) / self._observation_step_scale

    def _scale_source_index(self, value):
        if value < 0:
            return -1.0
        return float(value) / self._observation_source_index_scale

    def _remaining_budget_fraction(self, budget, used):
        if budget is None:
            return 1.0
        if budget <= 0:
            return 0.0 if used <= 0 else -1.0
        return float((budget - used) / budget)

    def _checkpoint_interval_progress(self):
        interval = self._deployment_config.checkpoint_interval_ops
        if not interval or interval <= 0:
            return 0.0
        return float((self._deployment.state.applied_operations % interval) / interval)

    def _deployment_observation_features(self):
        state = self._deployment.state
        config = self._deployment_config
        peak_storage = state.peak_persistent_storage_bytes
        peak_ram = state.peak_ram_bytes
        return np.array([
            self._remaining_budget_fraction(config.storage_budget_bytes, peak_storage),
            self._remaining_budget_fraction(config.ram_budget_bytes, peak_ram),
            self._scale_bytes(config.bandwidth_bytes_per_s),
            1.0 if state.rollback_ready else 0.0,
            self._checkpoint_interval_progress(),
            self._scale_count(len(state.unsafe_overwrites)),
            self._scale_bytes(peak_storage),
            self._scale_bytes(peak_ram),
        ], dtype=np.float32)

    def _block_similarity_score(self, target_block, source_block):
        if not target_block and not source_block:
            return 1.0
        if not target_block or not source_block:
            return 0.0

        sample_size = min(len(target_block), len(source_block), self._similarity_sample_bytes)
        if sample_size <= 0:
            return 0.0

        sampled_target = target_block[:sample_size]
        sampled_source = source_block[:sample_size]
        byte_match_fraction = sum(
            1 for target_byte, source_byte in zip(sampled_target, sampled_source)
            if target_byte == source_byte
        ) / sample_size
        length_fraction = min(len(target_block), len(source_block)) / max(len(target_block), len(source_block))
        return float((0.8 * byte_match_fraction) + (0.2 * length_fraction))

    def _empty_similarity_candidate(self):
        return {
            "source_index": -1,
            "similarity_score": 0.0,
            "exact_match": False,
        }

    def _precompute_similarity_candidates(self):
        candidates_by_target = []
        for target_index, target_block in enumerate(self._new_blocks):
            candidates = []
            target_md5 = self._new_block_md5s[target_index]
            for source_index, source_block in enumerate(self._old_blocks):
                exact_match = target_md5 == self._old_block_md5s[source_index]
                similarity_score = 1.0 if exact_match else self._block_similarity_score(target_block, source_block)
                candidates.append({
                    "source_index": source_index,
                    "similarity_score": similarity_score,
                    "exact_match": exact_match,
                })
            candidates.sort(key=lambda candidate: (-candidate["similarity_score"], candidate["source_index"]))
            while len(candidates) < self._similarity_top_k:
                candidates.append(self._empty_similarity_candidate())
            candidates_by_target.append(candidates[:self._similarity_top_k])
        return candidates_by_target

    def _exact_match_mask(self):
        available_source_md5s = set(self._old_block_md5s)
        return np.array([
            1.0 if target_md5 in available_source_md5s else 0.0
            for target_md5 in self._new_block_md5s
        ], dtype=np.float32)

    # helper function for md5 number of blocks
    def md5_blocks(self, filename, block_size=4096):
        """Calculates the MD5 hash of each block in a file.

        Args:
        filename: The path to the file.
        block_size: The size of each block in bytes.

        Returns:
        A list of hexadecimal MD5 hashes, one for each block.
        """
        md5_hashes = []
        with open(filename, "rb") as f:
            while True:
                block = f.read(block_size)
                if not block:
                    break
                hasher = hashlib.md5()
                hasher.update(block)
                md5_hashes.append(hasher.hexdigest())
        return md5_hashes


    # helper function for reading blocks from file
    def read_blocks(self, filename, block_size=4096):
        """Calculates the MD5 hash of each block in a file.

        Args:
        filename: The path to the file.
        block_size: The size of each block in bytes.

        Returns:
        A list of hexadecimal MD5 hashes, one for each block.
        """
        blocks = []
        with open(filename, "rb") as f:
            while True:
                block = f.read(block_size)
                if not block:
                    break
                blocks.append(block)
        return blocks


    def _get_obs(self):
        target_position = np.zeros(self._block_num, dtype=np.float32)
        if 0 <= self._current_block_pos < self._block_num:
            target_position[self._current_block_pos] = 1.0

        action_taken = np.eye(self.action_space.n, dtype=np.float32)[self._current_action]
        block_counts = np.array([
            self._scale_count(len(self._old_block_md5s)),
            self._scale_count(len(self._new_block_md5s)),
        ], dtype=np.float32)

        remaining_mask = np.array(self._blocks_mask, dtype=np.float32)
        exact_match_mask = self._exact_match_mask()
        remaining_count = len(self._blocks_remaining)
        remaining_fraction = remaining_count / self._block_num if self._block_num else 0.0
        progress_fraction = self._steps / self._block_num if self._block_num else 0.0
        scalar_state = np.array([
            self._scale_step(remaining_count),
            remaining_fraction,
            self._block_size / self._observation_byte_scale,
            self._scale_bytes(self._encoding_cost),
            self._scale_bytes(self._mem_cost),
            self._scale_step(self._steps),
            progress_fraction,
        ], dtype=np.float32)
        deployment_state = self._deployment_observation_features()

        top_k_features = []
        for candidate in self.get_similarity_candidates(self._current_block_pos):
            top_k_features.extend([
                self._scale_source_index(candidate["source_index"]),
                candidate["similarity_score"],
                1.0 if candidate["exact_match"] else 0.0,
            ])
        top_k_features = np.array(top_k_features, dtype=np.float32)

        obs = np.concatenate((
            target_position,
            action_taken,
            block_counts,
            remaining_mask,
            exact_match_mask,
            scalar_state,
            deployment_state,
            top_k_features,
        )).astype(np.float32)
        return obs


    def _get_info(self):
        return {
            "old_file_md5s": self._old_block_md5s,
            "new_file_md5s": self._new_block_md5s,
            "encoding": self._encoding,
            "encoding_ops": self.get_encoding_ops(),
            "metrics": self.get_metrics(),
            "observation_features": self.get_observation_feature_names(),
            "current_similarity_candidates": self.get_similarity_candidates(),
            "action_families": self.get_action_family_names(),
            "action_mask": self.get_action_mask().tolist(),
        }


    def reset(self, seed=None, options=None):
        # super().reset(seed=seed)
        self._identical_block_found = False

        self._old_blocks = self.read_blocks(self._old_file, self._block_size)

        self._old_block_md5s = self.md5_blocks(self._old_file, self._block_size)
        self._configure_observation_normalizers()
        self._similarity_candidates = self._precompute_similarity_candidates()

        self._mem_cost = len(self._old_blocks) * self._block_size
        self._encoding_cost = 0

        self._reward = 0.0
        self._encoding = ''
        self._encoding_ops = []
        self._backup_index_to_id = {}
        self._deployment = DeploymentSimulator(
            self._old_blocks,
            self._block_size,
            config=self._deployment_config,
            expected_blocks=self._new_blocks,
        )
        self._steps = 0
        self._action_counts = self._empty_action_counts()
        self._last_budget_violation_count = 0
        self._last_unsafe_overwrite_count = 0
        self._last_flash_write_bytes = 0
        self._last_package_size_bytes = 0
        self._last_network_bytes = 0
        self._last_peak_ram_bytes = 0
        self._last_peak_storage_bytes = 0

        # Table for storing yet-to-be-processed target/new block indices.
        self._blocks_remaining = [i for i in range(self._block_num)]
        self._blocks_mask = [1 for i in range(self._block_num)]

        self._current_block_pos = 0
        
        self._current_block = self._old_blocks[0] if self._old_blocks else None
        if self._block_num == 0:
            self._finalize_structured_update()
            self._deployment.complete(expected_blocks=self._new_blocks)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def _record_operation(self, operation):
        self._deployment.apply_operation(operation)
        self._encoding_ops.append(operation)

    def _finalize_structured_update(self):
        installed_size = sum(len(block) for block in self._deployment.state.installed_blocks)
        if installed_size > self._new_size_bytes:
            self._record_operation(make_truncate_operation(new_size_bytes=self._new_size_bytes))
            self._sync_installed_from_deployment()

    def _parse_action_payload(self, action, next_block):
        if isinstance(action, dict):
            payload = dict(action)
            op_name = payload.get("op", payload.get("action", payload.get("operation")))
            if op_name is None:
                raise ValueError("structured action requires an 'op' field")
            if isinstance(op_name, int):
                action_id = int(op_name)
                if action_id not in self._action_to_direction:
                    raise ValueError(f"unknown action id {action_id}")
                op_name = self._action_to_direction[action_id]
            else:
                op_name = str(op_name)
            if op_name not in self.ACTION_NAME_TO_ID:
                raise ValueError(f"unknown action '{op_name}'")
            payload["op"] = op_name
            if next_block is not None and "target" not in payload:
                payload["target"] = next_block
            return payload, False

        action_id = int(action)
        if action_id not in self._action_to_direction:
            raise ValueError(f"unknown action id {action_id}")
        return {
            "op": self._action_to_direction[action_id],
            "target": next_block,
        }, action_id in {self.ACTION_M, self.ACTION_MB}

    def _structured_target(self, payload):
        target = payload.get("target")
        if target is None:
            return self._blocks_remaining[0] if self._blocks_remaining else None
        return int(target)

    def step_structured(self, action):
        return self.step(action)

    def _apply_keep_action(self, target_index):
        self._require_pending_target(target_index)
        if not self._can_keep(target_index):
            raise ValueError(f"keep is invalid for target {target_index}")
        self._record_operation(
            make_keep_operation(
                target=target_index,
                target_block=self._new_blocks[target_index],
            )
        )
        self._encoding += "K"
        self._encoding_cost += 1
        self._mark_target_processed(target_index)

    def _apply_copy_action(self, target_index, payload):
        self._require_pending_target(target_index)
        source_ref = None
        if "source" in payload or payload.get("source_area") == "backup":
            source_ref = self._resolve_source_ref(
                source=payload.get("source"),
                source_area=payload.get("source_area", "installed"),
                source_backup_id=payload.get("source_backup_id"),
            )
        else:
            source_ref = self._default_copy_source(target_index)

        if source_ref is None:
            raise ValueError(f"copy has no matching source for target {target_index}")
        target_block = self._new_blocks[target_index]
        if source_ref["block"] != target_block:
            raise ValueError(f"copy source does not match target {target_index}")

        self._record_operation(
            make_copy_operation(
                target=target_index,
                target_block=target_block,
                **self._operation_source_kwargs(source_ref),
            )
        )
        self._write_old_block(
            target_index,
            target_block,
            self._new_block_md5s[target_index],
        )
        self._encoding += f"C{source_ref['source']}"
        self._encoding_cost += 2 + len(str(source_ref["source"]))
        self._mark_target_processed(target_index)

    def _apply_delta_action(self, target_index, payload):
        self._require_pending_target(target_index)
        if "source" in payload or payload.get("source_area") == "backup":
            source_ref = self._resolve_source_ref(
                source=payload.get("source"),
                source_area=payload.get("source_area", "installed"),
                source_backup_id=payload.get("source_backup_id"),
            )
        else:
            source_ref = self._default_delta_source(target_index)
        if source_ref is None:
            raise ValueError(f"delta has no source for target {target_index}")

        target_block = self._new_blocks[target_index]
        patch_bytes = bsdiff4.diff(source_ref["block"], target_block)
        self._record_operation(
            make_delta_operation(
                target=target_index,
                patch_bytes=patch_bytes,
                target_block=target_block,
                codec=DeltaCodec.BSDIFF4,
                **self._operation_source_kwargs(source_ref),
            )
        )
        self._write_old_block(
            target_index,
            target_block,
            self._new_block_md5s[target_index],
        )
        self._encoding += f"D{source_ref['source']}#{patch_bytes!r}"
        self._encoding_cost += 3 + len(str(source_ref["source"])) + len(patch_bytes)
        self._mark_target_processed(target_index)

    def _apply_raw_insert_action(self, target_index, payload):
        self._require_pending_target(target_index)
        data = payload.get("data", self._new_blocks[target_index])
        if data != self._new_blocks[target_index]:
            raise ValueError(f"raw_insert payload does not match target {target_index}")
        self._record_operation(make_raw_insert_operation(target=target_index, data=data))
        self._write_old_block(
            target_index,
            data,
            self._new_block_md5s[target_index],
        )
        self._encoding += "R"
        self._encoding_cost += 2 + len(data)
        self._mark_target_processed(target_index)

    def _apply_backup_action(self, payload, default_source=None):
        source = payload.get("source", default_source)
        if source is None:
            source = self._structured_target(payload)
        if source is None:
            raise ValueError("backup action requires a source index")
        source = int(source)
        if source < 0 or source >= len(self._deployment.state.installed_blocks):
            raise ValueError(
                f"backup source {source} is out of range for {len(self._deployment.state.installed_blocks)} installed blocks"
            )
        backup_id = payload.get("backup_id", self._default_backup_id(source))
        self._record_operation(make_backup_operation(source=source, backup_id=backup_id))
        self._backup_index_to_id[source] = backup_id
        self._mem_cost += self._block_size
        self._encoding += "B"
        self._encoding_cost += 1

    def _apply_append_action(self, target_index, payload):
        self._require_pending_target(target_index)
        if target_index != len(self._old_blocks):
            raise ValueError(
                f"append target {target_index} must equal current block count {len(self._old_blocks)}"
            )
        data = payload.get("data", self._new_blocks[target_index])
        if data != self._new_blocks[target_index]:
            raise ValueError(f"append payload does not match target {target_index}")
        self._record_operation(make_append_operation(data=data, target=target_index))
        self._write_old_block(
            target_index,
            data,
            self._new_block_md5s[target_index],
        )
        self._encoding += "A"
        self._encoding_cost += 2 + len(data)
        self._mark_target_processed(target_index)

    def _apply_delete_action(self, payload):
        target = int(payload.get("target", self._block_num))
        count = int(payload.get("count", 1))
        if count <= 0:
            raise ValueError("delete count must be positive")
        if target < self._block_num:
            raise ValueError("delete is restricted to installed tail blocks beyond the target image")
        if target < 0 or target + count > len(self._old_blocks):
            raise ValueError(
                f"delete range [{target}, {target + count}) is out of range for {len(self._old_blocks)} installed blocks"
            )
        self._record_operation(make_delete_operation(target=target, count=count))
        self._sync_installed_from_deployment()
        self._encoding += "X"
        self._encoding_cost += 1

    def _apply_truncate_action(self, payload):
        new_size_bytes = int(payload.get("new_size_bytes", self._new_size_bytes))
        if new_size_bytes < 0:
            raise ValueError("truncate new_size_bytes cannot be negative")
        if new_size_bytes > len(blocks_to_bytes(self._old_blocks)):
            raise ValueError("truncate cannot grow the installed image")
        self._record_operation(make_truncate_operation(new_size_bytes=new_size_bytes))
        self._sync_installed_from_deployment()
        self._encoding += "T"
        self._encoding_cost += 1

    def _apply_checkpoint_action(self, payload):
        self._record_operation(make_checkpoint_operation(step_id=payload.get("step_id")))
        self._encoding += "P"
        self._encoding_cost += 1

    def _apply_verify_action(self, payload):
        sha256 = payload.get("sha256")
        size_bytes = payload.get("size_bytes", self._new_size_bytes)
        if sha256 is None and int(size_bytes) == self._new_size_bytes:
            sha256 = hashlib.sha256(blocks_to_bytes(self._new_blocks)).hexdigest()
        self._record_operation(make_verify_operation(sha256=sha256, size_bytes=size_bytes))
        self._encoding += "V"
        self._encoding_cost += 1

    def _apply_commit_action(self):
        if not self.is_transformed():
            raise ValueError("commit requires the installed image to match the target image")
        self._record_operation(make_commit_operation())
        for target_index in list(self._blocks_remaining):
            self._mark_target_processed(target_index)
        self._encoding += "Q"
        self._encoding_cost += 1

    def _apply_rollback_action(self, payload):
        self._record_operation(make_rollback_operation(backup_id=payload.get("backup_id")))
        self._sync_installed_from_deployment()
        self._encoding += "Z"
        self._encoding_cost += 1

    def _apply_structured_action(self, payload):
        op_name = payload["op"]
        target_index = self._structured_target(payload)

        if op_name == "keep":
            self._apply_keep_action(target_index)
        elif op_name == "copy":
            self._apply_copy_action(target_index, payload)
        elif op_name == "delta":
            self._apply_delta_action(target_index, payload)
        elif op_name == "raw_insert":
            self._apply_raw_insert_action(target_index, payload)
        elif op_name == "backup":
            self._apply_backup_action(payload)
        elif op_name == "append":
            self._apply_append_action(target_index, payload)
        elif op_name == "delete":
            self._apply_delete_action(payload)
        elif op_name == "truncate":
            self._apply_truncate_action(payload)
        elif op_name == "checkpoint":
            self._apply_checkpoint_action(payload)
        elif op_name == "verify":
            self._apply_verify_action(payload)
        elif op_name == "commit":
            self._apply_commit_action()
        elif op_name == "rollback":
            self._apply_rollback_action(payload)
        else:
            raise ValueError(f"structured operation '{op_name}' is not supported by OTAEnv")

    def _source_reference(self, block_index):
        backup_id = self._backup_index_to_id.get(block_index)
        if backup_id is None:
            return {"source": block_index}
        return {
            "source": block_index,
            "source_area": "backup",
            "source_backup_id": backup_id,
        }

    def _delta_source_indices(self, block_index):
        source_indices = []
        seen = set()
        for candidate in self.get_similarity_candidates(block_index):
            source_index = candidate["source_index"]
            if source_index in seen:
                continue
            if 0 <= source_index < len(self._old_blocks):
                source_indices.append(source_index)
                seen.add(source_index)
        if not source_indices and self._old_blocks:
            source_indices.append(min(block_index, len(self._old_blocks) - 1))
        return source_indices


    def modify(self):
        if (
            self._current_block_pos < len(self._old_blocks)
            and self._old_blocks[self._current_block_pos] == self._current_block
        ):
            self._identical_block_found = True
            self._encoding += "MK"
            self._record_operation(
                make_keep_operation(
                    target=self._current_block_pos,
                    target_block=self._current_block,
                )
            )
            self._encoding_cost += 1
            return

        match_block_index = [i for i,x in enumerate(self._old_blocks) if x==self._current_block]
        if len(match_block_index) > 0:
            self._identical_block_found = True
            # find the matching block
            self._encoding += 'MC' # stand for copy modify
            self._encoding += str(match_block_index[0])
            self._record_operation(
                make_copy_operation(
                    target=self._current_block_pos,
                    target_block=self._current_block,
                    **self._source_reference(match_block_index[0]),
                )
            )
            self._write_old_block(
                self._current_block_pos,
                self._old_blocks[match_block_index[0]],
                self._new_block_md5s[self._current_block_pos],
            )

            # update encoding cost: MC + first matched block index
            self._encoding_cost += 2 + len(str(match_block_index[0]))

        else:
            # no indentical block found
            self._identical_block_found = False
            if not self._old_blocks:
                self._encoding += "MR"
                self._record_operation(
                    make_raw_insert_operation(
                        target=self._current_block_pos,
                        data=self._current_block,
                    )
                )
                self._encoding_cost += 2 + len(self._current_block)
                self._write_old_block(
                    self._current_block_pos,
                    self._current_block,
                    self._new_block_md5s[self._current_block_pos],
                )
                return

            # perform delta update with the closest block
            min_patch_bytes = b''
            min_patch_size = float('inf')
            closest_block_index = 0
            for i in self._delta_source_indices(self._current_block_pos):
                patch_bytes = bsdiff4.diff(self._old_blocks[i], self._current_block)
                if len(patch_bytes) < min_patch_size:
                    min_patch_bytes = patch_bytes
                    min_patch_size = len(patch_bytes)
                    closest_block_index = i

            self._encoding += "MD" # stand for delta modify
            self._encoding += str(closest_block_index)
            self._encoding += '#' # separating character
            self._encoding += str(min_patch_bytes)
            self._record_operation(
                make_delta_operation(
                    target=self._current_block_pos,
                    patch_bytes=min_patch_bytes,
                    target_block=self._current_block,
                    codec=DeltaCodec.BSDIFF4,
                    **self._source_reference(closest_block_index),
                )
            )


            self._encoding_cost += 3 + len(str(closest_block_index)) + min_patch_size

            self._write_old_block(
                self._current_block_pos,
                self._current_block,
                self._new_block_md5s[self._current_block_pos],
            )


    def _apply_legacy_action(self, action_name, target_index):
        self._require_pending_target(target_index)
        self._current_block_pos = target_index
        self._current_block = self._new_blocks[target_index]

        self._mark_target_processed(target_index)

        backup_block = None
        backup_md5 = None
        backup_id = None
        if (
            self._current_block_pos < len(self._old_blocks)
            and self._current_block_pos < len(self._deployment.state.installed_blocks)
        ):
            backup_block = self._old_blocks[self._current_block_pos]
            backup_md5 = self._old_block_md5s[self._current_block_pos]
            backup_id = f"step-{self._steps + 1}-target-{self._current_block_pos}"

        if action_name == "M":
            self.modify()
        elif action_name == "MB":
            if backup_block is not None:
                self._record_operation(
                    make_backup_operation(
                        source=self._current_block_pos,
                        backup_id=backup_id,
                    )
                )
            self.modify()
            if backup_block is not None:
                backup_index = len(self._old_blocks)
                self._old_blocks.append(backup_block)
                self._old_block_md5s.append(backup_md5)
                self._backup_index_to_id[backup_index] = backup_id
            self._mem_cost += self._block_size
            self._encoding_cost += 1
        else:
            raise ValueError(f"legacy action '{action_name}' is not supported")

    def _flash_write_budget_bytes(self):
        return max(1, self._old_size_bytes + self._new_size_bytes)

    def _reward_constraint_scale_bytes(self):
        return max(1, self._new_size_bytes, self._old_size_bytes, self._block_size)

    def _calculate_constraint_reward_penalty(self, metrics):
        constraints = self._reward_constraints
        if not constraints.enabled():
            return 0.0

        scale = self._reward_constraint_scale_bytes()
        package_delta = max(0, metrics["package_size_bytes"] - self._last_package_size_bytes)
        network_delta = max(0, metrics["network_bytes"] - self._last_network_bytes)
        flash_delta = max(0, metrics["flash_write_bytes"] - self._last_flash_write_bytes)
        peak_ram_delta = max(0, metrics["peak_ram_bytes"] - self._last_peak_ram_bytes)
        peak_storage_delta = max(
            0,
            metrics["peak_persistent_storage_bytes"] - self._last_peak_storage_bytes,
        )
        budget_delta = max(
            0,
            metrics["budget_violation_count"] - self._last_budget_violation_count,
        )

        penalty = 0.0
        penalty += constraints.package_size_weight * (package_delta / scale)
        penalty += constraints.network_bytes_weight * (network_delta / scale)
        penalty += constraints.flash_write_weight * (flash_delta / scale)
        penalty += constraints.peak_ram_weight * (peak_ram_delta / scale)
        penalty += constraints.peak_storage_weight * (peak_storage_delta / scale)
        penalty += constraints.budget_violation_weight * budget_delta
        if constraints.rollback_required_penalty > 0.0 and not metrics["rollback_ready"]:
            penalty += constraints.rollback_required_penalty
        return penalty

    def _dense_step_reward(self):
        return (1000 - self._encoding_cost * 0.1 - self._mem_cost * 0.01) / 1000000.0

    def _terminal_replay_reward(self):
        replay_result = self.validate_encoding()
        if replay_result.valid:
            return self.VALID_REPLAY_BONUS
        error_count = len(getattr(replay_result, "errors", []))
        return -(
            self.INVALID_REPLAY_PENALTY
            + min(error_count, 5) * self.REPLAY_ERROR_PENALTY
        )

    def _calculate_step_reward(self, done):
        metrics = self.get_deployment_metrics()
        flash_write_bytes = metrics["flash_write_bytes"]
        budget_violation_count = metrics["budget_violation_count"]
        unsafe_overwrite_count = metrics["unsafe_overwrite_count"]

        new_budget_violations = max(
            0,
            budget_violation_count - self._last_budget_violation_count,
        )
        new_unsafe_overwrites = max(
            0,
            unsafe_overwrite_count - self._last_unsafe_overwrite_count,
        )
        flash_write_delta = max(0, flash_write_bytes - self._last_flash_write_bytes)

        reward = self._dense_step_reward()
        reward -= new_budget_violations * self.BUDGET_VIOLATION_PENALTY
        reward -= budget_violation_count * self.BUDGET_VIOLATION_STATE_PENALTY
        reward -= new_unsafe_overwrites * self.UNSAFE_OVERWRITE_PENALTY
        if not metrics["rollback_ready"]:
            reward -= self.ROLLBACK_UNREADY_STATE_PENALTY

        flash_budget = self._flash_write_budget_bytes()
        reward -= self.FLASH_WRITE_COST_WEIGHT * (flash_write_delta / flash_budget)
        previous_excess_flash = max(0, self._last_flash_write_bytes - flash_budget)
        current_excess_flash = max(0, flash_write_bytes - flash_budget)
        reward -= self.EXCESS_FLASH_WRITE_PENALTY * (
            (current_excess_flash - previous_excess_flash) / flash_budget
        )

        if done:
            try:
                reward += self._terminal_replay_reward()
            except Exception:
                reward -= self.INVALID_REPLAY_PENALTY

        reward -= self._calculate_constraint_reward_penalty(metrics)

        self._last_budget_violation_count = budget_violation_count
        self._last_unsafe_overwrite_count = unsafe_overwrite_count
        self._last_flash_write_bytes = flash_write_bytes
        self._last_package_size_bytes = metrics["package_size_bytes"]
        self._last_network_bytes = metrics["network_bytes"]
        self._last_peak_ram_bytes = metrics["peak_ram_bytes"]
        self._last_peak_storage_bytes = metrics["peak_persistent_storage_bytes"]
        return reward

    def step(self, action, next_block=None):
        payload, legacy_action = self._parse_action_payload(action, next_block)
        action_name = payload["op"]
        action_id = self.ACTION_NAME_TO_ID[action_name]

        target_index = self._structured_target(payload)

        if legacy_action:
            self._apply_legacy_action(action_name, target_index)
        else:
            self._apply_structured_action(payload)

        if target_index is not None and 0 <= target_index < len(self._new_blocks):
            self._current_block_pos = target_index
            self._current_block = self._new_blocks[target_index]
        self._current_action = action_id
        self._steps += 1
        self._action_counts[action_name] += 1

        done = False
        if len(self._blocks_remaining) == 0:
            self._finalize_structured_update()
            self._deployment.complete(expected_blocks=self._new_blocks)
            done = True
        elif action_name == "commit" and self.is_transformed():
            self._deployment.complete(expected_blocks=self._new_blocks)
            done = True

        step_reward = self._calculate_step_reward(done)
        self._reward += step_reward
        observation = self._get_obs()
        info = self._get_info()

        # print(self._blocks_remaining)

        if done:
            print("all blocks processed...")
            print("memory cost: %s" % self._mem_cost)
            print("encoding size: %s" % self._encoding_cost)
            print("reward: %f" % self._reward)

        return observation, step_reward, done, info


    def close(self):
        print("exiting OTA Environment...")
        return

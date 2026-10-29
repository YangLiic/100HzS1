"""Model-independent, immutable expert seed sets shared across checkouts and hosts."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import time
import uuid
import yaml

DEFAULT_ROOT = Path("/DRoboticsAILab/users/licheng.yang/robotwin2.0/100HzS1/seed_registry")
SET_SIZE = 100


def registry_root(value=None):
    return Path(value or os.environ.get("S1_SEED_ROOT", DEFAULT_ROOT)).resolve()


def validate_name(value):
    if not re.fullmatch(r"[a-zA-Z0-9_][a-zA-Z0-9_.-]*", value) or value in {".", ".."}:
        raise ValueError("Task/profile must be a plain name")
    return value


def read_verified(path, metadata, task, profile):
    data = Path(path).read_bytes()
    values = [int(line) for line in data.decode().splitlines() if line.strip()]
    meta = json.loads(Path(metadata).read_text())
    if (len(values) != SET_SIZE or len(set(values)) != SET_SIZE
            or any(seed < 0 for seed in values) or meta.get("verified") is not True
            or meta.get("count") != SET_SIZE or meta.get("task") != task
            or meta.get("task_config") != profile
            or meta.get("seed_sha256") != hashlib.sha256(data).hexdigest()):
        raise ValueError("A shared seed set must contain exactly 100 distinct verified seeds for this task/profile")
    return values, meta


class SeedRegistry:
    def __init__(self, robotwin, root=None):
        self.robotwin, self.root = Path(robotwin).resolve(), registry_root(root)

    def directory(self, task, profile):
        return self.root / validate_name(task) / validate_name(profile)

    def scene_identity(self, task, profile):
        root = self.robotwin
        files = [root/f"envs/{validate_name(task)}.py", root/"envs/_base_task.py",
                 root/"envs/_GLOBAL_CONFIGS.py", root/"script/expert_check.py",
                 root/f"task_config/{validate_name(profile)}.yml",
                 root/"task_config/_camera_config.yml", root/"task_config/_embodiment_config.yml"]
        files += sorted((root/"envs/utils").glob("*.py"))
        files += sorted((root/"envs/robot").glob("*.py"))
        hashes = {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        settings = yaml.safe_load((root/f"task_config/{profile}.yml").read_text())
        embodiments = yaml.safe_load((root/"task_config/_embodiment_config.yml").read_text())
        for kind in set(settings["embodiment"][:2] if len(settings["embodiment"])==3 else settings["embodiment"]):
            config = root / embodiments[kind]["file_path"] / "config.yml"
            hashes[f"robot_config/{kind}"] = hashlib.sha256(config.read_bytes()).hexdigest()
        encoded = json.dumps(hashes, sort_keys=True).encode()
        return {"task":task,"task_config":profile,"files":hashes,
                "sha256":hashlib.sha256(encoded).hexdigest()}

    def _bundled(self, task, profile):
        path = self.robotwin/"task_config/seedlists"/f"{task}_{profile}.txt"
        return path, path.with_suffix(".expert.json")

    def get(self, task, profile):
        identity = self.scene_identity(task, profile)
        directory = self.directory(task, profile)/"set"
        path, meta = directory/"seeds.txt", directory/"summary.json"
        commit_path = directory/"COMMITTED.json"
        if commit_path.exists():
            commit = json.loads(commit_path.read_text())
            if hashlib.sha256(meta.read_bytes()).hexdigest() != commit["metadata_sha256"]:
                raise ValueError("Committed seed metadata was modified")
            values, info = read_verified(path, meta, task, profile)
            if info["seed_sha256"] != commit["seed_sha256"]:
                raise ValueError("Committed seed values were modified")
            if info.get("scene_identity", {}).get("sha256") != identity["sha256"]:
                raise ValueError("Scene/robot/camera definitions changed; refusing to reuse or overwrite the locked seed set")
            bundled, summary = self._bundled(task, profile)
            if bundled.exists() and summary.exists():
                _, bundled_info = read_verified(bundled, summary, task, profile)
                if bundled_info["seed_sha256"] != info["seed_sha256"]:
                    raise ValueError("Bundled seeds conflict with the shared locked seed set")
            return path, info
        return None

    def bundled(self, task, profile):
        path, metadata = self._bundled(task, profile)
        if not path.exists() and not metadata.exists():
            return None
        _, info = read_verified(path, metadata, task, profile)
        identity = self.scene_identity(task, profile)
        if info.get("scene_identity") and info["scene_identity"]["sha256"] != identity["sha256"]:
            raise ValueError("Bundled scene/robot/camera definitions changed")
        # Legacy verified lists record a subset of these physical-scene sources.
        for name, digest in info.get("source_sha256", {}).items():
            if name in identity["files"] and identity["files"][name] != digest:
                raise ValueError(f"Verified scene source changed: {name}")
        return path, info

    def resolve(self, task, profile):
        """Read-only lookup. It never selects new seeds or takes a GPU."""
        return self.get(task, profile) or self.bundled(task, profile)

    @contextmanager
    def _lock(self, task, profile, timeout):
        parent = self.directory(task, profile)
        parent.mkdir(parents=True, exist_ok=True)
        lock = parent/".selecting"
        token = uuid.uuid4().hex
        deadline = time.monotonic()+timeout
        owned = False
        try:
            while True:
                try:
                    lock.mkdir()
                    owned = True
                    (lock/"owner.json").write_text(json.dumps(dict(token=token,pid=os.getpid(),host=socket.gethostname())))
                    break
                except FileExistsError:
                    # Published sets are immediately reusable even while the owner exits.
                    if self.get(task, profile):
                        break
                    owner_path = lock/"owner.json"
                    if owner_path.exists():
                        try:
                            owner = json.loads(owner_path.read_text())
                            if owner.get("host")==socket.gethostname():
                                try:
                                    os.kill(owner["pid"],0)
                                except ProcessLookupError:
                                    raise RuntimeError(f"Interrupted seed selection lock: {lock}; recover its build state before removing the stale lock")
                        except (FileNotFoundError,json.JSONDecodeError):
                            pass
                    if time.monotonic()>=deadline:
                        raise TimeoutError(f"Another process is preparing this task's seeds: {lock}")
                    time.sleep(.1)
            yield
        finally:
            if owned:
                # No lock stealing: only this owner may remove this directory.
                current = json.loads((lock/"owner.json").read_text())
                if current["token"]==token:
                    shutil.rmtree(lock)

    def ensure(self, task, profile, select, timeout=3600):
        found = self.get(task, profile)
        if found:
            return found
        with self._lock(task, profile, timeout):
            found = self.get(task, profile)
            if found:
                return found
            source = self.bundled(task, profile)
            if source is None:
                build = self.directory(task, profile)/"build"
                if (build/"EXPERT_SEEDS_COMPLETED").exists():
                    source = build/"seeds.txt", json.loads((build/"summary.json").read_text())
                else:
                    source = select(build)
            path, info = source
            _, info = read_verified(path, Path(path).with_name("summary.json") if Path(path).name=="seeds.txt"
                                    else Path(path).with_suffix(".expert.json"), task, profile)
            identity = self.scene_identity(task, profile)
            if info.get("scene_identity") and info["scene_identity"]["sha256"] != identity["sha256"]:
                raise ValueError("Completed screening has different scene definitions")
            for name, digest in info.get("source_sha256", {}).items():
                if name in identity["files"] and identity["files"][name] != digest:
                    raise ValueError(f"Completed expert screening has different scene source: {name}")
            marker = Path(path).with_name("EXPERT_SEEDS_COMPLETED")
            if marker.exists() and marker.read_text().strip() != info["seed_sha256"]:
                raise ValueError("Expert completion marker does not match the seed values")
            info = dict(info, scene_identity=identity,
                        seed_set_scope="task_and_scene_profile", policy_checkpoint=None)
            directory = self.directory(task, profile)/"set"
            directory.mkdir(exist_ok=True)
            def write_atomic(target, data):
                temporary = target.with_name("."+target.name+"."+uuid.uuid4().hex+".tmp")
                try:
                    with temporary.open("wb") as handle:
                        handle.write(data)
                        handle.flush()
                        os.fsync(handle.fileno())
                    temporary.replace(target)
                finally:
                    temporary.unlink(missing_ok=True)
            # This shared filesystem supports atomic file replacement but not directory rename.
            metadata_bytes = (json.dumps(info,indent=2)+"\n").encode()
            write_atomic(directory/"seeds.txt",Path(path).read_bytes())
            write_atomic(directory/"summary.json",metadata_bytes)
            commit = dict(seed_sha256=info["seed_sha256"],
                          metadata_sha256=hashlib.sha256(metadata_bytes).hexdigest())
            write_atomic(directory/"COMMITTED.json",(json.dumps(commit)+"\n").encode())
            return self.get(task, profile)

    def export(self, task, profile):
        """Explicitly publish a locked set for version control; never overwrite another set."""
        found = self.get(task, profile)
        if found is None:
            raise FileNotFoundError("Prepare the shared seed set first")
        path, info = found
        target, metadata = self._bundled(task, profile)
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() or metadata.exists():
            _, existing = read_verified(target, metadata, task, profile)
            if existing["seed_sha256"] != info["seed_sha256"]:
                raise FileExistsError("Refusing to overwrite the existing bundled seed set")
            return target
        target.write_bytes(path.read_bytes())
        metadata.write_text(json.dumps(info,indent=2)+"\n")
        return target

#!/usr/bin/env python3
"""
Cross Table Dependency Tracker

Tracks file dependencies for cross table builds to avoid unnecessary rebuilds.
Uses SHA256 hashes of source files to detect changes.

Supports:
- level.ai hash
- level.spawn hash
- original_spawn hash (optional, for merged graph points)
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass, asdict, field

from utils import log, logWarning


@dataclass
class CrossTableDependencies:
    """Dependencies for a single cross table"""
    level_name: str
    level_ai_hash: str
    level_spawn_hash: str
    cross_table_path: str
    timestamp: float  # Unix timestamp of build
    original_spawn_hash: str = ""  # Optional: hash of original_spawn file
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'CrossTableDependencies':
        # Handle old format without original_spawn_hash
        if 'original_spawn_hash' not in data:
            data['original_spawn_hash'] = ""
        return cls(**data)


class DependencyTracker:
    """
    Track cross table build dependencies
    
    Stores hashes of input files (level.ai, level.spawn, original_spawn) to detect changes.
    Stored in build/.cross_table_deps.json
    """
    
    def __init__(self, build_dir: Path):
        """
        Initialize tracker
        
        Args:
            build_dir: Build directory (e.g., build/ or tmp/)
        """
        self.build_dir = Path(build_dir)
        self.build_dir.mkdir(parents=True, exist_ok=True)
        self.deps_file = self.build_dir / ".cross_table_deps.json"
        self.dependencies: Dict[str, CrossTableDependencies] = {}
        self._load()
    
    def _load(self):
        """Load dependencies from JSON file"""
        if self.deps_file.exists():
            try:
                with open(self.deps_file, 'r') as f:
                    data = json.load(f)
                
                for level_name, dep_data in data.items():
                    self.dependencies[level_name] = CrossTableDependencies.from_dict(dep_data)
            except Exception as e:
                logWarning(f"Could not load dependencies: {e}")
                self.dependencies = {}
    
    def _save(self):
        """Save dependencies to JSON file"""
        try:
            data = {
                name: dep.to_dict() 
                for name, dep in self.dependencies.items()
            }
            
            with open(self.deps_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logWarning(f"Could not save dependencies: {e}")
    
    @staticmethod
    def _hash_file(filepath: Path) -> str:
        """
        Calculate SHA256 hash of a file
        
        Args:
            filepath: Path to file
        
        Returns:
            Hex string of SHA256 hash
        """
        if not filepath or not filepath.exists():
            return ""
        
        sha256 = hashlib.sha256()
        
        try:
            with open(filepath, 'rb') as f:
                # Read in chunks for large files
                while chunk := f.read(8192):
                    sha256.update(chunk)
            
            return sha256.hexdigest()
        except Exception:
            return ""
    
    def needs_rebuild(self, level_name: str, 
                      level_ai_path: Path, 
                      level_spawn_path: Path,
                      cross_table_path: Path,
                      original_spawn: Optional[Path] = None) -> tuple[bool, str]:
        """
        Check if cross table needs to be rebuilt
        
        Args:
            level_name: Internal level name
            level_ai_path: Path to level.ai
            level_spawn_path: Path to level.spawn
            cross_table_path: Path to cross table output
            original_spawn: Optional path to original spawn file
        
        Returns:
            (needs_rebuild, reason)
        """
        # Check if cross table exists
        if not cross_table_path.exists():
            return True, "cross table doesn't exist"
        
        # Check if we have dependency info
        if level_name not in self.dependencies:
            return True, "no dependency info (first build)"
        
        # Get stored dependencies
        deps = self.dependencies[level_name]
        
        # Calculate current hashes
        current_ai_hash = self._hash_file(level_ai_path)
        current_spawn_hash = self._hash_file(level_spawn_path)
        current_original_hash = self._hash_file(original_spawn) if original_spawn else ""
        
        # Check if files have changed
        if current_ai_hash != deps.level_ai_hash:
            return True, "level.ai changed"
        
        if current_spawn_hash != deps.level_spawn_hash:
            return True, "level.spawn changed"
        
        if current_original_hash != deps.original_spawn_hash:
            if current_original_hash and not deps.original_spawn_hash:
                return True, "original_spawn added"
            elif not current_original_hash and deps.original_spawn_hash:
                return True, "original_spawn removed"
            else:
                return True, "original_spawn changed"
        
        # Check if cross table path changed (compare resolved paths to handle relative path variations)
        if cross_table_path.resolve() != Path(deps.cross_table_path).resolve():
            return True, "output path changed"
        
        # No rebuild needed
        return False, "up to date"
    
    def update(self, level_name: str,
               level_ai_path: Path,
               level_spawn_path: Path,
               cross_table_path: Path,
               original_spawn: Optional[Path] = None):
        """
        Update dependency info after successful build
        
        Args:
            level_name: Internal level name
            level_ai_path: Path to level.ai
            level_spawn_path: Path to level.spawn
            cross_table_path: Path to cross table output
            original_spawn: Optional path to original spawn file
        """
        import time
        
        deps = CrossTableDependencies(
            level_name=level_name,
            level_ai_hash=self._hash_file(level_ai_path),
            level_spawn_hash=self._hash_file(level_spawn_path),
            cross_table_path=str(cross_table_path),
            timestamp=time.time(),
            original_spawn_hash=self._hash_file(original_spawn) if original_spawn else ""
        )
        
        self.dependencies[level_name] = deps
        self._save()
    
    def get_status_report(self) -> str:
        """Get a human-readable status report"""
        if not self.dependencies:
            return "No cross tables built yet"
        
        lines = [f"Tracked cross tables: {len(self.dependencies)}"]
        
        for name, deps in sorted(self.dependencies.items()):
            import time
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S', 
                                     time.localtime(deps.timestamp))
            has_original = "+" if deps.original_spawn_hash else "-"
            lines.append(f"  {name}: built {timestamp} [original:{has_original}]")
        
        return "\n".join(lines)


def test_tracker():
    """Test the dependency tracker"""
    from pathlib import Path
    
    # Create test tracker
    tracker = DependencyTracker(Path("build"))
    
    # Test files
    level_ai = Path("levels/test/level.ai")
    level_spawn = Path("levels/test/level.spawn")
    cross_table = Path("build/test.gct")
    original_spawn = Path("tmp/oldspawnfiles/test.spawn")
    
    # Check if rebuild needed
    needs_rebuild, reason = tracker.needs_rebuild(
        "test_level",
        level_ai,
        level_spawn,
        cross_table,
        original_spawn=original_spawn
    )
    
    log(f"Needs rebuild: {needs_rebuild} - {reason}")

    # Simulate successful build
    if needs_rebuild:
        log("Building cross table...")
        # ... build happens here ...

        # Update dependencies
        tracker.update("test_level", level_ai, level_spawn, cross_table,
                      original_spawn=original_spawn)
        log("Dependencies updated")

    # Check again
    needs_rebuild, reason = tracker.needs_rebuild(
        "test_level",
        level_ai,
        level_spawn,
        cross_table,
        original_spawn=original_spawn
    )

    log(f"Needs rebuild: {needs_rebuild} - {reason}")

    # Show status
    log("\n" + tracker.get_status_report())


if __name__ == '__main__':
    test_tracker()
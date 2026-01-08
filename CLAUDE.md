# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

S.T.A.L.K.E.R. Game Graph Compilation Tool - builds and merges game graph data from the X-Ray Engine.
Processes multiple game levels and creates a unified `all.spawn` file containing spawn points, patrol paths, game graph topology, and cross-table references.
Includes a visualiser tool for displaying all.spawn information in a 3D visualiser.

## Running the Build

```bash
# Main build - runs the complete pipeline
./build_anomaly.sh

# With options
./build_anomaly.sh --force      # Force rebuild all cross tables
./build_anomaly.sh --dry-run    # Show what would be built

# Launch visualiser
./visualise.sh
```

No formal test suite exists. Debug/validation scripts are in `debug_scripts/`.

## Architecture

The build follows a 5-phase pipeline (scripts in `compiler/`):

```
┌─────────────────────────────────────────────────────────────┐
│                    PHASE 1: EXTRACTION                       │
│  extraction/spawn_entity_extractor.py                        │
│  - Extract spawn entities from level.spawn files             │
│  - Merge with original spawn data                            │
│  - Apply blacklist filtering                                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  PHASE 2: GAME GRAPH BUILD                   │
│  crosstables/build_cross_table.py → .gct files               │
│  game_graph_merger.py → unified vertices, edges, death points│
│  graph/game_graph.py → GameGraph object with all mappings    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   PHASE 3: GVID REMAPPING                    │
│  remapping/spawn_remapper.py → update entity GVIDs           │
│  remapping/patrol_remapper.py → update patrol GVIDs          │
│  (All remapping uses GameGraph for position lookups)         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   PHASE 4: SERIALIZATION                     │
│  serialization/game_graph_serializer.py → chunk 4            │
│  serialization/header_serializer.py → chunk 0                │
│  serialization/all_spawn_writer.py → final all.spawn         │
└─────────────────────────────────────────────────────────────┘
```

Files in `compiler/extractedanomalyspawns/` are generated separately using scripts in `recovery_scripts/`.

### Key Packages (in `compiler/`)

| Package | Purpose |
|---------|---------|
| `extraction/` | Phase 1: Read-only extraction of spawn entities |
| `graph/` | GameGraph dataclass - central object for all GVID resolution |
| `crosstables/` | Cross table building and remapping |
| `remapping/` | GVID remapping for spawns and patrols |
| `serialization/` | Final binary output generation |
| `patrols/` | Patrol path extraction and merging |
| `parsers/` | Binary format parsers for X-Ray engine files |
| `levels/` | Level configuration and level.game parsing |
| `utils/` | Shared utilities (logging, GUID generation, etc.) |

### Key Modules

| Module | Purpose |
|--------|---------|
| `build_all_spawn.py` | Master orchestrator (GameGraphBuilder class) |
| `game_graph_merger.py` | Merges per-level graphs; GameVertex/GameEdge/DeathPoint |
| `spawn_graph_builder.py` | Builds spawn chunk; M_SPAWN/M_UPDATE packets |
| `graph/game_graph.py` | GameGraph - caches level.ai, cross tables, provides GVID lookups |
| `remapping/spawn_remapper.py` | Updates entity GVIDs based on position |
| `remapping/patrol_remapper.py` | Updates patrol point GVIDs |
| `extraction/spawn_entity_extractor.py` | Extracts/merges spawn entities |
| `crosstables/build_cross_table.py` | Builds .gct files; LevelGraphNavigator pathfinding |

### Configuration Files (at project root)

- **levels.ini**: Defines all levels with IDs (0-33), world offsets, paths to level data
- **spawn_blacklist.ini**: Entity exclusion list (exact matches and wildcards)

### Binary File Formats (X-Ray Engine)

- `.spawn`: Chunked binary (header, spawn graph, artifacts, patrols, game graph)
- `.ai`: Level AI graph (vertices, edges, spatial indexing)
- `.gct`: Cross table (level vertex → game vertex mapping)
- `.game`: Compiled level data (waypoints, patrols)

## Directory Structure

- `levels.ini`: Level configuration (at project root)
- `spawn_blacklist.ini`: Entity exclusion patterns (at project root)
- `compiler/`: Build scripts and modules
  - `extraction/`: Spawn entity extraction (Phase 1)
  - `graph/`: GameGraph data structure
  - `crosstables/`: Cross table building and remapping
  - `remapping/`: GVID remapping for spawns and patrols
  - `serialization/`: Binary output serializers
  - `patrols/`: Patrol path handling
  - `parsers/`: Binary format parsers
  - `levels/`: Level configuration parsing
  - `utils/`: Shared utilities (logging, GUID, binary helpers)
  - `extractedanomalyspawns/`: Original extracted spawn files from game
  - `recovery_scripts/`: Tools to extract spawn/patrols/edges from original files
- `levels/`: Level directories with level-specific data files
- `gamedata/spawns/`: Build output (all.spawn)
- `.tmp/`: Build cache (.gct files)
- `build.log`: Build output log (generated at project root)
- `debug_scripts/`: Analysis and validation utilities
- `visualiser/`: 3D node graph inspector

## Key Concepts

- **GameGraph**: Central data structure holding all game graph info, caches level.ai and cross table data, provides GVID resolution methods
- **GVID (Game Vertex ID)**: Global identifier for vertices across all levels (0 to N-1)
- **Cross Tables**: Map level-local vertex IDs to global game graph vertex IDs
- **Death Points**: Spawn/respawn locations sampled at 10% from game vertices
- **Entity Blacklist**: Filters unwanted entities using patterns in spawn_blacklist.ini
- **Level Offsets**: Cumulative vertex counts for local→global GVID conversion
- **M_UPDATE Packets**: Category-based packet sizes for weapons, armor, etc.

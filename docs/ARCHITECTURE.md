# Architecture

Technical documentation for the L.A.M.B.D.A build pipeline.

## Codebase Structure

```
compiler/
├── build_all_spawn.py          # Master orchestrator (GameGraphBuilder)
├── project_paths.py            # Path resolution (ProjectPaths dataclass)
├── constants.py                # Shared constants
├── game_graph_merger.py        # Merges per-level graphs
├── config/                     # Mod configuration and file processing
│   ├── mod_config.py           #   Mod INI parser
│   ├── mod_copier.py           #   File copier with exclusion logic
│   └── tag_rewriter.py         #   LVID/GVID tag processing
├── converters/                 # Spawn data converters/exporters
├── crosstables/                # Cross table building
├── extraction/                 # Spawn entity extraction and merging
├── generation/                 # Death point generation
├── graph/                      # GameGraph data structure
├── levels/                     # Level configuration parser
├── parsers/                    # Binary format parsers
├── patrols/                    # Patrol path handling
├── remapping/                  # GVID remapping
├── serialization/              # Binary output (all.spawn writer)
├── spawn_graph/                # Spawn graph building
│   └── builder.py              # Builds spawn chunk
└── utils/                      # Logging, shared utilities
```

## Build Pipeline

The build follows a 6-phase pipeline:

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
│                  PHASE 2: CROSS TABLES                       │
│  crosstables/builder.py → .gct files                        │
│  - Build LVID → GVID mapping for each level                  │
│  - Cache in .tmp/ directory                                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                PHASE 3: GAME GRAPH MERGE                     │
│  game_graph_merger.py → unified vertices, edges, death points│
│  graph/game_graph.py → GameGraph object with all mappings    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   PHASE 4: GVID REMAPPING                    │
│  remapping/spawn_remapper.py → update entity GVIDs           │
│  remapping/patrol_remapper.py → update patrol GVIDs          │
│  (All remapping uses GameGraph for position lookups)         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   PHASE 5: SERIALIZATION                     │
│  serialization/game_graph_serializer.py → chunk 4            │
│  serialization/header_serializer.py → chunk 0                │
│  serialization/all_spawn_writer.py → final all.spawn         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  PHASE 6: POST-PROCESSING                    │
│  - Remap level changers                                      │
│  - Copy mod-specific files                                   │
└─────────────────────────────────────────────────────────────┘
```

## Key Concepts

### LVID (Level Vertex ID)

Local vertex ID within a single level's AI navigation mesh (`level.ai`). Each level has its own set of LVIDs starting from 0. Used for NPC pathfinding within a level.

### GVID (Game Vertex ID)

Global game vertex ID ranging from 0 to N-1 across all levels. Used for cross-level navigation and spawn point assignment. The game graph contains sparse vertices created via graph points in the level editor.

### Cross Table

Maps LVID → GVID for a level. Enables translation between local (level) and global (game) coordinate systems. Built from `level.ai` data and graph point definitions.

### GameGraph

Central data structure holding all game graph vertices, edges, and mappings across all levels. Provides methods for GVID resolution based on 3D position.

### Death Points

Spawn/respawn locations for the player. Randomly selected from approximately 10% of game graph vertices.

### Level Files Relationship

| File | Contents | Vertex Type |
|------|----------|-------------|
| `level.ai` | AI navigation mesh (millions of vertices) | LVID |
| `level.spawn` | Spawn entities with positions | References LVID |
| `level.game` | Waypoints and patrol paths | References LVID |
| `.gct` | Cross table (built by pipeline) | LVID → GVID |

## Binary File Formats

The X-Ray Engine uses several binary formats. Reference source files from the engine.

> For comprehensive documentation of the all.spawn format including entity types, M_UPDATE packet sizes, GVID calculations, and X-Ray source references, see [all.spawn Format](ALL_SPAWN_FORMAT.md).

### Game Graph

Defines the high-level navigation graph for cross-level travel.

**Source References:**
- `xrGame/game_graph.h`
- `xrServerEntities/game_graph_space.h`

**Structure:**
- Header with version, vertex count, edge count, death point count
- Array of game vertices (position, LVID, level ID, vertex type, edges)
- Array of game edges (target vertex, distance)
- Array of death points (GVID, level ID)

### Level Graph (level.ai)

Contains the detailed AI navigation mesh for a single level.

**Source References:**
- `xrGame/level_graph.h`
- `xrGame/level_graph_space.h`

**Structure:**
- Header with version, vertex count, cell size
- Array of navigation vertices (position, neighbor links, cover data)

### Cross Table (.gct)

Maps level vertices to game graph vertices.

**Source References:**
- `xrGame/game_level_cross_table.h`

**Structure:**
- Header with version, node count, GVID range
- Array of cross table cells (GVID, distance to graph point)

### ALife Objects

Spawn entities for NPCs, items, and objects.

**Source References:**
- `xrServerEntities/xrServer_Objects_ALife.h`
- `xrServerEntities/xrServer_Objects_ALife_Items.h`

**Structure:**
- M_SPAWN packet (creation data: class, position, GVID, LVID, flags)
- M_UPDATE packet (state data: varies by entity type)

### Spawn Registry

Container format for all.spawn file.

**Source References:**
- `xrGame/alife_spawn_registry.h`
- `xrGame/alife_spawn_registry_header.h`

**Chunks:**
- Chunk 0: Header (version, GUID, level count, object count)
- Chunk 1: Spawn graph (entities)
- Chunk 2: Artifacts (obsolete)
- Chunk 3: Patrols
- Chunk 4: Game graph

## Configuration Files

See [README.md](../README.md#configuration) for configuration file documentation (`levels.ini`, `spawn_blacklist.ini`, `level_changers.ini`, mod configs).

## Changes from Vanilla

### Dynamic Item Spawning

Vanilla Anomaly used hardcoded locations with hardcoded LVID and GVID values for dynamic item spawns. This approach breaks when the game graph is rebuilt, as the vertex IDs change.

**Solution:** During the build process, a `space_restrictor` entity is inserted into every level. This restrictor has an attached script that spawns items using XYZ world coordinates from `dynamic_item_spawn_locations.ltx` instead of vertex IDs.

The game engine resolves the correct LVID/GVID at runtime from the coordinates.

### Dynamic Anomalies

Vanilla used hardcoded vertex IDs for dynamic anomaly placement. This has the same problem as dynamic items when the game graph changes.

The build now inserts space restrictors that reference `dynamic_anomaly_locations.ltx`, spawning anomalies by world coordinates rather than pre-baked vertex IDs.

This coordinate-based approach ensures dynamic spawns remain valid regardless of game graph changes.

## Key Modules

| Module | Purpose |
|--------|---------|
| `build_all_spawn.py` | Master orchestrator (GameGraphBuilder class) |
| `project_paths.py` | Path resolution (ProjectPaths frozen dataclass) |
| `game_graph_merger.py` | Merges per-level graphs; GameVertex/GameEdge/DeathPoint |
| `spawn_graph/builder.py` | Builds spawn chunk; M_SPAWN/M_UPDATE packets |
| `graph/game_graph.py` | GameGraph - caches level.ai, cross tables, provides GVID lookups |
| `remapping/spawn_remapper.py` | Updates entity GVIDs based on position |
| `remapping/patrol_remapper.py` | Updates patrol point GVIDs |
| `extraction/spawn_entity_extractor.py` | Extracts/merges spawn entities |
| `crosstables/builder.py` | Builds .gct files; LevelGraphNavigator pathfinding |
| `config/mod_copier.py` | Copies mod overlay files to gamedata |
| `config/tag_rewriter.py` | Processes LVID/GVID tags in mod files |

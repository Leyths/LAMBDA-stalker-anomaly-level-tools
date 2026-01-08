# Architecture

Technical documentation for the L.A.M.B.D.A build pipeline.

## Codebase Structure

```
compiler/
├── build_all_spawn.py          # Master orchestrator (GameGraphBuilder)
├── game_graph_merger.py        # Merges per-level graphs
├── spawn_graph_builder.py      # Builds spawn chunk
├── extraction/                 # Phase 1: Spawn entity extraction
├── graph/                      # GameGraph data structure
├── crosstables/                # Cross table building
├── remapping/                  # GVID remapping
├── serialization/              # Binary output
├── patrols/                    # Patrol path handling
├── parsers/                    # Binary format parsers
├── levels/                     # Level configuration
└── utils/                      # Shared utilities
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
│  crosstables/build_cross_table.py → .gct files               │
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

Spawn/respawn locations for the player. Sampled at approximately 10% from game graph vertices.

### Level Files Relationship

| File | Contents | Vertex Type |
|------|----------|-------------|
| `level.ai` | AI navigation mesh (millions of vertices) | LVID |
| `level.spawn` | Spawn entities with positions | References LVID |
| `level.game` | Waypoints and patrol paths | References LVID |
| `.gct` | Cross table (built by pipeline) | LVID → GVID |

## Binary File Formats

The X-Ray Engine uses several binary formats. Reference source files from the engine:

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

### levels.ini

Defines all levels with their properties:

```ini
[level01]
name = k00_marsh                                    # Internal level name
caption = "k00_marsh"                               # Display name (optional)
offset = 1050.0, 1000.0, 0.0                        # World space offset (x, y, z)
path = ../levels/k00_marsh                          # Path to level folder
id = 01                                             # Unique level ID (0-255)
original_spawn = extractedanomalyspawns/k00_marsh.spawn      # Original spawn data
original_patrols = extractedanomalyspawns/k00_marsh.patrols  # Original patrol data
original_edges = extractedanomalyspawns/k00_marsh.edges.json # Original graph edges
connect_orphans_automatically = true                # Auto-connect orphan graph nodes (optional)
```

### spawn_blacklist.ini

Entity names to exclude from the final all.spawn. Supports exact names and prefix wildcards:

```ini
# Exact match
zat_b39_anomaly_protect_helmet

# Wildcard - matches any entity starting with "debug_"
debug_*
```

### level_changers.ini

Configuration for cross-level teleporters. Each entry defines destination, arrival position, and camera orientation:

```ini
[level_name]
entity_name.dest = destination_level
entity_name.pos = x, y, z           # Local coordinates on destination level
entity_name.dir = pitch, yaw, roll  # Camera orientation in radians
```

Direction values are in radians (1.57 rad = 90°, 3.14 rad = 180°).

Level changers **not** listed in this file are removed from all.spawn.

## Changes from Vanilla

### Dynamic Item Spawning

Vanilla Anomaly used hardcoded locations with hardcoded LVID and GVID values for dynamic item spawns. This approach breaks when the game graph is rebuilt, as the vertex IDs change.

**Solution:** During the build process, a `space_restrictor` entity is inserted into every level. This restrictor has an attached script that spawns items using XYZ world coordinates from `dynamic_item_spawn_locations.ltx` instead of vertex IDs. The game engine resolves the correct LVID/GVID at runtime from the coordinates.

### Dynamic Anomalies

Similarly, vanilla used hardcoded vertex IDs for dynamic anomaly placement. The build now inserts space restrictors that reference `dynamic_anomaly_locations.ltx`, spawning anomalies by world coordinates rather than pre-baked vertex IDs.

This coordinate-based approach ensures dynamic spawns remain valid regardless of game graph changes.

## Key Modules

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

# all.spawn File Format

Technical documentation for the S.T.A.L.K.E.R. all.spawn binary format.

## Overview

### What is all.spawn?

The `all.spawn` file is the central spawn registry for the S.T.A.L.K.E.R. X-Ray engine's ALife (Artificial Life) system. It contains:

- **Spawn entities** - All NPCs, items, objects, zones, and level changers that exist in the game world
- **Patrol paths** - Waypoint networks for NPC movement patterns
- **Game graph** - High-level navigation topology connecting all levels
- **Cross tables** - Mappings between local and global vertex coordinate systems

When the game loads, it reads all.spawn to populate the world with entities at their designated positions. The ALife system then simulates these entities across all levels, even those the player hasn't visited.

### Why Rebuild all.spawn?

The all.spawn file must be rebuilt when:

- Adding new levels to the game
- Removing or modifying existing levels
- Changing the game graph topology (edges between levels)
- Modifying spawn entities or patrol paths

The file contains **global vertex IDs (GVIDs)** that reference positions in the merged game graph. When levels are added or removed, all GVIDs must be recalculated to maintain valid references.

### New Game vs Loading a Save

The engine uses all.spawn differently depending on whether the player starts a new game or loads an existing save.

**New Game:**
1. Engine loads all.spawn via [`CALifeSpawnRegistry::load`](https://bitbucket.org/anomalymod/xray-monolith/src/master/src/xrServerEntities/alife_spawn_registry.cpp)
2. The spawn registry reads all chunks: header, spawn graph, patrols, and game graph
3. Every entity in the spawn graph is instantiated with its M_SPAWN data (position, section, flags)
4. M_UPDATE packets provide initial state (ammo counts, condition, AI graph positions)
5. The ALife simulator begins managing all entities across all levels

**Loading a Save:**
1. Engine loads all.spawn to get the game graph structure (chunk 4) and patrol paths (chunk 3)
2. Entity **state** is loaded from the save file via [`CALifeObjectRegistry::load`](https://bitbucket.org/anomalymod/xray-monolith/src/master/src/xrServerEntities/alife_object_registry.cpp), not from all.spawn
3. The save contains current positions, inventories, health, and AI state for all entities
4. M_SPAWN/M_UPDATE packets in all.spawn are **not re-read** for existing entities - only the save data matters
5. The spawn registry is used as a template for any newly created entities (respawns, script spawns)

**What Gets Used From Where:**

| Data | New Game Source | Save Load Source |
|------|-----------------|------------------|
| Game graph topology | all.spawn chunk 4 | all.spawn chunk 4 |
| Patrol paths | all.spawn chunk 3 | all.spawn chunk 3 |
| Entity initial state | all.spawn M_SPAWN/M_UPDATE | Save file |
| Entity current state | N/A (no prior state) | Save file |
| Cross tables | all.spawn chunk 4 | all.spawn chunk 4 |

**Why This Matters for Modding:**

- **Save compatibility:** If you rebuild all.spawn with different GVIDs, existing saves may crash because the save references old GVID values for entity positions and AI pathfinding state
- **Entity IDs:** The `m_tSpawnID` in each entity must match its vertex ID in the spawn graph. Saves reference entities by spawn ID via [`CALifeSpawnRegistry::spawns()`](https://bitbucket.org/anomalymod/xray-monolith/src/master/src/xrServerEntities/alife_spawn_registry.h)
- **Game graph changes:** Adding/removing levels changes GVID offsets, breaking save references to entities that store graph positions

> **Important:** Saves are generally incompatible with rebuilt all.spawn files. Players should start a new game after installing mods that modify the game graph or spawn data.

**Relevant Source Files:**

| Component | Source File |
|-----------|-------------|
| Spawn registry loading | [alife_spawn_registry.cpp](https://bitbucket.org/anomalymod/xray-monolith/src/master/src/xrServerEntities/alife_spawn_registry.cpp) |
| Object registry (save/load) | [alife_object_registry.cpp](https://bitbucket.org/anomalymod/xray-monolith/src/master/src/xrServerEntities/alife_object_registry.cpp) |
| ALife simulator | [alife_simulator.cpp](https://bitbucket.org/anomalymod/xray-monolith/src/master/src/xrServerEntities/alife_simulator.cpp) |
| Game graph loading | [game_graph_inline.h](https://bitbucket.org/anomalymod/xray-monolith/src/master/src/xrServerEntities/game_graph_inline.h) |

## File Structure

The all.spawn file consists of 5 chunks in a standard X-Ray chunked format:

| Chunk | Name | Description |
|-------|------|-------------|
| 0 | Header | Version, GUIDs, entity/level counts |
| 1 | Spawn Graph | Entity spawn data (M_SPAWN + M_UPDATE packets) |
| 2 | Artefacts | Historical chunk - not used by Anomaly |
| 3 | Patrol Paths | Waypoint networks for NPC movement |
| 4 | Game Graph | Navigation vertices, edges, death points, cross tables |

### Chunk 0: Header

The header identifies the all.spawn version and provides counts for validation.

```
Structure (44 bytes):
  u32     version         // XRAI_CURRENT_VERSION (10)
  GUID    spawn_guid      // 16 bytes - unique identifier for this spawn
  GUID    graph_guid      // 16 bytes - must match game graph GUID
  u32     spawn_count     // Number of spawn entities
  u32     level_count     // Number of levels
```

The `graph_guid` must match the GUID in the game graph chunk. The engine uses this to verify that spawn data and game graph data are synchronized.

**Source:** `compiler/serialization/header_serializer.py`

### Chunk 1: Spawn Graph

The spawn graph contains all spawn entities organized as a vertex graph. Each entity is a vertex with optional edges to child entities.

```
Structure:
  Sub-chunk 0: vertex_count (u32)
  Sub-chunk 1: Vertices
    For each vertex:
      Sub-chunk 0: vertex_id (u16)
      Sub-chunk 1: CServerEntityWrapper
        Sub-chunk 0: M_SPAWN packet (u16 size + data)
        Sub-chunk 1: M_UPDATE packet (u16 size + data)
  Sub-chunk 2: Edges
    For each vertex with edges:
      vertex_id (u16)
      edge_count (u32)
      For each edge: target_vertex_id (u16), weight (f32)
```

The vertex_id must match the `m_tSpawnID` field inside the M_SPAWN packet. If these don't match, the engine crashes when trying to access spawn data.

**Source:** `compiler/spawn_graph/builder.py`

### Chunk 2: Artefacts

> **Note:** While vanilla all.spawn contains artefact spawn data in this chunk, the Anomaly X-Ray engine doesn't appear to use it. This is likely a historical artefact from the original Shadow of Chornobyl. In Anomaly, artefact spawning is handled via Lua scripting on `smart_zone` entities that designate anomaly fields. This builder leaves chunk 2 empty.

### Chunk 3: Patrol Paths

Contains waypoint networks (patrols) that define paths for NPC movement, guard posts, and scripted sequences.

Each patrol consists of:
- A name (used by scripts to reference the patrol)
- A list of waypoints with positions, flags, and connections
- Optional waypoint properties (look points, wait times, etc.)

**Source:** `compiler/patrols/patrol_loader.py`

### Chunk 4: Game Graph

The game graph defines high-level navigation topology for cross-level travel. It contains sparse vertices (typically placed at key locations via graph points in the level editor) connected by edges.

```
Structure:
  Header:
    u8      version         // Graph version
    u16     vertex_count    // Total vertices across all levels
    u32     edge_count      // Total edges
    u8      death_point_count // Respawn locations
    GUID    graph_guid      // Must match header graph_guid

  Level descriptors:
    For each level: name, offset, id, section, GUID

  Vertices:
    For each vertex:
      position (3 floats)
      level_vertex_id (u32)
      level_id (u8)
      vertex_type (4 bytes)
      edge_offset (u32)
      edge_count (u8)

  Edges:
    For each edge:
      target_vertex_id (u16)
      distance (f32)

  Death points:
    For each death point:
      game_vertex_id (u16)
      level_id (u8)

  Cross tables:
    For each level: embedded .gct data (LVID -> GVID mapping)
```

**Source:** `compiler/serialization/game_graph_serializer.py`

### Online vs Offline ALife Simulation

The game graph exists primarily to support the ALife system's simulation of entities across the entire game world. The engine divides entities into "online" and "offline" states based on their distance from the player.

**Online:**
- Entity is within the player's online radius (defined in `alife.ltx`, default 750m)
- Uses the detailed `level.ai` navigation mesh (millions of vertices)
- Full pathfinding with obstacle avoidance
- Complete AI behaviors, combat, animations, physics
- Entities have precise LVID positions
- Computationally expensive - limited to nearby entities

**Offline:**
- Entity is outside the online radius (can be same level or different level)
- Uses the sparse game graph (hundreds of vertices total per level)
- Simplified movement: entities traverse between game graph vertices
- No detailed pathfinding - just graph edge traversal using distances
- Entities tracked by GVID position only
- Time-scaled simulation - offline entities "catch up" when checked

**Switching Between States:**

Entities constantly transition between online and offline as the player moves:

1. **Going Offline:** When the player moves away, nearby NPCs switch offline. Their current position is mapped to the nearest GVID and they continue on the game graph.

2. **Coming Online:** When the player approaches an offline entity's GVID location, the entity switches online. It spawns at a valid LVID position near the graph vertex.

3. **Cross-Level Travel:** Offline NPCs can traverse game graph edges that cross level boundaries. When an NPC's current GVID is on the player's level and within range, it switches online there.

**How Offline Movement Works:**

When an offline NPC travels (e.g., a stalker walking from Cordon to Rostok):

1. ALife calculates a path through the game graph using edge connections
2. The NPC's position updates to successive GVIDs based on edge distances and elapsed time
3. When crossing level boundaries, the NPC's `level_id` updates to the new level

**Why This Matters:**

- **Graph connectivity is critical** - if areas aren't connected by graph edges, NPCs cannot pathfind between them (online or offline)
- **Missing vertices** cause NPCs to get stuck or take bizarre routes
- **Edge distances** affect offline travel time calculations
- **M_UPDATE packets** for monsters/stalkers contain `m_tNextGraphID` and `m_tPrevGraphID` - the GVIDs the entity is traveling between while offline
- **Sparse graphs** mean NPCs may appear to "teleport" short distances when switching online, as they snap from GVID to nearest valid LVID

> **Performance Warning:** A disconnected game graph causes severe performance problems. When the ALife system cannot find a path between two graph vertices, it repeatedly attempts pathfinding in a loop, consuming CPU cycles and causing stuttering or freezes. Always ensure your game graph is fully connected - use the visualiser to verify all levels have edges connecting them to adjacent levels.

**Relevant Source Files:**

| Component | Source File |
|-----------|-------------|
| Online/offline switching | [alife_switch_manager.cpp](https://bitbucket.org/anomalymod/xray-monolith/src/master/src/xrServerEntities/alife_switch_manager.cpp) |
| ALife movement manager | [alife_movement_manager.cpp](https://bitbucket.org/anomalymod/xray-monolith/src/master/src/xrServerEntities/alife_movement_manager.cpp) |
| Graph engine | [graph_engine.h](https://bitbucket.org/anomalymod/xray-monolith/src/master/src/xrServerEntities/graph_engine.h) |
| Monster abstract (GVID fields) | [xrServer_Objects_ALife_Monsters.cpp](https://bitbucket.org/anomalymod/xray-monolith/src/master/src/xrServerEntities/xrServer_Objects_ALife_Monsters.cpp) |

## Entity Types

Spawn entities are categorized by their server class and serialization requirements.

### Static Entities

Entities that don't move and have minimal M_UPDATE data (typically 0 bytes).

| Section | Description |
|---------|-------------|
| `level_changer` | Cross-level teleporter |
| `graph_point` | Game graph vertex marker (not in final spawn) |
| `smart_terrain` | ALife simulation zone |
| `space_restrictor` | Invisible boundary/trigger zone |
| `script_zone` | Script-triggered area |
| `camp_zone` | NPC camping area |
| `smart_cover` | AI cover position |
| `inventory_box` | Container/stash |
| `campfire` | Interactive campfire |
| `hanging_lamp` | Light source |

### Dynamic Entities (Stalkers/Monsters)

Living entities with AI behavior. These have GVIDs in their M_UPDATE packets for pathfinding state.

**Stalkers:**
- `stalker` - Human NPCs (CSE_ALifeHumanStalker)
- Prefixes: `sim_default_*`

**Monsters:**
- `bloodsucker`, `boar`, `chimera`, `controller`, `burer`
- `dog_red`, `dog_black`, `psy_dog`, `flesh`, `snork`
- `pseudo_gigant`, `poltergeist`, `tushkano`, `zombie`
- Prefixes: `m_*` (e.g., `m_bloodsucker_e`)

M_UPDATE packet for monsters contains `m_tNextGraphID` and `m_tPrevGraphID` at fixed offsets, which must be remapped when rebuilding the game graph.

**Source:** `compiler/spawn_graph/entity_categories.py`

### Items

Items have varying M_UPDATE sizes based on their type.

| Category | M_UPDATE Size | Examples |
|----------|---------------|----------|
| Simple weapons | Varies | `wpn_knife` |
| Weapons with GL | 15 bytes | `wpn_ak74`, `wpn_groza` |
| Shotguns | 13 bytes | `wpn_spas12`, `wpn_bm16` |
| Outfits | 5 bytes | `stalker_outfit`, `exo_outfit` |
| Helmets | 5 bytes | `helm_battle`, `helm_tactic` |
| Physics objects | 2 bytes | `physic_object`, items starting with `bochka_`, `box_wood_` |

**Source:** `compiler/spawn_graph/update_packets.py`

### Anomaly Zones

Zone entities require a special fix - the SDK exports them without the `last_spawn_time` field required by the engine. The builder appends a null byte (0x00) to fix this.

Prefixes: `zone_*`, `generator_*`, `fireball_*`, `torrid_*`, `anomal_*`

## GVIDs and Cross Tables

### What Are GVIDs?

**GVID (Game Vertex ID)** is a global identifier for a position in the game graph. GVIDs range from 0 to N-1 where N is the total number of game graph vertices across all levels.

**LVID (Level Vertex ID)** is a local identifier within a single level's AI navigation mesh (`level.ai`). Each level has its own LVIDs starting from 0.

The relationship:
```
GVID = local_game_vertex + level_offset
```

Where `level_offset` is the cumulative count of game vertices from all preceding levels.

### Cross Tables

Cross tables (`.gct` files) provide the mapping from LVID to GVID for each level. When an entity is placed at a world position:

1. Find the nearest LVID in level.ai (millions of navigation vertices)
2. Look up the corresponding GVID in the cross table
3. Add the level offset to get the final GVID

```
Structure (.gct):
  Header:
    u32   version
    u32   level_vertex_count   // Number of level.ai vertices
    u32   game_vertex_count    // Number of graph points on this level
    GUID  level_guid
    GUID  game_guid

  Cells (one per level vertex):
    u16   game_vertex_id       // Local GVID (before offset)
    f32   distance             // Distance to nearest graph point
```

**Source:** `compiler/crosstables/builder.py`

### What Happens When GVIDs Are Wrong

Incorrect GVIDs cause severe problems:

- **Crashes on save/load** - Engine tries to access non-existent vertices
- **Pathfinding failures** - NPCs cannot navigate to destinations
- **Teleport bugs** - Level changers send players to wrong locations
- **ALife simulation errors** - Entities spawn in wrong levels

### Why GVIDs Change

GVIDs must be recalculated whenever:

- Levels are added or removed (changes all offsets)
- Game graph vertices are added/removed on any level
- Cross tables are rebuilt (happens when level.ai changes)

This is why L.A.M.B.D.A must remap all entity GVIDs during the build process.

**Source:** `compiler/remapping/spawn_remapper.py`

## Historical Context: Why Merging Is Required

### Missing level.spawn Files

Two Anomaly levels lack source `level.spawn` files:

- **l01_escape** (Cordon)
- **l10_red_forest** (Red Forest)

These levels only exist as extracted data from the vanilla Anomaly all.spawn. The original spawn files were either never distributed or were lost.

### Preserving Entity State

When merging with original spawn data, the builder preserves:

- **custom_data** - Script-specific configuration in INI format
- **M_UPDATE packets** - Entity state data (ammo counts, health, etc.)
- **Entity positioning** - Original coordinates if SDK coordinates are less precise

The merge logic:
```python
if len(old_spawn) > len(spawn_packet):
    # Old packet has more data (custom_data) - use it
    final_spawn = old_spawn
    final_update = old_update  # Preserve state
else:
    # Use new spawn, but keep old update packet
    final_spawn = spawn_packet
    final_update = old_update
```

### Anomaly Mod Ecosystem Compatibility

Many Anomaly mods expect specific entities to exist at specific positions. The `original_spawn` configuration in `levels.ini` maintains compatibility:

```ini
[level01]
name = l01_escape
original_spawn = extractedanomalyspawns/l01_escape.spawn
original_patrols = extractedanomalyspawns/l01_escape.patrols
original_edges = extractedanomalyspawns/l01_escape.edges.json
```

This ensures entities that mods reference (by name or position) remain present in rebuilt all.spawn files.

**Source:** `compiler/extraction/spawn_entity_extractor.py`

## X-Ray Engine Source References

The all.spawn format is defined in the X-Ray Monolith source code:

| Entity Category | Server Class | Source File |
|-----------------|--------------|-------------|
| Base spawn entity | `CSE_ALifeObject` | [xrServer_Objects_ALife.h](https://bitbucket.org/anomalymod/xray-monolith/src/master/src/xrServerEntities/xrServer_Objects_ALife.h) |
| Level changer | `CSE_ALifeLevelChanger` | [xrServer_Objects_ALife.cpp](https://bitbucket.org/anomalymod/xray-monolith/src/master/src/xrServerEntities/xrServer_Objects_ALife.cpp) |
| Stalker | `CSE_ALifeHumanStalker` | [xrServer_Objects_ALife_Monsters.h](https://bitbucket.org/anomalymod/xray-monolith/src/master/src/xrServerEntities/xrServer_Objects_ALife_Monsters.h) |
| Monsters | `CSE_ALifeMonsterBase` | [xrServer_Objects_ALife_Monsters.h](https://bitbucket.org/anomalymod/xray-monolith/src/master/src/xrServerEntities/xrServer_Objects_ALife_Monsters.h) |
| Weapons | `CSE_ALifeItemWeapon*` | [xrServer_Objects_ALife_Items.h](https://bitbucket.org/anomalymod/xray-monolith/src/master/src/xrServerEntities/xrServer_Objects_ALife_Items.h) |
| Items | `CSE_ALifeItem` | [xrServer_Objects_ALife_Items.h](https://bitbucket.org/anomalymod/xray-monolith/src/master/src/xrServerEntities/xrServer_Objects_ALife_Items.h) |
| Zones | `CSE_ALifeAnomalousZone` | [xrServer_Objects_ALife.h](https://bitbucket.org/anomalymod/xray-monolith/src/master/src/xrServerEntities/xrServer_Objects_ALife.h) |
| Smart terrain | `CSE_ALifeSmartZone` | [xrServer_Objects_ALife.h](https://bitbucket.org/anomalymod/xray-monolith/src/master/src/xrServerEntities/xrServer_Objects_ALife.h) |
| Space restrictor | `CSE_ALifeSpaceRestrictor` | [xrServer_Objects_ALife.h](https://bitbucket.org/anomalymod/xray-monolith/src/master/src/xrServerEntities/xrServer_Objects_ALife.h) |
| Object factory | (registration) | [object_factory_register.cpp](https://bitbucket.org/anomalymod/xray-monolith/src/master/src/xrServerEntities/object_factory_register.cpp) |
| CLSID definitions | (class IDs) | [clsid_game.h](https://bitbucket.org/anomalymod/xray-monolith/src/master/src/xrCore/clsid_game.h) |
| Spawn registry | `CALifeSpawnRegistry` | [alife_spawn_registry.h](https://bitbucket.org/anomalymod/xray-monolith/src/master/src/xrServerEntities/alife_spawn_registry.h) |
| Graph manager | `CGameGraph` | [game_graph.h](https://bitbucket.org/anomalymod/xray-monolith/src/master/src/xrServerEntities/game_graph.h) |

## Quick Reference

### Chunk IDs

| ID | Name | Purpose |
|----|------|---------|
| 0 | Header | Version, GUIDs, counts |
| 1 | Spawn Graph | Entity M_SPAWN/M_UPDATE data |
| 2 | Artefacts | Unused in Anomaly |
| 3 | Patrols | Waypoint networks |
| 4 | Game Graph | Navigation topology + cross tables |

### Entity M_UPDATE Sizes

| Category | Size | Notes |
|----------|------|-------|
| Static (zones, restrictors) | 0 bytes | No state data |
| Physics objects | 2 bytes | `num_items = 0` |
| Outfits/Helmets | 5 bytes | Condition data |
| Shotguns | 13 bytes | Ammo state |
| Weapons with GL | 15 bytes | Ammo + grenade state |
| Monsters/Stalkers | 48+ bytes | AI state including GVIDs at offsets 44/46 |

### Key Source Files (This Project)

| Topic | Source File |
|-------|-------------|
| Header format | `compiler/serialization/header_serializer.py` |
| Spawn graph structure | `compiler/spawn_graph/builder.py` |
| Entity categories | `compiler/spawn_graph/entity_categories.py` |
| M_UPDATE packets | `compiler/spawn_graph/update_packets.py` |
| GVID remapping | `compiler/remapping/spawn_remapper.py` |
| Cross table builder | `compiler/crosstables/builder.py` |
| Game graph serializer | `compiler/serialization/game_graph_serializer.py` |
| Entity merging | `compiler/extraction/spawn_entity_extractor.py` |

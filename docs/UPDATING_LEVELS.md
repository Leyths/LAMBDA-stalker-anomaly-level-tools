# Updating an Existing Level

This guide covers how to integrate a modified or rebuilt level into your all.spawn.

## Prerequisites

- S.T.A.L.K.E.R. SDK (Level Editor)
- Exported level files from the SDK
- L.A.M.B.D.A build tools set up

## Step 1: Export Level from SDK

1. Build/compile your level in the SDK
2. Export the level
3. Locate the exported files:
   - `level.ai` - AI navigation mesh
   - `level.spawn` - Spawn entities
   - `level.game` - Waypoints and patrols

## Step 2: Configure levels.ini

1. Copy `level.ai`, `level.spawn`, `level.game` to a folder (e.g., `levels/my_modified_level/`)

2. Update `levels.ini` to point to your new files:

```ini
[level05]
name = k01_darkscape
caption = "k01_darkscape"
offset = 3050.0, 1000.0, 0.0
path = ../levels/my_modified_darkscape    # Point to your folder
id = 05
# See Step 3 for original_* options
```

## Step 3: Original Data (Optional)

Decide whether to keep or disable the original spawn/patrol/edge data:

```ini
# OPTION A: Keep originals (better mod compatibility)
original_spawn = extractedanomalyspawns/k01_darkscape.spawn
original_patrols = extractedanomalyspawns/k01_darkscape.patrols
original_edges = extractedanomalyspawns/k01_darkscape.edges.json

# OPTION B: Disable originals (for total overhauls)
# original_spawn = extractedanomalyspawns/k01_darkscape.spawn
# original_patrols = extractedanomalyspawns/k01_darkscape.patrols
# original_edges = extractedanomalyspawns/k01_darkscape.edges.json

# Optional: Auto-connect orphan graph nodes (see Step 6.5)
connect_orphans_automatically = true
```

**Keep originals** if you want to maintain compatibility with other mods or preserve the vanilla experience.

**Disable originals** for total level overhauls where vanilla spawns/patrols may conflict with your changes.

## Step 4: Configure Level Changers

Edit `level_changers.ini` to set up teleporter connections:

1. Update coordinates for **inbound** connections to your map (where players arrive)
2. Add entries for any **new** level changers you've placed on your map

```ini
[k01_darkscape]
ds_level_changer_to_darkvalley.dest = l04_darkvalley
ds_level_changer_to_darkvalley.pos = -44.82, 0.43, -542.58
ds_level_changer_to_darkvalley.dir = 0.00, -0.09, 0.00

ds_level_changer_to_escape.dest = l01_escape
ds_level_changer_to_escape.pos = 346.87, 15.01, -28.19
ds_level_changer_to_escape.dir = 0.00, 0.79, 0.00
```

## Step 5: Initial Build

Run the build:

```bash
./build_anomaly.sh
# or
./build_gamma.sh
```

## Step 6: Create Graph Edges (if originals disabled)

If you disabled `original_edges`, you need to create new graph connections manually.

### Find Connection Points

1. Launch the visualiser:

Mac
   ```bash
   cd LAMBDA-stalker-anomaly-level-tools
   ./visualise.sh
   ```

Windows
   ```bash
   cd LAMBDA-stalker-anomaly-level-tools
   ./visualise.bat
   ```

[See this](../README.md#installation) if you encounter an error about open3d not being installed.

2. Open your modified map and find **graph nodes** (blue orbs) near the edges that should connect to adjacent maps

3. Note down their X/Y/Z coordinates

4. Open adjacent maps and find corresponding edge nodes that will connect back to your map

### Create edges.json

Create a new `<level>.edges.json` file. Use existing files in `compiler/extractedanomalyspawns/` as reference.

Example for a modified Darkscape with two outbound connections:

```json
{
  "level_name": "k01_darkscape",
  "level_id": 5,
  "edge_count": 2,
  "intra_level_edges": 0,
  "inter_level_edges": 2,
  "edges": [
    {
      "source_x": -236.4866,
      "source_y": 46.5132,
      "source_z": -280.0966,
      "target_x": 391.7981,
      "target_y": 15.1667,
      "target_z": -70.5833,
      "distance": 253.8893,
      "target_level": "l01_escape"
    },
    {
      "source_x": 64.3509,
      "source_y": 50.9274,
      "source_z": 284.4554,
      "target_x": -47.3282,
      "target_y": 0.387,
      "target_z": -601.7765,
      "distance": 468.4778,
      "target_level": "l04_darkvalley"
    }
  ]
}
```

- `source_*`: Coordinates on YOUR map
- `target_*`: Coordinates on the DESTINATION map
- `target_level`: Internal name of destination level

### Update Adjacent Maps

For each adjacent map, update its edges.json to include the reverse connection back to your map.

Example addition to Cordon's (`l01_escape`) edges.json:

```json
{
  "source_x": 391.7981,
  "source_y": 15.1667,
  "source_z": -70.5833,
  "target_x": -236.4866,
  "target_y": 46.5132,
  "target_z": -280.0966,
  "distance": 253.8893,
  "target_level": "k01_darkscape"
}
```

Note how source/target coordinates are swapped compared to your map's file.

### Point levels.ini to New Edges

```ini
original_edges = path/to/your/k01_darkscape.edges.json
```

## Step 6.5: Auto-Connect Disconnected Nodes (Optional)

Modified levels sometimes end up with disconnected graph nodes - vertices that exist but aren't connected to the main navigation graph. This can happen when:

- New areas are added that weren't properly connected during level compilation
- The level editor didn't generate all necessary graph edges
- Graph nodes exist at positions where edges couldn't be automatically computed

Disconnected nodes cause NPC pathfinding failures - NPCs simply cannot navigate to or through these areas.

### Detecting Disconnected Nodes

You may have disconnected nodes if:
- NPCs refuse to walk to certain areas of your map
- The visualiser shows isolated blue orbs with no connecting lines
- Build logs mention disconnected vertices

### Automatic Fix

Add this flag to your level's entry in `levels.ini`:

```ini
connect_orphans_automatically = true
```

This enables automatic connection of disconnected graph nodes, which:

1. **Detects** all vertices belonging to your level
2. **Finds** vertices that are disconnected from the main graph
3. **Connects** them to the nearest reachable vertex using a greedy nearest-neighbor algorithm
4. **Creates** bidirectional edges to ensure full connectivity

The algorithm minimizes total edge distance added while ensuring all level vertices form a single connected component.

### When to Use

- **Enable** when your modified level has pathfinding issues or disconnected areas
- **Disable** (default) for unmodified levels or when you want manual control over graph edges

### Important: Manual Connections Recommended

For best results, manually connect important graph points in the SDK, especially around buildings and complex geometry. The automatic algorithm uses straight-line distance calculations which can result in poor AI pathfinding behaviour - NPCs may try to walk through walls or take nonsensical routes.

The recommended workflow is:

1. **Manually connect** key graph nodes within your level using the SDK (doorways, building entrances, important waypoints)
2. **Enable** `connect_orphans_automatically = true` to let the algorithm connect your manually-connected nodes to the wider world graph

This way you maintain control over critical pathfinding areas while avoiding the tedious work of connecting every single node on your level to adjacent maps.

### Build Output

When enabled, you'll see output like:

```
  Connecting orphan vertices...
    k01_darkscape: 3 vertices need connection
    Connected vertex 1234 to 5678 (45.2m)
```

## Step 7: Final Build and Verify

1. Rebuild:
   ```bash
   ./build_anomaly.sh
   ```

2. Open the visualiser and verify:
   - Graph nodes appear correctly on your map
   - Edges connect to adjacent maps (visible as lines between nodes)
   - Level changers are positioned correctly

## Step 8: Package Your Mod

**Important:** You must include the entire `gamedata/` folder generated by L.A.M.B.D.A as a base for your mod. This folder contains essential scripts for:
- Spawn location handling
- Dynamic item spawning
- Dynamic anomaly spawning

Without these scripts, the game will crash.

Then add your level-specific files:
- `gamedata/levels/<level_name>/` - Your level files (level.ai, level.spawn, level.game, geometry, textures, etc.)

## Troubleshooting

**NPCs can't pathfind to/from my level**
- Check that graph edges exist connecting your map to adjacent maps
- Verify edge coordinates match actual graph node positions

**Level changer doesn't work**
- Ensure the entity name in level_changers.ini matches exactly
- Verify destination coordinates are valid (not inside geometry)

**Spawns from vanilla conflict with my changes**
- Disable original_spawn in levels.ini
- Add conflicting entities to spawn_blacklist.ini

**Spawning at fake_start (a room with four pillars) instead of on a level**
- A mod is trying to spawn the player at a GVID that no longer exists due to your level changes
- Find the mod responsible and update its GVID/LVID values to match your new game graph
- See [Tag-Based File Rewriting](mods-system.md#tag-based-file-rewriting) for how to add tags that automatically recompute these values during build

**Game crashes when saving or transitioning between levels**
- A mod is spawning an item or NPC at a GVID/LVID that no longer exists in your modified game graph
- Check your mod load order for mods that dynamically spawn entities (items, NPCs, anomalies)
- Find the offending mod and update its hardcoded GVID/LVID values
- See [Tag-Based File Rewriting](mods-system.md#tag-based-file-rewriting) for how to convert hardcoded values to automatically-computed tags

# Visualiser

A 3D inspection tool for viewing level vertices, spawns, and game graph data.

## Overview

The visualiser provides a read-only view of game data for analysis and debugging. It renders level AI navigation meshes, spawn objects, and game graph vertices in an interactive 3D environment.

**Launch command:**
```bash
./visualise.sh
```

This is a read-only tool - it does not modify any game files.

## Search Functionality

The visualiser includes search capabilities for locating specific data points.

### Search Types

Click the search dropdown to select a search type:

![Search options dropdown](images/show-search.png)

Three search types are available:

| Search Type | Search By |
|-------------|-----------|
| Find AI Node | LVID or coordinates |
| Find Spawn | Name or coordinates |
| Find Game Graph Vertex | GVID or coordinates |

### Searching

Enter your search term in the text input:

![Search input box](images/search-box.png)

**Partial matching:** Spawn searches support partial name matching. For example, searching for `level_changer` will find all level changer entities in the current level.

### Search Results

Results are displayed in a list that can be clicked to navigate to the item:

![Search results list](images/search-result.png)

## What Can Be Visualized

### Level Vertices (AI Nodes)

Navigation mesh points from `level.ai` files. These are the vertices NPCs use for pathfinding within a level. Vertices are colored by their cover score value.

### Spawn Objects

Spawn entities from `level.spawn` files. Over 20 entity types are rendered with custom 3D representations:

- Level changers
- Items (weapons, ammo, artifacts)
- Anomaly zones
- Restrictor zones
- NPCs and monsters
- Smart terrain markers
- And more

### Game Graph Vertices

Global navigation vertices from the game graph:

- **Blue spheres**: Local graph vertices (navigation within the level)
- **Purple spheres**: Inter-level connection vertices (cross-level travel points)

## Navigation Controls

| Control | Action |
|---------|--------|
| Left-drag | Rotate camera |
| Right-drag | Pan camera |
| Ctrl+Click | Select item |
| Space | Focus camera on selection |
| Arrow Keys | Navigate between linked vertices |

## Information Panel

Selecting an item displays detailed information in the side panel.

### Level Vertex Details

- Position (x, y, z)
- Cover score
- Associated GVID mapping
- Neighbor links

### Spawn Object Details

- Entity name
- Section type
- Position (x, y, z)
- Rotation
- Associated LVID and GVID
- Level changer destinations (if applicable)

### Game Graph Vertex Details

- GVID
- Position (x, y, z)
- Level ID and level mappings
- Connected edges

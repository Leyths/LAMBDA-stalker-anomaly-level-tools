# Mods System Documentation

This document describes the mod configuration system used by the S.T.A.L.K.E.R. spawn builder.

## Overview

The mods system allows you to create variant files for different base mods (e.g., Anomaly, GAMMA) and have them automatically processed during the build. Key features include:

- Automatic file copying from mod folders to gamedata
- Tag-based file rewriting for dynamic LVID/GVID computation
- Support for multiple base mod configurations

## Included Mod Files

This project ships with a **minimum set of mod files** required to allow most all.spawn rebuilds to work with:

- **GAMMA 0.94**
- **Vanilla Anomaly**

The included mods contain only the files that require tag-based rewriting (LVID/GVID computation). Other mod files (scripts, configs, textures, etc.) should already exist in your game installation.

> **Warning:** If levels are radically changed or removed from your game installation, mods that reference those levels may break regardless of tag rewriting. The tag system computes vertex IDs based on the game graph, so missing or restructured levels will cause lookup failures and the default values from the tags will be used instead.

## Folder Structure

```
stalkertool/
├── mods/
│   ├── Mod Name 1/
│   │   └── configs/
│   │       └── plugins/
│   │           └── some_file.ltx
│   └── Mod Name 2/
│       └── ...
├── anomaly.ini          # Anomaly mod configuration
├── gamma.ini            # GAMMA mod configuration
└── compiler/
    └── config/
        ├── mod_config.py    # Configuration parser
        ├── mod_copier.py    # File copier with tag rewriting
        └── tag_rewriter.py  # Tag processing logic
```

## INI Configuration Format

Each base mod has a configuration file (e.g., `anomaly.ini`, `gamma.ini`) that defines which mods are enabled and which files need tag processing.

### Section Format

Each mod is defined as a section where the section name is the mod folder name:

```ini
[Mod Name]
include = true
rewrite_files = configs/plugins/file1.ltx
                configs/plugins/file2.ltx
```

### Configuration Options

| Option | Description |
|--------|-------------|
| `include` | `true` or `false` - Whether the mod is enabled |
| `rewrite_files` | List of files (relative to mod folder) that need tag processing |

### Example Configuration

```ini
; anomaly.ini
[config]

[Dynamic Items And Anomalies Anomaly]
include = true

[_New Game Start Locations Anomaly]
include = true
rewrite_files = configs/plugins/new_game_start_locations.ltx
```

## Tag-Based File Rewriting

The tag system allows you to embed metadata in files that is processed during build. This is useful for computing LVID (Level Vertex ID) and GVID (Global Vertex ID) values based on coordinates.

### Tag Types

| Tag Format | Purpose | Output Behavior |
|------------|---------|-----------------|
| `{r%key:value}` | Read context tag | Stripped entirely |
| `{w%key:value}` | Write/rewrite tag | Value computed or preserved |

### Context Tags (`{r%...}`)

- `{r%level:level_name}` - Sets the current level context for subsequent LVID/GVID lookups
- Context persists until a new `{r%level:...}` tag is encountered
- Can appear on its own line or inline with section headers

### Rewrite Tags (`{w%...}`)

| Tag | Description | Output |
|-----|-------------|--------|
| `{w%lvid:default}` | Level Vertex ID | Computed from position |
| `{w%gvid:default}` | Global Vertex ID | Computed from position |
| `{w%x:value}` | X coordinate | Value preserved |
| `{w%y:value}` | Y coordinate | Value preserved |
| `{w%z:value}` | Z coordinate | Value preserved |

### Processing Logic

The tag rewriter uses a two-pass approach:

**Pass 1 - Parse & Collect:**
- Scans file sequentially, tracking:
  - Current level context (from most recent `{r%level:...}`)
  - Current position (from most recent `{w%x:...}`, `{w%y:...}`, `{w%z:...}`)
  - All tag locations and their associated context

**Pass 2 - Compute & Replace:**
- For each lvid/gvid tag, computes new value using recorded level + position
- Generates output with tags stripped and values replaced

### Example: INI File

**Source (with tags):**
```ini
[tc_central_tower]{r%level:k02_trucks_cemetery}
lvid    =    {w%lvid:598272}
gvid    =    {w%gvid:5200}
x       =    {w%x:40.89}
y       =    {w%y:21.18}
z       =    {w%z:-30.61}
```

**Output (after processing):**
```ini
[tc_central_tower]
lvid    =    598272
gvid    =    5200
x       =    40.89
y       =    21.18
z       =    -30.61
```

Note: The actual LVID and GVID values will be computed based on the position and may differ from the defaults.

### Multiple Coordinate Sets

When a file has multiple positions within the same level context, each lvid/gvid uses the most recently encountered x/y/z values:

```ini
{r%level:jupiter}
[spawn1]
x = {w%x:100}
y = {w%y:5}
z = {w%z:200}
lvid = {w%lvid:0}  ; uses (100, 5, 200)
gvid = {w%gvid:0}

[spawn2]
x = {w%x:300}
y = {w%y:2}
z = {w%z:400}
lvid = {w%lvid:0}  ; uses (300, 2, 400)
gvid = {w%gvid:0}
```

### Error Handling

| Situation | Behavior |
|-----------|----------|
| Missing level context | Logs error, uses default value from tag |
| Missing position | Logs error, uses default value from tag |
| GameGraph lookup failure | Logs warning, uses default value from tag |

## Creating a New Mod

1. **Create the mod folder:**
   ```
   mods/My New Mod/
   └── configs/
       └── plugins/
           └── my_file.ltx
   ```

2. **Add to configuration:**
   ```ini
   [My New Mod]
   include = true
   rewrite_files = configs/plugins/my_file.ltx
   ```

3. **Add tags to files (if needed):**
   - Add `{r%level:...}` tags to set level context
   - Add `{w%...}` tags around values that need processing

## Build Process Flow

1. **Load Configuration:** Parse the base mod's INI file (e.g., `anomaly.ini`)
2. **Build Game Graph:** Process all levels to create the complete game graph
3. **Copy Mod Files:** For each enabled mod:
   - If file is in `rewrite_files`: Process with TagRewriter
   - Otherwise: Copy directly to gamedata
4. **Output:** Files are placed in `gamedata/` preserving directory structure

## Level Names Reference

Common level names used in S.T.A.L.K.E.R.:

| Level Name | Description |
|------------|-------------|
| `l01_escape` | Cordon |
| `l02_garbage` | Garbage |
| `l03_agroprom` | Agroprom |
| `l03u_agr_underground` | Agroprom Underground |
| `l04_darkvalley` | Dark Valley |
| `l05_bar` | Bar |
| `l06_rostok` | Rostok/Wild Territory |
| `l07_military` | Army Warehouses |
| `l08_yantar` | Yantar |
| `l09_deadcity` | Dead City |
| `l10_radar` | Brain Scorcher |
| `l10_red_forest` | Red Forest |
| `l11_pripyat` | Pripyat (SoC) |
| `l12_stancia` | CNPP |
| `zaton` | Zaton (CoP) |
| `jupiter` | Jupiter (CoP) |
| `pripyat` | Pripyat (CoP) |
| `k00_marsh` | Great Swamps (CS) |
| `k01_darkscape` | Darkscape (CS) |
| `k02_trucks_cemetery` | Truck Cemetery (CS) |

"""
Entity section categories for M_UPDATE packet creation.

Defines which sections belong to which entity categories for proper
M_UPDATE packet generation and handling.
"""

# [Weapons with Grenade Launcher]
W_GL_SECTIONS = {
    'wpn_ak74', 'wpn_ak74u', 'wpn_abakan', 'wpn_groza', 'wpn_sig550',
    'wpn_g36', 'wpn_fn2000', 'wpn_l85', 'wpn_lr300', 'wpn_pkm', 'wpn_vintorez',
    'wpn_abakan_m1', 'wpn_abakan_m2', 'wpn_ak74_m1', 'wpn_l85_m1', 'wpn_l85_m2',
    'wpn_lr300_m1', 'wpn_sig_m1', 'wpn_sig_m2', 'wpn_vintorez_up', 'wpn_groza_m1',
    'wpn_ak74_arena', 'wpn_ak74u_arena', 'wpn_g36_arena', 'wpn_fn2000_arena',
    'wpn_groza_arena', 'wpn_ak74_minigame', 'wpn_ak74_up', 'wpn_ak74_up2',
    'wpn_ak74u_minigame', 'wpn_bozar', 'wpn_g36_up2', 'wpn_lr300_minigame',
    'wpn_lr300_up2', 'wpn_sig_no_draw_sound', 'wpn_sig_with_scope',
    'wpn_sig550_minigame', 'wpn_sig550_up2', 'wpn_ak74u_snag', 'wpn_fn2000_nimble',
    'wpn_g36_nimble', 'wpn_groza_nimble', 'wpn_groza_specops', 'wpn_pkm_zulus',
    'wpn_sig550_luckygun', 'wpn_vintorez_nimble'
}

# [Shotguns]
SHOTGUN_SECTIONS = {
    'wpn_spas12', 'wpn_wincheaster1300', 'wpn_bm16', 'wpn_toz34', 'wpn_rg-6',
    'wpn_rg6', 'wpn_protecta', 'wpn_spas12_m1', 'wpn_winchester_m1',
    'wpn_rg6_m1', 'wpn_bm16_arena', 'wpn_spas12_arena', 'wpn_toz34_arena',
    'wpn_protecta_nimble', 'wpn_spas12_nimble', 'wpn_wincheaster1300_trapper'
}

# [Simple Weapons]
SIMPLE_WEAPON_SECTIONS = {'wpn_knife'}

# [Outfits]
OUTFIT_SECTIONS = {
    'stalker_outfit', 'specops_outfit', 'svoboda_light_outfit', 'svoboda_heavy_outfit',
    'cs_heavy_outfit', 'scientific_outfit', 'exo_outfit', 'novice_outfit',
    'dolg_heavy_outfit', 'military_outfit', 'dolg_outfit', 'cs_light_outfit',
    'stalker_outfit_barge', 'svoboda_exo_outfit', 'stalker_outfit_up_stalk',
    'dolg_scientific_outfit', 'ecolog_outfit', 'killer_outfit', 'monolit_outfit',
    'military_commander_outfit', 'military_stalker_outfit', 'protection_outfit',
    'soldier_outfit', 'bandit_outfit', 'outfit_bandit_m1', 'outfit_dolg_m1',
    'outfit_exo_m1', 'outfit_novice_m1', 'outfit_specnaz_m1', 'outfit_stalker_m1',
    'outfit_stalker_m2', 'outfit_svoboda_m1', 'gar_quest_novice_outfit',
    'mar_quest_novice_outfit_1'
}

# [Helmets]
HELMET_SECTIONS = {
    'helm_battle', 'helm_hardhat', 'helm_hardhat_snag', 'helm_protective',
    'helm_respirator', 'helm_respirator_joker', 'helm_tactic', 'helmet'
}

# [Stationary MGun]
ST_MGUN_SECTIONS = {'stationary_mgun'}

# [Physics Objects] (Write num_items=0 for M_UPDATE - required for ACDC compatibility)
# These are cse_alife_object_physic classes (O_PHYS_S and O_DSTR_S clsids)
PHYSICS_PREFIXES = (
    'physic_', 'bochka_', 'balon_', 'box_wood_', 'box_metall_',
)
PHYSICS_SECTIONS = {
    # O_PHYS_S - physic_object class
    'physic_object', 'agru_door', 'red_forest_bridge',
    # O_DSTR_S - physic_destroyable_object and derivatives
    'physic_destroyable_object', 'oby_physic_destroyable',
    # Common O_DSTR_S items from scan.pm
    'lojka', 'rupor', 'tarelka1', 'tarelka2', 'komp_klava', 'komp_monitor',
    'komp_block', 'krujka', 'child_bench', 'med_stolik_01', 'ognetushitel',
    'tiski', 'vedro_01', 'table_lamp_01', 'kanistra_01', 'kanistra_02',
    'debris_01', 'teapot_1', 'shooting_target_1', 'bottle_3l',
    'priemnik_gorizont', 'kastrula', 'kastrula_up', 'notebook', 'miska',
    'fire_vedro', 'freezer', 'transiver', 'tv_1', 'wheel_litter_01',
    'wheel_litter_01_braked', 'kolyaska_01', 'kolyaska_01_braked',
    'kolyaska_wheel_01_braked', 'bidon',
    # Furniture and misc physics
    'box_paper', 'stul_wood_01', 'stul_metal_01',
    # Tools
    'hammer', 'lopata', 'molot',
    # Skeleton object
    'ph_skeleton_object',
}

# [Static Objects] (Write 0 Bytes for M_UPDATE)
STATIC_PREFIXES = (
    'breakable_', 'climable_', 'lights_', 'mounted_',
    'zone_', 'fireball_', 'generator_', 'torrid_', 'smart_'
)
STATIC_SECTIONS = {
    'breakable_object', 'climable_object', 'mounted_weapon',
    'inventory_box', 'campfire', 'hanging_lamp', 'lights_hanging_lamp',
    'lights_signal_light', 'smart_terrain', 'space_restrictor', 'graph_point',
    'level_changer', 'camp_zone', 'smart_cover', 'anomal_zone', 'm_car',
    'script_zone', 'm_trader', 'm_lesnik', 'helicopter'
}

# [Monster/Stalker sections] - These have m_tNextGraphID/m_tPrevGraphID in M_UPDATE
# CSE_ALifeMonsterAbstract::UPDATE_Write writes these GVIDs at offsets 44/46
# Source: gamesources/engine/src/xrServerEntities/object_factory_register.cpp
#
# NOTE: 'actor' is NOT here - CSE_ALifeCreatureActor inherits CSE_ALifeCreatureAbstract,
#       NOT CSE_ALifeMonsterAbstract. Actor's M_UPDATE has mstate (u16) at offset 44.
# NOTE: 'phantom', 'crow', 'trader' are NOT here - they don't have M_UPDATE GVIDs
MONSTER_PREFIXES = ('m_', 'sim_default_')
MONSTER_SECTIONS = {
    # Human stalker (CSE_ALifeHumanStalker -> CSE_ALifeHumanAbstract -> CSE_ALifeMonsterAbstract)
    'stalker',

    # Base monster names (CSE_ALifeMonsterBase -> CSE_ALifeMonsterAbstract)
    'flesh', 'chimera', 'dog_red', 'bloodsucker', 'boar', 'dog_black',
    'psy_dog', 'psy_dog_phantom', 'burer', 'pseudo_gigant', 'controller',
    'poltergeist', 'zombie', 'fracture', 'snork', 'cat', 'tushkano',

    # Other monster types
    'rat',          # CSE_ALifeMonsterRat
    'flesh_group',  # CSE_ALifeGroupTemplate<CSE_ALifeMonsterBase>

    # Common monster class IDs (prefixed versions caught by MONSTER_PREFIXES too)
    'm_bloodsucker_e', 'm_boar_e', 'm_burer_e', 'm_cat_e', 'm_chimera_e',
    'm_controller_e', 'm_dog_e', 'm_flesh_e', 'm_fracture_e', 'm_gigant_e',
    'm_poltergeist_e', 'm_pseudodog_e', 'm_psy_dog_e', 'm_psy_dog_phantom_e',
    'm_rat_e', 'm_snork_e', 'm_tushkano_e', 'm_zombie_e',
}

# Anomaly zone prefixes for fixing spawn packets
ANOMALY_PREFIXES = (
    'zone_', 'generator_', 'fireball_', 'torrid_', 'anomal_'
)


def is_monster_section(section: str) -> bool:
    """Check if section is a monster/stalker type with GVIDs in M_UPDATE."""
    if section in MONSTER_SECTIONS:
        return True
    return any(section.startswith(p) for p in MONSTER_PREFIXES)


def is_physics_section(section: str) -> bool:
    """Check if section is a physics object."""
    if section in PHYSICS_SECTIONS:
        return True
    return any(section.startswith(p) for p in PHYSICS_PREFIXES)


def is_static_section(section: str) -> bool:
    """Check if section is a static object."""
    if section in STATIC_SECTIONS:
        return True
    return any(section.startswith(p) for p in STATIC_PREFIXES)


def is_anomaly_section(section: str) -> bool:
    """Check if section is an anomaly zone."""
    return any(section.startswith(p) for p in ANOMALY_PREFIXES)

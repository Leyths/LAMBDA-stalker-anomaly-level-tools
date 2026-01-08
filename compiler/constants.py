"""
Constants used across the compiler modules.

Consolidates magic numbers and shared values to improve code readability
and maintainability.
"""

# X-Ray Engine version for AI files
XRAI_CURRENT_VERSION = 10

# Tolerance for matching vertices by position (in meters)
# Used when matching edge source/target positions to game graph vertices
VERTEX_MATCH_TOLERANCE = 2.0

# Epsilon for floating point comparisons
# Used in level graph row length calculations
FLOAT_EPSILON = 0.00001

# M_SPAWN packet version flag
M_SPAWN_VERSION_FLAG = 0x0020

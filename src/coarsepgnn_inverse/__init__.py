"""可追溯的 PU 逆向设计：先审计和冻结评价，再开展训练。"""

TARGET_UNITS = {
    "tg": "degC",
    "tm": "degC",
    "tensile_strength_ultimate": "MPa",
    "elongation_at_break": "%",
}
ROLES = ("diisocyanate", "polyol", "hydroxyl_chain_extender")


TOXIC_TERMS = {
    "idiot",
    "stupid",
    "dumb",
    "ugly",
    "fat",
    "crazy",
    "hate",
    "kill",
    "violent",
    "nasty",
    "trash",
    "loser",
}

LOW_QUALITY_TERMS = {
    "thing",
    "stuff",
    "something",
    "object",
    "someone",
    "people",
}

DEMOGRAPHIC_GROUP_PATTERNS = {
    "women": {"woman", "women", "female", "lady", "ladies", "girl", "girls"},
    "men": {"man", "men", "male", "gentleman", "boy", "boys"},
    "children": {"child", "children", "kid", "kids", "toddler", "baby"},
    "elderly": {"elderly", "old", "senior"},
    "race_black": {"black", "african", "african-american"},
    "race_white": {"white", "caucasian"},
    "race_asian": {"asian", "east-asian", "south-asian"},
    "race_latino": {"latino", "latina", "hispanic"},
}

# Simple keyword-based stereotype anchors for audit.
STEREOTYPE_RULES = {
    "women_with_cooking": {
        "group": "women",
        "keywords": {"kitchen", "cooking", "cook", "cleaning", "housewife"},
    },
    "men_with_sports": {
        "group": "men",
        "keywords": {"sports", "football", "baseball", "soccer", "basketball"},
    },
    "women_with_shopping_fashion": {
        "group": "women",
        "keywords": {"shopping", "dress", "fashion", "makeup", "handbag"},
    },
    "men_with_construction_technical": {
        "group": "men",
        "keywords": {"construction", "engineer", "mechanic", "tools", "workshop"},
    },
    "race_black_with_athletics": {
        "group": "race_black",
        "keywords": {"basketball", "football", "athlete", "sports"},
    },
    "race_asian_with_technical": {
        "group": "race_asian",
        "keywords": {"computer", "engineer", "math", "technology"},
    },
}


"""Metaphor categories for MLM scoring.

Card et al. 2022 used six categories (animals, cargo, disease, flood/tide,
machines, vermin). We add invasion/military and threat as separate categories
(splitting what was a combined "invasion_threat" earlier) since they're
conceptually distinct: invasion is a military metaphor, threat is general
danger language.

Each category's words must be filtered against the BERT vocabulary at runtime
to keep only single-WordPiece tokens; multi-piece words don't have a clean
P(w | context) under MLM scoring.
"""


METAPHOR_CATEGORIES: dict[str, list[str]] = {
    "animal": [
        "animal", "animals", "beast", "beasts", "brute", "brutes",
        "creature", "creatures", "critter", "critters",
        "herd", "herds", "swarm", "swarms", "flock", "flocks",
        "predator", "predators", "prey",
    ],
    "cargo": [
        "cargo", "freight", "shipment", "shipments",
        "load", "loads", "lading", "payload", "payloads",
        "consignment", "consignments", "goods", "baggage",
    ],
    "disease": [
        "disease", "diseases", "infection", "infections",
        "virus", "viruses", "contagion", "plague", "plagues",
        "epidemic", "epidemics", "sickness",
    ],
    "flood_tide": [
        "flood", "floods", "wave", "waves", "tide", "tides",
        "flow", "flows", "stream", "streams", "deluge",
        "torrent", "torrents", "inundation", "overflow",
        "outpouring", "surge", "surges",
    ],
    "machine": [
        "machine", "machines", "mechanism", "mechanisms",
        "engine", "engines", "machinery", "motor", "motors",
        "automaton", "automatons", "robot", "robots",
        "cyborg", "cyborgs",
    ],
    "vermin": [
        "vermin", "rats", "rat", "parasite", "parasites",
        "pest", "pests", "varmint", "varmints",
        "roach", "roaches", "lice",
    ],
    "invasion": [
        "invasion", "invasions", "invader", "invaders",
        "army", "armies", "soldier", "soldiers",
        "attacker", "attackers", "enemy", "enemies",
        "force", "forces", "troop", "troops",
    ],
    "threat": [
        "threat", "threats", "danger", "dangers",
        "menace", "menaces", "hazard", "hazards",
        "risk", "risks", "burden", "burdens",
    ],
}


def filter_categories_to_single_wordpiece(
    categories: dict[str, list[str]],
    tokenizer,
) -> tuple[dict[str, list[str]], dict[str, list[int]]]:
    """Keep only words that tokenize to a single WordPiece in the given
    tokenizer. Returns (filtered_words_per_category, token_ids_per_category)."""
    filtered_words: dict[str, list[str]] = {}
    token_ids: dict[str, list[int]] = {}
    for cat, words in categories.items():
        kept_words: list[str] = []
        kept_ids: list[int] = []
        for w in words:
            ids = tokenizer.encode(w, add_special_tokens=False)
            if len(ids) == 1:
                kept_words.append(w)
                kept_ids.append(ids[0])
        filtered_words[cat] = kept_words
        token_ids[cat] = kept_ids
    return filtered_words, token_ids

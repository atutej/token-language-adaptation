from functools import partial


def convert_choice(choice):
    return choice[0].lower() + choice[1:]


def doc_to_text(doc, connector):
    # Drop the period
    conn = connector[doc["question"]]
    return doc["premise"].strip()[:-1] + f" {conn}"


def doc_to_choice(doc):
    return [convert_choice(doc["choice1"]), convert_choice(doc["choice2"])]


doc_to_text_hi = partial(
    doc_to_text,
    connector={
        "cause": "क्योंकि",
        "effect": "इसलिए",
    },
)

doc_to_text_ta = partial(
    doc_to_text,
    connector={
        "cause": "ஏனெனில்",
        "effect": "அதனால்",
    },
)

doc_to_text_te = partial(
    doc_to_text,
    connector={
        "cause": "ఎందుకంటే",
        "effect": "కాబట్టి",
    },
)

doc_to_text_ur = partial(
    doc_to_text,
    connector={
        "cause": "کیونکہ",
        "effect": "تو",
    },
)
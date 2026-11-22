"""Knowledge base and retrieval.

Paper: Section 3.4. Sources: design/nameplate data, normal operating ranges,
maintenance records and expert rules.

Owner: Member C
"""


def load_documents(folder):
    raise NotImplementedError


def build_index(chunks, model_name):
    raise NotImplementedError


def retrieve(rule_text, k):
    raise NotImplementedError

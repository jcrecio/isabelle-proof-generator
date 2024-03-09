import re

pattern = r'text\s*\\<open>(.*?)\\<close>'

def isabelle2mistral(isabelle_text: str):
    pattern = r'text\s*\\<open>(.*?)\\<close>'

    matches = re.findall(pattern, isabelle_text, re.DOTALL)

    parts = ["[INST]Proof the following Isabelle/HOL statements: ",]
    for match in matches:
        parts.append(match.strip())
    parts.append("[/INST]")

    index = isabelle_text.find("begin") + len("begin")
    formal_part = isabelle_text[index:]

    parts.append(formal_part.strip())

    return ''.join(parts)
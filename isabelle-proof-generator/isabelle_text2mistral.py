import re

pattern = r'text\s*\\<open>(.*?)\\<close>'

def isabelle2mistral(isabelle_text: str):
    pattern = r'text\s*\\<open>(.*?)\\<close>'

    matches = re.findall(pattern, isabelle_text, re.DOTALL)

    parts = []
    for match in matches:
        parts.append(match.strip())

    question = ''.join(parts)

    index = isabelle_text.find("begin") + len("begin")
    formal_part = isabelle_text[index:]

    answer = ''.join(formal_part.strip())

    return [question, answer]
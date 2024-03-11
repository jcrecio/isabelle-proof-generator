import re

pattern = r'text\s*\\<open>(.*?)\\<close>'

def isabelle2mistral(isabelle_text: str):
    pattern = r'text\s*\\<open>(.*?)\\<close>'
    matches = re.findall(pattern, isabelle_text, re.DOTALL)

    parts = []
    for match in matches:
        parts.append(match.strip())

    question = ''.join(parts)
    if len(parts) == 0:
        pattern = r'\(\*([\s\S]*?)\*\)'
        matches = re.findall(pattern, isabelle_text)
        for match in matches:
            parts.append(match.strip())

    if len(parts) == 0:
        pattern = r'\\begin{abstract}(.*?)\\end{abstract}'
        match = re.search(pattern, isabelle_text, re.DOTALL)
        for match in matches:
            parts.append(match.strip())

    if len(parts) == 0:
        pattern = r'subsection\\<open>(.*?)\\<close>'
        matches = re.findall(pattern, isabelle_text)
        for match in matches:
            parts.append(match.strip())

    index = isabelle_text.find("begin") + len("begin")
    formal_part = isabelle_text[index:]

    answer = ''.join(formal_part.strip())

    return [question, answer]
"""Rule-based coding of disclosure statements (tool, purpose, intensity), shared by E6 and E7."""
import re

TOOLS = [("ChatGPT", r"chat ?gpt|openai|gpt-?[345]"), ("Claude", r"\bclaude\b"), ("Gemini", r"gemini"),
         ("DeepSeek", r"deepseek"), ("Copilot", r"copilot"), ("Grammarly", r"grammarly"), ("DeepL", r"deepl"),
         ("PaperPal", r"paperpal"), ("QuillBot", r"quillbot"), ("Other named LLM", r"doubao|kimi|perplexity|wordtune|writefull")]
NON_LLM = {"Grammarly", "DeepL", "PaperPal", "QuillBot"}
PURPOSES = [("drafting/structuring", r"\bdrafting\b|structur|summari[sz]|paragraph|organi[sz]ation|phrasing|paraphras"),
            ("translation", r"translat"),
            ("polishing", r"readab|polish|clarity|fluen|language edit|editing|refine|improv|concise|wording|flow|style"),
            ("grammar", r"grammar|spelling|proofread|grammatical")]


def code(sentence):
    s = sentence.lower()
    tools = [n for n, rx in TOOLS if re.search(rx, s)] or ["Unspecified"]
    purposes = [n for n, rx in PURPOSES if re.search(rx, s)]
    intensity = 3 if "drafting/structuring" in purposes else 2 if {"polishing", "translation"} & set(purposes) \
        else 1 if purposes else 0
    return {"tools": tools, "purposes": purposes, "intensity": intensity,
            "non_llm_only": set(tools) <= NON_LLM, "chatgpt": "ChatGPT" in tools}

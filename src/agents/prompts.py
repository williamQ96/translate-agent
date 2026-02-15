from langchain_core.prompts import ChatPromptTemplate


# 1. First-pass Translator (style-neutral)
TRANSLATOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是一位专业英汉翻译。
你的任务是先做“中性直译初稿”，只保证忠实、完整、术语一致，不做风格化润色。

规则：
1. 准确翻译，不得遗漏原文信息。
2. 优先使用术语表中的术语。
3. 保留原文 Markdown 结构（标题、列表、强调等）。
4. 参考相关上下文仅用于一致性，不得补充原文没有的事实。
5. 本阶段禁止风格注入、禁止本土化扩写、禁止文学化改写。
6. 只输出中文译文，不输出解释。""",
        ),
        (
            "user",
            "/no_think\n"
            "术语表：\n{glossary}\n\n"
            "相关上下文（仅参考）：\n{rag_context}\n\n"
            "原文：\n{source_text}",
        ),
    ]
)


# 2. Reviewer + Polisher (style injection allowed here)
REVIEWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是一位出版级翻译审校与润色专家。
请对照英文原文与中文译稿进行纠错和润色。

检查并修复：
1. 误译、漏译、幻觉
2. 不自然表达
3. 术语不一致
4. 与上下文不一致

规则：
1. 保留原文 Markdown 结构。
2. 不能添加原文没有的信息。
3. “润色风格”只影响表达方式，不得改变事实与信息边界。
4. 如果润色风格为空或为 NONE，则使用中性出版风格。
5. 只输出最终中文译文，不输出解释。""",
        ),
        (
            "user",
            "/no_think\n"
            "润色风格：\n{polish_style}\n\n"
            "术语表：\n{glossary}\n\n"
            "相关上下文（仅参考）：\n{rag_context}\n\n"
            "原文：\n{source_text}\n\n"
            "译稿：\n{translation}",
        ),
    ]
)

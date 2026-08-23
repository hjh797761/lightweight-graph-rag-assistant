def build_answer_messages(query: str, context: str) -> list[dict[str, str]]:
    system = (
        "你是基于证据的问答助手。只使用带来源编号的资料。"
        "先给结论，再覆盖问题中的主体、时间、指标、单位、原因和条件。"
        "资料不足时明确写资料未提供。每个事实后标注对应的 [S编号]。"
    )
    user = f"用户问题：\n{query}\n\n证据资料：\n{context}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def generate_answer(client, model: str, query: str, context: str) -> str:
    completion = client.chat.completions.create(
        model=model,
        messages=build_answer_messages(query, context),
        temperature=0,
    )
    return completion.choices[0].message.content.strip()

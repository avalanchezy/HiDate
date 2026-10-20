"""Natural LawQA query identity and exact reader prompt budgeting."""


CHOICE_SYSTEM = ("法令に関する四択問題に答えてください。参考資料がある場合は資料を用いて判断し、"
                 "回答は a、b、c、d のいずれか一文字だけを出力してください。")


def external_queries(questions, corpus_dates):
    return [{"query_id": question["ファイル名"] + "|corpus_asof=" + date,
        "original_question_id": question["ファイル名"], "query_text": question["問題文"],
        "query_variant": "natural_question", "source_role": "external", "split": "external",
        "target_asof": date} for question in questions for date in corpus_dates]


def pack_choice_evidence(question, passages, tokenizer, prompt_budget=4096, max_passages=4):
    # The complete question and all released options are always included.
    def messages(selected):
        references = "\n\n".join(f"参考資料 {i + 1}\n{row['text']}" for i, row in enumerate(selected))
        content = ((references + "\n\n") if references else "") + "問題\n" + question["問題文"] + "\n\n選択肢\n" + question["選択肢"]
        return [{"role": "system", "content": CHOICE_SYSTEM}, {"role": "user", "content": content}]

    def tokens(selected):
        return len(tokenizer.apply_chat_template(messages(selected), tokenize=True, add_generation_prompt=True))

    if tokens([]) > prompt_budget:
        raise ValueError("The complete question and options exceed the fixed reader prompt budget")
    chosen = []
    for passage in passages[:max_passages]:
        text = passage["text"]
        item = {"chunk_id": passage["chunk_id"], "text": text, "truncated": False,
                "included_characters": len(text), "original_characters": len(text)}
        if tokens(chosen + [item]) <= prompt_budget:
            chosen.append(item)
            continue
        # Bound only the evidence prefix. Character-level search also handles
        # tokenization changes at the evidence/template boundary exactly.
        lower, upper = 0, len(text)
        while lower < upper:
            middle = (lower + upper + 1) // 2
            trial = {**item, "text": text[:middle]}
            if tokens(chosen + [trial]) <= prompt_budget:
                lower = middle
            else:
                upper = middle - 1
        if lower:
            chosen.append({**item, "text": text[:lower], "truncated": True,
                           "included_characters": lower})
        break
    return {"messages": messages(chosen), "prompt_tokens": tokens(chosen), "evidence": chosen,
            "prompt_budget": prompt_budget, "max_passages": max_passages}

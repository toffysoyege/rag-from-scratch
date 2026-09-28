from rag.pipeline import answer


def main():
    print("Ask about CellFix (type 'quit' to exit).\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"quit", "exit", "q"}:
            break
        if not question:
            continue
        reply, hits = answer(question)
        print(f"\nAssistant: {reply}")
        sources = sorted({h["source"] for h in hits})
        print(f"(retrieved from: {', '.join(sources)})\n")


if __name__ == "__main__":
    main()

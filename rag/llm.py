from rag.config import CHAT_MODEL, client

def chat(messages, temperature=0.0):
    """Send a list of {role, content} messages and return the reply text."""
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    reply = chat([
        {"role": "system", "content": "You are a concise, helpful assistant."},
        {"role": "user", "content": "How long is CellFix's repair warranty?"},
    ])
    print(reply)

from llm_client import make_client


def main():
    client = make_client(mock=False)
    messages = [
        {"role": "system", "content": "You are a brief assistant."},
        {
            "role": "user",
            "content": (
                "For this conversation, my made-up name is Velmora. "
                "Reply briefly to confirm."
            ),
        },
    ]
    response = client.chat("llama3.2:3b", messages, temperature=0)
    print("Model replied:", response.text)

    fresh_messages = [
        {"role": "system", "content": "You are a brief assistant."},
        {"role": "user", "content": "What is my made-up name?"},
    ]
    fresh_response = client.chat("llama3.2:3b", fresh_messages, temperature=0)
    print("Fresh call without history:", fresh_response.text)

    history_messages = messages + [
        {"role": "assistant", "content": response.text},
        {"role": "user", "content": "What is my made-up name?"},
    ]
    history_response = client.chat("llama3.2:3b", history_messages, temperature=0)
    print("Call with history:", history_response.text)


if __name__ == "__main__":
    main()
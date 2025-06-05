import logging
import time  # Added this import
from proxy_call import OpenaiCall

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')


def test_openai_call():
    client = OpenaiCall()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, can you respond with 'Test successful'?"}
    ]

    try:
        # Test chat
        response = client.chat(messages)
        print("Chat response:", response)

        # Test stream_chat
        print("Stream response:", end=" ")
        for chunk in client.stream_chat(messages):
            print(chunk, end="", flush=True)
        print()

        # Test embedding with retries
        for attempt in range(3):
            try:
                embedding = client.embedding("Test input")
                print("Embedding response:", embedding.data[0].embedding[:5], "... (truncated)")
                break
            except Exception as e:
                if attempt < 2:
                    logging.warning(f"Embedding attempt {attempt + 1} failed: {str(e)}. Retrying...")
                    time.sleep(attempt + 1)
                else:
                    logging.error(f"Embedding failed after 3 attempts: {str(e)}")

    except Exception as e:
        logging.error(f"Test failed: {str(e)}")


if __name__ == "__main__":
    test_openai_call()
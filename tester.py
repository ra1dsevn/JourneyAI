from model.utils.proxy_call import OpenaiCall

proxy = OpenaiCall()
messages = [{"role": "user", "content": "Hello, can you help me?"}]
try:
    response = proxy.chat(messages=messages, model="gpt-4o")
    print(response)
except Exception as e:
    print(f"Error: {e}")
from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:11434/v1/',
    api_key='ollama',
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            'role': 'user',
            'content': 'Say this is a test',
        },
        {
            'role': 'assistant',
            'content': 'Yes this is a test.'
        },
        {
            'role': 'user',
            'content': 'why do you think i am testing you?'
        }
    ],
    model='llama3.2',
)

print(chat_completion.choices[0].message.content)

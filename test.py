from openai import OpenAI

client = OpenAI(
    api_key="sk-BgGbzEZmbUE8nRce4f3794Bb211b4c52Aa9f7f4706A32eA8",
    base_url="https://vip.yi-zhan.top/v1",
)

response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[
        {"role": "system", "content": "你是个通用助手。"},
        {"role": "user", "content": "简单介绍一下自己"},
    ],
)

print(response.choices[0].message.model_dump().keys())
print(response.choices[0].message.content)

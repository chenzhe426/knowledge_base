from app.agent.agent import create_kb_agent

agent = create_kb_agent()

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "FastAPI 的核心特点是什么？",
            }
        ]
    }
)

print(result)
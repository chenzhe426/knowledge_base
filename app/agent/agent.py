from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

from app.agent.tools import TOOLS
from app.config import OLLAMA_BASE_URL, OLLAMA_MODEL


def create_kb_agent():
    model = init_chat_model(
        model=f"ollama:{OLLAMA_MODEL}",
        base_url=OLLAMA_BASE_URL,
        temperature=0,
    )

    agent = create_agent(
        model=model,
        tools=TOOLS,
        system_prompt=(
            "你是一个知识库问答助手。"
            "优先使用工具检索知识库后再回答。"
            "如果用户的问题涉及事实、概念解释、文档内容、总结、对比、出处定位，"
            "应先调用合适的工具。"
            "回答时尽量依据工具返回结果，不要编造。"
        ),
    )
    return agent
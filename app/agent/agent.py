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
            "你是一个知识库问答助手。\n"
            "优先使用工具检索知识库后再回答。\n"
            "如果用户的问题涉及事实、概念解释、文档内容、总结、对比、出处定位，应先调用合适的工具。\n"
            "系统已经提供了当前对话历史，你必须结合历史理解用户的追问。\n"
            "如果当前问题像是在追问上一轮内容，要结合已有对话历史理解代词、省略和上下文。\n"
            "回答时尽量依据工具返回结果，不要编造。"
        ),
    )
    return agent
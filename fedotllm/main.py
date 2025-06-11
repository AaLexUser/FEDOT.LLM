from pathlib import Path
from typing import Callable, List

import pandas as pd
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk # Import AIMessage
from langchain_core.runnables.schema import StreamEvent
from typing_extensions import Any, AsyncIterator

from fedotllm.agents.supervisor import SupervisorAgent
from fedotllm.agents.agent_wrapper.agent_wrapper import AgentWrapper
from fedotllm.agents.automl.automl_chat import AutoMLAgentChat
from fedotllm.agents.researcher.researcher import ResearcherAgent
from fedotllm.data import Dataset
from fedotllm.llm import AIInference, OpenaiEmbeddings
from fedotllm.agents.translator import TranslatorAgent

class FedotAI:
    def __init__(
        self,
        task_path: Path | str,
        inference: AIInference = AIInference(),
        embeddings: OpenaiEmbeddings = OpenaiEmbeddings(),
        handlers: List[Callable[[StreamEvent], None]] = [],
        workspace: Path | str | None = None,
    ):
        if isinstance(task_path, str):
            task_path = Path(task_path)
        self.task_path = task_path.resolve()
        assert task_path.is_dir(), (
            "Task path does not exist, please provide a valid directory."
        )

        self.inference = inference
        self.embeddings = embeddings
        self.handlers = handlers

        if isinstance(workspace, str):
            workspace = Path(workspace)
        self.workspace = workspace
        
    async def ainvoke(self,  message: str):
        if not self.workspace:
            self.workspace = Path(
                f"fedotllm-output-{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
            )

        dataset = Dataset.from_path(self.task_path)
        translator_agent = TranslatorAgent()
        translated_message = translator_agent.translate_input_to_english(message)

        automl_agent = AutoMLAgentChat(
            inference=self.inference, dataset=dataset, workspace=self.workspace
        ).create_graph()

        researcher_agent = AgentWrapper(
            ResearcherAgent(inference=self.inference, embeddings=self.embeddings)
        ).create_graph()

        entry_point = SupervisorAgent(
            inference=self.inference,
            automl_agent=automl_agent,
            researcher_agent=researcher_agent,
        ).create_graph()

        raw_response = await entry_point.ainvoke({"messages": [HumanMessage(content=translated_message)]})

        if raw_response and 'messages' in raw_response and isinstance(raw_response['messages'], list) and len(raw_response['messages']) > 0:
            last_message = raw_response['messages'][-1]
            # Ensure it's an AIMessage and has content to translate
            if isinstance(last_message, AIMessage) and hasattr(last_message, 'content'):
                ai_message_content = last_message.content
                translated_output = translator_agent.translate_output_to_source_language(ai_message_content)

                # Create a new AIMessage with the translated content, preserving other attributes
                raw_response['messages'][-1] = AIMessage(
                    content=translated_output,
                    id=last_message.id if hasattr(last_message, 'id') else None, # AIMessage has id
                    response_metadata=last_message.response_metadata if hasattr(last_message, 'response_metadata') else {},
                    tool_calls=last_message.tool_calls if hasattr(last_message, 'tool_calls') else [],
                    # tool_call_chunks might not be directly on AIMessage but on AIMessageChunk
                    # usage_metadata is also often on AIMessage
                    usage_metadata=last_message.usage_metadata if hasattr(last_message, 'usage_metadata') else None
                )

        return raw_response

    async def ask(self, message: str) -> AsyncIterator[Any]:
        if not self.workspace:
            self.workspace = Path(
                f"fedotllm-output-{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
            )

        dataset = Dataset.from_path(self.task_path)
        translator_agent = TranslatorAgent()
        translated_message = translator_agent.translate_input_to_english(message)

        automl_agent = AutoMLAgentChat(
            inference=self.inference, dataset=dataset, workspace=self.workspace
        ).create_graph()

        researcher_agent = AgentWrapper(
            ResearcherAgent(inference=self.inference, embeddings=self.embeddings)
        ).create_graph()

        entry_point = SupervisorAgent(
            inference=self.inference,
            automl_agent=automl_agent,
            researcher_agent=researcher_agent,
        ).create_graph()

        async for event in entry_point.astream_events(
            {"messages": [HumanMessage(content=translated_message)]},
            version="v2",
        ):
            if event["event"] == "on_chat_model_stream" and event.get("data", {}).get("chunk"):
                chunk = event["data"]["chunk"]
                # Ensure chunk is AIMessageChunk and has content
                if isinstance(chunk, AIMessageChunk) and hasattr(chunk, 'content'):
                    translated_chunk_content = translator_agent.translate_output_to_source_language(chunk.content)
                    # Create a new AIMessageChunk with translated content
                    event["data"]["chunk"] = AIMessageChunk(
                        content=translated_chunk_content,
                        id=chunk.id if hasattr(chunk, 'id') else None,
                        response_metadata=chunk.response_metadata if hasattr(chunk, 'response_metadata') else {},
                        tool_calls=chunk.tool_calls if hasattr(chunk, 'tool_calls') else [],
                        tool_call_chunks=chunk.tool_call_chunks if hasattr(chunk, 'tool_call_chunks') else [],
                        usage_metadata=chunk.usage_metadata if hasattr(chunk, 'usage_metadata') else None
                    )

            for handler in self.handlers:
                handler(event)
            yield event

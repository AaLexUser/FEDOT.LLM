from pathlib import Path
from typing import Callable, List, Optional # Added Optional

import pandas as pd
from langchain_core.messages import HumanMessage
from langchain_core.runnables.schema import StreamEvent
from typing_extensions import Any, AsyncIterator

from fedotllm.agents.supervisor import SupervisorAgent
from fedotllm.agents.agent_wrapper.agent_wrapper import AgentWrapper
from fedotllm.agents.automl.automl_chat import AutoMLAgentChat
from fedotllm.agents.researcher.researcher import ResearcherAgent
from fedotllm.data import Dataset
from fedotllm.llm import AIInference, OpenaiEmbeddings


class FedotAI:
    def __init__(
        self,
        task_path: Path | str,
        inference: Optional[AIInference] = None, # Default to None
        embeddings: Optional[OpenaiEmbeddings] = None, # Default to None
        handlers: Optional[List[Callable[[StreamEvent], None]]] = None, # Optional list of handlers
        workspace: Path | str | None = None,
    ):
        if isinstance(task_path, str):
            task_path = Path(task_path)
        self.task_path = task_path.resolve()
        assert self.task_path.is_dir(), ( # Used self.task_path here
            "Task path does not exist or is not a directory." # Simplified message slightly
        )

        self.inference = inference if inference is not None else AIInference()
        self.embeddings = embeddings if embeddings is not None else OpenaiEmbeddings()
        self.handlers = handlers if handlers is not None else [] # Ensure handlers is a list

        if isinstance(workspace, str):
            workspace = Path(workspace)
        self.workspace = workspace
        
    async def ainvoke(self,  message: str):
        if not self.workspace:
            self.workspace = Path(
                f"fedotllm-output-{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
            )

        dataset = Dataset.from_path(self.task_path)

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
        return await entry_point.ainvoke({"messages": [HumanMessage(content=message)]})

    async def ask(self, message: str) -> AsyncIterator[Any]:
        if not self.workspace:
            self.workspace = Path(
                f"fedotllm-output-{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
            )

        dataset = Dataset.from_path(self.task_path)

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
            {"messages": [HumanMessage(content=message)]},
            version="v2",
        ):
            for handler in self.handlers:
                handler(event)
            yield event

import pytest
from unittest.mock import MagicMock, patch, ANY, call, partial
from pathlib import Path

from fedotllm.agents.automl.automl import AutoMLAgent
from fedotllm.agents.automl.state import AutoMLAgentState
from fedotllm.llm import AIInference
from fedotllm.data import Dataset
from langgraph.types import Command
# from langgraph.graph import StateGraph, START, END # To be mocked

# Fixtures
@pytest.fixture
def mock_inference(mocker):
    return MagicMock(spec=AIInference)

@pytest.fixture
def mock_dataset(mocker):
    # Mocking specific attributes/methods of Dataset that might be used by nodes
    dataset = MagicMock(spec=Dataset)
    dataset.dataset_eda.return_value = "Mocked EDA"
    # Add other necessary mocked attributes or methods if nodes access them
    return dataset

@pytest.fixture
def mock_workspace(mocker):
    return MagicMock(spec=Path)


# Test Class for AutoMLAgent
@patch('fedotllm.agents.automl.automl.generate_report')
@patch('fedotllm.agents.automl.automl.extract_metrics')
@patch('fedotllm.agents.automl.automl.run_tests')
@patch('fedotllm.agents.automl.automl.fix_solution')
@patch('fedotllm.agents.automl.automl.evaluate')
@patch('fedotllm.agents.automl.automl.generate_code')
@patch('fedotllm.agents.automl.automl.insert_templates')
@patch('fedotllm.agents.automl.automl.select_skeleton')
@patch('fedotllm.agents.automl.automl.generate_automl_config')
@patch('fedotllm.agents.automl.automl.problem_reflection')
@patch('fedotllm.agents.automl.automl.if_bug') # This is a condition function
@patch('fedotllm.agents.automl.automl.END', new_callable=lambda: "END_NODE_AUTOML")
@patch('fedotllm.agents.automl.automl.START', new_callable=lambda: "START_NODE_AUTOML")
@patch('fedotllm.agents.automl.automl.StateGraph')
class TestAutoMLAgent:

    def test_automl_agent_init(self, mock_sg, mock_start, mock_end, mock_if_bug,
                               mock_pr, mock_gac, mock_ss, mock_it, mock_gc,
                               mock_ev, mock_fs, mock_rt, mock_em, mock_gr, # Patched node functions
                               mock_inference, mock_dataset, mock_workspace):
        agent = AutoMLAgent(inference=mock_inference, dataset=mock_dataset, workspace=mock_workspace)
        assert agent.inference == mock_inference
        assert agent.dataset == mock_dataset
        assert agent.workspace == mock_workspace

    def test_automl_agent_init_state(self, mock_sg, mock_start, mock_end, mock_if_bug,
                                     mock_pr, mock_gac, mock_ss, mock_it, mock_gc,
                                     mock_ev, mock_fs, mock_rt, mock_em, mock_gr,
                                     mock_inference, mock_dataset, mock_workspace):
        agent = AutoMLAgent(inference=mock_inference, dataset=mock_dataset, workspace=mock_workspace)

        # Create a dummy state; only 'messages' is accessed by init_state if inherited, but AutoMLAgentState is simple
        initial_state_dict = {"messages": []} # Or any other required fields for AutoMLAgentState
        state_instance = AutoMLAgentState(**initial_state_dict)

        command = agent.init_state(state_instance)

        assert isinstance(command, Command)
        expected_update = {
            "reflection": None, "fedot_config": None, "skeleton": None,
            "raw_code": None, "code": None, "observation": None,
            "fix_attempts": 0, "metrics": "", "pipeline": "", "report": ""
        }
        assert command.update == expected_update

    def test_automl_agent_create_graph(self, mock_state_graph_constructor,
                                       mock_start_node_str, mock_end_node_str, mock_if_bug_func,
                                       mock_problem_reflection_node,
                                       mock_generate_automl_config_node,
                                       mock_select_skeleton_node,
                                       mock_insert_templates_node,
                                       mock_generate_code_node,
                                       mock_evaluate_node,
                                       mock_fix_solution_node,
                                       mock_run_tests_node,
                                       mock_extract_metrics_node,
                                       mock_generate_report_node,
                                       mock_inference, mock_dataset, mock_workspace):

        mock_workflow_instance = MagicMock(name="StateGraphInstance")
        mock_state_graph_constructor.return_value = mock_workflow_instance

        mock_compiled_graph_final = MagicMock(name="FinalCompiledGraphWithConfig")
        mock_compiled_workflow = MagicMock(name="InitialCompiledGraph")
        mock_workflow_instance.compile.return_value = mock_compiled_workflow
        mock_compiled_workflow.with_config.return_value = mock_compiled_graph_final

        agent = AutoMLAgent(inference=mock_inference, dataset=mock_dataset, workspace=mock_workspace)
        compiled_graph = agent.create_graph()

        mock_state_graph_constructor.assert_called_once_with(AutoMLAgentState)

        # Assert add_node calls
        add_node_calls = mock_workflow_instance.add_node.call_args_list

        # Helper to check partial calls
        def check_partial_call(actual_call, expected_node_name, expected_func, expected_kwargs):
            assert actual_call[0][0] == expected_node_name
            assert isinstance(actual_call[0][1], partial)
            assert actual_call[0][1].func == expected_func
            for key, value in expected_kwargs.items():
                assert actual_call[0][1].keywords.get(key) == value

        check_partial_call(next(c for c in add_node_calls if c[0][0] == "problem_reflection"),
                           "problem_reflection", mock_problem_reflection_node,
                           {"inference": mock_inference, "dataset": mock_dataset})
        check_partial_call(next(c for c in add_node_calls if c[0][0] == "generate_automl_config"),
                           "generate_automl_config", mock_generate_automl_config_node, {"inference": mock_inference, "dataset": mock_dataset})
        check_partial_call(next(c for c in add_node_calls if c[0][0] == "select_skeleton"),
                           "select_skeleton", mock_select_skeleton_node, {"dataset": mock_dataset, "workspace": mock_workspace}) # Corrected kwargs

        # insert_templates is added directly, not as a partial
        assert call("insert_templates", mock_insert_templates_node) in add_node_calls

        check_partial_call(next(c for c in add_node_calls if c[0][0] == "generate_code"),
                           "generate_code", mock_generate_code_node, {"inference": mock_inference, "dataset": mock_dataset})
        check_partial_call(next(c for c in add_node_calls if c[0][0] == "evaluate_main"), # Name in graph
                           "evaluate_main", mock_evaluate_node, {"workspace": mock_workspace})
        check_partial_call(next(c for c in add_node_calls if c[0][0] == "fix_solution_main"), # Name in graph
                           "fix_solution_main", mock_fix_solution_node, {"inference": mock_inference, "dataset": mock_dataset})
        check_partial_call(next(c for c in add_node_calls if c[0][0] == "run_tests"),
                           "run_tests", mock_run_tests_node, {"workspace": mock_workspace, "inference": mock_inference})
        check_partial_call(next(c for c in add_node_calls if c[0][0] == "extract_metrics"),
                           "extract_metrics", mock_extract_metrics_node, {"workspace": mock_workspace})
        check_partial_call(next(c for c in add_node_calls if c[0][0] == "generate_report"),
                           "generate_report", mock_generate_report_node, {"inference": mock_inference})

        assert call("init_state", agent.init_state) in add_node_calls


        # Assert add_edge calls for direct edges
        mock_workflow_instance.add_edge.assert_any_call(mock_start_node_str, "init_state")
        mock_workflow_instance.add_edge.assert_any_call("init_state", "problem_reflection")
        mock_workflow_instance.add_edge.assert_any_call("problem_reflection", "generate_automl_config")
        mock_workflow_instance.add_edge.assert_any_call("generate_automl_config", "select_skeleton")
        mock_workflow_instance.add_edge.assert_any_call("select_skeleton", "generate_code") # Corrected edge
        mock_workflow_instance.add_edge.assert_any_call("generate_code", "insert_templates") # Corrected edge
        # ... (from insert_templates it's conditional)
        # evaluate_main is connected via conditional edge from insert_templates
        # ... (from evaluate_main it's conditional)
        # ... (from run_tests it's conditional)
        mock_workflow_instance.add_edge.assert_any_call("extract_metrics", "generate_report")
        mock_workflow_instance.add_edge.assert_any_call("generate_report", mock_end_node_str)
        mock_workflow_instance.add_edge.assert_any_call("fix_solution_main", "insert_templates") # Corrected loop back and node name


        # Assert add_conditional_edges calls
        conditional_edge_calls = mock_workflow_instance.add_conditional_edges.call_args_list

        # insert_templates conditional edge
        cond_insert_templates = next(c for c in conditional_edge_calls if c[0][0] == "insert_templates")
        assert callable(cond_insert_templates[0][1])
        assert cond_insert_templates[0][2] == {True: "generate_code", False: "evaluate_main"} # Corrected path map keys

        # evaluate_main conditional edge
        cond_evaluate_main = next(c for c in conditional_edge_calls if c[0][0] == "evaluate_main")
        assert cond_evaluate_main[0][1] == mock_if_bug_func
        assert cond_evaluate_main[0][2] == {True: "fix_solution_main", False: "run_tests"} # Corrected node name

        # run_tests conditional edge
        cond_run_tests = next(c for c in conditional_edge_calls if c[0][0] == "run_tests")
        assert cond_run_tests[0][1] == mock_if_bug_func
        assert cond_run_tests[0][2] == {True: "fix_solution_main", False: "extract_metrics"} # Corrected node name

        mock_workflow_instance.compile.assert_called_once()
        mock_compiled_workflow.with_config.assert_called_once_with(config={"run_name": "AutoMLAgent"})
        assert compiled_graph == mock_compiled_graph_final

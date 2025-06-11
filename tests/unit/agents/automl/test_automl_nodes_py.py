import pytest
from unittest.mock import MagicMock, patch, call, ANY
from pathlib import Path

from fedotllm.agents.automl.nodes import (
    problem_reflection,
    generate_automl_config,
    select_skeleton,
    generate_code,
    insert_templates,
    evaluate,
    if_bug,
    fix_solution
)
from fedotllm.agents.automl.state import AutoMLAgentState
from fedotllm.agents.automl.structured import FedotConfig
from fedotllm.llm import AIInference
from fedotllm.data import Dataset
from langgraph.types import Command
from fedotllm.enviroments.types import Observation # For creating mock observations
# from fedotllm.settings.config_loader import get_settings # For patching if_bug if needed for config


# Fixtures
@pytest.fixture
def mock_inference(mocker):
    return MagicMock(spec=AIInference)

@pytest.fixture
def mock_dataset(mocker):
    dataset = MagicMock(spec=Dataset)
    dataset.path = MagicMock(spec=Path)
    dataset.path.absolute.return_value = "/mock/dataset/path/abs"
    dataset.path.__str__.return_value = "/mock/dataset/path/str"
    return dataset

@pytest.fixture
def automl_nodes_mock_workspace(mocker):
    return MagicMock(spec=Path)

@pytest.fixture
def initial_state():
    return AutoMLAgentState(messages=[], description="test description")

# Tests for problem_reflection
@patch('fedotllm.agents.automl.nodes.prompts')
def test_problem_reflection_success(mock_prompts, mock_inference, mock_dataset, initial_state):
    mock_dataset.dataset_preview.return_value = "Dataset Preview Text"
    mock_dataset.dataset_eda.return_value = "Dataset EDA Text"
    mock_prompts.automl.problem_reflection_prompt.return_value = "Generated Reflection Prompt"
    mock_inference.query.return_value = "Test Reflection Output"

    state_dict = dict(initial_state)

    result_command = problem_reflection(state_dict, mock_inference, mock_dataset)

    mock_prompts.automl.problem_reflection_prompt.assert_called_once_with(
        user_description="test description",
        data_files_and_content="Dataset Preview Text",
        dataset_eda="Dataset EDA Text"
    )
    mock_inference.query.assert_called_once_with("Generated Reflection Prompt")
    assert isinstance(result_command, Command)
    assert result_command.update == {"reflection": "Test Reflection Output"}

# Tests for generate_automl_config
@patch('fedotllm.agents.automl.nodes.prompts')
def test_generate_automl_config_success(mock_prompts, mock_inference, initial_state, mock_dataset):
    current_state_dict = dict(initial_state)
    current_state_dict["reflection"] = "Test Reflection"

    mock_prompts.automl.generate_configuration_prompt.return_value = "Generated Config Prompt"
    mock_fedot_config_instance = FedotConfig(
        problem="classification",
        preset="fast_train",
        metric="accuracy",
        cv_folds=5,
        predict_method="predict",
        timeout=1.0
    )
    mock_inference.create.return_value = mock_fedot_config_instance

    result_command = generate_automl_config(current_state_dict, mock_inference, mock_dataset)

    mock_prompts.automl.generate_configuration_prompt.assert_called_once_with(
        reflection="Test Reflection"
    )
    mock_inference.create.assert_called_once_with("Generated Config Prompt", response_model=FedotConfig)
    assert isinstance(result_command, Command)
    assert result_command.update == {"fedot_config": mock_fedot_config_instance}

# Tests for select_skeleton
@patch('fedotllm.agents.automl.nodes.render_template')
@patch('fedotllm.agents.automl.nodes.load_template')
def test_select_skeleton_success(mock_load_template, mock_render_template, mock_dataset, initial_state, automl_nodes_mock_workspace):
    mock_fedot_config = FedotConfig(
        problem="classification",
        preset="fast_train",
        metric="accuracy",
        cv_folds=5,
        predict_method="predict",
        timeout=1.0
    )
    current_state_dict = dict(initial_state)
    current_state_dict["fedot_config"] = mock_fedot_config

    resolved_workspace_path = Path("/resolved/workspace")
    automl_nodes_mock_workspace.resolve.return_value = resolved_workspace_path

    mock_load_template.return_value = "Raw Skeleton Template Content"
    mock_render_template.return_value = "Rendered Skeleton Code String"

    result_command = select_skeleton(current_state_dict, mock_dataset, automl_nodes_mock_workspace)

    expected_template_name = "skeleton"
    mock_load_template.assert_called_once_with(expected_template_name)

    mock_render_template.assert_called_once_with(
        template="Raw Skeleton Template Content",
        dataset_path=mock_dataset.path,
        work_dir_path=resolved_workspace_path
    )
    assert isinstance(result_command, Command)
    assert result_command.update == {"skeleton": "Rendered Skeleton Code String"}

@patch('fedotllm.agents.automl.nodes.load_template')
def test_select_skeleton_unknown_predict_method(mock_load_template_ignored, mock_dataset, initial_state, automl_nodes_mock_workspace):
    mock_fedot_config = FedotConfig.model_construct(
        problem="classification",
        preset="fast_train",
        metric="accuracy",
        cv_folds=5,
        predict_method="unknown_method",
        timeout=1.0
    )
    current_state_dict = dict(initial_state)
    current_state_dict["fedot_config"] = mock_fedot_config

    with pytest.raises(ValueError, match="Unknown predict method: unknown_method"):
        select_skeleton(current_state_dict, mock_dataset, automl_nodes_mock_workspace)

# Tests for generate_code
@patch('fedotllm.agents.automl.nodes.extract_code')
@patch('fedotllm.agents.automl.nodes.prompts')
def test_generate_code_success(mock_prompts, mock_extract_code, mock_inference, mock_dataset, initial_state):
    current_state_dict = dict(initial_state)
    current_state_dict["reflection"] = "Test Reflection"
    current_state_dict["skeleton"] = "Test Skeleton"

    mock_prompts.automl.code_generation_prompt.return_value = "Generated CodeGen Prompt"
    mock_inference.query.return_value = "Raw LLM Code Output"
    mock_extract_code.return_value = "Extracted Python Code"

    result_command = generate_code(current_state_dict, mock_inference, mock_dataset)

    mock_prompts.automl.code_generation_prompt.assert_called_once_with(
        reflection="Test Reflection",
        skeleton="Test Skeleton",
        dataset_path="/mock/dataset/path/abs"
    )
    mock_inference.query.assert_called_once_with("Generated CodeGen Prompt")
    mock_extract_code.assert_called_once_with("Raw LLM Code Output")
    assert isinstance(result_command, Command)
    assert result_command.update == {"raw_code": "Extracted Python Code"}

# Tests for insert_templates
@patch('fedotllm.agents.automl.nodes.fix_code')
@patch('fedotllm.agents.automl.nodes.render_template')
@patch('fedotllm.agents.automl.nodes.load_template')
def test_insert_templates_success(mock_load_template, mock_render_template, mock_fix_code, initial_state):
    mock_fedot_config = FedotConfig(
        problem="classification",
        preset="fast_train",
        metric="accuracy",
        cv_folds=5,
        predict_method="predict",
        timeout=1.0
    )
    current_state_dict = dict(initial_state)
    current_state_dict["raw_code"] = "from automl import train_model, evaluate_model, automl_predict\n# Rest of code"
    current_state_dict["fedot_config"] = mock_fedot_config

    def load_template_side_effect(template_name):
        if template_name == "fedot_train.py": return "Raw Template: fedot_train.py"
        if template_name == "fedot_evaluate.py": return "Raw Template: fedot_evaluate.py"
        if template_name == "fedot_predict.py": return "Raw Template: fedot_predict.py"
        return None
    mock_load_template.side_effect = load_template_side_effect

    render_calls = []
    def render_template_side_effect(template, **params):
        rendered_output = f"Rendered: {template} with {params}"
        render_calls.append(call(template=template, **params))
        return rendered_output
    mock_render_template.side_effect = render_template_side_effect

    mock_fix_code.return_value = "Final Fixed Code"

    result_command = insert_templates(current_state_dict)

    expected_load_calls = [
        call("fedot_train.py"),
        call("fedot_evaluate.py"),
        call("fedot_predict.py")
    ]
    mock_load_template.assert_has_calls(expected_load_calls, any_order=False)

    assert len(render_calls) == 3
    assert render_calls[0].kwargs['template'] == "Raw Template: fedot_train.py"
    assert render_calls[0].kwargs['problem'] == str(mock_fedot_config.problem)
    assert render_calls[1].kwargs['template'] == "Raw Template: fedot_evaluate.py"
    assert render_calls[1].kwargs['predict_method'] == "predict(features=input_data)"
    assert render_calls[2].kwargs['template'] == "Raw Template: fedot_predict.py"
    assert render_calls[2].kwargs['predict_method'] == "predict(features=input_data)"

    combined_code_arg = mock_fix_code.call_args[0][0]
    assert "Rendered: Raw Template: fedot_train.py" in combined_code_arg
    assert "Rendered: Raw Template: fedot_evaluate.py" in combined_code_arg
    assert "Rendered: Raw Template: fedot_predict.py" in combined_code_arg
    assert "\n# Rest of code" in combined_code_arg
    assert "from automl import train_model, evaluate_model, automl_predict" not in combined_code_arg

    mock_fix_code.assert_called_once_with(ANY, remove_all_unused_imports=True, remove_unused_variables=True)

    assert isinstance(result_command, Command)
    assert result_command.update == {"code": "Final Fixed Code"}

@patch('fedotllm.agents.automl.nodes.fix_code')
@patch('fedotllm.agents.automl.nodes.render_template')
@patch('fedotllm.agents.automl.nodes.load_template')
def test_insert_templates_exception_handling(mock_load_template, mock_render_template, mock_fix_code, initial_state):
    mock_fedot_config = FedotConfig(
        problem="classification",
        preset="fast_train",
        metric="accuracy",
        cv_folds=5,
        predict_method="predict",
        timeout=1.0
    )
    current_state_dict = dict(initial_state)
    current_state_dict["raw_code"] = "Some code"
    current_state_dict["fedot_config"] = mock_fedot_config

    mock_load_template.side_effect = Exception("Template loading failed")

    result_command = insert_templates(current_state_dict)

    assert isinstance(result_command, Command)
    assert result_command.update == {"code": None}
    mock_fix_code.assert_not_called()

# Tests for evaluate
@patch('fedotllm.agents.automl.nodes.execute_code')
@patch('fedotllm.agents.automl.nodes._generate_code_file')
def test_evaluate_success(mock_generate_code_file, mock_execute_code, automl_nodes_mock_workspace, initial_state):
    current_state_dict = dict(initial_state)
    current_state_dict["code"] = "valid python code"

    mock_code_file_path = MagicMock(spec=Path)
    mock_generate_code_file.return_value = mock_code_file_path

    mock_observation = Observation(stdout="Run successful", stderr="", error=False, msg="OK")
    mock_execute_code.return_value = mock_observation

    result_command = evaluate(current_state_dict, automl_nodes_mock_workspace)

    mock_generate_code_file.assert_called_once_with("valid python code", automl_nodes_mock_workspace)
    mock_execute_code.assert_called_once_with(path_to_run_code=mock_code_file_path)
    assert isinstance(result_command, Command)
    assert result_command.update == {"observation": mock_observation}

@patch('fedotllm.agents.automl.nodes.execute_code')
@patch('fedotllm.agents.automl.nodes._generate_code_file')
def test_evaluate_execution_error(mock_generate_code_file, mock_execute_code, automl_nodes_mock_workspace, initial_state):
    current_state_dict = dict(initial_state)
    current_state_dict["code"] = "buggy python code"

    mock_code_file_path = MagicMock(spec=Path)
    mock_generate_code_file.return_value = mock_code_file_path

    mock_observation = Observation(stdout="", stderr="Syntax Error", error=True, msg="Error")
    mock_execute_code.return_value = mock_observation

    result_command = evaluate(current_state_dict, automl_nodes_mock_workspace)

    mock_generate_code_file.assert_called_once_with("buggy python code", automl_nodes_mock_workspace)
    mock_execute_code.assert_called_once_with(path_to_run_code=mock_code_file_path)
    assert isinstance(result_command, Command)
    assert result_command.update == {"observation": mock_observation}

# Tests for if_bug
@patch('fedotllm.agents.automl.nodes.get_settings')
def test_if_bug_is_true(mock_get_settings, initial_state):
    mock_settings_instance = MagicMock()
    mock_settings_instance.config.fix_tries = 3
    mock_get_settings.return_value = mock_settings_instance

    current_state_dict = dict(initial_state)
    current_state_dict["observation"] = Observation(error=True, msg="Error occurred", stdout="", stderr="Traceback...")
    current_state_dict["fix_attempts"] = 1

    assert if_bug(current_state_dict) is True

@patch('fedotllm.agents.automl.nodes.get_settings')
def test_if_bug_is_false_no_error(mock_get_settings, initial_state):
    mock_settings_instance = MagicMock()
    mock_settings_instance.config.fix_tries = 3
    mock_get_settings.return_value = mock_settings_instance

    current_state_dict = dict(initial_state)
    current_state_dict["observation"] = Observation(error=False, msg="OK", stdout="Success", stderr="")
    current_state_dict["fix_attempts"] = 1

    assert if_bug(current_state_dict) is False

@patch('fedotllm.agents.automl.nodes.get_settings')
def test_if_bug_is_false_max_attempts_reached(mock_get_settings, initial_state):
    mock_settings_instance = MagicMock()
    mock_settings_instance.config.fix_tries = 3
    mock_get_settings.return_value = mock_settings_instance

    current_state_dict = dict(initial_state)
    current_state_dict["observation"] = Observation(error=True, msg="Error occurred", stdout="", stderr="Traceback...")
    current_state_dict["fix_attempts"] = 3

    assert if_bug(current_state_dict) is False

# Tests for fix_solution
@patch('fedotllm.agents.automl.nodes.extract_code')
@patch('fedotllm.agents.automl.nodes.prompts')
def test_fix_solution_success(mock_prompts, mock_extract_code, mock_inference, mock_dataset, initial_state):
    # mock_dataset.path.absolute is already configured in the fixture

    current_state_dict = dict(initial_state)
    current_state_dict["reflection"] = "Test Reflection"
    current_state_dict["raw_code"] = "buggy code"
    current_state_dict["observation"] = Observation(stdout="out", stderr="err", error=True, msg="Error details")
    current_state_dict["fix_attempts"] = 0

    mock_prompts.automl.fix_solution_prompt.return_value = "Generated Fix Prompt"
    mock_inference.query.return_value = "Raw Fixed Code Output"
    mock_extract_code.return_value = "Extracted Fixed Code"

    result_command = fix_solution(current_state_dict, mock_inference, mock_dataset)

    mock_prompts.automl.fix_solution_prompt.assert_called_once_with(
        reflection="Test Reflection",
        dataset_path="/mock/dataset/path/abs",
        code_recent_solution="buggy code",
        msg="Error details",
        stderr="err",
        stdout="out"
    )
    mock_inference.query.assert_called_once_with("Generated Fix Prompt")
    mock_extract_code.assert_called_once_with("Raw Fixed Code Output")
    assert isinstance(result_command, Command)
    assert result_command.update == {"raw_code": "Extracted Fixed Code", "fix_attempts": 1}

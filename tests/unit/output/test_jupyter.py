import pytest
from unittest.mock import patch, MagicMock, call

from IPython.display import Markdown as IPythonMarkdown  # To avoid clash with pytest-markdown

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables.schema import StreamEvent

from fedotllm.output.jupyter import JupyterOutput


@pytest.fixture
def mock_display():
    with patch("IPython.display.display") as mock_d:
        yield mock_d


@pytest.fixture
def mock_markdown():
    # Patching where Markdown is *used* by the module under test
    with patch("fedotllm.output.jupyter.Markdown", spec=IPythonMarkdown) as mock_m:
        # When Markdown is called, return a mock object that can be tracked
        mock_m.side_effect = lambda x: MagicMock(spec=IPythonMarkdown, content=x)
        yield mock_m


@pytest.fixture
def mock_clear_output():
    with patch("IPython.display.clear_output") as mock_c:
        yield mock_c


@pytest.fixture
def jupyter_output_instance():
    return JupyterOutput()


class TestJupyterOutput:

    def test_messages_handler_init(self, jupyter_output_instance):
        handler = jupyter_output_instance.messages_handler()
        assert callable(handler)

    @pytest.mark.parametrize("agent_name", ["SupervisorAgent", "ResearcherAgent", "AutoMLAgent"])
    def test_messages_handler_processes_subscribed_agents(self, agent_name, jupyter_output_instance, mock_markdown):
        handler = jupyter_output_instance.messages_handler()
        ai_message = AIMessage(content="AI says hello", id="ai1")
        human_message = HumanMessage(content="Human says hi", name="Tester", id="human1")

        event_data = {
            "output": {
                "messages": [ai_message, human_message]
            }
        }
        # StreamEvent is a TypedDict, create as a dict literal
        event: StreamEvent = {"data": event_data, "name": agent_name, "event": "on_chain_stream", "run_id": "test_run"}

        handler(event)

        mock_markdown.assert_called_once()
        assert len(jupyter_output_instance.display_content) == 1

        # Content is now on the mock object returned by side_effect, stored in display_content
        processed_content_str = jupyter_output_instance.display_content[0].content

        assert "Supervisor" in processed_content_str
        assert "AI says hello" in processed_content_str
        assert "Tester" in processed_content_str # For HumanMessage with name
        assert "Human says hi" in processed_content_str

        # The content attribute of the mock object in display_content should be the string passed to Markdown()
        # This is implicitly tested by checking processed_content_str above.

    def test_messages_handler_ignores_other_agents(self, jupyter_output_instance, mock_markdown):
        handler = jupyter_output_instance.messages_handler()
        event: StreamEvent = {
            "data": {"output": {"messages": [AIMessage(content="ignore me", id="ai_ignore")]}},
            "name": "OtherAgent",
            "event": "on_chain_stream",
            "run_id": "test_run"
        }
        handler(event)
        mock_markdown.assert_not_called()
        assert len(jupyter_output_instance.display_content) == 0

    def test_messages_handler_handles_no_messages_in_output(self, jupyter_output_instance, mock_markdown):
        handler = jupyter_output_instance.messages_handler()
        event: StreamEvent = {
            "data": {"output": {}},
            "name": "SupervisorAgent",
            "event": "on_chain_stream",
            "run_id": "test_run"
        } # No 'messages' key
        handler(event)
        mock_markdown.assert_not_called()
        assert len(jupyter_output_instance.display_content) == 0

    def test_messages_handler_handles_messages_not_list(self, jupyter_output_instance, mock_markdown):
        handler = jupyter_output_instance.messages_handler()
        ai_message = AIMessage(content="Single AI message", id="single_ai")
        event: StreamEvent = {
            "data": {"output": {"messages": ai_message}},
            "name": "SupervisorAgent",
            "event": "on_chain_stream",
            "run_id": "test_run"
        }
        handler(event)
        mock_markdown.assert_called_once()
        assert len(jupyter_output_instance.display_content) == 1
        processed_content_str = jupyter_output_instance.display_content[0].content
        assert "Supervisor" in processed_content_str
        assert "Single AI message" in processed_content_str

    def test_messages_handler_formats_headers_correctly(self, jupyter_output_instance, mock_markdown):
        handler = jupyter_output_instance.messages_handler()
        ai_msg = AIMessage(content="AI content", id="h_ai")
        human_msg_named = HumanMessage(content="Human named content", name="SpecificAgent", id="h_h1")
        human_msg_unnamed = HumanMessage(content="Human unnamed content", id="h_h2")

        event: StreamEvent = {
            "data": {"output": {"messages": [ai_msg, human_msg_named, human_msg_unnamed]}},
            "name": "SupervisorAgent",
            "event": "on_chain_stream",
            "run_id": "test_run"
        }
        handler(event)

        mock_markdown.assert_called_once()
        assert len(jupyter_output_instance.display_content) == 1
        rendered_markdown = jupyter_output_instance.display_content[0].content

        assert " Supervisor ".center(50, "=") in rendered_markdown
        assert " SpecificAgent ".center(50, "=") in rendered_markdown
        assert " HumanMessage ".center(50, "=") in rendered_markdown
        assert "AI content" in rendered_markdown
        assert "Human named content" in rendered_markdown
        assert "Human unnamed content" in rendered_markdown

    def test_messages_handler_ignores_duplicate_messages(self, jupyter_output_instance, mock_markdown):
        handler = jupyter_output_instance.messages_handler()
        msg1 = AIMessage(content="Unique message 1", id="m1")
        msg_dup = AIMessage(content="Duplicate", id="m_dup")
        msg2 = HumanMessage(content="Unique message 2", id="m2")

        event1_data = {"output": {"messages": [msg1, msg_dup]}}
        event1: StreamEvent = {
            "data": event1_data,
            "name": "SupervisorAgent",
            "event": "on_chain_stream",
            "run_id": "test_run1"
        }

        event2_data = {"output": {"messages": [msg_dup, msg2]}} # msg_dup is repeated
        event2: StreamEvent = {
            "data": event2_data,
            "name": "ResearcherAgent",
            "event": "on_chain_stream",
            "run_id": "test_run2"
        }

        handler(event1)
        mock_markdown.assert_called_once()
        assert len(jupyter_output_instance.display_content) == 1
        md_content_1 = jupyter_output_instance.display_content[0].content
        assert "Unique message 1" in md_content_1
        assert "Duplicate" in md_content_1

        # Reset display_content to isolate effect of next event processing by handler
        jupyter_output_instance.display_content = []
        mock_markdown.reset_mock() # Reset call_count for the next assertion

        handler(event2)
        mock_markdown.assert_called_once() # Check call for event2
        assert len(jupyter_output_instance.display_content) == 1
        md_content_2 = jupyter_output_instance.display_content[0].content
        assert "Duplicate" not in md_content_2 # Crucial: duplicate content should not be re-added
        assert "Unique message 2" in md_content_2


    def test_messages_handler_no_data_in_event(self, jupyter_output_instance, mock_markdown):
        handler = jupyter_output_instance.messages_handler()
        event: StreamEvent = {
            "data": {},
            "name": "SupervisorAgent",
            "event": "on_chain_stream",
            "run_id": "test_run"
        }
        handler(event)
        mock_markdown.assert_not_called()
        assert len(jupyter_output_instance.display_content) == 0

    def test_messages_handler_no_output_in_data(self, jupyter_output_instance, mock_markdown):
        handler = jupyter_output_instance.messages_handler()
        event: StreamEvent = {
            "data": {"some_other_key": "value"},
            "name": "SupervisorAgent",
            "event": "on_chain_stream",
            "run_id": "test_run"
        }
        handler(event)
        mock_markdown.assert_not_called()
        assert len(jupyter_output_instance.display_content) == 0

    def test_messages_handler_empty_messages_list(self, jupyter_output_instance, mock_markdown):
        handler = jupyter_output_instance.messages_handler()
        event_data = {"output": {"messages": []}}
        event: StreamEvent = {
            "data": event_data,
            "name": "SupervisorAgent",
            "event": "on_chain_stream",
            "run_id": "test_run"
        }
        handler(event)
        mock_markdown.assert_not_called()
        assert len(jupyter_output_instance.display_content) == 0

    def test_display_handler_init(self, jupyter_output_instance):
        handler = jupyter_output_instance.display_handler()
        assert callable(handler)

    @patch("fedotllm.output.jupyter.clear_output") # Patch where clear_output is used
    @patch("fedotllm.output.jupyter.display")    # Patch where display is used
    def test_display_handler_calls_display_and_clear(self, mock_display_in_module, mock_clear_output_in_module, jupyter_output_instance):
        # Fixtures mock_display and mock_clear_output are not used due to direct module patching.
        handler = jupyter_output_instance.display_handler()
        mock_md_obj1 = MagicMock(spec=IPythonMarkdown) # These are just dummy objects for the list
        mock_md_obj2 = MagicMock(spec=IPythonMarkdown)
        jupyter_output_instance.display_content = [mock_md_obj1, mock_md_obj2]

        dummy_event: StreamEvent = {"data": {}, "name": "AnyAgent", "event": "on_chain_end", "run_id": "test_run"}
        handler(dummy_event)

        mock_clear_output_in_module.assert_called_once_with(wait=True)
        mock_display_in_module.assert_has_calls([call(mock_md_obj1), call(mock_md_obj2)])
        assert len(jupyter_output_instance.display_content) == 0

    @patch("fedotllm.output.jupyter.clear_output") # Patch where clear_output is used
    @patch("fedotllm.output.jupyter.display")    # Patch where display is used
    def test_display_handler_no_content(self, mock_display_in_module, mock_clear_output_in_module, jupyter_output_instance):
        # Fixtures mock_display and mock_clear_output are not used.
        handler = jupyter_output_instance.display_handler()
        jupyter_output_instance.display_content = [] # Ensure empty

        dummy_event: StreamEvent = {"data": {}, "name": "AnyAgent", "event": "on_chain_end", "run_id": "test_run"}
        handler(dummy_event)

        mock_clear_output_in_module.assert_not_called()
        mock_display_in_module.assert_not_called()
        assert len(jupyter_output_instance.display_content) == 0

    def test_subscribe_property(self, jupyter_output_instance):
        subscribe_list = jupyter_output_instance.subscribe
        assert isinstance(subscribe_list, list)
        assert len(subscribe_list) == 2
        # Check if they are callable (as they are handler functions)
        assert callable(subscribe_list[0]) # messages_handler()
        assert callable(subscribe_list[1]) # display_handler()

        # More robust check: ensure they are indeed the *results* of those methods.
        # This can be done by patching the handler methods themselves on the instance.
        # However, Pydantic models might resist direct patching of methods if they are not fields.
        # A simpler, effective test is to verify the identity of the returned functions.

        # Get fresh handlers
        handler_one = jupyter_output_instance.messages_handler()
        handler_two = jupyter_output_instance.display_handler()

        # Store these, then access subscribe
        jupyter_output_instance._test_msg_handler_ref = handler_one
        jupyter_output_instance._test_disp_handler_ref = handler_two

        # Patch the methods on the class to return these specific instances when called via subscribe
        # This is a bit more involved; for now, let's trust that property calls the methods.
        # The main check is that it returns callables of the right type.
        # If more rigor is needed, one might need to mock the methods on the instance *before* property access,
        # but this is tricky with Pydantic's __setattr__.

        # A practical check:
        # The property @property def subscribe(self): return [self.messages_handler(), self.display_handler()]
        # will indeed call these methods on the instance.
        # We can verify the *type* of functions returned, but not easily their exact identity without complex mocking.
        # The callable checks are good. Let's assume for now this is sufficient to test the property's intent.
        pass # Keeping the callable checks.

    def test_messages_handler_multiple_events_accumulate_markdown(self, jupyter_output_instance, mock_markdown):
        handler = jupyter_output_instance.messages_handler()
        ai_message1 = AIMessage(content="First AI message", id="ai_m1")
        event1: StreamEvent = {
            "data": {"output": {"messages": [ai_message1]}},
            "name": "SupervisorAgent",
            "event": "on_chain_stream",
            "run_id": "test_run1"
        }

        handler(event1)
        mock_markdown.assert_called_once()
        assert len(jupyter_output_instance.display_content) == 1
        md_obj1_content_str = jupyter_output_instance.display_content[0].content
        assert "First AI message" in md_obj1_content_str

        # Reset mock_markdown for the next distinct call check
        mock_markdown.reset_mock()

        human_message1 = HumanMessage(content="First Human message", id="hu_m1")
        event2: StreamEvent = {
            "data": {"output": {"messages": [human_message1]}},
            "name": "ResearcherAgent",
            "event": "on_chain_stream",
            "run_id": "test_run2"
        }
        handler(event2) # Call handler again with new event
        mock_markdown.assert_called_once()
        assert len(jupyter_output_instance.display_content) == 2 # A new Markdown object is appended

        # Check the content of the *second* markdown object.
        # The side_effect of mock_markdown ensures that the .content attribute is set.
        md_obj2_actual_content = jupyter_output_instance.display_content[1].content
        assert "First Human message" in md_obj2_actual_content

        # Verify content of display_content by checking the mock objects' .content attribute
        assert jupyter_output_instance.display_content[0].content == md_obj1_content_str
        # Ensure the second object's content is what was just processed
        assert jupyter_output_instance.display_content[1].content == md_obj2_actual_content


    def test_messages_handler_human_message_no_name(self, jupyter_output_instance, mock_markdown):
        handler = jupyter_output_instance.messages_handler()
        human_no_name = HumanMessage(content="No name human", id="h_no_name")
        event: StreamEvent = {
            "data": {"output": {"messages": [human_no_name]}},
            "name": "AutoMLAgent",
            "event": "on_chain_stream",
            "run_id": "test_run"
        }
        handler(event)

        mock_markdown.assert_called_once()
        assert len(jupyter_output_instance.display_content) == 1
        rendered_markdown = jupyter_output_instance.display_content[0].content
        assert " HumanMessage ".center(50, "=") in rendered_markdown
        assert "No name human" in rendered_markdown

    def test_messages_handler_event_data_not_dict(self, jupyter_output_instance, mock_markdown):
        handler = jupyter_output_instance.messages_handler()
        # Event data['output'] is not a dict
        event: StreamEvent = {
            "data": {"output": "just a string"},
            "name": "SupervisorAgent",
            "event": "on_chain_stream",
            "run_id": "test_run"
        }
        handler(event)
        mock_markdown.assert_not_called()
        assert len(jupyter_output_instance.display_content) == 0

    def test_messages_handler_messages_is_not_message_type_or_list(self, jupyter_output_instance, mock_markdown):
        handler = jupyter_output_instance.messages_handler()
        # Event data['output']['messages'] is not a Message or list of Messages
        event: StreamEvent = {
            "data": {"output": {"messages": "not a message object"}},
            "name": "SupervisorAgent",
            "event": "on_chain_stream",
            "run_id": "test_run"
        }
        handler(event)
        mock_markdown.assert_not_called()
        assert len(jupyter_output_instance.display_content) == 0

        jupyter_output_instance.display_content = [] # reset
        event_invalid_list: StreamEvent = {
            "data": {"output": {"messages": ["string in a list"]}},
            "name": "SupervisorAgent",
            "event": "on_chain_stream",
            "run_id": "test_run_invalid_list"
        }
        handler(event_invalid_list)
        mock_markdown.assert_not_called()
        assert len(jupyter_output_instance.display_content) == 0

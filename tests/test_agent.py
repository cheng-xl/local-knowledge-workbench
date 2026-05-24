import operator
from agent_graph import (
    build_graph, _node_context, _pick_tool, _last_user_msg, _tool_results,
    route_after_decide, should_continue, AgentState,
)


class TestGraphStructure:
    def test_has_all_nodes(self):
        graph = build_graph()
        compiled = graph.compile()
        nodes = set(compiled.get_graph().nodes.keys())
        expected = {"retrieve", "decide", "execute", "reflect",
                    "human_confirm", "answer", "__start__", "__end__"}
        missing = expected - nodes
        assert not missing, f"Missing nodes: {missing}"

    def test_graph_compiles(self):
        graph = build_graph()
        compiled = graph.compile()
        assert compiled is not None


class TestHelpers:
    def test_last_user_msg(self):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        assert _last_user_msg(msgs)["content"] == "hello"

    def test_last_user_msg_none(self):
        assert _last_user_msg([]) is None
        assert _last_user_msg([{"role": "tool", "content": "x"}]) is None

    def test_tool_results(self):
        msgs = [
            {"role": "user", "content": "q"},
            {"role": "tool", "content": "result", "name": "calc"},
        ]
        results = _tool_results(msgs)
        assert len(results) == 1
        assert results[0]["name"] == "calc"

    def test_node_context(self):
        state: AgentState = {
            "messages": [
                {"role": "user", "content": "test query"},
                {"role": "assistant", "content": "test answer"},
            ],
            "retrieved_docs": [],
            "tool_calls": [],
            "need_human_confirm": False,
            "loop_count": 0,
            "decision": "",
        }
        ctx = _node_context(state)
        assert ctx["last_user"]["content"] == "test query"
        assert "test query" in ctx["history"]
        assert "test answer" in ctx["history"]
        assert ctx["loop_count"] == 0


class TestToolRouting:
    def test_pick_calculator(self):
        assert _pick_tool("计算 12*13") == "calculator"
        assert _pick_tool("1+1等于几") == "calculator"

    def test_pick_datetime(self):
        assert _pick_tool("今天是星期几") == "datetime_tool"
        assert _pick_tool("现在几点了") == "datetime_tool"

    def test_pick_none(self):
        assert _pick_tool("你是谁") is None


class TestRouterFunctions:
    def test_route_after_decide(self):
        state = {"decision": "tool", "messages": [], "retrieved_docs": [],
                 "tool_calls": [], "need_human_confirm": False,
                 "loop_count": 0}
        assert route_after_decide(state) == "tool"

    def test_route_default(self):
        # state without decision key should return default "answer"
        state = {"messages": [], "retrieved_docs": [],
                 "tool_calls": [], "need_human_confirm": False,
                 "loop_count": 0}
        assert route_after_decide(state) == "answer"

    def test_should_continue(self):
        state = {"decision": "continue", "messages": [], "retrieved_docs": [],
                 "tool_calls": [], "need_human_confirm": False,
                 "loop_count": 0}
        assert should_continue(state) == "continue"


class TestMessageReducer:
    def test_operator_add_accumulates(self):
        a = [{"role": "user", "content": "q1"}]
        b = [{"role": "assistant", "content": "a1"}]
        result = operator.add(a, b)
        assert len(result) == 2

from agent_graph import build_graph


def test_graph_has_all_nodes():
    graph = build_graph()
    compiled = graph.compile()
    # All 6 core nodes should be present
    nodes = compiled.get_graph().nodes
    node_names = list(nodes.keys())
    expected = {"retrieve", "decide", "execute", "reflect", "human_confirm", "answer", "__start__", "__end__"}
    missing = expected - set(node_names)
    assert not missing, f"Missing nodes: {missing}"


def test_graph_compiles():
    graph = build_graph()
    compiled = graph.compile()
    assert compiled is not None

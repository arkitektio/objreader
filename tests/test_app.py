"""Offline checks of the app's declaration: no backend and no OBF file."""

import app as M


def test_the_app_declares_its_action():
    """One action, taking an .obf/.msr mikro file and handed a mikro client."""
    assert M.app.services == ["mikro"]
    declaration = M.app.registry.to_implement_agent_input().model_dump(
        mode="json", exclude_none=True
    )
    [implementation] = declaration["implementations"]
    assert implementation["interface"] == "convert_obf_file_to_datasets"
    args = {arg["key"]: arg for arg in implementation["definition"]["args"]}
    assert args["file"]["identifier"] == "@mikro/file"
    assert args["file"]["requires"] == [
        {"key": "@mikro/extension", "operator": "IN", "value": ["obf", "msr"]}
    ]
    assert args["levels"]["default"] == 6
    # The client and the task are injected, not ports.
    assert "mikro" not in args and "task" not in args

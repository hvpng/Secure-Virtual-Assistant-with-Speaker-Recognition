from __future__ import annotations

import pytest

from module_a.scripts.evaluate_export import build_parser as evaluation_parser
from module_a.scripts.train_ecapa import build_parser as training_parser


@pytest.mark.parametrize("factory", [training_parser, evaluation_parser])
def test_cli_help(factory, capsys):
    with pytest.raises(SystemExit) as raised:
        factory().parse_args(["--help"])
    assert raised.value.code == 0
    assert "ECAPA" in capsys.readouterr().out


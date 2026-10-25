import pllm


def test_version_is_a_string():
    assert isinstance(pllm.__version__, str)

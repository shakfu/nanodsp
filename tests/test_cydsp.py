"""Tests for nanodsp nanobind extension module."""

from nanodsp._core import add, greet


def test_add():
    """Test add function."""
    assert add(1, 2) == 3
    assert add(-1, 1) == 0
    assert add(0, 0) == 0


def test_greet():
    """Test greet function."""
    assert greet("World") == "Hello, World!"
    assert greet("Python") == "Hello, Python!"

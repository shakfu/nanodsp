"""Tests for nanodsp.mix module (signalsmith mixing utilities)."""

import numpy as np
from nanodsp._core import mix


class TestHadamard:
    def test_construction(self):
        h = mix.Hadamard(4)
        assert h is not None
        assert isinstance(h, mix.Hadamard)

    def test_in_place_shape(self):
        h = mix.Hadamard(4)
        inp = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        out = h.in_place(inp)
        assert out.shape == (4,)
        assert out.dtype == np.float32

    def test_involution(self):
        """Hadamard applied twice should return to original (it is self-inverse)."""
        h = mix.Hadamard(4)
        inp = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        once = h.in_place(inp)
        twice = h.in_place(once)
        assert twice.shape == inp.shape
        np.testing.assert_allclose(twice, inp, atol=1e-5)

    def test_scaling_factor(self):
        h = mix.Hadamard(4)
        sf = h.scaling_factor()
        assert isinstance(sf, float)
        # For size 4, scaling should be 1/sqrt(4) = 0.5
        assert abs(sf - 0.5) < 0.01

    def test_energy_preservation(self):
        """Scaled Hadamard should preserve energy."""
        h = mix.Hadamard(8)
        inp = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32)
        out = h.in_place(inp)
        in_energy = np.sum(inp**2)
        out_energy = np.sum(out**2)
        assert out.shape == (8,)
        # Since it's a scaled orthogonal matrix, energy scales by scaling_factor^2 * N
        # For orthogonal: ||Hx||^2 = ||x||^2
        np.testing.assert_allclose(out_energy, in_energy, rtol=1e-4)

    def test_size_8(self):
        h = mix.Hadamard(8)
        inp = np.zeros(8, dtype=np.float32)
        inp[0] = 1.0
        out = h.in_place(inp)
        assert out.shape == (8,)
        # All entries should have equal magnitude
        magnitudes = np.abs(out)
        np.testing.assert_allclose(magnitudes, magnitudes[0], atol=1e-6)

    def test_uniform_spread(self):
        """An impulse in one channel should spread equally to all channels."""
        n = 4
        h = mix.Hadamard(n)
        inp = np.zeros(n, dtype=np.float32)
        inp[0] = 1.0
        out = h.in_place(inp)
        assert out.shape == (n,)
        # All outputs should have same magnitude
        mags = np.abs(out)
        np.testing.assert_allclose(mags, mags[0], atol=1e-6)


class TestHouseholder:
    def test_construction(self):
        h = mix.Householder(4)
        assert h is not None
        assert isinstance(h, mix.Householder)

    def test_in_place_shape(self):
        h = mix.Householder(4)
        inp = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        out = h.in_place(inp)
        assert out.shape == (4,)
        assert out.dtype == np.float32

    def test_reflection_property(self):
        """Householder applied twice should return to original."""
        h = mix.Householder(4)
        inp = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        once = h.in_place(inp)
        twice = h.in_place(once)
        assert twice.shape == inp.shape
        np.testing.assert_allclose(twice, inp, atol=1e-5)

    def test_energy_preservation(self):
        """Householder reflection preserves energy."""
        h = mix.Householder(4)
        inp = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        out = h.in_place(inp)
        np.testing.assert_allclose(np.sum(out**2), np.sum(inp**2), rtol=1e-5)

    def test_different_from_input(self):
        h = mix.Householder(4)
        inp = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        out = h.in_place(inp)
        assert not np.allclose(out, inp)


class TestCheapEnergyCrossfade:
    def test_returns_tuple(self):
        to_c, from_c = mix.cheap_energy_crossfade(0.5)
        assert isinstance(to_c, float)
        assert isinstance(from_c, float)

    def test_endpoints(self):
        to_c_0, from_c_0 = mix.cheap_energy_crossfade(0.0)
        to_c_1, from_c_1 = mix.cheap_energy_crossfade(1.0)
        # At x=0: full "from", zero "to"
        assert abs(to_c_0) < 0.01
        assert abs(from_c_0 - 1.0) < 0.01
        # At x=1: full "to", zero "from"
        assert abs(to_c_1 - 1.0) < 0.01
        assert abs(from_c_1) < 0.01

    def test_midpoint_equal_power(self):
        to_c, from_c = mix.cheap_energy_crossfade(0.5)
        # At midpoint, both should be approximately equal
        assert abs(to_c - from_c) < 0.1

    def test_energy_preservation(self):
        """to^2 + from^2 should be approximately 1 for energy-preserving crossfade."""
        for x in [0.0, 0.25, 0.5, 0.75, 1.0]:
            to_c, from_c = mix.cheap_energy_crossfade(x)
            energy = to_c**2 + from_c**2
            assert abs(energy - 1.0) < 0.1, f"Energy at x={x}: {energy}"

    def test_monotonic_interior(self):
        """'to' coefficient should generally increase with x (interior region)."""
        to_values = []
        # Test interior points where the cheap approximation is well-behaved
        for x_int in range(1, 10):
            x = x_int / 10.0
            to_c, _ = mix.cheap_energy_crossfade(x)
            to_values.append(to_c)
        for i in range(1, len(to_values)):
            assert to_values[i] >= to_values[i - 1] - 0.01

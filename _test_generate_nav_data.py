import generate_nav_data as g


def test_main():
    dat = g.main()
    assert dat['nn_input'].shape == (80, 200, 200, 5, 3)
    assert dat['nn_output'].shape == (80, 1000_000)

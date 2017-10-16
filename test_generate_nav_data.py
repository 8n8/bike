import generate_nav_data as g


def test_main():
    dat = g.main()
    assert dat['nn_input'].shape == (20, 200, 200, 5, 3)
    assert dat['nn_output'].shape == (20, 1000_000)

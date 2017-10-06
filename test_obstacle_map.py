import numpy as np
import obstacle_map as o


def test_output_shape():
    i = np.random.randint(0, 256, size=(1, 200, 200, 3), dtype=np.uint8)
    midout = o.first_net().predict(i)
    assert midout.shape == (1, 2940)
    midin = np.stack(
        [midout for _ in range(5)], axis=1).reshape(1, 2940, 5, 1)
    assert midin.shape == (1, 2940, 5, 1)
    out = o.second_net().predict(midin)
    assert out.shape == (1, 5000)

from ..sydra.generate_dataset import generate_sample


def test_generate_sample():
    signals, config = generate_sample()

    assert len(signals) == len(config["mic_coordinates"])
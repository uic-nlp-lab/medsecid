#!./pyvirenv/bin/python3

from zensols.cli import ConfigurationImporterCliHarness


def init():
    # reset random state for consistency before any other packages are
    # imported
    from zensols.deeplearn import TorchConfig
    TorchConfig.init()
    # initialize the NLP system
    from zensols import deepnlp
    deepnlp.init()


def silencewarn():
    """Silence warnings from sklearn metrics about missing labels."""
    import warnings
    m = 'Recall is ill-defined and being set to 0.0 in labels with no true.*'
    warnings.filterwarnings("ignore", message=m)


if (__name__ == '__main__'):
    init()
    silencewarn()
    harness = ConfigurationImporterCliHarness(
        src_dir_name='src',
        package_resource='sidmodel',
        config_path='models/glove50.conf',
        proto_args='hyperparams',
        proto_factory_kwargs={'reload_pattern': r'^sidmodel'},
    )
    harness.run()

import matplotlib.pyplot as plt
from zensols.cli import NotebookHarness


class AppNotebookHarness(NotebookHarness):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            root_dir='..',
            src_dir_name='src',
            package_resource='sidmodel',
            app_config_resource='resources/app.conf',
            factory_kwargs={'reload_pattern': r'^sidmodel\.(embanalysis)'},
            **kwargs)

    def get_app(self):
        from sidmodel import Application
        app: Application = self('proto --config ../models/glove50.conf')
        return app

    def get_analyzer(self, **kwargs):
        plot_type: str = kwargs.get('plot_type', 'section')
        app = self.get_app()
        analyzer = app.config_factory('embedded_analyzer')
        if plot_type == 'age':
            analyzer.plot_tfidf = False
        else:
            analyzer.plot_tfidf = True
            analyzer.keep_age_types = {'adult', 'pediatric'}
        for k, v in kwargs.items():
            setattr(analyzer, k, v)
        return analyzer


def create_subplots(rows: int = 1, cols: int = 1, pad: float = 5.,
                    add_height: int = 0, height: int = None,
                    width: int = 20):
    """Create the matplotlib plot axes."""
    if height is None:
        height = 5 * (rows + add_height)
    params = dict(ncols=cols, nrows=rows, sharex=False)
    if width is not None:
        params.update(dict(figsize=(width, height)))
    fig, axs = plt.subplots(**params)
    if width is None:
        fig.canvas.layout.width = '100%'
        fig.canvas.layout.height = '900px'
    else:
        fig.tight_layout(pad=pad)
    return fig, axs

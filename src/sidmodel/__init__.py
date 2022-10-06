from zensols.util import APIError


class AnnotationError(APIError):
    pass


from .corpus import *
from .majorsent import *
from .model import *
from .facade import *
from .embanalysis import *
from .resanalysis import *
from .app import *

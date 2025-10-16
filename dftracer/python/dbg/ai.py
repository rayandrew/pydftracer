from dftracer.python.dbg.logger import DFTRACER_ENABLE, dft_fn as dft_fn_orig, dftracer as dftracer_orig

import dftracer.python.ai_common as ai_common
ai_common.dft_fn = dft_fn_orig
ai_common.dftracer = dftracer_orig

from dftracer.python.ai_common import *

from dftracer.python.ai_init import *
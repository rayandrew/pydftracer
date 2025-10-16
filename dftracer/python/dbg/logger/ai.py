from dftracer.dbg.logger.logger import DFTRACER_ENABLE, dft_fn as dft_fn_orig, dftracer as dftracer_orig

import dftracer.logger.ai_common as ai_common
ai_common.dft_fn = dft_fn_orig
ai_common.dftracer = dftracer_orig

from dftracer.logger.ai_common import *

from dftracer.logger.ai_init import *
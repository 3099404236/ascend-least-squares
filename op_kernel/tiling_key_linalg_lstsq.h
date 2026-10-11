#pragma once
#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(LinalgLstsq,
    ASCENDC_TPL_DATATYPE_DECL(DT_A, C_DT_FLOAT),
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_A, C_DT_FLOAT),
    ),
);

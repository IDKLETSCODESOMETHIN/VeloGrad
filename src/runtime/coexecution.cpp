// =================================================================================================
// PROJECT: VeloGrad Engine Core (Bare-Metal Neural Compute Framework)
// MODULE: Spatial Multitasking Implementation (VeloGrad v4.3)
// COPYRIGHT (C) 2026 VeloGrad Architecture & Systems Core Team
// AGPLv3 + Commercial Royalty Waiver Dual License
// =================================================================================================

#include "velograd/coexecution.h"

namespace velograd {
namespace coexecution {

SpatialMultitasker& SpatialMultitasker::instance() {
    static SpatialMultitasker multitasker;
    return multitasker;
}

SpatialMultitasker::SpatialMultitasker() = default;

SpatialMultitasker::~SpatialMultitasker() = default;

} // namespace coexecution
} // namespace velograd

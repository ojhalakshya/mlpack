/**
 * @file dropout_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the PixelShuffle class, which implements rearrangement of
 * data from depth into of spatial data.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_PIXEL_SHUFFLE_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_PIXEL_SHUFFLE_IMPL_HPP

// In case it hasn't yet been included.
#include "pixel_shuffle.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
Dropout<InputDataType, OutputDataType>::Dropout(
    const double ratio) :
    ratio(ratio),
    scale(1.0 / (1.0 - ratio)),
    deterministic(false)
{
/**
 * @file pixel_shuffle.hpp
 * @author Lakshya Ojha
 *
 * Pixel-shuffling is the operation of taking groups of values along
 * the *channel* dimension and regrouping them into blocks of pixels
 * along the ``H`` and ``W`` dimensions, thereby effectively multiplying
 * those dimensions by a constant factor in size.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_PIXEL_SHUFFLE_HPP
#define MLPACK_METHODS_ANN_LAYER_PIXEL_SHUFFLE_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {


/**
 * The dropout layer is a regularizer that randomly with probability 'ratio'
 * sets input values to zero and scales the remaining elements by factor 1 /
 * (1 - ratio) rather than during test time so as to keep the expected sum same.
 * In the deterministic mode (during testing), there is no change in the input.
 *
 * Note: During training you should set deterministic to false and during
 * testing you should set deterministic to true.
 *
 * For more information, see the following.
 *
 * @code
 * ///////////////////////////////////////////////////////////
 * @endcode
 *
 * @tparam InputDataType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputDataType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template<typename InputDataType = arma::mat,
         typename OutputDataType = arma::mat>
class PixelShuffle
{
 public:
  /**
   * Create the PixelShuffle object.
   */
  PixelShuffle();

  /**
   * Ordinary feed forward pass of the PixelShuffle layer.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  template<typename eT>
  void Forward(const arma::Mat<eT>& input, arma::Mat<eT>& output);

  /**
   * Ordinary feed backward pass of the PixelShuffle layer.
   *
   * @param input The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  template<typename eT>
  void Backward(const arma::Mat<eT>& /* input */,
                const arma::Mat<eT>& gy,
                arma::Mat<eT>& g);

  //! Get the output parameter.
  OutputDataType const& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the detla.
  OutputDataType const& Delta() const { return delta; }
  //! Modify the delta.
  OutputDataType& Delta() { return delta; }

  //! The value of the deterministic parameter.
  bool Deterministic() const { return deterministic; }
  //! Modify the value of the deterministic parameter.
  bool& Deterministic() { return deterministic; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */);

 private:
  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

  //! Locally-stored mast object.
  OutputDataType mask;

  //! If true Pixelshuffle and scaling is disabled, see notes above.
  bool deterministic;
}; // class PixelShuffle

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "pixel_shuffle_impl.hpp"

#endif
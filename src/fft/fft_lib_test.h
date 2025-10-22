/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2025 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/
#ifndef FFT_LIB_TEST_H
#define FFT_LIB_TEST_H

/*******************************************************************************
 * \brief Function to test the local FFT backend (1-3D).
 * \author Frederick Stein
 ******************************************************************************/
int fft_test_local();

/*******************************************************************************
 * \brief Function to test the distributed FFT backend (2-3D, 1D not used).
 * \author Frederick Stein
 ******************************************************************************/
int fft_test_distributed();

/*******************************************************************************
 * \brief Function to test the local transposition operation.
 * \author Frederick Stein
 ******************************************************************************/
int fft_test_transpose();

#endif /* FFT_LIB_TEST_H */

// EOF

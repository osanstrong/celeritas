
//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/Complex.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"

namespace celeritas
{

struct complex
{
    real_type real;
    real_type imag;

    inline CELER_FUNCTION complex(real_type r, real_type i)
    {
        this->real = r;
        this->imag = i;
    }

    inline CELER_FUNCTION complex() { complex(0.0, 0.0); }

    // static inline CELER_FUNCTION complex ::std::sqrt(complex const& val) {
    //     return complex{0,0};
    // }

    inline CELER_FUNCTION complex sqrt() const
    {
        real_type r = std::sqrt(0.5 * (this->abs() + this->real));
        real_type i = std::copysign(
            std::sqrt(0.5 * (this->abs() - this->real)), this->imag);
        return complex{r, i};
    }

    inline CELER_FUNCTION real_type abs() const
    {
        return hypot<real_type>(this->real, this->imag);
    }

    inline CELER_FUNCTION complex conj() const
    {
        return complex{this->real, -this->imag};
    }

    inline CELER_FUNCTION complex operator+(complex const& other) const
    {
        return complex{this->real + other.real, this->imag + other.imag};
    }

    inline CELER_FUNCTION complex operator-(complex const& other) const
    {
        return complex{this->real - other.real, this->imag - other.imag};
    }

    inline CELER_FUNCTION complex operator+(real_type other) const
    {
        return complex{this->real + other, this->imag};
    }

    inline CELER_FUNCTION complex operator-(real_type other) const
    {
        return (*this) + (-other);
    }

    inline CELER_FUNCTION complex operator*(real_type factor_real) const
    {
        return complex{this->real * factor_real, this->imag * factor_real};
    }

    // inline CELER_FUNCTION complex operator*(real_type r, complex c) const {
    //     return c * r;
    // }

    inline CELER_FUNCTION complex operator/(real_type divisor_real) const
    {
        return complex{this->real / divisor_real, this->imag / divisor_real};
    }

    inline CELER_FUNCTION complex operator/(complex divisor_complex) const
    {
        complex d = divisor_complex;
        real_type d2 = d.real * d.real - d.imag * d.imag;
        complex c = d.conj();
        return ((*this) * c) / d2;
    }

    inline CELER_FUNCTION complex operator*(complex const& other) const
    {
        real_type r1 = this->real, i1 = this->imag, r2 = other.real,
                  i2 = other.real;
        return complex{r1 * r2 - i1 * i2, i1 * r2 + r1 * i2};
    }

    inline CELER_FUNCTION complex& operator=(real_type right)
    {
        real = right;
        imag = 0.0;
        return *this;
    }

    inline CELER_FUNCTION complex& operator*=(real_type right)
    {
        real *= right;
        imag *= right;
        return *this;
    }
};

// class Complex {
//     public:

//         //// CONSTRUCTORS ////

//         inline CELER_FUNCTION Complex(real_type r, real_type i);

//         inline CELER_FUNCTION Complex(): Complex{0.,0.} {}

//         //// DATA ////

//         real_type real;
//         real_type imag;

//         //// OPERATORS ////

//         inline CELER_FUNCTION real_type abs() const;

//         inline CELER_FUNCTION Complex conj() const {
//             return Complex{this->real, -this->imag};
//         }

//         inline CELER_FUNCTION Complex operator+(Complex const& other) const
//         {
//             return Complex{this->real + other.real, this->imag +
//             other.imag};
//         }

//         inline CELER_FUNCTION Complex operator+(real_type other) const {
//             return Complex{this->real + other, this->imag};
//         }

//         inline CELER_FUNCTION Complex operator-(real_type other) const {
//             return (*this) + (-other);
//         }

//         // inline CELER_FUNCTION Complex operator*(real_type factor_real)
//         const {
//         //     return Complex{this->real * factor_real, this->imag *
//         factor_real};
//         // }

//         // inline CELER_FUNCTION Complex operator*(real_type r, Complex c)
//         const {
//         //     return c * r;
//         // }

//         inline CELER_FUNCTION Complex operator/(real_type divisor_real)
//         const {
//             return Complex{this->real / divisor_real, this->imag /
//             divisor_real};
//         }

//         inline CELER_FUNCTION Complex operator*(Complex const& other) const
//         {
//             real_type r1 = this->real, i1 = this-> imag, r2 = other.real, i2
//             = other.real; return Complex{r1*r2 - i1*i2, i1*r2 + r1*i2};
//         }

//         inline CELER_FUNCTION Complex operator=(real_type other) {
//             return Complex{other, 0.};
//         }
// };

// CELER_FUNCTION Complex::Complex(real_type r, real_type i)  : real{r},
// imag{i} {}

// CELER_FUNCTION real_type Complex::abs() const {
//     return hypot<real_type>(real, imag);
// }

}  // namespace celeritas

// namespace std {
//     inline CELER_FUNCTION real_type fabs(complex const& val) {
//         return val.abs();
//     }
// }

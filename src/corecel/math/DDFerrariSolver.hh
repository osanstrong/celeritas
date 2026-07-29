//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/DDFerrariSolver.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Constants.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/NumericLimits.hh"
#include "corecel/math/PolyEvaluator.hh"
#include "corecel/math/SoftEqual.hh"
#include "corecel/math/gdd.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Find positive, real roots for quartic functions using Ferarri.
 *
 * The quartic solver uses the Ferrari method \citep{polyanin-ferrari-2007,
 * https://doi.org/10.1201/9781420010510}, using the underlying quadratic and
 * cubic solutions from
 * \citet{press-nr-1992, https://dl.acm.org/doi/10.5555/1403886} .
 *
 * The quartic equation
 * \f[
   a x^4 + b x^3 + c x^2 + d x + e = 0
 * \f]
 * has four solutions mathematically, but we only require solutions which are
 * both real and positive. This equation is also subject to multiple cases of
 * catastrophic precision-limitation-based error both fundamentally and as a
 * consequence of the particular algorithm chosen. This solver implements the
 * Ferrari method, which is well-established and simple, but more
 * prone to numerical error than contemporary methods to be explored such as
 * Algorithm 1010 \citep{orellana-alg1010-2020,
 * https://doi.org/10.1145/3386241}.
 *
 * The input argument to an instance of this class is an array \c abcde that
 * corresponds to {a, b, c, d, e}. An overload using a four-element array
 * \c abcd solves the degenerate case where \f$ e = 0 \f$.
 *
 * The result is an array of 4 real numbers, where each is either a positive
 * valid intersection or the sentinel result \c infinity.
 */
class DDFerrariSolver
{
  public:
    //!@{
    //! \name Type aliases
    using Real4 = Array<gdd_real, 4>;
    using Real5 = Array<gdd_real, 5>;

    using result_type = Array<real_type, 4>;
    //!@}

  public:
    // Construct with given tolerance
    inline CELER_FUNCTION DDFerrariSolver(real_type tolerance);

    //! Construct with default tolerance equal to Orange `Tolerance`.
    inline CELER_FUNCTION DDFerrariSolver() : DDFerrariSolver{default_tol_} {}

    // Solver for fully general case
    inline CELER_FUNCTION result_type operator()(Real5 const& abcde) const;

    // Solver for surface case
    inline CELER_FUNCTION result_type operator()(Real4 const& abcd) const;

    // General solver, given normal floats
    inline CELER_FUNCTION result_type operator()(
        Array<real_type, 5> const& abcde) const
    {
        auto [a, b, c, d, e] = abcde;
        return operator()(Real5{
            make_gdd(a), make_gdd(b), make_gdd(c), make_gdd(d), make_gdd(e)});
    }

    // Surface case, given normal floats
    inline CELER_FUNCTION result_type operator()(
        Array<real_type, 4> const& abcd) const
    {
        auto [a, b, c, d] = abcd;
        return operator()(
            Real4{make_gdd(a), make_gdd(b), make_gdd(c), make_gdd(d)});
    }

  private:
    //// TYPES ////

    using Real2 = Array<gdd_real, 2>;
    using Real3 = Array<gdd_real, 3>;

    //// STATIC DATA ////

    //! Default tolerance for quadric solve, taken from Orange `Tolerance`.
    static constexpr real_type default_tol_
        = (std::is_same_v<real_type, double> ? 1e-5 : 5e-2f);

    //! No positive real solution (aka "no intersection")
    static constexpr real_type no_solution_
        = NumericLimits<real_type>::infinity();

    //// DATA ////

    // Soft zero for biquadratic and degenerate cubic detection
    SoftZero<real_type> const soft_zero_;

    //// HELPER FUNCTIONS ////
    // Try to place real at given index in list, return next free index
    inline CELER_FUNCTION int
    place_root(result_type& roots, real_type new_root, int free_index) const;

    // Find roots of special reduced quartic which is biquadratic
    inline CELER_FUNCTION result_type calc_biquadratic_roots(
        gdd_real qb, gdd_real p, gdd_real r) const;

    // Find all roots of normalized cubic (unsorted, dominant first)
    inline CELER_FUNCTION Real3 real_roots_normalized_cubic(
        gdd_real b, gdd_real c, gdd_real d) const;

    // Find real quadratic roots
    inline CELER_FUNCTION Real2 real_roots_normalized_quadratic(
        gdd_real b, gdd_real c) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct a solver instance with a specified tolerance for degenerate cases,
 * such as the particle starting on the surface.
 */
CELER_FUNCTION
DDFerrariSolver::DDFerrariSolver(real_type tolerance) : soft_zero_{tolerance}
{
}

//---------------------------------------------------------------------------//
/*!
 * Find all positive roots of the polynomial with given a, b, c, d, e.
 *
 * Replaces negative or complex roots with sentry value no_solution_.
 *
 * Allows user to specify if the operation takes place on the surface, in which
 * case it will solve as a cubic to avoid the root at 0.
 */
CELER_FUNCTION auto DDFerrariSolver::operator()(Real5 const& abcde) const
    -> result_type
{
    using namespace celeritas::literals;

    CELER_EXPECT(abcde[0] != 0);
    auto [a, b, c, d, e] = abcde;

    // Normalize coefficients
    gdd_real ba = b / a, ca = c / a, da = d / a, ea = e / a;

    constexpr real_type half{0.5};
    gdd_real qb = 0.25 * ba;

    // Incomplete quartic
    gdd_real p
        = PolyEvaluator<gdd_real, 2>{-half * ca, make_gdd(0), make_gdd(3)}(qb);
    gdd_real q = PolyEvaluator<gdd_real, 3>{
        half * da, negative(ca), make_gdd(0), make_gdd(4)}(qb);
    gdd_real r = PolyEvaluator<gdd_real, 4>{
        negative(ea), da, negative(ca), make_gdd(0), make_gdd(3)}(qb);

    // Edge case: equation is biquadratic
    if (soft_zero_(q.x))
    {
        return calc_biquadratic_roots(qb, p, r);
    }

    // One real root of subsidiary cubic
    Real3 z = DDFerrariSolver::real_roots_normalized_cubic(
        p, r, p * r - half * q * q);
    gdd_real z0 = (z[1].x == no_solution_)
                      ? z[0]
                      : max<gdd_real>(z[0], max(z[1], z[2]));

    gdd_real s2 = 2 * p + 2 * z0;
    if (s2 >= 0)
    {
        gdd_real s = gdd::sqrt(s2);
        gdd_real t;
        real_type tol = 1e-6;
        if (std::fabs(s.x) < tol)
        {
            t = sqr(z0) + r;
        }
        else
        {
            t = -q / s;
        }
        auto const [r0, r1] = real_roots_normalized_quadratic(s * half, z0 + t);
        auto const [r2, r3]
            = real_roots_normalized_quadratic(-s * half, z0 - t);

        result_type roots(
            no_solution_, no_solution_, no_solution_, no_solution_);
        int idx = 0;
        idx = place_root(roots, (r0 - qb).x, idx);
        idx = place_root(roots, (r1 - qb).x, idx);
        idx = place_root(roots, (r2 - qb).x, idx);
        place_root(roots, (r3 - qb).x, idx);

        return roots;
    }
    else
    {
        return result_type(
            no_solution_, no_solution_, no_solution_, no_solution_);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Solve a quartic polynomial where coefficient \c e is known to be 0.
 *
 * Solves as cubic equation, and does not return the known root of 0.
 */
CELER_FUNCTION auto DDFerrariSolver::operator()(Real4 const& abcd) const
    -> result_type
{
    auto [a, b, c, d] = abcd;
    // Normalize coefficients
    gdd_real ba = b / a, ca = c / a, da = d / a;
    Real3 cubic_roots = real_roots_normalized_cubic(ba, ca, da);
    auto [z0, z1, z2] = cubic_roots;

    result_type roots(no_solution_, no_solution_, no_solution_, no_solution_);

    int idx = 0;
    idx = place_root(roots, z0.x, idx);
    idx = place_root(roots, z1.x, idx);
    place_root(roots, z2.x, idx);
    return roots;
}

//---------------------------------------------------------------------------//
/*!
 * Attempt to put a value into the given list at given index, returning where
 * to place the next item.
 *
 * If the given value is no_solution_ or is not positive, does not place
 * the root, and returns the same index for the next one.
 */
CELER_FUNCTION int DDFerrariSolver::place_root(
    result_type& roots, real_type new_root, int free_index) const
{
    if (!(new_root == no_solution_ || std::isnan(new_root) || new_root <= 0))
    {
        roots[free_index] = new_root;
        free_index += 1;
    }
    return free_index;
}

//---------------------------------------------------------------------------//
/*!
 * Solve special case of Ferrari where reduced quartic is also biquadratic.
 *
 * In this special case, the normal solution won't work, and must instead be
 * solved as a quadratic equation: The square roots of each quadratic solution
 * then go on to form potential quartic solutions, for up to four roots.
 */
CELER_FUNCTION auto DDFerrariSolver::calc_biquadratic_roots(
    gdd_real qb, gdd_real p, gdd_real r) const -> result_type
{
    auto ir = real_roots_normalized_quadratic(-p, -r);
    result_type roots(no_solution_, no_solution_, no_solution_, no_solution_);
    int idx = 0;
    if (ir[1] != no_solution_ && ir[1] > 0)
    {
        gdd_real sqrt_ir1 = gdd::sqrt(ir[1]);
        gdd_real from_pos1 = sqrt_ir1 - qb;
        idx = place_root(roots, from_pos1.x, idx);
        if (from_pos1 > 0)
        {
            idx = place_root(roots, (-sqrt_ir1 - qb).x, idx);
        }
    }
    if (ir[0] != no_solution_ && ir[0] > 0)
    {
        gdd_real sqrt_ir0 = gdd::sqrt(ir[0]);
        gdd_real from_pos0 = sqrt_ir0 - qb;
        idx = place_root(roots, from_pos0.x, idx);
        if (from_pos0 > 0)
        {
            place_root(roots, (-sqrt_ir0 - qb).x, idx);
        }
    }
    return roots;
}

//---------------------------------------------------------------------------//
/*!
 * Solve for the real roots of a cubic function.
 *
 * Specifically, the cubic function
 * \f[
   a x^3 + b x^2 + c x + d
 * \f]
 * where a is assumed to already be 1, and is not provided to the
 * function.
 *
 * \return The real roots of the given cubic equation, with the dominant at
 * index 0.
 */
CELER_FUNCTION auto DDFerrariSolver::real_roots_normalized_cubic(
    gdd_real b, gdd_real c, gdd_real d) const -> Real3
{
    using namespace celeritas::literals;

    constexpr gdd_real half = gdd_real{0.5, 0};
    constexpr gdd_real third = gdd_real{1, 0} / gdd_real{3, 0};
    gdd_real third_b = b * third;

    // Intermediate values
    gdd_real q = sqr(third_b) - third * c;
    gdd_real r = half
                 * PolyEvaluator<gdd_real, 3>{d, -c, make_gdd(0), make_gdd(2)}(
                     third_b);

    gdd_real q3 = sqr(q) * q;
    gdd_real r2 = sqr(r);

    gdd_real discrim = r2 - q3;

    if (soft_zero_(q.x) && soft_zero_(r.x) && soft_zero_(discrim.x))
    {
        return Real3(
            -gdd::cbrt(d), make_gdd(no_solution_), make_gdd(no_solution_));
    }
    else if (discrim < 0)  // All roots real, calculate with trigomonetry
    {
        gdd_real theta = gdd::acos(r / gdd::sqrt(q3));
        gdd_real n2_root_q = -2_r * gdd::sqrt(q);
        // NOTE: Using positive 2*sqrt(q) seems correct for [1,-2,-2,0,8]?
        gdd_real twth_pi = constants::pi * 2_r * third;
        gdd_real third_theta = theta * third;

        gdd_real z0 = n2_root_q * gdd::cos(third_theta) - third_b;
        gdd_real z1 = n2_root_q * gdd::cos(third_theta + twth_pi) - third_b;
        gdd_real z2 = n2_root_q * gdd::cos(third_theta - twth_pi) - third_b;

        return Real3(z0, z1, z2);
    }
    else  // One real and two complex roots, solve for real root with Cardano
    {
        gdd_real nr_a = -signum(r.x) * gdd::cbrt(abs(r) + gdd::sqrt(discrim));
        gdd_real nr_b = nr_a == 0 ? gdd_real{0, 0} : q / nr_a;
        gdd_real z0 = nr_a + nr_b - third_b;
        return Real3(z0, make_gdd(no_solution_), make_gdd(no_solution_));
    }
}

//---------------------------------------------------------------------------//
/*!
 * Solve for the real roots of a quadratic function.
 *
 * Specifically, the quadratic function
 * \f[
   a x^2 + (hb*2) x + c
 * \f]
 * where a is assumed to already be 1 and not provided.
 *
 * \return A pair of roots. If roots are imaginary, returns 2x
 * no_solution_.
 */
CELER_FUNCTION auto DDFerrariSolver::real_roots_normalized_quadratic(
    gdd_real hb, gdd_real c) const -> Real2
{
    gdd_real qb2 = sqr(hb);
    if (soft_zero_((qb2 - c).x))
    {
        // One critical root
        return Real2(-hb, make_gdd(no_solution_));
    }
    else if (qb2 > c)
    {
        // Two real roots
        gdd_real ht = gdd::sqrt(qb2 - c);
        return Real2(-hb - ht, -hb + ht);
    }
    else
    {
        return Real2(gdd_real{no_solution_, no_solution_},
                     gdd_real{no_solution_, no_solution_});
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

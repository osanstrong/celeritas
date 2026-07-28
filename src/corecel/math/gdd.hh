

#pragma once
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
// #include <crt/host_defines.h>


namespace celeritas {

namespace gdd {
// { Inline function definitions

#define _GQD_SPLITTER            (134217729.0)                   // = 2^27 + 1
#define _GQD_SPLIT_THRESH        (6.69692879491417e+299)         // = 2^996

/****************Basic Funcitons *********************/

//computs fl( a + b ) and err( a + b ), assumes |a| > |b|

inline constexpr CELER_FUNCTION
double quick_two_sum(double a, double b, double &err) {

    if (b == 0.0) {
        err = 0.0;
        return (a + b);
    }

    double s = a + b;
    err = b - (s - a);

    return s;
}

inline CELER_FUNCTION
double two_sum(double a, double b, double &err) {

    if ((a == 0.0) || (b == 0.0)) {
        err = 0.0;
        return (a + b);
    }

    double s = a + b;
    double bb = s - a;
    err = (a - (s - bb)) + (b - bb);

    return s;
}


//computes fl( a - b ) and err( a - b ), assumes |a| >= |b|

inline CELER_FUNCTION
double quick_two_diff(double a, double b, double &err) {
    if (a == b) {
        err = 0.0;
        return 0.0;
    }

    double s;

    /*
    if(fabs((a-b)/a) < GPU_D_EPS) {
            s = 0.0;
            err = 0.0;
            return s;
    }
     */

    s = a - b;
    err = (a - s) - b;
    return s;
}

//computes fl( a - b ) and err( a - b )

inline constexpr CELER_FUNCTION
double two_diff(double a, double b, double &err) {
    if (a == b) {
        err = 0.0;
        return 0.0;
    }

    double s = a - b;

    /*
    if(fabs((a-b)/a) < GPU_D_EPS) {
            s = 0.0;
            err = 0.0;
            return s;
    }
     */

    double bb = s - a;
    err = (a - (s - bb)) - (b + bb);
    return s;
}

// Computes high word and lo word of a 

inline constexpr CELER_FUNCTION
void split(double a, double &hi, double &lo) {
    double temp = 0;
    if (a > _GQD_SPLIT_THRESH || a < -_GQD_SPLIT_THRESH) {
        a *= 3.7252902984619140625e-09; // 2^-28
        temp = _GQD_SPLITTER * a;
        hi = temp - (temp - a);
        lo = a - hi;
        hi *= 268435456.0; // 2^28
        lo *= 268435456.0; // 2^28
    } else {
        temp = _GQD_SPLITTER * a;
        hi = temp - (temp - a);
        lo = a - hi;
    }
}

/* Computes fl(a*b) and err(a*b). */
inline constexpr CELER_FUNCTION
double two_prod(double a, double b, double &err) {

    double a_hi = 0, a_lo = 0, b_hi = 0, b_lo = 0;
    double p = a * b;
    split(a, a_hi, a_lo);
    split(b, b_hi, b_lo);

    //err = (a_hi*b_hi) - p + (a_hi*b_lo) + (a_lo*b_hi) + (a_lo*b_lo); 
    err = (a_hi * b_hi) - p + (a_hi * b_lo) + (a_lo * b_hi) + (a_lo * b_lo);

    return p;
}

/* Computes fl(a*a) and err(a*a).  Faster than the above method. */
inline CELER_FUNCTION
double two_sqr(double a, double &err) {
    double hi, lo;
    double q = a * a;
    split(a, hi, lo);
    err = ((hi * hi - q) + 2.0 * hi * lo) + lo * lo;
    return q;
}

/* Computes the nearest integer to d. */
inline CELER_FUNCTION
double nint(double d) {
    if (d == floor(d))
        return d;
    return floor(d + 0.5);
}

/* Double add, round nearest; if compiling on device uses CUDA functions
 * to avoid fused multiply-add operations*/
inline constexpr CELER_FUNCTION __attribute__((optimize("fp-contract=off")))
double dadd_rn(double a, double b) {
    #if CELER_DEVICE_COMPILE
        return __dadd_rn(a, b);
    #else 
        return a + b;
    #endif
}

/* Double multiply, round nearest; if compiling on device uses CUDA functions
 * to avoid fused multiply-add operations*/
inline constexpr CELER_FUNCTION __attribute__((optimize("fp-contract=off")))
double dmul_rn(double a, double b) {
    #if CELER_DEVICE_COMPILe
        return __dmul_rn(a, b);
    #else
        return a * b;
    #endif
}


// }
}

struct gdd_real {
    // using real_type = double;

    real_type x;
    real_type y;

    // inline CELER_FUNCTION gdd_real(real_type u, real_type l) : x(u), y(l) {}

    // inline CELER_FUNCTION gdd_real(real_type u) : x(u) {}

    // inline CELER_FUNCTION gdd_real() {}


};

/**
 * Generator functions
 */
inline CELER_FUNCTION gdd_real make_gdd(real_type x, real_type y) {return gdd_real{x,y};}

inline CELER_FUNCTION gdd_real make_gdd(real_type x) {return gdd_real{x, 0.0};}

/**
 * Constants
 */

#define _dd_eps (4.93038065763132e-32)  // 2^-104
#define _dd_e gdd_real{2.718281828459045091e+00, 1.445646891729250158e-16}
#define _dd_log2 gdd_real{6.931471805599452862e-01, 2.319046813846299558e-17}
#define _dd_2pi gdd_real{6.283185307179586232e+00, 2.449293598294706414e-16}
#define _dd_pi gdd_real{3.141592653589793116e+00, 1.224646799147353207e-16}
#define _dd_pi2 gdd_real{1.570796326794896558e+00, 6.123233995736766036e-17}
#define _dd_pi16 gdd_real{1.963495408493620697e-01, 7.654042494670957545e-18}
#define _dd_pi4 gdd_real{7.853981633974482790e-01, 3.061616997868383018e-17}    
#define _dd_3pi4 gdd_real{2.356194490192344837e+00, 9.1848509936051484375e-17}

static const int n_dd_inv_fact = 15;
using gdd_fact = Array<gdd_real, n_dd_inv_fact>;
using gdd_four = Array<gdd_real, 4>;

static constexpr gdd_fact dd_inv_fact{
        gdd_real{1.66666666666666657e-01, 9.25185853854297066e-18},
        gdd_real{4.16666666666666644e-02, 2.31296463463574266e-18},
        gdd_real{8.33333333333333322e-03, 1.15648231731787138e-19},
        gdd_real{1.38888888888888894e-03, -5.30054395437357706e-20},
        gdd_real{1.98412698412698413e-04, 1.72095582934207053e-22},
        gdd_real{2.48015873015873016e-05, 2.15119478667758816e-23},
        gdd_real{2.75573192239858925e-06, -1.85839327404647208e-22},
        gdd_real{2.75573192239858883e-07, 2.37677146222502973e-23},
        gdd_real{2.50521083854417202e-08, -1.44881407093591197e-24},
        gdd_real{2.08767569878681002e-09, -1.20734505911325997e-25},
        gdd_real{1.60590438368216133e-10, 1.25852945887520981e-26},
        gdd_real{1.14707455977297245e-11, 2.06555127528307454e-28},
        gdd_real{7.64716373181981641e-13, 7.03872877733453001e-30},
        gdd_real{4.77947733238738525e-14, 4.39920548583408126e-31},
        gdd_real{2.81145725434552060e-15, 1.65088427308614326e-31}
    };

static constexpr gdd_four d_dd_sin_table{
        gdd_real{1.950903220161282758e-01, -7.991079068461731263e-18},
        gdd_real{3.826834323650897818e-01, -1.005077269646158761e-17},
        gdd_real{5.555702330196021776e-01, 4.709410940561676821e-17},
        gdd_real{7.071067811865475727e-01, -4.833646656726456726e-17}
    };

static constexpr gdd_four d_dd_cos_table{
        gdd_real{9.807852804032304306e-01, 1.854693999782500573e-17},
        gdd_real{9.238795325112867385e-01, 1.764504708433667706e-17},
        gdd_real{8.314696123025452357e-01, 1.407385698472802389e-18},
        gdd_real{7.071067811865475727e-01, -4.833646656726456726e-17}
    };
/**
 * arithmetic operators
 * comparison
 */
///////////////////// Addition /////////////////////

inline CELER_FUNCTION
gdd_real negative(const gdd_real &a) {
    return gdd_real{-a.x, -a.y};
}

/* double-double = double + double */
inline CELER_FUNCTION
gdd_real dd_add(double a, double b) {
    double s, e;
    s = gdd::two_sum(a, b, e);
    return gdd_real{s, e};
}

/* double-double + double */
inline CELER_FUNCTION
gdd_real operator+(const gdd_real &a, double b) {
    double s1, s2;
    s1 = gdd::two_sum(a.x, b, s2);
    s2 += a.y;
    s1 = gdd::quick_two_sum(s1, s2, s2);
    return gdd_real{s1, s2};
}

/* double + double-double */
inline CELER_FUNCTION
gdd_real operator+(const double &a, gdd_real b) {
    return b + a;
}

inline CELER_FUNCTION
gdd_real sloppy_add(const gdd_real &a, const gdd_real &b) {
    double s, e;

    s = gdd::two_sum(a.x, b.x, e);
    e += (a.y + b.y);
    s = gdd::quick_two_sum(s, e, e);
    return gdd_real{s, e};
}

inline CELER_FUNCTION
gdd_real operator+(const gdd_real &a, const gdd_real &b) {
    return sloppy_add(a, b);
}

/*********** Subtractions *********/

inline CELER_FUNCTION
gdd_real operator-(const gdd_real &a, const gdd_real &b) {

    double s, e;
    s = gdd::two_diff(a.x, b.x, e);
    //return gdd_real(s, e);
    e += a.y;
    e -= b.y;
    s = gdd::quick_two_sum(s, e, e);
    return gdd_real{s, e};

    /*
      double s1, s2, t1, t2;
      s1 = two_diff(a.x, b.x, s2);
      t1 = two_diff(a.y, b.y, t2);
      s2 += t1;
      s1 = quick_two_sum(s1, s2, s2);
      s2 += t2;
      s1 = quick_two_sum(s1, s2, s2);
      return gdd_real(s1, s2);
     */
}

/* double-double - double */
inline CELER_FUNCTION
gdd_real operator-(const gdd_real &a, double b) {
    double s1, s2;
    s1 = gdd::two_diff(a.x, b, s2);
    s2 += a.y;
    s1 = gdd::quick_two_sum(s1, s2, s2);
    return gdd_real{s1, s2};
}

/* double - double-double */
inline CELER_FUNCTION
gdd_real operator-(double a, const gdd_real &b) {
    double s1, s2;
    s1 = gdd::two_diff(a, b.x, s2);
    s2 -= b.y;
    s1 = gdd::quick_two_sum(s1, s2, s2);
    return gdd_real{s1, s2};
}

/* negative */
inline CELER_FUNCTION
gdd_real operator-(const gdd_real &right) {
    return negative(right);
}

/*********** Squaring **********/
inline CELER_FUNCTION
gdd_real sqr(const gdd_real &a) {
    double p1, p2;
    double s1, s2;
    p1 = gdd::two_sqr(a.x, p2);
    // p2 += (2.0 * a.x * a.y);
    p2 = gdd::dadd_rn(p2, gdd::dmul_rn(gdd::dmul_rn(2.0, a.x), a.y));
    // p2 += (a.y * a.y);
    p2 = gdd::dadd_rn(p2, gdd::dmul_rn(a.y, a.y));
    s1 = gdd::quick_two_sum(p1, p2, s2);
    return gdd_real{s1, s2};
}

inline CELER_FUNCTION
gdd_real sqr(double a) {
    double p1, p2;
    p1 = gdd::two_sqr(a, p2);
    return gdd_real{p1, p2};
}

/****************** Multiplication ********************/

// /* double-double * (2.0 ^ exp) */ // Currently unused; uses math_functions.h
// inline CELER_FUNCTION
// gdd_real ldexp(const gdd_real &a, int exp) {
//     return gdd_real(ldexp(a.x, exp), ldexp(a.y, exp));
// }

/* double-double * double,  where double is a power of 2. */
inline constexpr CELER_FUNCTION
gdd_real mul_pwr2(const gdd_real &a, double b) {
    return gdd_real{a.x * b, a.y * b};
}

/* double-double * double-double */
inline constexpr CELER_FUNCTION
gdd_real operator*(const gdd_real &a, const gdd_real &b) {
    double p2 = 0;

    double p1 = gdd::two_prod(a.x, b.x, p2);
    p2 += (a.x * b.y + a.y * b.x);
    //p2 += __dadd_rn(__dmul_rn(a.x, b.y), __dmul_rn(a.y, b.x));

    p1 = gdd::quick_two_sum(p1, p2, p2);

    return gdd_real{p1, p2};
}

/* double-double * double */
inline constexpr CELER_FUNCTION
gdd_real operator*(const gdd_real &a, double b) {
    double p2 = 0;

    double p1 = gdd::two_prod(a.x, b, p2);
    p2 = gdd::dadd_rn(p2, (gdd::dmul_rn(a.y, b)));
    // p2 = p2 + (a.y * b);
    p1 = gdd::quick_two_sum(p1, p2, p2);
    return gdd_real{p1, p2};
}

/* double * double-double */
inline CELER_FUNCTION
gdd_real operator*(double a, const gdd_real &b) {
    return (b * a);
}

/******************* Division *********************/

inline constexpr CELER_FUNCTION
gdd_real sloppy_div(const gdd_real &a, const gdd_real &b) {
    double s2 = 0;

    double q1 = a.x / b.x; /* approximate quotient */

    /* compute  this - q1 * dd */
    gdd_real r = b * q1;
    double s1 = gdd::two_diff(a.x, r.x, s2);
    s2 -= r.y;
    s2 += a.y;

    /* get next approximation */
    double q2 = (s1 + s2) / b.x;

    /* renormalize */
    r.x = gdd::quick_two_sum(q1, q2, r.y);
    return r;
}

/* double-double / double-double */
inline constexpr CELER_FUNCTION
gdd_real operator/(const gdd_real &a, const gdd_real &b) {
    return sloppy_div(a, b);
}

/* double-double / double */
inline CELER_FUNCTION
gdd_real operator/(const gdd_real &a, double b) {

    double q1, q2;
    double p1, p2;
    double s, e;
    gdd_real r;

    q1 = a.x / b; /* approximate quotient. */

    /* Compute  this - q1 * d */
    p1 = gdd::two_prod(q1, b, p2);
    s = gdd::two_diff(a.x, p1, e);
    e = e + a.y;
    e = e - p2;

    /* get next approximation. */
    q2 = (s + e) / b;

    /* renormalize */
    r.x = gdd::quick_two_sum(q1, q2, r.y);

    return r;
}

inline CELER_FUNCTION
bool is_zero(const gdd_real &a) {
    return (a.x == 0.0);
}

inline CELER_FUNCTION
bool is_one(const gdd_real &a) {
    return (a.x == 1.0 && a.y == 0.0);
}

/*  this > 0 */
inline CELER_FUNCTION
bool is_positive(const gdd_real &a) {
    return (a.x > 0.0);
}

/* this < 0 */
inline CELER_FUNCTION
bool is_negative(const gdd_real &a) {
    return (a.x < 0.0);
}

/* Cast to double. */
inline CELER_FUNCTION
double to_double(const gdd_real &a) {
    return a.x;
}

/************* Comparison ***************/

/* double-double <= double-double */
inline CELER_FUNCTION
bool operator<=(const gdd_real &a, const gdd_real &b) {
    return (a.x < b.x || (a.x == b.x && a.y <= b.y));
}



/*********** Equality Comparisons ************/

/* double-double == double */
inline CELER_FUNCTION bool operator==(const gdd_real &a, double b) {
    return (a.x == b && a.y == 0.0);
}

/* double-double == double-double */
inline CELER_FUNCTION bool operator==(const gdd_real &a, const gdd_real &b) {
    return (a.x == b.x && a.y == b.y);
}

/* double == double-double */
inline CELER_FUNCTION bool operator==(double a, const gdd_real &b) {
    return (a == b.x && b.y == 0.0);
}

/*********** Greater-Than Comparisons ************/

/* double-double > double */
inline CELER_FUNCTION bool operator>(const gdd_real &a, double b) {
    return (a.x > b || (a.x == b && a.y > 0.0));
}

/* double-double > double-double */
inline CELER_FUNCTION bool operator>(const gdd_real &a, const gdd_real &b) {
    return (a.x > b.x || (a.x == b.x && a.y > b.y));
}

/* double > double-double */
inline CELER_FUNCTION bool operator>(double a, const gdd_real &b) {
    return (a > b.x || (a == b.x && b.y < 0.0));
}

/*********** Less-Than Comparisons ************/

/* double-double < double */
inline CELER_FUNCTION bool operator<(const gdd_real &a, double b) {
    return (a.x < b || (a.x == b && a.y < 0.0));
}

/* double-double < double-double */
inline CELER_FUNCTION bool operator<(const gdd_real &a, const gdd_real &b) {
    return (a.x < b.x || (a.x == b.x && a.y < b.y));
}

/* double < double-double */
inline CELER_FUNCTION bool operator<(double a, const gdd_real &b) {
    return (a < b.x || (a == b.x && b.y > 0.0));
}

/*********** Greater-Than-Or-Equal-To Comparisons ************/

/* double-double >= double */
inline CELER_FUNCTION bool operator>=(const gdd_real &a, double b) {
    return (a.x > b || (a.x == b && a.y >= 0.0));
}

/* double-double >= double-double */
inline CELER_FUNCTION bool operator>=(const gdd_real &a, const gdd_real &b) {
    return (a.x > b.x || (a.x == b.x && a.y >= b.y));
}

/* double >= double-double */
//inline CELER_FUNCTION bool operator>=(double a, const gdd_real &b) {
//  return (b <= a);
//}

/*********** Less-Than-Or-Equal-To Comparisons ************/

/* double-double <= double */
inline CELER_FUNCTION bool operator<=(const gdd_real &a, double b) {
    return (a.x < b || (a.x == b && a.y <= 0.0));
}

/* double >= double-double */
inline CELER_FUNCTION bool operator>=(double a, const gdd_real &b) {
    return (b <= a);
}


/* double-double <= double-double */
//inline CELER_FUNCTION bool operator<=(const gdd_real &a, const gdd_real &b) {
//  return (a.x[0] < b.x[0] || (a.x[0] == b.x[0] && a.x[1] <= b.x[1]));
//}

/* double <= double-double */
inline CELER_FUNCTION bool operator<=(double a, const gdd_real &b) {
    return (b >= a);
}

/*********** Not-Equal-To Comparisons ************/

/* double-double != double */
inline CELER_FUNCTION bool operator!=(const gdd_real &a, double b) {
    return (a.x != b || a.y != 0.0);
}

/* double-double != double-double */
inline CELER_FUNCTION bool operator!=(const gdd_real &a, const gdd_real &b) {
    return (a.x != b.x || a.y != b.y);
}

/* double != double-double */
inline CELER_FUNCTION bool operator!=(double a, const gdd_real &b) {
    return (a != b.x || b.y != 0.0);
}

/************* Miscellaneous ***********/

inline CELER_FUNCTION
gdd_real nint(const gdd_real &a) {
    double hi = gdd::nint(a.x);
    double lo;

    if (hi == a.x) {
        /* High word is an integer already.  Round the low word.*/
        lo = gdd::nint(a.y);

        /* Renormalize. This is needed if x[0] = some integer, x[1] = 1/2.*/
        hi = gdd::quick_two_sum(hi, lo, lo);
    } else {
        /* High word is not an integer. */
        lo = 0.0;
        if (std::fabs(hi - a.x) == 0.5 && a.y < 0.0) {
            /* There is a tie in the high word, consult the low word 
               to break the tie. */
            hi -= 1.0; /* NOTE: This does not cause INEXACT. */
        }
    }

    return make_gdd(hi, lo);
}

inline CELER_FUNCTION
gdd_real abs(const gdd_real &a) {
    return (a.x < 0.0) ? negative(a) : a;
}

inline CELER_FUNCTION
gdd_real fabs(const gdd_real &a) {
    return abs(a);
}

/* double / double-double */
inline CELER_FUNCTION
gdd_real operator/(double a, const gdd_real &b) {
    return make_gdd(a) / b;
}

// inline CELER_FUNCTION
// gdd_real inv(const gdd_real &a) {
//     return 1.0 / a;
// }
/****** Sqrt, cbrt ****/

namespace gdd {
/* Computes the square root of the double-double number a.
   NOTE: dd must be a non-negative number.                   */
inline CELER_FUNCTION
gdd_real sqrt(const gdd_real &a) {
    if (is_zero(a))
        return make_gdd(0.0);

    //TODO: should make an error
    if (is_negative(a)) {
        //return _nan;
        return make_gdd(0.0);
    }

    double x = 1.0 / std::sqrt(a.x);
    double ax = a.x * x;

    return dd_add(ax, (a - sqr(ax)).x * (x * 0.5));
    //return a - sqr(ax);
}
/** Computes the cube root of the double-double number a.
 * NOTE: This is VERY much a placeholder until we have an actual algo
 */
inline CELER_FUNCTION gdd_real cbrt(const gdd_real &a) {
    return make_gdd(std::cbrt(a.x));
}
}

/******** Trig functions */

namespace gdd {

/* Computes sin(a) using Taylor series.
   Assumes |a| <= pi/32.                           */
inline CELER_FUNCTION
gdd_real sin_taylor(const gdd_real &a) {
    const double thresh = 0.5 * std::fabs(to_double(a)) * _dd_eps;
    gdd_real r, s, t, x;

    if (is_zero(a)) {
        return make_gdd(0.0);
    }

    int i = 0;
    x = negative(sqr(a)); //-sqr(a);
    s = a;
    r = a;
    do {
        r = r*x;
        t = r * dd_inv_fact[i];
        s = s + t;
        i += 2;
    } while (i < n_dd_inv_fact && std::fabs(to_double(t)) > thresh);

    return s;
}

inline CELER_FUNCTION
gdd_real cos_taylor(const gdd_real &a) {
    const double thresh = 0.5 * _dd_eps;
    gdd_real r, s, t, x;
    int i = 1;

    if (is_zero(a)) {
        return make_gdd(1.0);
    }

    x = negative(sqr(a));
    r = x;
    s = 1.0 + mul_pwr2(r, 0.5);
    do {
        r = r*x;
        t = r * dd_inv_fact[i];
        s = s + t;
        i += 2;
    } while (i < n_dd_inv_fact && std::fabs(to_double(t)) > thresh);

    return s;
}

inline CELER_FUNCTION
void sincos_taylor(const gdd_real &a, gdd_real &sin_a, gdd_real &cos_a) {
    if (is_zero(a)) {
        sin_a.x = 0.0;
        sin_a.y = 0.0;
        cos_a.x = 1.0;
        cos_a.y = 0.0;
        return;
    }

    sin_a = sin_taylor(a);
    cos_a = sqrt(1.0 - sqr(sin_a));
}

// inline CELER_FUNCTION
// gdd_real sin(const gdd_real &a) {

//     if (is_zero(a)) {
//         return make_gdd(0.0);
//     }

//     // approximately reduce modulo 2*pi
//     gdd_real z = nint(a / _dd_2pi);
//     gdd_real r = a - _dd_2pi * z;

//     // approximately reduce modulo pi/2 and then modulo pi/16.
//     gdd_real t;
//     double q = floor(r.x / _dd_pi2.x + 0.5);
//     t = r - _dd_pi2 * q;
//     int j = (int) (q);
//     q = floor(t.x / _dd_pi16.x + 0.5);
//     t = t - _dd_pi16 * q;
//     int k = (int) (q);
//     int abs_k = std::abs(k);

//     if (j < -2 || j > 2) {
//         //dd_real::error("(dd_real::sin): Cannot reduce modulo pi/2.");
//         r.x = r.y = 0.0;
//         return r;
//     }

//     if (abs_k > 4) {
//         //dd_real::error("(dd_real::sin): Cannot reduce modulo pi/16.");
//         r.x = r.y = 0.0;
//         return r;
//     }

//     if (k == 0) {
//         switch (j) {
//             case 0:
//                 return sin_taylor(t);
//             case 1:
//                 return cos_taylor(t);
//             case -1:
//                 return negative(cos_taylor(t));
//             default:
//                 return negative(sin_taylor(t));
//         }
//     }

//     gdd_real u = d_dd_cos_table[abs_k - 1];
//     gdd_real v = d_dd_sin_table[abs_k - 1];
//     gdd_real sin_t, cos_t;
//     sincos_taylor(t, sin_t, cos_t);
//     if (j == 0) {
//         if (k > 0) {
//             r = u * sin_t + v * cos_t;
//         } else {
//             r = u * sin_t - v * cos_t;
//         }
//     } else if (j == 1) {
//         if (k > 0) {
//             r = u * cos_t - v * sin_t;
//         } else {
//             r = u * cos_t + v * sin_t;
//         }
//     } else if (j == -1) {
//         if (k > 0) {
//             r = v * sin_t - u * cos_t;
//         } else if (k < 0) {
//             //r = -u * cos_t - v * sin_t;
//             r = negative(u * cos_t) - v * sin_t;
//         }
//     } else {
//         if (k > 0) {
//             //r = -u * sin_t - v * cos_t;
//             r = negative(u * sin_t) - v * cos_t;
//         } else {
//             r = v * cos_t - u * sin_t;
//         }
//     }

//     return r;
// }

inline CELER_FUNCTION
gdd_real cos(const gdd_real &a) {

    if (is_zero(a)) {
        return make_gdd(1.0);
    }

    // approximately reduce modulo 2*pi
    gdd_real z = nint(a / _dd_2pi);
    gdd_real r = a - z * _dd_2pi;

    // approximately reduce modulo pi/2 and then modulo pi/16
    gdd_real t;
    double q = floor(r.x / _dd_pi2.x + 0.5);
    t = r - _dd_pi2 * q;
    int j = (int) (q);
    q = floor(t.x / _dd_pi16.x + 0.5);
    t = t - _dd_pi16 * q;
    int k = (int) (q);
    int abs_k = std::abs(k);

    if (j < -2 || j > 2) {
        //dd_real::error("(dd_real::cos): Cannot reduce modulo pi/2.");
        //return dd_real::_nan;
        return make_gdd(0.0);
    }

    if (abs_k > 4) {
        //dd_real::error("(dd_real::cos): Cannot reduce modulo pi/16.");
        //return dd_real::_nan;
        return make_gdd(0.0);
    }

    if (k == 0) {
        switch (j) {
            case 0:
                return cos_taylor(t);
            case 1:
                return negative(sin_taylor(t));
            case -1:
                return sin_taylor(t);
            default:
                return negative(cos_taylor(t));
        }
    }

    gdd_real sin_t, cos_t;
    sincos_taylor(t, sin_t, cos_t);
    gdd_real u = d_dd_cos_table[abs_k - 1];
    gdd_real v = d_dd_sin_table[abs_k - 1];

    if (j == 0) {
        if (k > 0) {
            r = u * cos_t - v * sin_t;
        } else {
            r = u * cos_t + v * sin_t;
        }
    } else if (j == 1) {
        if (k > 0) {
            r = negative(u * sin_t) - v * cos_t;
        } else {
            r = v * cos_t - u * sin_t;
        }
    } else if (j == -1) {
        if (k > 0) {
            r = u * sin_t + v * cos_t;
        } else {
            r = u * sin_t - v * cos_t;
        }
    } else {
        if (k > 0) {
            r = v * sin_t - u * cos_t;
        } else {
            r = negative(u * cos_t) - v * sin_t;
        }
    }

    return r;
}

inline CELER_FUNCTION
void sincos(const gdd_real &a, gdd_real &sin_a, gdd_real &cos_a) {

    if (is_zero(a)) {
        sin_a = make_gdd(0.0);
        cos_a = make_gdd(1.0);
        return;
    }

    // approximately reduce modulo 2*pi
    gdd_real z = nint(a / _dd_2pi);
    gdd_real r = a - _dd_2pi * z;

    // approximately reduce module pi/2 and pi/16
    gdd_real t;
    double q = floor(r.x / _dd_pi2.x + 0.5);
    t = r - _dd_pi2 * q;
    int j = (int) (q);
    int abs_j = std::abs(j);
    q = floor(t.x / _dd_pi16.x + 0.5);
    t = t - _dd_pi16 * q;
    int k = (int) (q);
    int abs_k = std::abs(k);

    if (abs_j > 2) {
        //dd_real::error("(dd_real::sincos): Cannot reduce modulo pi/2.");
        //cos_a = sin_a = dd_real::_nan;
        cos_a = sin_a = make_gdd(0.0);
        return;
    }

    if (abs_k > 4) {
        //dd_real::error("(dd_real::sincos): Cannot reduce modulo pi/16.");
        //cos_a = sin_a = dd_real::_nan;
        cos_a = sin_a = make_gdd(0.0);
        return;
    }

    gdd_real sin_t, cos_t;
    gdd_real s, c;

    sincos_taylor(t, sin_t, cos_t);

    if (abs_k == 0) {
        s = sin_t;
        c = cos_t;
    } else {
        gdd_real u = d_dd_cos_table[abs_k - 1];
        gdd_real v = d_dd_sin_table[abs_k - 1];

        if (k > 0) {
            s = u * sin_t + v * cos_t;
            c = u * cos_t - v * sin_t;
        } else {
            s = u * sin_t - v * cos_t;
            c = u * cos_t + v * sin_t;
        }
    }

    if (abs_j == 0) {
        sin_a = s;
        cos_a = c;
    } else if (j == 1) {
        sin_a = c;
        cos_a = negative(s);
    } else if (j == -1) {
        sin_a = negative(c);
        cos_a = s;
    } else {
        sin_a = negative(s);
        cos_a = negative(c);
    }
}

// inline CELER_FUNCTION
// gdd_real tan(const gdd_real &a) {
//     gdd_real s, c;
//     sincos(a, s, c);
//     return s / c;
// }

inline CELER_FUNCTION
gdd_real atan2(const gdd_real &y, const gdd_real &x) {

    if (is_zero(x)) {

        if (is_zero(y)) {
            /* Both x and y is zero. */
            //dd_real::error("(dd_real::atan2): Both arguments zero.");
            //return dd_real::_nan;
            return make_gdd(0.0);
        }

        return (is_positive(y)) ? _dd_pi2 : negative(_dd_pi2);
    } else if (is_zero(y)) {
        return (is_positive(x)) ? make_gdd(0.0) : _dd_pi;
    }

    if (x == y) {
        return (is_positive(y)) ? _dd_pi4 : negative(_dd_3pi4);
    }

    if (x == negative(y)) {
        return (is_positive(y)) ? _dd_3pi4 : negative(_dd_pi4);
    }

    gdd_real r = sqrt(sqr(x) + sqr(y));
    gdd_real xx = x / r;
    gdd_real yy = y / r;

    /* Compute double precision approximation to atan. */
    gdd_real z = make_gdd(std::atan2(to_double(y), to_double(x)));
    gdd_real sin_z, cos_z;

    if (std::abs(xx.x) > std::abs(yy.x)) {
        /* Use Newton iteration 1.  z' = z + (y - sin(z)) / cos(z)  */
        sincos(z, sin_z, cos_z);
        z = z + (yy - sin_z) / cos_z;
    } else {
        /* Use Newton iteration 2.  z' = z - (x - cos(z)) / sin(z)  */
        sincos(z, sin_z, cos_z);
        z = z - (xx - cos_z) / sin_z;
    }

    return z;
}

// inline CELER_FUNCTION
// gdd_real atan(const gdd_real &a) {
//     return atan2(a, make_gdd(1.0));
// }

// inline CELER_FUNCTION
// gdd_real asin(const gdd_real &a) {
//     gdd_real abs_a = abs(a);

//     if (abs_a > 1.0) {
//         //dd_real::error("(dd_real::asin): Argument out of domain.");
//         //return dd_real::_nan;
//         return make_gdd(0.0);
//     }

//     if (is_one(abs_a)) {
//         return (is_positive(a)) ? _dd_pi2 : negative(_dd_pi2);
//     }

//     return atan2(a, sqrt(1.0 - sqr(a)));
// }

inline CELER_FUNCTION
gdd_real acos(const gdd_real &a) {
    gdd_real abs_a = abs(a);

    if (abs_a > 1.0) {
        //dd_real::error("(dd_real::acos): Argument out of domain.");
        //return dd_real::_nan;
        return make_gdd(0.0);
    }

    if (is_one(abs_a)) {
        return (is_positive(a)) ? make_gdd(0.0) : _dd_pi;
    }

    return atan2(sqrt(1.0 - sqr(a)), a);
}

}

}
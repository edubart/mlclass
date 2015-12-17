#ifndef OPTIMIZERS_HPP
#define OPTIMIZERS_HPP

#include "prereqs.hpp"

#include <dlib/optimization.h>

namespace mlight {

typedef dlib::matrix<double,0,1> dlibvec;

template<typename T,long NR, long NC>
const arma::mat from_dlibmat(const dlib::matrix<T,NR,NC>& in) {
    return arma::mat((double*)&in(0,0), (arma::uword)in.nr(), (arma::uword)in.nc(), false, false);
}

template<typename T,long NR, long NC>
const arma::mat from_dlibmat(const dlib::matrix<T,NR,NC>& in, uint nr, uint nc) {
    //return arma::reshape(arma::mat((double*)&in(0,0), (arma::uword)in.nr(), (arma::uword)in.nc(), false, false), nr, nc);
    return arma::mat((double*)&in(0,0), (arma::uword)nr, (arma::uword)nc, false, false);
}

/*
class OptimizerCG {
public:
    OptimizerCG(double tolerance = 1e-7, ulong maxIterations = 0) :
        m_maxIterations(maxIterations), m_tolerance(tolerance) { }

    template<typename OptimizerFunctionType>
    void optimize(OptimizerFunctionType&& function, arma::mat& params) {
        dlibvec starting_point(dlib::mat(params));
        dlib::find_min(dlib::cg_search_strategy(),
                       dlib::gradient_norm_stop_strategy(m_tolerance, m_maxIterations),
        [&](const dlibvec& in) {
            return function.evaluate(from_dlibmat(in));
        },
        [&](const dlibvec& in) {
            return dlib::mat(function.gradient(from_dlibmat(in)));
        }, starting_point, -1);
        params = from_dlibmat(starting_point);
    }

private:
    ulong m_maxIterations;
    double m_tolerance;
};
*/

class OptimizerCG {
public:
    OptimizerCG(double tolerance = 1e-7, ulong maxIterations = 0) :
        m_maxIterations(maxIterations), m_tolerance(tolerance) { }

    template<typename OptimizerFunctionType>
    void optimize(OptimizerFunctionType&& function, arma::mat& params) {
        uint nr = params.n_rows;
        uint nc = params.n_cols;
        dlibvec starting_point(dlib::mat(arma::vectorise(params).eval()));
        dlib::find_min(dlib::cg_search_strategy(),
                       dlib::gradient_norm_stop_strategy(m_tolerance, m_maxIterations),
        [&](const dlibvec& in) {
            double d = function.evaluate(from_dlibmat(in, nr, nc));
            assert(std::isfinite(d));
            return d;
        },
        [&](const dlibvec& in) {
            return dlib::mat(arma::vectorise(function.gradient(from_dlibmat(in, nr, nc))).eval());
        }, starting_point, -1);
        params = from_dlibmat(starting_point, nr, nc);
    }

private:
    ulong m_maxIterations;
    double m_tolerance;
};

class OptimizerLBFGS {
public:
   OptimizerLBFGS(ulong maxSize = 10, double tolerance = 1e-7, ulong maxIterations = 0) :
        m_maxSize(maxSize), m_maxIterations(maxIterations), m_tolerance(tolerance) { }

    template<typename OptimizerFunctionType>
    void optimize(OptimizerFunctionType&& function, arma::mat& params) {
        uint nr = params.n_rows;
        uint nc = params.n_cols;
        dlibvec starting_point(dlib::mat(arma::vectorise(params).eval()));
        dlib::find_min(dlib::lbfgs_search_strategy(m_maxSize),
                       dlib::gradient_norm_stop_strategy(m_tolerance, m_maxIterations),
        [&](const dlibvec& in) {
            double d = function.evaluate(from_dlibmat(in, nr, nc));
            assert(std::isfinite(d));
            return d;
        },
        [&](const dlibvec& in) {
            return dlib::mat(arma::vectorise(function.gradient(from_dlibmat(in, nr, nc))).eval());
        }, starting_point, -1);
        params = from_dlibmat(starting_point, nr, nc);
    }

private:
    ulong m_maxSize;
    ulong m_maxIterations;
    double m_tolerance;
};


class OptimizerGD {
public:
    OptimizerGD(double alpha = 0.03, double tolerance = 1e-5, ulong maxIterations = 0) :
        m_maxIterations(maxIterations), m_alpha(alpha), m_tolerance(tolerance) { }

    template<typename OptimizerFunctionType>
    void optimize(OptimizerFunctionType&& function, arma::mat& params) {
        for(ulong it=1; it <= m_maxIterations || m_maxIterations == 0; ++it) {
            arma::mat grad = function.gradient(params);
            params = params - m_alpha*grad;
            if(m_tolerance > 0 && arma::norm(grad) <= m_tolerance)
                break;
        }
    }

private:
    ulong m_maxIterations;
    double m_alpha;
    double m_tolerance;
};

class OptimizerGDCheck {
public:
    OptimizerGDCheck(double alpha = 0.03, double tolerance = 1e-5, ulong maxIterations = 0) :
        m_maxIterations(maxIterations), m_alpha(alpha), m_tolerance(tolerance) { }

    template<typename OptimizerFunctionType>
    void optimize(OptimizerFunctionType&& function, arma::mat& params,
                                   double eps = 1e-4, double maxApproxError = 1e-7) {
        for(ulong it=1; it <= m_maxIterations || m_maxIterations == 0; ++it) {
            arma::mat grad = function.gradient(params);
            arma::mat gradApprox = grad;

            for(uint i=0;i<params.n_rows;++i) {
                for(uint j=0;j<params.n_cols;++j) {
                    arma::mat tmp = params;
                    tmp(i,j) = params(i,j) - eps;
                    double f1 = function.evaluate(tmp);
                    tmp(i,j) = params(i,j) + eps;
                    double f2 = function.evaluate(tmp);
                    gradApprox(i,j) = (f2 - f1)/(2.0*eps);
                }
            }

            double gradDiff = arma::norm(arma::vectorise(gradApprox - grad))/arma::norm(arma::vectorise(gradApprox + grad));
            if(gradDiff > maxApproxError || !std::isnormal(gradDiff)) {
                dump << it << "gradient error" << std::scientific << gradDiff;
                dump << "grad: " << grad;
                dump << "approx grad: " << gradApprox;
            }

            params = params - m_alpha*grad;
            if(m_tolerance > 0 && arma::norm(grad) <= m_tolerance)
                break;
        }
    }

private:
    ulong m_maxIterations;
    double m_alpha;
    double m_tolerance;
};

}

#endif

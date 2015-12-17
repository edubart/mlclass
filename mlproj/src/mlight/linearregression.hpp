#ifndef LINEARREGRESSION_HPP
#define LINEARREGRESSION_HPP

#include "prereqs.hpp"

namespace mlight {

class LinearRegressionCostFunction {
public:
    LinearRegressionCostFunction(const arma::mat& X, const arma::mat& y, double lambda) : X(X), y(y), lambda(lambda) {
        double m = X.n_rows;
        ke = 1.0/(2.0*m);
        kg = 1.0/m;
        reg.zeros(X.n_cols,y.n_cols);
        reg.fill(lambda/m);
        reg.row(0).fill(0.0);
    }
    double evaluate(const arma::mat& theta) const {
        return ke*(arma::accu(arma::square(X*theta - y)) + lambda*arma::accu(arma::square(theta.rows(1, theta.n_rows-1))));
    }
    arma::mat gradient(const arma::mat& theta) const {
        return kg*(X.t() * (X*theta - y)) + theta % reg;
    }
private:
    const arma::mat& X;
    const arma::mat& y;
    arma::mat reg;
    double lambda;
    double ke;
    double kg;
};

class LinearRegression
{
public:
    template<typename Optimizer>
    void fit(arma::mat X, const arma::mat& y, Optimizer optimizer, double lambda = 0) {
        m_lambda = lambda;
        prepareNormalization(X);
        prepareFeatures(X);
        m_theta.zeros(X.n_cols,y.n_cols);
        optimizer.optimize(LinearRegressionCostFunction(X,y,m_lambda), m_theta);
    }

    void fitNormal(arma::mat X, const arma::mat& y) {
        prepareNormalization(X);
        prepareFeatures(X);
        arma::mat reg = arma::eye(X.n_cols, X.n_cols);
        reg(0,0) = 0;
        m_theta = arma::pinv(X.t()*X + m_lambda * reg)*X.t()*y;
    }

    arma::mat predict(arma::mat X) {
        prepareFeatures(X);
        return X*m_theta;
    }

    double cost(arma::mat X, const arma::mat& y) {
        prepareFeatures(X);
        return LinearRegressionCostFunction(X,y,m_lambda).evaluate(m_theta);
    }

    double score(arma::mat X, const arma::mat& y) {
        arma::mat u = y - predict(std::move(X));
        arma::mat v = y.each_row() - arma::mean(y);
        return 1.0 - arma::accu(u.t()*u)/arma::accu(v.t()*v);
    }

    double score2(arma::mat X, const arma::mat& y, double tolerance = 0.5) {
        arma::mat u = arma::square(y - predict(std::move(X)));
        return arma::find(u <= tolerance*tolerance).eval().n_elem/(double)u.n_elem;
    }

private:
    void prepareNormalization(arma::mat& X) {
        m_mu = arma::mean(X);
        m_sigma = arma::stddev(X);
    }

    void prepareFeatures(arma::mat& X) {
        X = (X.each_row() - m_mu).each_row()/m_sigma;
        X = X.cols(arma::find(m_sigma > 0));
        X = arma::join_horiz(arma::ones<arma::vec>(X.n_rows,1), X);
    }

    arma::mat m_mu;
    arma::mat m_sigma;
    arma::vec m_theta;
    double m_lambda = 0;
};

}

#endif

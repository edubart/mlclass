#ifndef LOGISTICREGRESSION_HPP
#define LOGISTICREGRESSION_HPP

#include "prereqs.hpp"

namespace mlight {

class LogisticRegressionCostFunction {
public:
    LogisticRegressionCostFunction(const arma::mat& X, const arma::mat& y, double lambda) : X(X), y(y) {
        double m = X.n_rows;
        ke = lambda/(2.0*m);
        kg = 1.0/m;
        reg.zeros(X.n_cols,y.n_cols);
        reg.fill(lambda/m);
        reg.row(0).fill(0.0);
    }
    double evaluate(const arma::mat& theta) const {
        arma::mat h = sigmoid(X*theta);
        const double c = std::numeric_limits<double>::min();
        return -kg*arma::accu((arma::log(h+c).t()*y+arma::log((1.0-h).eval()+c).t()*(1.0-y))) +
                ke*arma::accu(arma::square(theta.rows(1, theta.n_rows-1)));
    }
    arma::mat gradient(const arma::mat& theta) const {
        return kg*(X.t() * (sigmoid(X*theta) - y)) + theta % reg;
    }
private:
    const arma::mat& X;
    const arma::mat& y;
    arma::mat reg;
    double ke;
    double kg;
};

class LogisticRegression
{
public:
    template<typename Optimizer>
    void fit(arma::mat X, const arma::mat& y, Optimizer optimizer, double lambda = 0) {
        m_lambda = lambda;
        prepareNormalization(X);
        prepareFeatures(X);
        m_theta.zeros(X.n_cols,y.n_cols);
        optimizer.optimize(LogisticRegressionCostFunction(X,y,m_lambda), m_theta);
    }

    arma::mat predict(arma::mat X) {
        prepareFeatures(X);
        return arma::round(sigmoid(X*m_theta));
    }

    arma::mat predictProbability(arma::mat X) {
        prepareFeatures(X);
        return sigmoid(X*m_theta);
    }

    double cost(arma::mat X, const arma::mat& y) {
        prepareFeatures(X);
        return LogisticRegressionCostFunction(X,y,m_lambda).evaluate(m_theta);
    }

    double score(arma::mat X, const arma::mat& y) {
        return arma::accu(predict(X) == y)/(double)y.n_elem;
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
    arma::mat m_theta;
    double m_lambda = 0;
};

}

#endif

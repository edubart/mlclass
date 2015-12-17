#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include "prereqs.hpp"

namespace mlight {

class NeuralNetwork
{
public:
    // alpha: learning rate
    // lambda: regularizer
    double fit(arma::mat X, const arma::mat& y, std::deque<int> layers, double alpha = 0.03, double lambda = 0.0, ulong maxIterations = 100000) {
        this->lambda = lambda;

        layers.push_front(X.n_cols);
        layers.push_back(y.n_cols);

        prepareTheta(layers);
        prepareNormalization(X);
        prepareFeatures(X);
        return gradientDescent(X,y,alpha,maxIterations);
    }

    arma::mat predict(arma::mat X) {
        prepareFeatures(X);
        arma::mat p = hypothesis(X);
        for(uint i=0;i<X.n_rows;++i) {
            arma::uword j = 0;
            double v = p.row(i).max(j);
            p.row(i).fill(0);
            if(v > 0)
                p(i,j) = 1;
        }
        return p;
    }

    arma::mat predictProbability(arma::mat X) {
        prepareFeatures(X);
        return hypothesis(X);
    }

    double score(arma::mat X, const arma::mat& y) {
        return arma::accu(predict(X) == y)/(double)y.n_elem;
    }

private:
    void prepareNormalization(arma::mat& X) {
        mu = arma::mean(X);
        sigma = arma::stddev(X);
        sigma.transform([](double& e) { return (e == 0) ? 1 : e;});
    }

    void prepareFeatures(arma::mat& X) {
        // normalize entire X
        X = (X.each_row() - mu).each_row()/sigma;
    }

    void prepareTheta(std::deque<int> layers) {
        theta.resize(layers.size()-1);
        for(uint i=0;i<layers.size()-1;++i) {
            int lout = layers[i+1];
            int lin = layers[i];
            double eps = std::sqrt(6.0)/std::sqrt(lout+lin);
            theta[i] = (arma::randu<arma::mat>(lout, lin+1) *  (2.0 * eps)) - eps;
        }
    }

    arma::mat sigmoid(const arma::mat& x) {
        return 1.0/(1.0 + arma::exp(-x));
    }

    arma::mat sigmoidGradient(const arma::mat& x) {
        arma::mat t = sigmoid(x);
        return t % (1-t);
    }

    arma::mat hypothesis(const arma::mat& X) {
        arma::mat a = X;
        for(uint i=0;i<theta.size();++i) {
            a = arma::join_horiz(arma::ones<arma::vec>(a.n_rows,1), a);
            a = sigmoid(a * theta[i].t());
        }
        return a;
    }

    double gradientDescent(const arma::mat& X, const arma::mat& y, double alpha, ulong maxIterations) {
        int L = theta.size();
        int m = X.n_rows;
        double k = alpha/m;
        double k_reg = 1.0 - (alpha*lambda)/m;
        std::vector<arma::mat> theta_reg(L);
        std::vector<arma::mat> a(L);
        std::vector<arma::mat> d(L);
        std::vector<arma::mat> D(L);
        for(int i=0;i<L;++i) {
            theta_reg[i].resize(theta[i].n_rows, theta[i].n_cols);
            theta_reg[i].fill(k_reg);
            theta_reg[i].col(0).fill(1.0);
        }
        for(int i=0;i<L;++i)
            a[i].ones(theta[i].n_cols,m);
        a[0].rows(1, a[0].n_rows-1) = X.t();
        for(ulong it=1;it<=maxIterations; ++it) {
            for(int i=1;i<L;++i)
                a[i].rows(1, a[i].n_rows-1) = sigmoid(theta[i-1] * a[i-1]);
            d[L-1] = sigmoid(theta[L-1] * a[L-1]) - y.t();
            d[L-2] = (theta[L-1].t() * d[L-1]) % sigmoidGradient(a[L-1]);
            for(int i=L-3;i>=0;--i)
                d[i] = (theta[i+1].t() * d[i+1].rows(1, d[i+1].n_rows-1)) % sigmoidGradient(a[i+1]);
            D[L-1] = d[L-1]*a[L-1].t() * k;
            for(int i=0;i<L-1;++i)
                D[i] = d[i].rows(1,d[i].n_rows-1)*a[i].t() * k;
            for(int i=0;i<L;++i)
                theta[i] = theta[i] % theta_reg[i] - D[i];
            //TODO: gradient checking
        }
        return computeCost(X, y);
    }

    //TODO: cost function
    double computeCost(const arma::mat& X, const arma::mat& y) {
        //arma::mat h = hypothesis(X);
        //arma::mat theta_reg = theta.rows(1, theta.n_rows-1);
        //return arma::accu((arma::log(h).t()*y+arma::log(1.0-h).t()*(1.0-y))*(-1.0/X.n_rows) + (theta_reg.t()*theta_reg)*(lambda/(2.0*X.n_rows)));
        return 0;
    }

    arma::mat mu;
    arma::mat sigma;
    double lambda;
    std::vector<arma::mat> theta;
};

}

#endif


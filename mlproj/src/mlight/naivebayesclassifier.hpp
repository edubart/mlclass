#ifndef NAIVEBAYESCLASSIFIER_HPP
#define NAIVEBAYESCLASSIFIER_HPP

#include "prereqs.hpp"

namespace mlight {

class NaiveBayesClassifier
{
public:
    void fit(arma::mat X, const arma::vec& y) {
        std::map<double, arma::mat> cX;
        for(double c : arma::unique(y).eval()) {
            cX[c] = X.rows(find(y == c));
            classes.insert(c);
        }

        for(double c : classes) {
            pXc_mu[c] = arma::mean(cX[c]).t();
            pXc_sigma[c] = arma::stddev(cX[c]).t();
            pXc_sigma[c].transform([](double& e) { return (e == 0) ? 1 : e;});
        }
    }

    double predict(const arma::mat& x) {
        double bestP = 0;
        double bestC = 0;
        for(double c : classes) {
            double p = arma::prod(computeProbability(x, pXc_mu[c], pXc_sigma[c]));
            if(p > bestP) {
                bestP = p;
                bestC = c;
            }
        }
        return bestC;
    }

    double score(const arma::mat& X, const arma::vec& y) {
        uint valid = 0;
        for(uint i=0;i<X.n_rows;++i) {
            if(predict(X.row(i).t()) == y(i))
                valid++;
        }
        double accuracy = valid/(double)X.n_rows;
        return accuracy;
    }

private:
    arma::vec computeProbability(const arma::vec& x, const arma::vec& mu, const arma::vec& sigma) {
        static double K = sqrt(2.0*arma::datum::pi);
        return arma::exp(-arma::square(x-mu) / (2.0*arma::square(sigma)))/(sigma * K);
    }

    std::set<double> classes;
    std::map<double, arma::vec> pXc_mu;
    std::map<double, arma::vec> pXc_sigma;
};

}

#endif

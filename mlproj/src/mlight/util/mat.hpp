#ifndef MAT_HPP
#define MAT_HPP

#include <cmath>
#include <armadillo>

namespace mlight {

inline double binom(int n, int m)
{
    double cnm = 1.0;
    if(m * 2 > n)
        m = n - m;
    for(int i = 1; i <= m; n--, i++) {
        cnm /= i;
        cnm *= n;
    }
    return cnm;
}

inline arma::mat sigmoid(const arma::mat& x) {
    return 1.0/(1.0 + arma::exp(-x));
}

inline arma::mat map_features(arma::mat features, int degree)
{
    features = arma::join_horiz(arma::ones<arma::vec>(features.n_rows,1), features);
    arma::mat out;
    int numFeatures = (int)binom(features.n_cols+degree-1,degree) - 1;
    out.zeros(features.n_rows, numFeatures);

    std::vector<int> p(degree, 0);
    p.back() = 1;
    int c = 0;
    while(true) {
        out.col(c) = features.col(p[0]);
        for(int j = 1;j<degree;++j)
            out.col(c) %= features.col(p[j]);
        c++;
        int i = degree-1;
        while(p[i] == (int)features.n_cols-1)
            i--;
        if(i < 0)
            break;
        p[i]++;
        for(int j = i+1;j<degree;++j)
            p[j] = p[i];
    }
    assert(numFeatures == c);
    return out;
}

inline arma::mat map_labels(const arma::vec& labels)
{
    std::map<double,int> classes;
    int index = 0;
    for(double c : arma::unique(labels).eval()) {
        if(c == 0)
            continue;
         classes[c] = index++;
    }

    arma::mat out;
    out.zeros(labels.n_rows, classes.size());
    for(uint i=0;i<labels.n_rows;++i) {
        double c = labels(i);
        if(c == 0)
            continue;
        out(i,classes[c]) = 1;
    }

    return out;
}

template<typename... Args>
arma::mat load_mat(Args... args)
{
    arma::mat data;
    data.load(args...);
    return data;
}

inline void split_dataset(const arma::mat& dataset,
                               arma::mat& features, arma::mat& labels,
                               int numLabels = 1)
{
    features = dataset(0, 0, arma::size(dataset.n_rows, dataset.n_cols - numLabels));
    labels = dataset(0, dataset.n_cols - numLabels, arma::size(dataset.n_rows, numLabels));
}

inline void split_dataset(const arma::mat& dataset,
                               arma::mat& train_features, arma::mat& train_labels,
                               arma::mat& test_features, arma::mat& test_labels,
                               int numLabels = 1,
                               double train_ratio = 0.67)
{
    int trainRows = dataset.n_rows * train_ratio;
    int testRows = dataset.n_rows - trainRows;
    train_features = dataset(0, 0, arma::size(trainRows, dataset.n_cols - numLabels));
    train_labels = dataset(0, dataset.n_cols - numLabels, arma::size(trainRows, numLabels));
    test_features = dataset(trainRows, 0, arma::size(testRows, dataset.n_cols - numLabels));
    test_labels = dataset(trainRows, dataset.n_cols - numLabels, arma::size(testRows, numLabels));
}

inline void split_dataset(const arma::mat& features, const arma::mat& labels,
                               arma::mat& train_features, arma::mat& train_labels,
                               arma::mat& test_features, arma::mat& test_labels,
                               double train_ratio = 0.67)
{
    assert(features.n_rows == labels.n_rows);
    int trainRows = features.n_rows * train_ratio;
    int testRows = features.n_rows - trainRows;
    train_features = features(0, 0, arma::size(trainRows, features.n_cols));
    train_labels = labels(0, 0, arma::size(trainRows, labels.n_cols));
    test_features = features(trainRows, 0, arma::size(testRows, features.n_cols));
    test_labels = labels(trainRows, 0, arma::size(testRows, labels.n_cols));
}

}

#endif

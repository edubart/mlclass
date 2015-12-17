#include <mlight/neuralnetwork.hpp>
#include <boost/test/unit_test.hpp>
#include <mlight/naivebayesclassifier.hpp>
#include <mlight/logisticregression.hpp>
#include <mlight/linearregression.hpp>
#include <mlight/optimizers.hpp>
#include <mlight/neuralnetwork.hpp>

BOOST_AUTO_TEST_SUITE(TibiaRandom);

// fast random implementation
#define LEEF_RAND_MAX 65535
unsigned long fast_next_rand_seed;
inline void fast_srand(unsigned long seed) {
    fast_next_rand_seed = seed;
}
inline int fast_rand() {
    fast_next_rand_seed = fast_next_rand_seed * 1103515245 + 12345;
    return ((unsigned)(fast_next_rand_seed/65536) % (LEEF_RAND_MAX+1));
}

// gather train data
arma::mat gather_data()
{
    const int NUM_FEATURES = 30;
    const int NUM_CASES = 10000;
    const int TOTAL_CASES = NUM_CASES + NUM_FEATURES;
    arma::mat mat(NUM_CASES, NUM_FEATURES+1);
    fast_srand(0);
    std::vector<int> vec(TOTAL_CASES);
    for(int i=0;i<TOTAL_CASES;++i) {
        vec[i] = (fast_rand() % 6) + 1;
    }
    for(int i=0;i<NUM_CASES;++i) {
        for(int j=0;j<NUM_FEATURES;++j) {
            mat(i,j) = vec[i+j];
        }
        bool ok = (vec[i+NUM_FEATURES] == 6);
        mat(i,NUM_FEATURES) = ok ? 1 : 0;
    }
    return arma::shuffle(mat);
}

BOOST_AUTO_TEST_CASE(TibiaRandomTestCase)
{
    arma::arma_rng::set_seed(0); // fixed random
    arma::mat dataset = gather_data();
    arma::wall_clock timer;
    arma::mat train_features, train_labels, test_features, test_labels;
    arma::mat features, labels;
    mlight::split_dataset(dataset, features, labels);
    features = mlight::map_features(features, 2);
    mlight::split_dataset(features, labels,
                          train_features, train_labels, test_features, test_labels);

    //::NeuralNetwork clf;
    mlight::LogisticRegression clf;
    //mlight::NaiveBayesClassifier clf;

    timer.tic();
    //clf.fit(train_features, train_labels, { 30 },  0.3, 0.0, 1000);
    clf.fit(train_features, train_labels, mlight::OptimizerLBFGS(), 30);
    //clf.fit(train_features, train_labels);
    double score_train = clf.score(train_features, train_labels);
    double score_test = clf.score(test_features, test_labels);
    mlight::pformat("[tibiarandom] Classifier: (train time: %.2f ms, train score: %.6f, test score: %.6f)", timer.toc() * 1000, score_train, score_test);
}

BOOST_AUTO_TEST_SUITE_END();

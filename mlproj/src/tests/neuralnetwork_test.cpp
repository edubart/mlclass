#include <mlight/neuralnetwork.hpp>
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(NeuralNetwork);

//TODO: cross validation set
//TODO: plot learning curves, plot lambda curves, plot features curves
//TODO: optimizers
BOOST_AUTO_TEST_CASE(NeuralNetworkTestDigits)
{
    arma::arma_rng::set_seed(0); // fixed random
    arma::wall_clock timer;
    arma::mat features, labels;
    mlight::split_dataset(mlight::load_mat("ex3data1.txt"), features, labels);
    labels = mlight::map_labels(labels);

    mlight::NeuralNetwork clf;

    timer.tic();
    clf.fit(features, labels, { 400 },  0.1, 0.1, 50);
    double score = clf.score(features, labels);
    //mlight::dump << clf.predict(test_features);
    mlight::pformat("[digits] Neural network classifier: (train time: %.2f ms, score: %.6f)", timer.toc() * 1000, score);
}

BOOST_AUTO_TEST_SUITE_END();

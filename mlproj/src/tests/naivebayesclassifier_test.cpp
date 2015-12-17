#include <mlight/naivebayesclassifier.hpp>
#include <boost/test/unit_test.hpp>

using mlight::dump;

BOOST_AUTO_TEST_SUITE(NaiveBayesClassifier);

BOOST_AUTO_TEST_CASE(NaiveBayesClassifierTestCase)
{
    arma::wall_clock timer;
    arma::mat train_features, train_labels, test_features, test_labels;
    mlight::split_dataset(mlight::load_mat("pima-indians-diabetes.data"),
                          train_features, train_labels, test_features, test_labels);

    mlight::NaiveBayesClassifier clf;

    timer.tic();
    clf.fit(train_features, train_labels);
    double score = clf.score(test_features, test_labels);
    mlight::pformat("[pima] Naive bayes classifier: (train time: %.2f ms, score: %.6f)", timer.toc() * 1000, score);

    BOOST_REQUIRE_CLOSE(score, 0.767717, 1e-3);
}

BOOST_AUTO_TEST_SUITE_END();

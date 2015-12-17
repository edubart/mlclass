#include <mlight/logisticregression.hpp>
#include <mlight/optimizers.hpp>
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(LogisticRegressionTest);

BOOST_AUTO_TEST_CASE(LogisticRegressionTestCase)
{
    arma::wall_clock timer;
    arma::mat features, labels;
    mlight::split_dataset(mlight::load_mat("ex2data1.txt"), features, labels);

    mlight::LogisticRegression lr;

    timer.tic();
    lr.fit(features, labels, mlight::OptimizerLBFGS(), 0);
    double cost = lr.cost(features, labels);
    double score = lr.score(features, labels);
    mlight::pformat("[students] Logistic regression (gradient): (train time: %.2f ms, score: %.6f, cost: %.2e)", timer.toc() * 1000, score, cost);
    BOOST_REQUIRE_GE(score, 0.89);
}

BOOST_AUTO_TEST_CASE(LogisticRegressionTestReg)
{
    arma::wall_clock timer;
    arma::mat features, labels;
    mlight::split_dataset(mlight::load_mat("ex2data2.txt"), features, labels);

    features = mlight::map_features(features, 6);

    mlight::LogisticRegression lr;

    timer.tic();
    lr.fit(features, labels, mlight::OptimizerLBFGS(), 0.1);
    double cost = lr.cost(features, labels);
    double score = lr.score(features, labels);
    mlight::pformat("[chips] Logistic regression (gradient): (train time: %.2f ms, score: %.6f, cost: %.2e)", timer.toc() * 1000, score, cost);
    BOOST_REQUIRE_GE(score, 0.83);
}


BOOST_AUTO_TEST_CASE(LogisticRegressionTestPima)
{
    arma::wall_clock timer;
    arma::mat features, labels;
    arma::mat train_features, train_labels, test_features, test_labels;
    mlight::split_dataset(mlight::load_mat("pima-indians-diabetes.data"), features, labels);
    labels = mlight::map_labels(labels);
    mlight::split_dataset(features, labels, train_features, train_labels, test_features, test_labels);

    mlight::LogisticRegression clf;

    timer.tic();
    clf.fit(train_features, train_labels, mlight::OptimizerLBFGS(), 0.0);
    double score = clf.score(test_features, test_labels);
    mlight::pformat("[pima] Logist regression classifier: (train time: %.2f ms, score: %.6f)", timer.toc() * 1000, score);

    BOOST_REQUIRE_GE(score, 0.80);
}

BOOST_AUTO_TEST_CASE(LogisticRegressionTestDigits)
{
    arma::wall_clock timer;
    arma::mat features, labels;
    mlight::split_dataset(mlight::load_mat("ex3data1.txt"), features, labels);
    labels = mlight::map_labels(labels);

    mlight::LogisticRegression clf;

    timer.tic();
    clf.fit(features, labels, mlight::OptimizerGD(0.1, 1e-1), 0.1);
    //clf.fit(features, labels, mlight::OptimizerLBFGS(10, 1e-1, 50), 0.1);
    //clf.fit(features, labels, mlight::OptimizerGDCheck(0.1, 1e-1), 0.1);
    double score = clf.score(features, labels);
    mlight::pformat("[digits] Logist regression classifier: (train time: %.2f ms, score: %.6f)", timer.toc() * 1000, score);

    BOOST_REQUIRE_GE(score, 0.96);
}

BOOST_AUTO_TEST_SUITE_END();

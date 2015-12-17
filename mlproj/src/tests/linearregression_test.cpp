#include <mlight/linearregression.hpp>
#include <mlight/optimizers.hpp>
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(LinearRegressionTest);

BOOST_AUTO_TEST_CASE(LinearRegressionTestCase)
{
    arma::wall_clock timer;
    arma::mat features, labels;
    mlight::split_dataset(mlight::load_mat("ex1data2.txt"), features, labels);

    //features = mlight::map_features(features, 3);

    mlight::LinearRegression lr;

    timer.tic();
    lr.fit(features, labels, mlight::OptimizerCG(1e-7, 100));
    double cost = lr.cost(features, labels);
    double score = lr.score(features, labels);
    mlight::pformat("[houses] Linear regression (gradient): (train time: %.2f ms, score: %.6f, cost: %.2e)", timer.toc() * 1000, score, cost);
    BOOST_REQUIRE_CLOSE(score, 0.73295, 1e-3);
    BOOST_REQUIRE_CLOSE(cost, 2.0433e+09, 1e-2);

    timer.tic();
    lr.fitNormal(features, labels);
    cost = lr.cost(features, labels);
    score = lr.score(features, labels);
    mlight::pformat("[houses] Linear regression (normal): (train time: %.2f ms, score: %.6f, cost: %.2e)", timer.toc() * 1000, score, cost);
    BOOST_REQUIRE_CLOSE(score, 0.73295, 1e-3);
    BOOST_REQUIRE_CLOSE(cost, 2.0433e+09, 1e-2);
}

BOOST_AUTO_TEST_SUITE_END();

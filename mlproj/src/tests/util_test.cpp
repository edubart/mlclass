#include <mlight/prereqs.hpp>
#include <boost/test/unit_test.hpp>

using mlight::dump;

BOOST_AUTO_TEST_SUITE(UtilTest);

BOOST_AUTO_TEST_CASE(BinomTest)
{
    BOOST_REQUIRE_EQUAL(mlight::binom(0,0),1);
    BOOST_REQUIRE_EQUAL(mlight::binom(0,1),1);
    BOOST_REQUIRE_EQUAL(mlight::binom(1,1),1);
    BOOST_REQUIRE_EQUAL(mlight::binom(1,0),1);
    BOOST_REQUIRE_EQUAL(mlight::binom(2,2),1);
    BOOST_REQUIRE_EQUAL(mlight::binom(2,1),2);
    BOOST_REQUIRE_EQUAL(mlight::binom(3,2),3);
    BOOST_REQUIRE_EQUAL(mlight::binom(4,1),4);
    BOOST_REQUIRE_EQUAL(mlight::binom(4,2),6);
    BOOST_REQUIRE_EQUAL(mlight::binom(5,2),10);
}

BOOST_AUTO_TEST_CASE(MapLabelsTest)
{
    arma::vec unmapped_labels = {1,1,0,0,2,3,2,1,4,5,5,5,10};
    arma::mat mapped_labels = mlight::map_labels(unmapped_labels);
    arma::mat expected_labels = {
        {1,0,0,0,0,0},
        {1,0,0,0,0,0},
        {0,0,0,0,0,0},
        {0,0,0,0,0,0},
        {0,1,0,0,0,0},
        {0,0,1,0,0,0},
        {0,1,0,0,0,0},
        {1,0,0,0,0,0},
        {0,0,0,1,0,0},
        {0,0,0,0,1,0},
        {0,0,0,0,1,0},
        {0,0,0,0,1,0},
        {0,0,0,0,0,1},
    };
    BOOST_CHECK(arma::accu(mapped_labels != expected_labels) == 0);
}

BOOST_AUTO_TEST_SUITE_END();

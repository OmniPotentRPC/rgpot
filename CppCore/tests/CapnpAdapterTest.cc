#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <capnp/message.h>
#include "rgpot/types/adapters/capnp/capnp_adapter.hpp"

TEST_CASE("CapnpAdapter: Position Conversion", "[rpc][adapter]") {
    size_t numAtoms = 2;
    rgpot::types::AtomMatrix nativePos(numAtoms, 3);
    nativePos(0, 0) = 1.0; nativePos(0, 1) = 2.0; nativePos(0, 2) = 3.0;
    nativePos(1, 0) = 4.0; nativePos(1, 1) = 5.0; nativePos(1, 2) = 6.0;

    ::capnp::MallocMessageBuilder message;
    auto builder = message.initRoot<::capnp::List<double>>(numAtoms * 3);
    rgpot::types::adapt::capnp::populatePositionsToCapnp(builder, nativePos);

    auto reader = builder.asReader();
    auto convertedPos = rgpot::types::adapt::capnp::convertPositionsFromCapnp(reader, numAtoms);

    REQUIRE(convertedPos.rows() == 2);
    
    // Check Row 0
    REQUIRE(convertedPos(0, 0) == Catch::Approx(1.0));
    REQUIRE(convertedPos(0, 1) == Catch::Approx(2.0));
    REQUIRE(convertedPos(0, 2) == Catch::Approx(3.0));

    // Check Row 1
    REQUIRE(convertedPos(1, 0) == Catch::Approx(4.0));
    REQUIRE(convertedPos(1, 1) == Catch::Approx(5.0));
    REQUIRE(convertedPos(1, 2) == Catch::Approx(6.0));
}

TEST_CASE("CapnpAdapter: Box Matrix Conversion", "[rpc][adapter]") {
    std::array<std::array<double, 3>, 3> nativeBox = {{
        {10.0, 0.0, 0.0},
        {0.0, 20.0, 0.0},
        {0.0, 0.0, 30.0}
    }};

    ::capnp::MallocMessageBuilder message;
    auto builder = message.initRoot<::capnp::List<double>>(9);
    rgpot::types::adapt::capnp::populateBoxMatrixToCapnp(builder, nativeBox);

    auto reader = builder.asReader();
    auto convertedBox = rgpot::types::adapt::capnp::convertBoxMatrixFromCapnp(reader);

    REQUIRE(convertedBox[0][0] == Catch::Approx(10.0));
    REQUIRE(convertedBox[1][1] == Catch::Approx(20.0));
    REQUIRE(convertedBox[2][2] == Catch::Approx(30.0));
    REQUIRE(convertedBox[0][1] == Catch::Approx(0.0));
}

TEST_CASE("CapnpAdapter: Atom Types Conversion", "[rpc][adapter]") {
    std::vector<int> atoms = {29, 1};
    
    ::capnp::MallocMessageBuilder message;
    auto builder = message.initRoot<::capnp::List<int>>(2);
    rgpot::types::adapt::capnp::populateAtomNumbersToCapnp(builder, atoms);

    auto reader = builder.asReader();
    auto converted = rgpot::types::adapt::capnp::convertAtomNumbersFromCapnp(reader);

    REQUIRE(converted.size() == 2);
    REQUIRE(converted[0] == 29);
    REQUIRE(converted[1] == 1);
}

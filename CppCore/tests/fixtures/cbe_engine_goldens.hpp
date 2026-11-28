#pragma once
// MIT License
// Copyright 2023--present rgpot developers
//
// Bit goldens of the CBE engine, in the engine's units (Angstrom, eV,
// eV/Angstrom per Angstrom^k). The inputs are the Bohr geometry and direction
// of the potlibc ch52008 jet goldens, scaled to Angstrom; every output is the exact
// double potlibc's ch52008 returns through CBEPot's unit conversion on x86-64
// Linux with glibc libm. The bit comparison is compiled in only when meson
// sees that host; other hosts keep a tolerance check. kSaddle is the saddle
// point of the H + CH4 surface,
// kDisplaced a geometry 0.05 Bohr away along a fixed pattern. Each Jet array
// holds the derivative orders 1, 2 and 3 in blocks of 18 values. A kernel
// evaluation costs the same number of forward-mode operations at every
// geometry.

#include <array>

namespace cbe_goldens {

// kSaddle
const std::array<double, 18> kSaddlePositions = {
    0x1.53be527a5b7bp+0, 0x1.9c5dfabfe6a6ap+0,
    0x1.60c1a918d431dp+0, 0x1.2cc4c032dab7dp-1,
    0x1.70d7fa3d9520ep-1, 0x1.37342f358371fp-1,
    0x1.e1648fdf0392fp-8, 0x1.c4370abac57fdp-4,
    0x1.4ceb0c5d218d9p+0, 0x1.4d69ce799ee69p+0,
    0x1.ff3ceac371c5ep-4, 0x1.475308411b06ap-5,
    -0x1.a2fc6f9a90073p-5, 0x1.51d20478b9b54p+0,
    -0x1.534600ba5ae2bp-5, 0x1.ca9ca0ac24cb3p+0,
    0x1.15ba90cb7ecf8p+1, 0x1.dc84b2b41fe8cp+0,
};
const std::array<double, 18> kSaddleDirection = {
    0x1.02d670b01700ep-1, -0x1.fa526fd63d62ep-8,
    -0x1.0711f6ed96d1dp-2, -0x1.09a94a56628c9p-2,
    0x1.80031e3bdb1f7p-3, 0x1.d71b5e45fdec9p-3,
    -0x1.07f1327f5734cp-3, -0x1.0edb0651d6334p-2,
    -0x1.3bafaa71a7accp-4, 0x1.c943f41477c75p-2,
    0x1.927aa23d1d40ep-3, -0x1.e3e154df91b59p-4,
    -0x1.09f54d6b080cdp-1, -0x1.55124f772079bp-6,
    0x1.fd1b7c63419fbp-3, 0x1.3b015413360d9p-2,
    -0x1.5494906967e8p-3, -0x1.f137018644e49p-3,
};
const double kSaddleEnergy = 0x1.4d193000e0f8ap-1;
const std::array<double, 18> kSaddleForce = {
    0x1.2d9e6269d01e9p-33, 0x1.ae2c0b68a1628p-32,
    0x1.7169c6d0e6b9ep-35, -0x1.8e4b31766c86fp-32,
    -0x1.2ed9e31ff9869p-30, 0x1.54624c19c92d6p-32,
    -0x1.463aabfa81993p-34, 0x1.aa5eafb95eb55p-35,
    -0x1.8186dfb87c945p-32, -0x1.841ef73537ad1p-34,
    0x1.b2843264a2ba2p-33, -0x1.eddde2f475303p-34,
    0x1.4b296c08fa62dp-32, 0x1.045073064c706p-32,
    0x1.43ae5e95a0a6cp-33, 0x1.7ba2cd7e54b29p-34,
    0x1.f4bab02183195p-33, -0x1.3b43b027ea05cp-35,
};
const std::array<double, 18> kSaddleHvp = {
    0x1.34345c0f32d54p+1, -0x1.664a6f3a0351ap-2,
    -0x1.570005e7c78dcp+0, -0x1.b941ed3fe93a4p+3,
    0x1.9df7609dbac1cp+3, 0x1.ba9b7b770d8adp+3,
    -0x1.9f73716bc984ep-1, -0x1.9edd9b53ce24ep-2,
    -0x1.6f25d09b1731fp+0, 0x1.e0ba4644b079p+3,
    -0x1.5a09950761924p+3, -0x1.3b8d007abb479p+3,
    -0x1.6d96f2c7c354dp+0, -0x1.ca30ba22e7893p-4,
    -0x1.4309e95d7952cp-2, -0x1.66dad4c6f78c8p+0,
    -0x1.41814e6d26a5fp+0, -0x1.c3170e10aa067p-1,
};
const std::array<double, 54> kSaddleJet = {
    0x1.34345c0f32d54p+1, -0x1.664a6f3a0351ap-2,
    -0x1.570005e7c78dcp+0, -0x1.b941ed3fe93a4p+3,
    0x1.9df7609dbac1cp+3, 0x1.ba9b7b770d8adp+3,
    -0x1.9f73716bc984ep-1, -0x1.9edd9b53ce24ep-2,
    -0x1.6f25d09b1731fp+0, 0x1.e0ba4644b079p+3,
    -0x1.5a09950761924p+3, -0x1.3b8d007abb479p+3,
    -0x1.6d96f2c7c354dp+0, -0x1.ca30ba22e7893p-4,
    -0x1.4309e95d7952cp-2, -0x1.66dad4c6f78c8p+0,
    -0x1.41814e6d26a5fp+0, -0x1.c3170e10aa067p-1,
    -0x1.487d62998515dp+3, -0x1.310ecce722f03p+2,
    -0x1.74adca7fe021fp+1, 0x1.ab21fa34280cfp+5,
    -0x1.7400fa63af8bap+5, -0x1.39f25221d3de6p+5,
    -0x1.3555bb2552e17p+2, -0x1.8475cacbbcedbp+1,
    0x1.695867695a9aap+2, -0x1.3189c89c1049ep+5,
    0x1.9000ac262f548p+5, 0x1.12e39c2236304p+5,
    -0x1.a9503a5c249ecp+1, 0x1.163c9739dd6f2p-1,
    -0x1.2535f5e723bfcp-1, 0x1.9c6e218b627ffp+1,
    0x1.e10922a38ee84p+1, 0x1.5c35d920cebb6p+1,
    0x1.961307605b404p-1, 0x1.fc03ec511b62bp+3,
    0x1.6cd6f9fac84cp+4, -0x1.0c4903addf0dap+4,
    0x1.41a7f6c049638p+7, 0x1.366fc8e7e6129p+5,
    0x1.68d6883e05cbdp+2, -0x1.263f2844689fep+3,
    0x1.265dd46319af3p+1, 0x1.43c79d8c6f5e6p+3,
    -0x1.3bfb37d5d1ad8p+7, -0x1.d91013833f0d9p+5,
    0x1.979846070bec1p+2, -0x1.112bce36b46c7p+1,
    -0x1.5b1c1e4eeb39p-2, -0x1.899c5b927fa72p+2,
    -0x1.d88b7e4d02143p+2, -0x1.1bd6bb5cf776ap+2,
};
const long long kSaddleOperations = 17172;
const double kSaddleScale = 0x1.41a7f6c049638p+7;
const double kSaddleOrderOneScale = 0x1.e0ba4644b079p+3;
// kDisplaced
const std::array<double, 18> kDisplacedPositions = {
    0x1.5971708332988p+0, 0x1.a31588ba2f879p+0,
    0x1.6554eb44a1f11p+0, 0x1.2d54f402e2203p-1,
    0x1.688e0b70a85d5p-1, 0x1.29f618343ecc5p-1,
    -0x1.06a191b5f844ep-6, 0x1.9bb2291a429dfp-4,
    0x1.4f07437337476p+0, 0x1.532c7868c8c64p+0,
    0x1.353ae605c8f81p-3, 0x1.d706094184ce3p-5,
    -0x1.9d9db0de663d3p-5, 0x1.4d9623d93b8eap+0,
    -0x1.13f2384de7c8dp-4, 0x1.c4ae962ee641fp+0,
    0x1.1483faba4b6cfp+1, 0x1.debc8a93faa79p+0,
};
const std::array<double, 18> kDisplacedDirection = {
    0x1.02d670b01700ep-1, -0x1.fa526fd63d62ep-8,
    -0x1.0711f6ed96d1dp-2, -0x1.09a94a56628c9p-2,
    0x1.80031e3bdb1f7p-3, 0x1.d71b5e45fdec9p-3,
    -0x1.07f1327f5734cp-3, -0x1.0edb0651d6334p-2,
    -0x1.3bafaa71a7accp-4, 0x1.c943f41477c75p-2,
    0x1.927aa23d1d40ep-3, -0x1.e3e154df91b59p-4,
    -0x1.09f54d6b080cdp-1, -0x1.55124f772079bp-6,
    0x1.fd1b7c63419fbp-3, 0x1.3b015413360d9p-2,
    -0x1.5494906967e8p-3, -0x1.f137018644e49p-3,
};
const double kDisplacedEnergy = 0x1.59a847f6191dap-1;
const std::array<double, 18> kDisplacedForce = {
    0x1.9a0f4f64fbf38p-4, 0x1.a542edda54156p-3,
    0x1.652bfdd377384p-3, -0x1.84ebd7515632ep+0,
    -0x1.6beba36c295fdp-4, 0x1.eb8fcd0ad54e5p-1,
    0x1.3214643192a1dp-1, 0x1.b558ea612afecp-2,
    -0x1.51e6c33235c71p-1, 0x1.5566e47de4447p-1,
    -0x1.5a7cf4a646b56p-1, -0x1.1756301b10d14p-1,
    0x1.8471886862f0bp-5, 0x1.39fa53864f501p-8,
    -0x1.0c6b545f2e31dp-5, 0x1.b69b1c007e8f5p-4,
    0x1.06250f1652f78p-3, 0x1.a946e09c32ecp-4,
};
const std::array<double, 18> kDisplacedHvp = {
    0x1.44b91c763cbf2p+1, 0x1.a3b46469d001bp-1,
    0x1.e3df4b315256cp-6, -0x1.14d76b58345f7p+4,
    0x1.e6c6fdb49a85p+3, 0x1.eee4e50eabf2ap+3,
    -0x1.a59de2f3e28cfp-2, -0x1.1726cee4aff8ep-1,
    -0x1.bb790b586b512p+0, 0x1.283db77dba76ap+4,
    -0x1.a78b3cad7c19bp+3, -0x1.77456eebd4a5ep+3,
    -0x1.4e7fe901920b2p+0, -0x1.ec80722453c49p-4,
    -0x1.186ed8e40bcfcp-2, -0x1.03f7ccc32820fp+1,
    -0x1.10ae65ec9f126p+1, -0x1.c2f66cb211694p+0,
};
const std::array<double, 54> kDisplacedJet = {
    0x1.44b91c763cbf2p+1, 0x1.a3b46469d001bp-1,
    0x1.e3df4b315256cp-6, -0x1.14d76b58345f7p+4,
    0x1.e6c6fdb49a85p+3, 0x1.eee4e50eabf2ap+3,
    -0x1.a59de2f3e28cfp-2, -0x1.1726cee4aff8ep-1,
    -0x1.bb790b586b512p+0, 0x1.283db77dba76ap+4,
    -0x1.a78b3cad7c19bp+3, -0x1.77456eebd4a5ep+3,
    -0x1.4e7fe901920b2p+0, -0x1.ec80722453c49p-4,
    -0x1.186ed8e40bcfcp-2, -0x1.03f7ccc32820fp+1,
    -0x1.10ae65ec9f126p+1, -0x1.c2f66cb211694p+0,
    -0x1.f785b02652054p+2, -0x1.73113ffe8dce6p+1,
    -0x1.bc882a9ed9598p-1, 0x1.f1a497853088dp+5,
    -0x1.ceea435f16efdp+5, -0x1.5d5907d2d2766p+5,
    -0x1.f093ce5838876p+1, -0x1.092158f091c56p+1,
    0x1.2a647ee6be127p+2, -0x1.89e1f1d3cf1d5p+5,
    0x1.dccf41eb6c60ap+5, 0x1.3da2beacebef8p+5,
    -0x1.bce4003408cb8p+1, 0x1.d9a3cc7cda2b5p-1,
    -0x1.8540149b0e75dp-1, 0x1.2058d3c2cea11p+1,
    0x1.2779bd0a91f6ep+1, 0x1.dc36917d98f14p-1,
    0x1.749bca46cf0f6p+1, 0x1.0028fedcd15f7p+4,
    0x1.63aca6eea673bp+4, -0x1.3df28e591ac25p+5,
    0x1.8ddb108b78b29p+7, 0x1.a60d1cbbdd3c2p+5,
    0x1.c20ff7d607851p+1, -0x1.d3a129726da33p+2,
    0x1.abbe292783326p+1, 0x1.05271d3a4eba7p+5,
    -0x1.89df20de1b565p+7, -0x1.203a3b46b7ccdp+6,
    0x1.80ec461c22246p+2, -0x1.93ac8132946dbp+1,
    0x1.d221a81bcbc34p-3, -0x1.55e69e342d316p+2,
    -0x1.e2aa8713392f8p+2, -0x1.9fe7ef02a6dcap+2,
};
const long long kDisplacedOperations = 17172;
const double kDisplacedScale = 0x1.8ddb108b78b29p+7;
const double kDisplacedOrderOneScale = 0x1.283db77dba76ap+4;

} // namespace cbe_goldens

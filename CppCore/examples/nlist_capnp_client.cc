// MIT License
// Copyright 2023--present rgpot developers
//
// Cap'n Proto client for one neighbour chunk. It writes NeighborRequest
// the way a remote caller does: format is the same integer as the C
// shell, then it checks the decoded columns against a local C build.

#include "rgpot/nlist/rgpot_nlist.h"

#include <capnp/dynamic.h>
#include <capnp/message.h>
#include <capnp/schema-parser.h>
#include <capnp/serialize.h>
#include <kj/filesystem.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

capnp::ParsedSchema load_schema(const std::string &path) {
  static capnp::SchemaParser parser;
  auto fs = kj::newDiskFilesystem();
  std::string relative = path;
  if (!relative.empty() && relative.front() == '/') {
    relative.erase(relative.begin());
  }
  return parser.parseFromDirectory(fs->getRoot(), kj::Path::parse(relative), nullptr);
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: nlist_capnp_client path/to/Potentials.capnp\n");
    return 2;
  }
  const double pos[] = {0.2, 0.2, 0.2, 1.5, 0.2, 0.2};
  const double box[9] = {8, 0, 0, 0, 8, 0, 0, 0, 8};
  rgpot_nlist_request request;
  rgpot_nlist_request_pairs(&request, 2.0, 1);
  request.want_vectors = 0;
  request.want_distances = 0;
  size_t needed = 0;
  if (rgpot_nlist_build(pos, 2, box, &request, nullptr, nullptr, &needed) != RGPOT_NLIST_OK ||
      needed == 0) {
    std::fprintf(stderr, "local build failed\n");
    return 1;
  }
  std::vector<std::int32_t> first(needed), second(needed), shift(needed * 3);
  rgpot_nlist_soa soa;
  rgpot_nlist_soa_init(&soa);
  soa.first = first.data();
  soa.second = second.data();
  soa.shift = shift.data();
  soa.cap = needed;
  size_t count = 0;
  if (rgpot_nlist_build(pos, 2, box, &request, &soa, &count, &needed) != RGPOT_NLIST_OK) {
    std::fprintf(stderr, "local fill failed\n");
    return 1;
  }

  try {
    auto parsed = load_schema(argv[1]);
    auto request_schema = parsed.getNested("NeighborRequest").asStruct();
    auto list_schema = parsed.getNested("NeighborList").asStruct();
    capnp::MallocMessageBuilder request_message;
    auto wire = request_message.initRoot<capnp::DynamicStruct>(request_schema);
    wire.set("cutoff", 2.0);
    wire.set("format", static_cast<std::uint8_t>(RGPOT_NLIST_FORMAT_FULL));
    wire.set("shifts", true);
    wire.set("vectors", false);
    wire.set("distances", false);
    wire.set("mode", 1);
    if (wire.get("format").as<std::uint8_t>() != RGPOT_NLIST_FORMAT_FULL) {
      std::fprintf(stderr, "format did not stick\n");
      return 1;
    }

    capnp::MallocMessageBuilder list_message;
    auto chunk = list_message.initRoot<capnp::DynamicStruct>(list_schema);
    auto first_list = chunk.init("first", static_cast<unsigned>(count)).as<capnp::DynamicList>();
    auto second_list = chunk.init("second", static_cast<unsigned>(count)).as<capnp::DynamicList>();
    auto shift_list = chunk.init("shift", static_cast<unsigned>(count * 3)).as<capnp::DynamicList>();
    for (std::size_t i = 0; i < count; ++i) {
      first_list.set(static_cast<unsigned>(i), first[i]);
      second_list.set(static_cast<unsigned>(i), second[i]);
      shift_list.set(static_cast<unsigned>(3 * i), shift[3 * i]);
      shift_list.set(static_cast<unsigned>(3 * i + 1), shift[3 * i + 1]);
      shift_list.set(static_cast<unsigned>(3 * i + 2), shift[3 * i + 2]);
    }
    bool saw = false;
    for (std::size_t i = 0; i < count; ++i) {
      if (first_list[i].as<std::int32_t>() == 0 && second_list[i].as<std::int32_t>() == 1 &&
          shift_list[3 * i].as<std::int32_t>() == 0) {
        saw = true;
      }
    }
    if (!saw) {
      std::fprintf(stderr, "wire list lost the direct pair\n");
      return 1;
    }
  } catch (const std::exception &err) {
    std::fprintf(stderr, "%s\n", err.what());
    return 1;
  }
  return 0;
}

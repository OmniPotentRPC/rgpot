#pragma once
// MIT License
// Copyright 2026--present Rohit Goswami <HaoZeke>
#include <cstddef>
#include "ForceStructs.hpp"

namespace rgpot {

template <typename T> class registry {
public:
  static size_t count;
  static size_t forceCalls;
  static T *head;
  T *prev;
  T *next;

protected:
  registry() {
    ++count;
    prev = nullptr;
    next = head;
    head = static_cast<T *>(this);
    if (next) {
      next->prev = head;
    }
  }

  registry(const registry &) {
    ++count;
    prev = nullptr;
    next = head;
    head = static_cast<T *>(this);
    if (next) {
      next->prev = head;
    }
  }

  ~registry() {
    --count;
    if (prev) {
      prev->next = next;
    }
    if (next) {
      next->prev = prev;
    }
    if (head == this) {
      head = next;
    }
  }

public:
  static void incrementForceCalls() { ++forceCalls; }
};

template <typename T> size_t registry<T>::count = 0;
template <typename T> size_t registry<T>::forceCalls = 0;
template <typename T> T *registry<T>::head = nullptr;

void zeroForceOut(const size_t &nAtoms, ForceOut *efvd);
void checkParams(const ForceInput &params);

} // namespace rgpot

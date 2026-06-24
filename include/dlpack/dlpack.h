#pragma once

// Torch vendors the upstream DLPack header under ATen/. rgpot's public C ABI
// expects the canonical <dlpack/dlpack.h> include path, so provide a thin
// compatibility shim when the standalone DLPack headers are not packaged.
#include <ATen/dlpack.h>

// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/plugin/PluginRegistry.hpp"
#include "rgpot/plugin/DynLib.hpp"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>

namespace fs = std::filesystem;

// Forward declarations of builtin plugin init functions
extern "C" const rgpot_plugin_descriptor_t *rgpot_plugin_init(void);

namespace rgpot::plugin {

PluginRegistry &PluginRegistry::instance() {
  static PluginRegistry reg;
  return reg;
}

// Called once by Meyer's singleton above
PluginRegistry::PluginRegistry() {
  // Register compiled-in builtins
  register_builtin(rgpot_plugin_init()); // LJ (always available)
}

PluginRegistry::~PluginRegistry() {
  for (auto &p : m_plugins) {
    if (p.handle) {
      dynlib::close(static_cast<dynlib::Handle>(p.handle));
    }
  }
}

void PluginRegistry::discover() {
  const char *env = std::getenv("RGPOT_PLUGIN_PATH");
  if (!env)
    return;

#ifdef _WIN32
  const char delim = ';';
#else
  const char delim = ':';
#endif

  std::istringstream stream(env);
  std::string dir;
  while (std::getline(stream, dir, delim)) {
    if (!dir.empty()) {
      scan_directory(dir);
    }
  }
}

void PluginRegistry::scan_directory(const std::string &dir) {
  if (!fs::is_directory(dir))
    return;

  const std::string ext = dynlib::so_extension();
  for (const auto &entry : fs::directory_iterator(dir)) {
    if (!entry.is_regular_file())
      continue;
    const auto &path = entry.path();
    if (path.extension() != ext)
      continue;
    load_plugin(path.string());
  }
}

bool PluginRegistry::load_plugin(const std::string &path) {
  auto handle = dynlib::open(path.c_str());
  if (!handle) {
    std::cerr << "[rgpot] failed to load " << path << ": " << dynlib::error()
              << "\n";
    return false;
  }

  auto init_fn =
      dynlib::loadSym<rgpot_plugin_init_fn>(handle, "rgpot_plugin_init");
  if (!init_fn) {
    dynlib::close(handle);
    return false; // not a plugin, silently skip
  }

  const auto *desc = init_fn();
  if (!desc) {
    std::cerr << "[rgpot] " << path << ": rgpot_plugin_init returned NULL\n";
    dynlib::close(handle);
    return false;
  }

  if (desc->api_version != RGPOT_PLUGIN_API_VERSION) {
    std::cerr << "[rgpot] " << path << ": API version mismatch (plugin="
              << desc->api_version
              << ", host=" << RGPOT_PLUGIN_API_VERSION << ")\n";
    dynlib::close(handle);
    return false;
  }

  // Check minimum descriptor size (all required fields must be present)
  size_t min_size = offsetof(rgpot_plugin_descriptor_t, capabilities);
  if (desc->descriptor_size < min_size) {
    std::cerr << "[rgpot] " << path << ": descriptor too small\n";
    dynlib::close(handle);
    return false;
  }

  if (!desc->name || !desc->create || !desc->destroy || !desc->calculate) {
    std::cerr << "[rgpot] " << path << ": missing required fields\n";
    dynlib::close(handle);
    return false;
  }

  // Check for name collision
  for (const auto &existing : m_plugins) {
    if (existing.desc->name == std::string(desc->name)) {
      std::cerr << "[rgpot] " << path << ": plugin '" << desc->name
                << "' already registered from " << existing.path << "\n";
      dynlib::close(handle);
      return false;
    }
  }

  m_plugins.push_back({static_cast<void *>(handle), desc, path});
  return true;
}

void PluginRegistry::register_builtin(const rgpot_plugin_descriptor_t *desc) {
  if (!desc || !desc->name)
    return;
  m_plugins.push_back({nullptr, desc, ""});
}

std::vector<std::string> PluginRegistry::list_plugins() const {
  std::vector<std::string> names;
  names.reserve(m_plugins.size());
  for (const auto &p : m_plugins) {
    names.emplace_back(p.desc->name);
  }
  return names;
}

const LoadedPlugin *PluginRegistry::find(const std::string &name) const {
  for (const auto &p : m_plugins) {
    if (p.desc->name == name) {
      return &p;
    }
  }
  return nullptr;
}

} // namespace rgpot::plugin

#pragma once
// MIT License
// Copyright 2023--present rgpot developers

#include <rgpot/plugin.h>

#include <string>
#include <vector>

namespace rgpot::plugin {

struct LoadedPlugin {
  void *handle;                          // dynlib handle (NULL for builtins)
  const rgpot_plugin_descriptor_t *desc; // descriptor from plugin_init
  std::string path;                      // file path (empty for builtins)
};

class PluginRegistry {
public:
  static PluginRegistry &instance();

  /// Scan RGPOT_PLUGIN_PATH and load all valid plugins.
  void discover();

  /// Load a single plugin from a shared library path.
  bool load_plugin(const std::string &path);

  /// Register a compiled-in plugin (no dlopen).
  void register_builtin(const rgpot_plugin_descriptor_t *desc);

  /// List registered plugin names.
  std::vector<std::string> list_plugins() const;

  /// Find a plugin by name.
  const LoadedPlugin *find(const std::string &name) const;

  PluginRegistry(const PluginRegistry &) = delete;
  PluginRegistry &operator=(const PluginRegistry &) = delete;

private:
  PluginRegistry();
  ~PluginRegistry();

  void scan_directory(const std::string &dir);

  std::vector<LoadedPlugin> m_plugins;
};

} // namespace rgpot::plugin

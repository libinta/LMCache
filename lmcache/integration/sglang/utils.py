# SPDX-License-Identifier: Apache-2.0
# Standard
import os

# First Party
from lmcache.logging import init_logger
from lmcache.v1.config import LMCacheEngineConfig

logger = init_logger(__name__)
ENGINE_NAME = "sglang-instance"


def is_false(value: str) -> bool:
    """Check if the given string value is equivalent to 'false'."""
    return value.lower() in ("false", "0", "no", "n", "off")


def lmcache_get_config(config_file: str = "") -> LMCacheEngineConfig:
    """Load the LMCache configuration.

    Args:
        config_file: Path to a YAML configuration file, or empty string
            to build the config from ``LMCACHE_*`` environment variables.

    Returns:
        A validated ``LMCacheEngineConfig``.
    """
    resolved_config_file = config_file or os.getenv("LMCACHE_CONFIG_FILE", "")

    if resolved_config_file:
        logger.info(f"Loading LMCache config file {resolved_config_file}")
        config = LMCacheEngineConfig.from_file(resolved_config_file)
        # Keep precedence aligned with other integrations: env overrides file.
        config.update_config_from_env()
    else:
        config = LMCacheEngineConfig.from_env()
    config.validate()
    return config

#
# Copyright (c) 2026, Paxa Labs
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Paxa text-to-speech services for Pipecat."""

from ._version import __version__
from .tts import PaxaHttpTTSService, PaxaTTSService, PaxaTTSSettings

__all__ = ["PaxaHttpTTSService", "PaxaTTSService", "PaxaTTSSettings", "__version__"]

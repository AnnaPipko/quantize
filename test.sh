#!/bin/sh

exec env PYTHONPATH=build/release:build/release_private pytest

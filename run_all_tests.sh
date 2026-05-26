#!/bin/bash

time coverage run --branch --source=src/MPP -m unittest_parallel --level test --coverage-branch --coverage-html htmlcov

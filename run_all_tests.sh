#!/bin/bash

time coverage run --branch --source=. -m unittest_parallel --level test --coverage-branch --coverage-html htmlcov

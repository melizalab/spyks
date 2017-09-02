#!/bin/bash

for mdl in models/*.yml; do
    spykscc "$mdl"
done

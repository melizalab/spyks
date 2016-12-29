# -*- coding: utf-8 -*-
# -*- mode: python -*-


def constructor(model):
    s = ["{}(params[{}])".format(n, i) for i,(n,v) in enumerate(model["parameters"])]
    return ", ".join(s)

def fields(model):
    return ", ".join(n for n,v in model["parameters"])

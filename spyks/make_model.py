# -*- coding: utf-8 -*-
# -*- mode: python -*-

nakl_params = ["C", "gna", "Ena", "gk", "Ek", "gl", "El", "vm", "dvm", "tm0", "tm1", "vmt", "dvmt", "vh", "dvh",
               "th0", "th1", "vht", "dvht", "vn", "dvn", "tn0", "tn1", "vnt", "dvnt", "Isa"]



def make_constructor(param_names):
    s = ["{}(params[{}])".format(p, i) for i,p in enumerate(param_names)]
    return ", ".join(s)

# -*- coding: utf-8 -*-
# -*- mode: python -*-

nakl_params = ["C", "gna", "Ena", "gk", "Ek", "gl", "El",
               "vm", "dvm", "tm0", "tm1", "vmt", "dvmt",
               "vh", "dvh", "th0", "th1", "vht", "dvht",
               "vn", "dvn", "tn0", "tn1", "vnt", "dvnt", "Isa"]

biocm_params = ["C", "g_na", "E_na", "g_k", "E_k", "g_l", "E_l",
                # Na parameters
                "vnam", "dvnam", "tnam0", "tnam1", "vnamt", "dvnamt",
                "vnah", "dvnah", "tnah0", "tnah1", "vnaht", "dvnaht",
                # KA parameters
                "vkam", "dvkam", "tkam0", "tkam1", "vkamt", "dvkamt",
                "vkah", "dvkah", "tkah0", "tkah1", "vkaht", "dvkaht",
                # KLT parameters - inactivation is a bit different
                "vkltm", "dvkltm", "tkltm0", "tkltm1", "vkltmt", "dvkltmt",

                # HCN parameters
                "vhcnh", "dvhcnh", "thcnh0", "thcnh1", "vhcnht", "dvhcnht",]


def make_constructor(param_names):
    s = ["{}(params[{}])".format(p, i) for i,p in enumerate(param_names)]
    return ", ".join(s)


def make_fields(param_names):
    return ", ".join(param_names)

# -*- coding: utf-8 -*-
#
# Hack around the limitation that C pointers cannot be passed to Python functions.
#
# http://grokbase.com/t/gg/cython-users/134b21rga8/passing-callback-pointers-to-python-and-back
#
# JJ 2016-02-29

cdef class PointerWrapper:
    cdef void* ptr
    cdef set_ptr(self, void * input)


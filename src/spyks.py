from __future__ import division, print_function
import cneurons
import numpy as np

class _neuron:
	model = None

	def apply_current(self,current,res):
		self.model.apply_current(np.asarray(current),res)

class hh(_neuron):
	 def __init__(self, C=1.0, gna=120.0, Ena=50.0, gk=20, Ek=-77.0, gl=0.3, 
	 	El=-54.4, vm=-40.0, dvm=15.0, tm0=0.1, tm1=0.4, vmt=-40.0, dvmt=15.0, 
	    vh=-60.0, dvh=-15.0, th0=1.0, th1=7.0, vht=-60, dvht=-15.0, vn=-55.0, 
	    dvn=30.0, tn0=1.0, tn1=5.0, vnt=-55.0, dvnt=30.0):

		self.model = cneurons.hh([gna,Ena,gk,Ek,gl,El,vm,dvm,tm0,tm1,vmt,dvmt,
								  vh,dvh,th0,th1,vht,dvht,vn,dvn,tn0,tn1,vnt,
								  dvnt,C]);
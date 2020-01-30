

from numpy import *

def forward_kinem_UR5(q):



	alpha1 = pi/2
	alpha2 = 0
	alpha3 = 0
	alpha4 = pi/2
	alpha5 = -pi/2
	alpha6 = 0

	d1 = 0.089159
	d2 = 0.0
	d3 = 0.0
	d4 = 0.10915
	d5 = 0.09465
	d6 = 0.0823

	a1 = 0.0
	a2 = -0.425
	a3 = -0.39225
	a4 = 0.0
	a5 = 0.0
	a6 = 0.0

	q1 = q[0]
	q2 = q[1]
	q3 = q[2]
	q4 = q[3]
	q5 = q[4]
	q6 = q[5]






	T1 = np.array([ [np.cos(q1), -np.sin(q1), 0.0, a1  ], [ np.sin(q1)*np.cos(alpha1), np.cos(q1)*np.cos(alpha1), -np.sin(alpha1), -d1*np.sin(alpha1)  ], [ np.sin(q1)*np.sin(alpha1), np.cos(q1)*np.sin(alpha1), np.cos(alpha1), d1*np.cos(alpha1)   ], [0.0, 0.0, 0.0, 1.0 ]   ])

	T2 = np.array([ [np.cos(q2), -np.sin(q2), 0.0, a2  ], [ np.sin(q2)*np.cos(alpha2), np.cos(q2)*np.cos(alpha2), -np.sin(alpha2), -d2*np.sin(alpha2)  ], [ np.sin(q2)*np.sin(alpha2), np.cos(q2)*np.sin(alpha2), np.cos(alpha2), d2*np.cos(alpha2)   ], [0.0, 0.0, 0.0, 1.0 ]   ])

	T3 = np.array([ [np.cos(q3), -np.sin(q3), 0.0, a3  ], [ np.sin(q3)*np.cos(alpha3), np.cos(q3)*np.cos(alpha3), -np.sin(alpha3), -d3*np.sin(alpha3)  ], [ np.sin(q3)*np.sin(alpha3), np.cos(q3)*np.sin(alpha3), np.cos(alpha3), d3*np.cos(alpha3)   ], [0.0, 0.0, 0.0, 1.0 ]   ])

	T4 = np.array([ [np.cos(q4), -np.sin(q4), 0.0, a4  ], [ np.sin(q4)*np.cos(alpha4), np.cos(q4)*np.cos(alpha4), -np.sin(alpha4), -d4*np.sin(alpha4)  ], [ np.sin(q4)*np.sin(alpha4), np.cos(q4)*np.sin(alpha4), np.cos(alpha4), d4*np.cos(alpha4)   ], [0.0, 0.0, 0.0, 1.0 ]   ])

	T5 = np.array([ [np.cos(q5), -np.sin(q5), 0.0, a5  ], [ np.sin(q5)*np.cos(alpha5), np.cos(q5)*np.cos(alpha5), -np.sin(alpha5), -d5*np.sin(alpha5)  ], [ np.sin(q5)*np.sin(alpha5), np.cos(q5)*np.sin(alpha5), np.cos(alpha5), d5*np.cos(alpha5)   ], [0.0, 0.0, 0.0, 1.0 ]   ])

	T6 = np.array([ [np.cos(q6), -np.sin(q6), 0.0, a6  ], [ np.sin(q6)*np.cos(alpha6), np.cos(q6)*np.cos(alpha6), -np.sin(alpha6), -d6*np.sin(alpha6)  ], [ np.sin(q6)*np.sin(alpha6), np.cos(q6)*np.sin(alpha6), np.cos(alpha6), d6*np.cos(alpha6)   ], [0.0, 0.0, 0.0, 1.0 ]   ])

	#T7 = np.array([ [np.cos(q7), -np.sin(q7), 0.0, a7  ], [ np.sin(q7)*np.cos(alpha7), np.cos(q7)*np.cos(alpha7), -np.sin(alpha7), -d7*np.sin(alpha7)  ], [ np.sin(q7)*np.sin(alpha7), np.cos(q7)*np.sin(alpha7), np.cos(alpha7), d7*np.cos(alpha7)   ], [0.0, 0.0, 0.0, 1.0 ]   ])

	#T8 = np.array([ [np.cos(q8), -np.sin(q8), 0.0, a8  ], [ np.sin(q8)*np.cos(alpha8), np.cos(q8)*np.cos(alpha8), -np.sin(alpha8), -d8*np.sin(alpha8)  ], [ np.sin(q8)*np.sin(alpha8), np.cos(q8)*np.sin(alpha8), np.cos(alpha8), d8*np.cos(alpha8)   ], [0.0, 0.0, 0.0, 1.0 ]   ])



	T_fin = T1.dot(T2).dot(T3).dot(T4).dot(T5).dot(T6)
	
	R1 = T1[0:3, 0:3]
	R2 = T2[0:3, 0:3]
	R3 = T3[0:3, 0:3]
	R4 = T4[0:3, 0:3]
	R5 = T5[0:3, 0:3]
	R6 = T6[0:3, 0:3]


	
	
		
	vec_x = T_fin[0][3]
	vec_y = T_fin[1][3]
	vec_z = T_fin[2][3]

	R_fin = T_fin[0:3, 0:3]

	trace_R_fin = trace(R_fin)
	

	return vec_x, vec_y, vec_z, R_fin



	






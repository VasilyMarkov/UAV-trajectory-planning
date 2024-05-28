import math
import matplotlib

linewidth=5
movie=False

def mod2pi(theta):
	return theta - 2.0 * math.pi * math.floor(theta / 2.0 / math.pi)


def pi_2_pi(angle):
	while(angle >= math.pi):
		angle = angle - 2.0 * math.pi

	while(angle <= -math.pi):
		angle = angle + 2.0 * math.pi

	return angle

def general_planner(planner, alpha, beta, d):
	sa = math.sin(alpha)
	sb = math.sin(beta)
	ca = math.cos(alpha)
	cb = math.cos(beta)
	c_ab = math.cos(alpha - beta)
	mode = list(planner)
	#print(mode)

	planner_uc = planner.upper()

	if planner_uc == 'LSL':
		tmp0 = d + sa - sb
		p_squared = 2 + (d * d) - (2 * c_ab) + (2 * d * (sa - sb))
		if p_squared < 0:
			return None
		tmp1 = math.atan2((cb - ca), tmp0)
		t = mod2pi(-alpha + tmp1)
		p = math.sqrt(p_squared)
		q = mod2pi(beta - tmp1)
		#  print(math.degrees(t), p, math.degrees(q))

	elif planner_uc == 'RSR':
		tmp0 = d - sa + sb
		p_squared = 2 + (d * d) - (2 * c_ab) + (2 * d * (sb - sa))
		if p_squared < 0:
			return None
		tmp1 = math.atan2((ca - cb), tmp0)
		t = mod2pi(alpha - tmp1)
		p = math.sqrt(p_squared)
		q = mod2pi(-beta + tmp1)

	elif planner_uc == 'LSR':
		p_squared = -2 + (d * d) + (2 * c_ab) + (2 * d * (sa + sb))
		if p_squared < 0:
			return None
		p = math.sqrt(p_squared)
		tmp2 = math.atan2((-ca - cb), (d + sa + sb)) - math.atan2(-2.0, p)
		t = mod2pi(-alpha + tmp2)
		q = mod2pi(-mod2pi(beta) + tmp2)

	elif planner_uc == 'RSL':
		p_squared = (d * d) - 2 + (2 * c_ab) - (2 * d * (sa + sb))
		if p_squared < 0:
			return None
		p = math.sqrt(p_squared)
		tmp2 = math.atan2((ca + cb), (d - sa - sb)) - math.atan2(2.0, p)
		t = mod2pi(alpha - tmp2)
		q = mod2pi(beta - tmp2)

	elif planner_uc == 'RLR':
		tmp_rlr = (6.0 - d * d + 2.0 * c_ab + 2.0 * d * (sa - sb)) / 8.0
		if abs(tmp_rlr) > 1.0:
			return None

		p = mod2pi(2 * math.pi - math.acos(tmp_rlr))
		t = mod2pi(alpha - math.atan2(ca - cb, d - sa + sb) + mod2pi(p / 2.0))
		q = mod2pi(alpha - beta - t + mod2pi(p))

	elif planner_uc == 'LRL':
		tmp_lrl = (6. - d * d + 2 * c_ab + 2 * d * (- sa + sb)) / 8.
		if abs(tmp_lrl) > 1:
			return None
		p = mod2pi(2 * math.pi - math.acos(tmp_lrl))
		t = mod2pi(-alpha - math.atan2(ca - cb, d + sa - sb) + p / 2.)
		q = mod2pi(mod2pi(beta) - alpha - t + mod2pi(p))

	else:
		print('bad planner:', planner)

	path = [t, p, q]

	# Lowercase directions are driven in reverse.
	for i in [0, 2]:
		if planner[i].islower():
			path[i] = (2 * math.pi) - path[i]
	# This will screw up whatever is in the middle.

	cost = sum(map(abs, path))

	return(path, mode, cost)
def dubins_path(start, end, radius):
	(sx, sy, syaw) = start
	(ex, ey, eyaw) = end
	c = radius

	ex = ex - sx
	ey = ey - sy

	lex = math.cos(syaw) * ex + math.sin(syaw) * ey
	ley = - math.sin(syaw) * ex + math.cos(syaw) * ey
	leyaw = eyaw - syaw
	D = math.sqrt(lex ** 2.0 + ley ** 2.0)
	d = D / c
	print('D:', D)

	theta = mod2pi(math.atan2(ley, lex))
	alpha = mod2pi(- theta)
	beta = mod2pi(leyaw - theta)

	#planners = ['RSr', 'rSR', 'rSr', 'LSL', 'RSR', 'LSR', 'RSL', 'RLR', 'LRL']
	planners = ['LSL', 'RSR', 'LSR', 'RSL', 'RLR', 'LRL']
	#planners = ['RSr']

	bcost = float("inf")
	bt, bp, bq, bmode = None, None, None, None

	for planner in planners:
		#t, p, q, mode = planner(alpha, beta, d)
		solution = general_planner(planner, alpha, beta, d)

		if solution is None:
			continue

		(path, mode, cost) = solution
		(t, p, q) = path
		if bcost > cost:
			# best cost
			bt, bp, bq, bmode = t, p, q, mode
			bcost = cost

	#  print(bmode)
	return(zip(bmode, [bt*c, bp*c, bq*c], [c] * 3))

